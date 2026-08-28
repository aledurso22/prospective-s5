"""Phase B24 -- interface-rank frontier. Determines the smallest
genuinely multidimensional temporal interface k for which width
becomes functionally useful while exact prospective credit remains
cheap.

Verified first (before any architecture work): k output "channels"
that are scalar multiples of ONE shared signal collapse EXACTLY to an
effective SISO system (b_eff = sum_j g_j*b_j) -- machine precision,
1.8e-15. This confirms k copies of the same broadcast (B23's mistake,
generalized) give zero benefit regardless of the nominal channel
count; genuine benefit requires channels carrying independent
information.

Architecture (Part C): n_l copies per layer, EACH with its own
k_in-dim input via a copy-specific mixing matrix V_l[q] (drawn from
the lower layer's full (n_{l-1}*k_out_{l-1})-dim output), processed by
a SHARED (tied) k_in-input, k_out-output canonical core (Part A: k_in
parallel v-chains sharing one denominator a, k_out independent
readouts via k_out distinct numerator vectors b^(1)..b^(k_out)).
Layer output is n_l*k_out scalar channels total.

Run:  python -m credit_memory.b24_interface_frontier
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Part A: k-input, k_out-output canonical core, shared denominator.
# ---------------------------------------------------------------------------
def make_core_coeffs(r, k_in, k_out, rng, mag_range=(0.5, 0.85)):
    roots = rng.uniform(mag_range[0], mag_range[1], r) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, r))
    coeffs = np.poly(roots)
    a = np.real(coeffs[1:])
    b = rng.randn(k_out, k_in, r) * 0.5  # b[j,i,:] = numerator for output j, input i
    return a, b


def miso_core_forward(a, b, U):
    """U: (k_in, T). Returns V: (k_in, T) (per-input delay states) and
    Y: (k_out, T) outputs."""
    r = len(a)
    k_in, T_ = U.shape
    k_out = b.shape[0]
    V = np.zeros((k_in, T_))
    vh = np.zeros((k_in, r))
    for t in range(T_):
        vt = U[:, t] - vh @ a  # (k_in,)
        V[:, t] = vt
        vh = np.concatenate([vt[:, None], vh[:, :-1]], axis=1)
    # outputs: y_j,t = sum_i sum_s b[j,i,s]*V[i,t-s]
    Y = np.zeros((k_out, T_))
    vh = np.zeros((k_in, r))
    for t in range(T_):
        Y[:, t] = np.einsum("jis,is->j", b, vh)
        vh = np.concatenate([V[:, t:t + 1], vh[:, :-1]], axis=1)
    return V, Y


# ---------------------------------------------------------------------------
# Local + denominator eligibility (Part A/B): exact closed form.
#   dy_j,t/db[j,i,s] = V[i,t-s]
#   dy_j,t/da[s]     = -w_j,t-s   (one w-chain PER OUTPUT j, since each
#                                   output has its own error signal)
# ---------------------------------------------------------------------------
def w_chains(a, Y):
    r = len(a)
    k_out, T_ = Y.shape
    W = np.zeros((k_out, T_))
    wh = np.zeros((k_out, r))
    for t in range(T_):
        wt = Y[:, t] - wh @ a
        W[:, t] = wt
        wh = np.concatenate([wt[:, None], wh[:, :-1]], axis=1)
    return W


def analytic_gradients_miso(a, b, U, Ytgt):
    """Exact reverse-mode adjoint (companion-form state-space per
    input channel, summed) -- independent reference for Part B."""
    r = len(a)
    k_in, T_ = U.shape
    k_out = b.shape[0]
    V, Y = miso_core_forward(a, b, U)
    err = (Y - Ytgt) / T_  # (k_out, T_)

    # companion-form per input channel i: state x_i (T_,r), shared A
    # (denominator), input via e_r, output via b[j,i,:] reversed per j.
    A = np.zeros((r, r)); A[:-1, 1:] = np.eye(r - 1); A[-1, :] = -a[::-1]
    Bvec = np.zeros(r); Bvec[-1] = 1.0

    x_delayed_list = []
    for i in range(k_in):
        x = np.zeros((T_, r)); xp = np.zeros(r)
        for t in range(T_):
            xp = A @ xp + Bvec * U[i, t]
            x[t] = xp
        x_delayed = np.concatenate([np.zeros((1, r)), x[:-1]], axis=0)
        x_delayed_list.append(x_delayed)

    # adjoint: lambda_i[t] = dL/dx_i[t], gets contributions from EVERY
    # output j's readout (C_j,i = b[j,i,::-1]) at t+1, plus A-recursion.
    grad_a = np.zeros(r)
    grad_b = np.zeros_like(b)
    lam_list = [np.zeros((T_, r)) for _ in range(k_in)]
    lam_next = [np.zeros(r) for _ in range(k_in)]
    for t in reversed(range(T_)):
        for i in range(k_in):
            direct = np.zeros(r)
            if t + 1 < T_:
                for j in range(k_out):
                    Cji = b[j, i, ::-1]
                    direct += err[j, t + 1] * Cji
                lam_t = direct + lam_next[i] @ A
            else:
                lam_t = np.zeros(r)
            lam_list[i][t] = lam_t
            lam_next[i] = lam_t

    for i in range(k_in):
        gA = np.zeros((r, r))
        for t in range(T_):
            gA += np.outer(lam_list[i][t], x_delayed_list[i][t])
        grad_a += -gA[-1, :][::-1]  # accumulate across i (shared a)
        for j in range(k_out):
            gC = np.einsum("t,tn->n", err[j], x_delayed_list[i])
            grad_b[j, i, :] = gC[::-1]
    return grad_a, grad_b


def miso_backward(a, b, U, dL_dY):
    """Generalized adjoint: accepts an ARBITRARY upstream gradient
    dL_dY (k_out,T) (not just an internal MSE error) and returns
    (dL_dU, grad_a, grad_b) -- dL_dU is required to chain the adjoint
    across copies (via V) and down across layers (Part F/G)."""
    r = len(a)
    k_in, T_ = U.shape
    k_out = b.shape[0]
    A = np.zeros((r, r)); A[:-1, 1:] = np.eye(r - 1); A[-1, :] = -a[::-1]
    Bvec = np.zeros(r); Bvec[-1] = 1.0

    x_delayed_list = []
    for i in range(k_in):
        x = np.zeros((T_, r)); xp = np.zeros(r)
        for t in range(T_):
            xp = A @ xp + Bvec * U[i, t]
            x[t] = xp
        x_delayed = np.concatenate([np.zeros((1, r)), x[:-1]], axis=0)
        x_delayed_list.append(x_delayed)

    grad_a = np.zeros(r)
    grad_b = np.zeros_like(b)
    dL_dU = np.zeros((k_in, T_))
    lam_list = [np.zeros((T_, r)) for _ in range(k_in)]
    lam_next = [np.zeros(r) for _ in range(k_in)]
    for t in reversed(range(T_)):
        for i in range(k_in):
            direct = np.zeros(r)
            if t + 1 < T_:
                for j in range(k_out):
                    Cji = b[j, i, ::-1]
                    direct += dL_dY[j, t + 1] * Cji
                lam_t = direct + lam_next[i] @ A
            else:
                lam_t = np.zeros(r)
            lam_list[i][t] = lam_t
            lam_next[i] = lam_t
            dL_dU[i, t] = lam_t[-1]  # d(x[t])/d(U[i,t]) = Bvec = e_r

    for i in range(k_in):
        gA = np.zeros((r, r))
        for t in range(T_):
            gA += np.outer(lam_list[i][t], x_delayed_list[i][t])
        grad_a += -gA[-1, :][::-1]
        for j in range(k_out):
            gC = np.einsum("t,tn->n", dL_dY[j], x_delayed_list[i])
            grad_b[j, i, :] = gC[::-1]
    return dL_dU, grad_a, grad_b


# ---------------------------------------------------------------------------
# Part F/G: deep exact credit. Chains miso_backward across copies (via V)
# and down across layers -- a genuine adjoint through the whole stack, not
# finite differences. Local per-copy adjoint state is O(k_in*r); total
# per-layer work is O(n*k_in*r) -- LINEAR in n, independent of the
# functional cap dim_below*min(k_in*k_out,r), which can saturate while
# credit cost keeps growing only linearly (Part G's requirement).
# ---------------------------------------------------------------------------
def layer_backward(layer, lower_out, dL_dOut):
    """dL_dOut: (n*k_out, T) upstream gradient for this layer's stacked
    output. Returns (grad_a, grad_b, grad_V[list], dL_d_lower_out)."""
    a, b, V, n, k_out = layer["a"], layer["b"], layer["V"], layer["n"], layer["k_out"]
    dim_below, T_ = lower_out.shape
    grad_a = np.zeros_like(a)
    grad_b = np.zeros_like(b)
    grad_V = []
    dL_d_lower = np.zeros((dim_below, T_))
    for q in range(n):
        U_q = V[q] @ lower_out
        dL_dY_q = dL_dOut[q * k_out:(q + 1) * k_out, :]
        dL_dU_q, ga_q, gb_q = miso_backward(a, b, U_q, dL_dY_q)
        grad_a += ga_q
        grad_b += gb_q
        grad_V.append(dL_dU_q @ lower_out.T)
        dL_d_lower += V[q].T @ dL_dU_q
    return grad_a, grad_b, grad_V, dL_d_lower


def stack_backward(layers, x, Ytgt):
    """Full analytic adjoint through the whole stack (forward pass
    cached, then layer_backward chained top to bottom). Exact -- no FD."""
    T_ = x.shape[1]
    outs = [x]
    cur = x
    for layer in layers:
        cur = layer_forward(layer, cur)
        outs.append(cur)
    dL_dOut = (outs[-1] - Ytgt) / T_
    grads = [None] * len(layers)
    for l in reversed(range(len(layers))):
        ga, gb, gV, dL_dOut = layer_backward(layers[l], outs[l], dL_dOut)
        grads[l] = dict(a=ga, b=gb, V=gV)
    return grads, outs[-1]


def prefix_gradients_miso(a, b, U, Ytgt):
    """Closed-form via the local eligibility formulas stated in Part A,
    verified against analytic_gradients_miso."""
    r = len(a)
    k_in, T_ = U.shape
    k_out = b.shape[0]
    V, Y = miso_core_forward(a, b, U)
    W = w_chains(a, Y)
    err = (Y - Ytgt) / T_

    grad_b = np.zeros_like(b)
    for j in range(k_out):
        for i in range(k_in):
            for s in range(r):
                V_tk = np.concatenate([np.zeros(s + 1), V[i, :-(s + 1)]]) if s + 1 <= T_ else np.zeros(T_)
                grad_b[j, i, s] = float(np.sum(err[j] * V_tk))

    grad_a = np.zeros(r)
    for s in range(r):
        total = 0.0
        for j in range(k_out):
            W_tk = np.concatenate([np.zeros(s + 1), W[j, :-(s + 1)]]) if s + 1 <= T_ else np.zeros(T_)
            total += float(np.sum(err[j] * (-W_tk)))
        grad_a[s] = total
    return grad_a, grad_b


# ---------------------------------------------------------------------------
# Part C: multi-copy architecture. n_l copies per layer, each with its
# OWN k_in-dim input via a copy-specific mixing V_l[q] (generic, drawn
# from the lower layer's full n_{l-1}*k_out_{l-1}-dim output), sharing
# ONE tied (a,b) core. Layer output: n_l*k_out scalar channels.
# ---------------------------------------------------------------------------
def build_layer(r, k_in, k_out, n, dim_below, seed):
    rng = np.random.RandomState(seed)
    a, b = make_core_coeffs(r, k_in, k_out, rng)
    V = [rng.randn(k_in, dim_below) / np.sqrt(dim_below) * 0.7 for _ in range(n)]
    return dict(a=a, b=b, V=V, r=r, k_in=k_in, k_out=k_out, n=n)


def layer_forward(layer, lower_out):
    """lower_out: (dim_below, T). Returns (n*k_out, T) stacked output."""
    a, b, V, n, k_out = layer["a"], layer["b"], layer["V"], layer["n"], layer["k_out"]
    outs = []
    for q in range(n):
        U_q = V[q] @ lower_out  # (k_in, T)
        _, Y_q = miso_core_forward(a, b, U_q)  # (k_out, T)
        outs.append(Y_q)
    return np.concatenate(outs, axis=0)  # (n*k_out, T)


def stack_forward(layers, x):
    """x: (dim_in, T). Returns final layer output (n_L*k_out_L, T)."""
    cur = x
    for layer in layers:
        cur = layer_forward(layer, cur)
    return cur


# ---------------------------------------------------------------------------
# Part D: functional-width test. Measures whether width n genuinely
# enlarges the attainable transfer-function-coefficient space at fixed
# k, via the rank of the Jacobian of outputs w.r.t. the mixing
# parameters V (not merely parameter count).
# ---------------------------------------------------------------------------
def scalar_collapse_check(rng, r=4, k_out=3, T_=25):
    """k output channels that are scalar multiples of ONE shared
    signal collapse EXACTLY to an effective SISO system with
    b_eff = sum_j g_j*b_j (generalization of B23's k=1 collapse)."""
    a, b_shared = make_core_coeffs(r, 1, 1, rng)
    gains = rng.randn(k_out)
    b = np.stack([g * b_shared[0] for g in gains], axis=0)  # (k_out,1,r)
    U = rng.randn(1, T_)
    _, Y = miso_core_forward(a, b, U)  # (k_out, T_)
    b_eff = (np.sum(gains) * b_shared[0]).reshape(1, 1, r)
    _, Y_eff = miso_core_forward(a, b_eff, U)  # (1, T_)
    # sum of the k scalar-multiple channels == one SISO core w/ b_eff=sum(g_j)*b_shared
    err = np.max(np.abs(np.sum(Y, axis=0) - Y_eff[0]))
    return err


def functional_rank(layer, lower_out, tol=1e-9):
    """CORRECTED measurement (an earlier version measured d(output)/dV
    Jacobian rank -- that tests local PARAMETER identifiability, a
    different question from whether the achievable FUNCTION CLASS
    grows with width; a full-rank Jacobian is fully consistent with
    every copy's output being a scalar multiple of one shared signal,
    exactly B23's k=1 case, since each V[q] still has a distinct
    -- if redundant-in-aggregate -- effect on its OWN copy).

    This measures the rank of the n*k_out ACHIEVED OUTPUT TRAJECTORIES
    themselves (stacked over copies and output channels, for the SAME
    fixed input): if copies are collapsible (B23's k=1 case), every
    row is a scalar multiple of one shared sequence -- rank 1,
    regardless of n. Genuine functional diversity shows rank growing
    with n (up to whatever cap the core's own structure permits)."""
    a, b, V, n, k_out = layer["a"], layer["b"], layer["V"], layer["n"], layer["k_out"]
    T_ = lower_out.shape[1]
    Y_full = layer_forward(layer, lower_out)  # (n*k_out, T_)
    S = np.linalg.svd(Y_full, compute_uv=False)
    rank = int(np.sum(S > tol * S[0])) if len(S) and S[0] > 0 else 0
    return rank, n * k_out, S


# ---------------------------------------------------------------------------
# Part H (light): a real trained-task check grounding the cap law -- does a
# k=1 (width-vacuous) approximator actually plateau below a k-matched one
# when fitting a target that genuinely needs a 2D interface?
# ---------------------------------------------------------------------------
def _clip(g, maxnorm=1.0):
    nrm = np.linalg.norm(g)
    return g if nrm <= maxnorm else g * (maxnorm / nrm)


def train_stack(layers, x, Ytgt, steps=1500, lr=0.01):
    losses = []
    for _ in range(steps):
        grads, out = stack_backward(layers, x, Ytgt)
        loss = 0.5 * np.sum((out - Ytgt) ** 2) / x.shape[1]
        if not np.isfinite(loss):
            losses.append(np.inf)
            break
        losses.append(loss)
        for l, layer in enumerate(layers):
            layer["a"] -= lr * _clip(grads[l]["a"])
            layer["b"] -= lr * _clip(grads[l]["b"].ravel()).reshape(grads[l]["b"].shape)
            for q in range(layer["n"]):
                layer["V"][q] -= lr * _clip(grads[l]["V"][q].ravel()).reshape(grads[l]["V"][q].shape)
    return losses


def main():
    rng = np.random.RandomState(0)

    print("=" * 70)
    print("PART A/B -- k-input canonical core, exact gradient verification")
    print("(prefix closed-form vs. independent analytic adjoint)")
    print("=" * 70)
    worst = 0.0
    for r in (2, 4, 8):
        for k_in in (1, 2, 4):
            for k_out in (1, 2, 4):
                rr = np.random.RandomState(100 + r + 10 * k_in + 100 * k_out)
                a, b = make_core_coeffs(r, k_in, k_out, rr)
                T_ = 30
                U = rr.randn(k_in, T_) * 0.5
                Ytgt = rr.randn(k_out, T_) * 0.3
                ga1, gb1 = analytic_gradients_miso(a, b, U, Ytgt)
                ga2, gb2 = prefix_gradients_miso(a, b, U, Ytgt)
                e = max(np.max(np.abs(ga1 - ga2)), np.max(np.abs(gb1 - gb2)))
                worst = max(worst, e)
    print(f"  max |prefix - analytic adjoint| over r in {{2,4,8}}, "
          f"k_in,k_out in {{1,2,4}}: {worst:.2e}  (independent-method agreement)")

    print()
    print("=" * 70)
    print("PART C -- scalar-collapse counter-example (generalizes B23's k=1)")
    print("=" * 70)
    errs = [scalar_collapse_check(rng) for _ in range(6)]
    print(f"  k output channels = scalar multiples of ONE shared signal collapse")
    print(f"  to an effective SISO core (b_eff=sum g_j b_j): max err = {max(errs):.2e}")

    print()
    print("=" * 70)
    print("PART D -- true stack sanity check: k=1 THROUGHOUT reproduces B23")
    print("(external scalar -> k=1 layer -> k=1 layer; rank must stay 1)")
    print("=" * 70)
    T_ = 25
    x = rng.randn(1, T_)
    for n0 in (1, 2, 4, 8):
        layer0 = build_layer(r=3, k_in=1, k_out=1, n=n0, dim_below=1, seed=10)
        out0 = layer_forward(layer0, x)
        layer1 = build_layer(r=4, k_in=1, k_out=1, n=8, dim_below=n0, seed=11)
        rank, dimP, _ = functional_rank(layer1, out0)
        print(f"  n0={n0:2d}: layer1 output rank={rank}  (n1*k_out={dimP})  [expect 1]")

    print()
    print("=" * 70)
    print("PART D/E -- functional-rank cap law: rank -> dim_below*min(k_in*k_out, r)")
    print("(single layer fed a genuinely dim_below-dim signal; large T_ to resolve)")
    print("=" * 70)
    T_ = 80
    dim_below = 8
    lower_out = np.random.RandomState(0).randn(dim_below, T_)
    configs = [
        dict(r=3, k_in=1, k_out=1, n=16, seed=1),
        dict(r=3, k_in=2, k_out=2, n=16, seed=2),
        dict(r=3, k_in=4, k_out=4, n=16, seed=3),
        dict(r=5, k_in=1, k_out=1, n=16, seed=4),
        dict(r=5, k_in=2, k_out=1, n=16, seed=5),
        dict(r=5, k_in=1, k_out=2, n=16, seed=6),
        dict(r=5, k_in=3, k_out=3, n=20, seed=7),
        dict(r=2, k_in=2, k_out=2, n=16, seed=8),
    ]
    print(f"  {'r':>3} {'k_in':>5} {'k_out':>6} {'n':>3} | {'rank':>5} {'n*k_out':>8} {'pred':>6}")
    all_ok = True
    for c in configs:
        layer = build_layer(r=c["r"], k_in=c["k_in"], k_out=c["k_out"], n=c["n"],
                             dim_below=dim_below, seed=c["seed"])
        rank, dimP, _ = functional_rank(layer, lower_out)
        pred = dim_below * min(c["k_in"] * c["k_out"], c["r"])
        ok = (rank == pred)
        all_ok &= ok
        print(f"  {c['r']:>3} {c['k_in']:>5} {c['k_out']:>6} {c['n']:>3} | "
              f"{rank:>5} {dimP:>8} {pred:>6}  {'OK' if ok else 'MISMATCH'}")
    print(f"  law cap = dim_below*min(k_in*k_out, r): {'ALL MATCH' if all_ok else 'SOME MISMATCH'}")

    print()
    print("=" * 70)
    print("PART F/G -- deep exact credit: analytic adjoint through the full")
    print("multi-copy, multi-layer stack (FD used only as secondary sanity check)")
    print("=" * 70)
    for L in (2, 3, 4):
        rng_l = np.random.RandomState(42)
        T_ = 20
        if L == 2:
            layers = [build_layer(r=3, k_in=1, k_out=2, n=2, dim_below=1, seed=1),
                      build_layer(r=3, k_in=2, k_out=1, n=3, dim_below=4, seed=2)]
        elif L == 3:
            layers = [build_layer(r=3, k_in=1, k_out=2, n=2, dim_below=1, seed=1),
                      build_layer(r=3, k_in=2, k_out=2, n=3, dim_below=4, seed=2),
                      build_layer(r=4, k_in=2, k_out=1, n=2, dim_below=6, seed=3)]
        else:
            layers = [build_layer(r=3, k_in=1, k_out=2, n=2, dim_below=1, seed=1),
                      build_layer(r=3, k_in=2, k_out=2, n=2, dim_below=4, seed=2),
                      build_layer(r=3, k_in=2, k_out=1, n=3, dim_below=4, seed=3),
                      build_layer(r=4, k_in=1, k_out=1, n=2, dim_below=3, seed=4)]
        x_l = rng_l.randn(1, T_)
        dim_out = layers[-1]["n"] * layers[-1]["k_out"]
        Ytgt_l = rng_l.randn(dim_out, T_) * 0.2
        grads, _ = stack_backward(layers, x_l, Ytgt_l)

        def loss_only(layers_):
            out = stack_forward(layers_, x_l)
            return 0.5 * np.sum((out - Ytgt_l) ** 2) / T_

        eps = 1e-6
        worst_fd = 0.0
        for l, layer in enumerate(layers):
            r = layer["r"]
            ai = rng_l.randint(0, r)
            orig = layer["a"][ai]
            layer["a"][ai] = orig + eps; Lp = loss_only(layers)
            layer["a"][ai] = orig - eps; Lm = loss_only(layers)
            layer["a"][ai] = orig
            worst_fd = max(worst_fd, abs((Lp - Lm) / (2 * eps) - grads[l]["a"][ai]))
        print(f"  L={L}: max |analytic adjoint - FD sanity check| (sampled 'a' entries) = {worst_fd:.2e}")

    print()
    print("=" * 70)
    print("PART H (light) -- trained-task check: k=1 (width-vacuous) vs k=2")
    print("(matched-interface) approximator fitting a genuinely 2D-interface target")
    print("=" * 70)
    rng_t = np.random.RandomState(11)
    T_ = 60
    x = rng_t.randn(1, T_) * 0.3
    target_layers = [build_layer(r=3, k_in=1, k_out=2, n=1, dim_below=1, seed=21),
                      build_layer(r=3, k_in=2, k_out=1, n=1, dim_below=2, seed=22)]
    Ytgt = stack_forward(target_layers, x)

    approxA = [build_layer(r=3, k_in=1, k_out=1, n=4, dim_below=1, seed=31),
               build_layer(r=3, k_in=1, k_out=1, n=1, dim_below=4, seed=32)]
    lossA = train_stack(approxA, x, Ytgt)

    approxB = [build_layer(r=3, k_in=1, k_out=2, n=1, dim_below=1, seed=41),
               build_layer(r=3, k_in=2, k_out=1, n=1, dim_below=2, seed=42)]
    lossB = train_stack(approxB, x, Ytgt)

    print(f"  k=1 (width-vacuous, n=4) approximator: final loss = {lossA[-1]:.6f}")
    print(f"  k=2 (interface-matched, n=1) approximator: final loss = {lossB[-1]:.6e}")
    print(f"  ratio (k=1 loss / k=2 loss) = {lossA[-1]/max(lossB[-1],1e-300):.1f}x worse")


if __name__ == "__main__":
    main()
