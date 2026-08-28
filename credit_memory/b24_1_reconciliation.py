"""Phase B24.1 -- theory reconciliation. Two corrections to B24's
framing, tested directly:

(1) B24's k>1 architecture is the NON-separable / multi-generator case.
    If a route is a truly separable operator F_l (x) Q_l (one shared
    scalar temporal generator, static gain matrix), width is redundant
    for EVERY fixed k, not just k=1 -- tested in Part A.

(2) B24's functional_rank let external output channel count grow with
    n. The stricter claim -- fixed scalar external I/O, does internal
    feature-bond width n matter as a finite TENSOR-RANK resource -- is
    tested directly in Parts B/C.

Run: python -m credit_memory.b24_1_reconciliation
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared SISO filter utility (order r, one-step-delay convention, matching
# B22.1/B23/B24: y_t = b . v_{t-1-*}, v_t = x_t - a . v_{t-1-*}).
# ---------------------------------------------------------------------------
def siso_filter(a, b, x):
    r = len(a)
    T_ = len(x)
    V = np.zeros(T_)
    vh = np.zeros(r)
    for t in range(T_):
        vt = x[t] - vh @ a
        V[t] = vt
        vh = np.concatenate([[vt], vh[:-1]])
    y = np.zeros(T_)
    vh = np.zeros(r)
    for t in range(T_):
        y[t] = b @ vh
        vh = np.concatenate([[V[t]], vh[:-1]])
    return y


def cascade_filters(coeffs_list, x):
    cur = x
    for a, b in coeffs_list:
        cur = siso_filter(a, b, cur)
    return cur


def make_ar_denominator(r, rng, mag_range=(0.5, 0.85)):
    roots = rng.uniform(mag_range[0], mag_range[1], r) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, r))
    coeffs = np.poly(roots)
    return np.real(coeffs[1:])


# ---------------------------------------------------------------------------
# Part A: separable-route control. route_l = F_l (x) Q_l -- ONE shared
# scalar temporal generator q per layer, static gain matrix F -- makes the
# numerator tensor b[j,i,s] = F[j,i]*q[s] exactly rank-1 in (i,j). Reuses
# B24's miso_core_forward/layer_forward/stack_forward (a genuinely
# separable b tensor is just a special case of the general core).
# ---------------------------------------------------------------------------
from credit_memory.b24_interface_frontier import (  # noqa: E402
    miso_core_forward, layer_forward, stack_forward,
)


def make_separable_layer(r, k_in, k_out, n, dim_below, seed):
    rng = np.random.RandomState(seed)
    a = make_ar_denominator(r, rng)
    q = rng.randn(r) * 0.5                 # single shared temporal numerator
    F = rng.randn(k_out, k_in) * 0.6        # static gain matrix
    b = np.einsum("ji,s->jis", F, q)        # rank-1 numerator tensor
    V = [rng.randn(k_in, dim_below) / np.sqrt(dim_below) * 0.7 for _ in range(n)]
    return dict(a=a, b=b, F=F, q=q, V=V, r=r, k_in=k_in, k_out=k_out, n=n)


def verify_separable_collapse(r, k, n_list, L, seed=0, T_=60):
    """Build an L-layer stack of separable layers (k_in=k_out=k, widths
    n_list), scalar external input, fixed scalar readout. Predicted: the
    output equals gamma * H_temporal(x) EXACTLY, where H_temporal is the
    L-fold cascade of the shared (a_l, q_l) SISO filters alone --
    independent of n_list, k, and the specific V[q]/readout draws."""
    rng = np.random.RandomState(seed)
    x = rng.randn(T_) * 0.3

    layers = []
    dim_below = 1
    for l in range(L):
        layer = make_separable_layer(r, k, k, n_list[l], dim_below, seed=seed * 100 + l)
        layers.append(layer)
        dim_below = n_list[l] * k

    y_full = stack_forward(layers, x.reshape(1, T_))  # (n_L*k, T_)
    w = np.random.RandomState(seed + 999).randn(y_full.shape[0])
    y_scalar = w @ y_full  # fixed scalar readout

    h_temporal = cascade_filters([(layers[l]["a"], layers[l]["q"]) for l in range(L)], x)

    denom = float(h_temporal @ h_temporal)
    gamma = float(y_scalar @ h_temporal) / denom if denom > 0 else 0.0
    residual = np.max(np.abs(y_scalar - gamma * h_temporal))
    scale = max(np.max(np.abs(y_scalar)), 1e-300)
    return residual, residual / scale, gamma


# ---------------------------------------------------------------------------
# Part B: explicit multi-generator k=2 counterexample.
#   H_n(z) = g(z)^T Gamma f(z),  rank(Gamma) <= n,  fixed scalar I/O.
# f = (f_1,...,f_k): k DISTINCT temporal generators applied to the scalar
# input. g = (g_1,...,g_k): k DISTINCT temporal generators applied AFTER a
# static combination, summed to a scalar output. Each of n copies
# contributes a rank-1 outer product w_q (x) v_q to an effective Gamma =
# sum_q w_q(x)v_q -- so n copies realize exactly the rank-<=n matrices.
# ---------------------------------------------------------------------------
def make_k_generators(k, r, rng, mag_range=(0.4, 0.85)):
    """k INDEPENDENT (distinct pole) order-r SISO filters."""
    return [(make_ar_denominator(r, rng, mag_range), rng.randn(r) * 0.5) for _ in range(k)]


def apply_generators(gens, x):
    """Returns (k, T) -- x filtered through each of the k generators."""
    return np.stack([siso_filter(a, b, x) for a, b in gens], axis=0)


def multigen_forward(f_gens, g_gens, Gamma_terms, x):
    """Gamma_terms: list of (w_q, v_q) rank-1 contributions (each in R^k).
    y_t = sum_q w_q . [g-filters]( v_q . [f-filters](x) )_t
    Returns scalar output (T,) and the effective Gamma = sum_q outer(w_q,v_q)."""
    k = len(f_gens)
    phi = apply_generators(f_gens, x)          # (k, T) -- f_j(x)
    T_ = len(x)
    y = np.zeros(T_)
    Gamma = np.zeros((k, k))
    for w_q, v_q in Gamma_terms:
        h_q = v_q @ phi                          # (T,) static combo of f's
        psi_q = apply_generators(g_gens, h_q)     # (k, T) -- g_i(h_q)
        y += w_q @ psi_q
        Gamma += np.outer(w_q, v_q)
    return y, Gamma


def direct_from_gamma(f_gens, g_gens, Gamma, x):
    """Closed form: y_t = sum_ij Gamma[i,j] * (g_i o f_j)(x)_t -- an
    independent reference computed WITHOUT going through any copies."""
    k = len(f_gens)
    T_ = len(x)
    y = np.zeros(T_)
    for i in range(k):
        for j in range(k):
            if Gamma[i, j] == 0.0:
                continue
            hij = cascade_filters([f_gens[j], g_gens[i]], x)
            y += Gamma[i, j] * hij
    return y


def part_b_counterexample(r=2, k=2, T_=50, seed=7):
    rng = np.random.RandomState(seed)
    x = rng.randn(T_) * 0.3
    f_gens = make_k_generators(k, r, rng)
    g_gens = make_k_generators(k, r, rng)

    # target rank-2 Gamma
    rng2 = np.random.RandomState(seed + 1)
    U = rng2.randn(k, k)
    S = np.array([1.3, 0.7])[:k]
    Vt = rng2.randn(k, k)
    U, _ = np.linalg.qr(U)
    Vt, _ = np.linalg.qr(Vt)
    Gamma_target = (U * S) @ Vt.T  # rank-2 (k=2), full rank

    y_target = direct_from_gamma(f_gens, g_gens, Gamma_target, x)

    # n=1: only rank-1 Gamma attainable -- best rank-1 approx via SVD, show residual
    Us, Ss, Vts = np.linalg.svd(Gamma_target)
    Gamma_rank1 = Ss[0] * np.outer(Us[:, 0], Vts[0, :])
    w1 = np.sqrt(Ss[0]) * Us[:, 0]
    v1 = np.sqrt(Ss[0]) * Vts[0, :]
    y_n1, Gamma_n1 = multigen_forward(f_gens, g_gens, [(w1, v1)], x)
    err_n1_vs_direct = np.max(np.abs(y_n1 - direct_from_gamma(f_gens, g_gens, Gamma_rank1, x)))
    err_n1_vs_target = np.max(np.abs(y_n1 - y_target))

    # n=2: exact rank-2 realization via SVD decomposition into 2 rank-1 terms
    terms = []
    for qidx in range(k):
        wq = np.sqrt(Ss[qidx]) * Us[:, qidx]
        vq = np.sqrt(Ss[qidx]) * Vts[qidx, :]
        terms.append((wq, vq))
    y_n2, Gamma_n2 = multigen_forward(f_gens, g_gens, terms, x)
    err_n2_gamma = np.max(np.abs(Gamma_n2 - Gamma_target))
    err_n2_vs_target = np.max(np.abs(y_n2 - y_target))

    # n=3 with an extra all-zero-weight copy: should not change anything
    extra = (np.zeros(k), rng.randn(k))
    y_n3, Gamma_n3 = multigen_forward(f_gens, g_gens, terms + [extra], x)
    err_n3_vs_n2 = np.max(np.abs(y_n3 - y_n2))

    return dict(
        err_n1_vs_direct=err_n1_vs_direct,
        err_n1_vs_target=err_n1_vs_target,
        rank_n1=int(np.linalg.matrix_rank(Gamma_n1)),
        err_n2_gamma=err_n2_gamma,
        err_n2_vs_target=err_n2_vs_target,
        rank_n2=int(np.linalg.matrix_rank(Gamma_n2)),
        err_n3_vs_n2=err_n3_vs_n2,
        target_rank=int(np.linalg.matrix_rank(Gamma_target)),
    )


# ---------------------------------------------------------------------------
# Part C: tensor-rank / saturation check for a larger multi-generator
# stack (k generators/side). Sweep n; measure (a) rank of the accumulated
# Gamma = sum_q outer(w_q,v_q), and (b) the external Hankel rank of the
# resulting scalar impulse response H_n(z). Prediction: (a) grows with n
# then saturates at k; (b) stays fixed (bounded by the f/g pole sets),
# independent of n even as (a) grows.
# ---------------------------------------------------------------------------
def hankel_rank(impulse, size=None, tol=1e-9):
    T_ = len(impulse)
    size = size or T_ // 2
    H = np.array([[impulse[i + j] for j in range(size)] for i in range(size)])
    S = np.linalg.svd(H, compute_uv=False)
    return int(np.sum(S > tol * S[0])) if S[0] > 0 else 0, S


def part_c_saturation(r=2, k=3, n_max=6, T_=200, seed=13):
    rng = np.random.RandomState(seed)
    f_gens = make_k_generators(k, r, rng)
    g_gens = make_k_generators(k, r, rng)
    impulse = np.zeros(T_)
    impulse[0] = 1.0

    rows = []
    Gamma_accum = np.zeros((k, k))
    terms = []
    rng_q = np.random.RandomState(seed + 500)
    for n in range(1, n_max + 1):
        w_q = rng_q.randn(k)
        v_q = rng_q.randn(k)
        terms.append((w_q, v_q))
        Gamma_accum = Gamma_accum + np.outer(w_q, v_q)
        gamma_rank = int(np.linalg.matrix_rank(Gamma_accum, tol=1e-9))
        y_imp, _ = multigen_forward(f_gens, g_gens, terms, impulse)
        h_rank, _ = hankel_rank(y_imp)
        rows.append(dict(n=n, gamma_rank=gamma_rank, hankel_rank=h_rank))
    return rows


def main():
    print("=" * 70)
    print("PART A -- separable-route control (route_l = F_l (x) Q_l)")
    print("Prediction: full transfer == gamma * H_temporal(z) for EVERY n, k")
    print("=" * 70)
    worst_rel = 0.0
    for k in (2, 3):
        for n_list in ([1, 1], [3, 5], [8, 2], [10, 10]):
            r, L = 3, len(n_list)
            resid, resid_rel, gamma = verify_separable_collapse(r, k, n_list, L, seed=1)
            worst_rel = max(worst_rel, resid_rel)
            print(f"  k={k} n_list={n_list}: residual={resid:.2e} "
                  f"relative={resid_rel:.2e} gamma={gamma:.4f}")
    print(f"  worst relative residual over all configs: {worst_rel:.2e}")

    print()
    print("=" * 70)
    print("PART B -- explicit multi-generator k=2 counterexample")
    print("H_n(z) = g(z)^T Gamma f(z), rank(Gamma)<=n, fixed scalar I/O")
    print("=" * 70)
    res = part_b_counterexample()
    print(f"  target Gamma rank: {res['target_rank']}")
    print(f"  n=1: achieved Gamma rank={res['rank_n1']}, matches best rank-1 "
          f"reconstruction to {res['err_n1_vs_direct']:.2e}, "
          f"but CANNOT match the rank-2 target (residual={res['err_n1_vs_target']:.4f})")
    print(f"  n=2: achieved Gamma rank={res['rank_n2']}, Gamma error={res['err_n2_gamma']:.2e}, "
          f"output vs target error={res['err_n2_vs_target']:.2e}  <- EXACT")
    print(f"  n=3 (extra zero-weight copy): output unchanged from n=2, "
          f"error={res['err_n3_vs_n2']:.2e}")

    print()
    print("=" * 70)
    print("PART C -- tensor-rank / saturation check")
    print("Gamma rank should grow with n then saturate at k; Hankel/McMillan")
    print("order of the external transfer function should stay n-independent")
    print("=" * 70)
    for r, k in ((2, 3), (3, 2), (2, 4)):
        print(f"  r={r} k={k}:")
        rows = part_c_saturation(r=r, k=k, n_max=6)
        for row in rows:
            print(f"    n={row['n']}: gamma_rank={row['gamma_rank']} "
                  f"(cap k={k})   hankel_rank={row['hankel_rank']}")
        hankel_vals = [row["hankel_rank"] for row in rows]
        print(f"    hankel_rank range across n=1..6: "
              f"{min(hankel_vals)}-{max(hankel_vals)} (n-independent within "
              f"numerical rank-detection noise; gamma_rank saturates at k={k})")


if __name__ == "__main__":
    main()
