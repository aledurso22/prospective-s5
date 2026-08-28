"""Phase B22.1 -- reconciliation + constructive closure.

Resolves a direct contradiction between B22's own "A-alone" rank
measurement (r^2) and the theory's prediction (2r-1, matching the
classical minimal-SISO identifiable-denominator-parameter count).

RESOLVED, not assumed: B22's `d_credit_A_alone` computed the rank of
H^(k)[a,b] = C A^a E_k A^b B treating a,b as INDEPENDENT indices -- a
genuinely r^2-dimensional "bilinear"/hidden-state-sensitivity object.
The true EXTERNAL Markov-parameter Jacobian dH_n/dA (a SINGLE time
index n = a+b+1) is a linear projection of that object (summing along
anti-diagonals a+b=const) and has rank exactly 2r-1, verified directly
below with a completely unambiguous singular-value gap (12 orders of
magnitude at r=4). The old r^2 result was correct for the object it
was actually measuring -- it was measuring the wrong object for "what
a future external teaching signal can distinguish."

Run:  python -m credit_memory.b22_1_reconciliation
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# System construction (reused pattern from b22_interface_credit)
# ---------------------------------------------------------------------------
def make_dense_siso(r, rng, mag_range=(0.6, 0.9)):
    A = rng.randn(r, r) / np.sqrt(r)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (rng.uniform(*mag_range) / (rho + 1e-9))
    B = rng.randn(r, 1)
    C = rng.randn(1, r)
    return A, B, C


def elementary_dirs(r):
    return [np.outer(np.eye(r)[j], np.eye(r)[k]) for j in range(r) for k in range(r)]


def powers_of(A, K):
    P = [np.eye(A.shape[0])]
    for _ in range(K):
        P.append(P[-1] @ A)
    return P


# ---------------------------------------------------------------------------
# Part A: rebuild the rank test from first principles -- TRUE Markov-
# parameter Jacobians, joint (A,B,C) and A-only.
# ---------------------------------------------------------------------------
def markov_jacobian_ranks(A, B, C, Nmax=None, tol=1e-9):
    r = A.shape[0]
    # generous default margin (same fix as B22's joint_credit): a tight
    # Nmax can leave the true rank boundary numerically ambiguous for a
    # fast-decaying random draw once r grows -- confirmed directly at
    # r=16 (Nmax=4r gave a spurious rank of 29 for both A-only and joint
    # instead of the true 31/32; resolved cleanly with a larger margin
    # and slower spectral decay).
    Nmax = Nmax or 8 * r
    P = powers_of(A, Nmax)

    def dHn_dA(E_k):
        dHn = np.zeros(Nmax)
        for n in range(1, Nmax + 1):
            s = 0.0
            for a in range(n):
                s += (C @ P[a] @ E_k @ P[n - 1 - a] @ B).item()
            dHn[n - 1] = s
        return dHn

    def dHn_dB(j):
        dB = np.zeros((r, 1)); dB[j, 0] = 1.0
        return np.array([(C @ P[n - 1] @ dB).item() for n in range(1, Nmax + 1)])

    def dHn_dC(j):
        dC = np.zeros((1, r)); dC[0, j] = 1.0
        return np.array([(dC @ P[n - 1] @ B).item() for n in range(1, Nmax + 1)])

    E = elementary_dirs(r)
    rows_A = [dHn_dA(Ek) for Ek in E]
    rows_B = [dHn_dB(j) for j in range(r)]
    rows_C = [dHn_dC(j) for j in range(r)]

    S_Aonly = np.linalg.svd(np.stack(rows_A), compute_uv=False)
    S_joint = np.linalg.svd(np.stack(rows_A + rows_B + rows_C), compute_uv=False)
    rank_Aonly = int(np.sum(S_Aonly > tol * S_Aonly[0]))
    rank_joint = int(np.sum(S_joint > tol * S_joint[0]))
    return rank_Aonly, S_Aonly, rank_joint, S_joint


# ---------------------------------------------------------------------------
# Part B: locate exactly what the old B22 object measured -- vary the
# queried object one thing at a time.
# ---------------------------------------------------------------------------
def old_b22_object_rank(A, B, C, tol=1e-9):
    """Reproduces b22_interface_credit.d_credit_A_alone exactly (the
    original object: full (a,b)-indexed, treating a,b independently)."""
    r = A.shape[0]
    P = powers_of(A, 2 * r - 1)
    rows = []
    for Ek in elementary_dirs(r):
        H = np.array([[(C @ P[a] @ Ek @ P[b] @ B).item()
                       for b in range(r)] for a in range(r)])
        rows.append(H.ravel())
    S = np.linalg.svd(np.stack(rows), compute_uv=False)
    return int(np.sum(S > tol * S[0])), S


def hidden_state_sensitivity_rank(A, B, C, T_=None, tol=1e-9):
    """B1: d x_t / dA -- sensitivity of the FULL hidden state (not yet
    passed through C) to A, zero-state (impulse at B, t=0), stacked
    over t and over all r^2 directions."""
    r = A.shape[0]
    T_ = T_ or 3 * r
    P = powers_of(A, T_)
    rows = []
    for Ek in elementary_dirs(r):
        # d x_t/dA = sum_{a+b=t-1} A^a Ek A^b B  (vector in R^r, per t)
        vecs = []
        for t in range(1, T_ + 1):
            v = np.zeros(r)
            for a in range(t):
                v = v + (P[a] @ Ek @ P[t - 1 - a] @ B).ravel()
            vecs.append(v)
        rows.append(np.concatenate(vecs))
    S = np.linalg.svd(np.stack(rows), compute_uv=False)
    return int(np.sum(S > tol * S[0])), S


def external_output_sensitivity_rank(A, B, C, Nmax=None, tol=1e-9):
    """B2: d y_t / dA, zero-state -- this IS the true Markov-parameter
    Jacobian from Part A, reproduced here under the "vary one thing"
    framing for direct comparison."""
    rank, S, _, _ = markov_jacobian_ranks(A, B, C, Nmax=Nmax, tol=tol)
    return rank, S


def nonzero_state_held_fixed_rank(A, B, C, x0, T_=None, tol=1e-9):
    """B4: current state held fixed at x0 (NOT zero, not itself a
    function of A) while A changes -- d y_t(x0)/dA for t>=1, x0 GENERIC
    (e.g. spanning a basis of R^r independently of A,B)."""
    r = A.shape[0]
    T_ = T_ or 2 * r
    P = powers_of(A, T_)
    rows = []
    for Ek in elementary_dirs(r):
        vec = []
        for t in range(1, T_ + 1):
            # d(C A^t x0)/dA[j,k] = C sum_{a+b=t-1} A^a Ek A^b x0
            s = 0.0
            for a in range(t):
                s += (C @ P[a] @ Ek @ P[t - 1 - a] @ x0).item()
            vec.append(s)
        rows.append(np.array(vec))
    S = np.linalg.svd(np.stack(rows), compute_uv=False)
    return int(np.sum(S > tol * S[0])), S


# ---------------------------------------------------------------------------
# Part C: gauge null-direction dimension.
# ---------------------------------------------------------------------------
def verify_joint_gauge_exact(A, B, C, seed=0, Kmax=None):
    r = A.shape[0]
    Kmax = Kmax or 3 * r
    rng = np.random.RandomState(seed)
    X = rng.randn(r, r)
    dA = X @ A - A @ X
    dB = X @ B
    dC = -C @ X
    P = powers_of(A, Kmax)
    errs = []
    for k in range(Kmax):
        term = (C @ P[k] @ dB).item()
        for a in range(k):
            term += (C @ P[a] @ dA @ P[k - 1 - a] @ B).item()
        term += (dC @ P[k] @ B).item()
        errs.append(term)
    return float(np.max(np.abs(errs)))


def residual_gauge_dimension(A, B, C, tol=1e-8):
    """Dimension of {X : XB=0, CX=0} -- predicted (r-1)^2 generically."""
    r = A.shape[0]
    # X is r x r; constraints: X@B=0 (r eqns, B is r x 1) and C@X=0
    # (r eqns, C is 1 x r) -- build the linear map X -> (XB, (CX)^T) and
    # find its null space dimension.
    rows = []
    for j in range(r):
        for k in range(r):
            E = np.zeros((r, r)); E[j, k] = 1.0
            xb = (E @ B).ravel()      # r values
            cx = (C @ E).ravel()      # r values
            rows.append(np.concatenate([xb, cx]))
    M = np.stack(rows)  # (r^2) x (2r), each row = effect of one E_jk
    # null space of X->(XB,CX) as a map FROM r^2-dim X-space: rank of M's
    # TRANSPOSE view -- we want dim{X: constraints=0} = r^2 - rank(M^T)
    rank = np.linalg.matrix_rank(M, tol=tol)
    return r * r - rank, rank


def verify_residual_gauge_kills_Aonly_response(A, B, C, tol=1e-8, seed=1):
    """For X with XB=0, CX=0, confirm dA=[X,A] gives zero Markov Jacobian
    contribution too (not just B,C fixed contributions, which are
    trivially zero by construction here)."""
    r = A.shape[0]
    rng = np.random.RandomState(seed)
    # find a residual-gauge X via null space of the constraint map
    rows = []
    for j in range(r):
        for k in range(r):
            E = np.zeros((r, r)); E[j, k] = 1.0
            rows.append(np.concatenate([(E @ B).ravel(), (C @ E).ravel()]))
    M = np.stack(rows)
    _, _, Vt = np.linalg.svd(M.T)
    rank = np.linalg.matrix_rank(M.T, tol=tol)
    null_basis = Vt[rank:]  # rows span the null space, in r^2-dim X-space
    if null_basis.shape[0] == 0:
        return None
    x_vec = null_basis[rng.randint(null_basis.shape[0])]
    X = x_vec.reshape(r, r)
    dA = X @ A - A @ X
    Kmax = 3 * r
    P = powers_of(A, Kmax)
    errs = []
    for k in range(Kmax):
        term = 0.0
        for a in range(k):
            term += (C @ P[a] @ dA @ P[k - 1 - a] @ B).item()
        errs.append(term)
    return float(np.max(np.abs(errs)))


# ---------------------------------------------------------------------------
# Part D: constructive 2r recurrence (V,W two-filter eligibility).
# ---------------------------------------------------------------------------
def make_transfer_coeffs(r, rng, mag_range=(0.5, 0.85)):
    """Random stable, coprime (generically) denominator/numerator."""
    roots = rng.uniform(mag_range[0], mag_range[1], r) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, r))
    coeffs = np.poly(roots)
    a = np.real(coeffs[1:])  # a_1..a_r
    b = rng.randn(r) * 0.5
    return a, b


def two_filter_forward(a, b, u):
    r = len(a)
    T_ = len(u)
    v = np.zeros(T_); y = np.zeros(T_); w = np.zeros(T_)
    vh = np.zeros(r)  # history buffer, most-recent-first
    for t in range(T_):
        vt = u[t] - float(a @ vh)
        yt = float(b @ vh)  # uses PAST v's only (v_{t-1}..v_{t-r})
        v[t] = vt
        y[t] = yt
        vh = np.concatenate([[vt], vh[:-1]])
    # second pass for w (needs y computed first, causal, same recursion)
    wh2 = np.zeros(r)
    for t in range(T_):
        wt = y[t] - float(a @ wh2)
        w[t] = wt
        wh2 = np.concatenate([[wt], wh2[:-1]])
    return v, y, w


def two_filter_eligibility_gradients(a, b, u, ytgt):
    """dy_t/db_k = v_{t-k}, dy_t/da_k = -w_{t-k} -- accumulate dL/da,
    dL/db for L=0.5*mean((y-ytgt)^2) via the theorized closed form."""
    r = len(a)
    T_ = len(u)
    v, y, w = two_filter_forward(a, b, u)
    err = (y - ytgt) / T_
    grad_b = np.zeros(r)
    grad_a = np.zeros(r)
    for t in range(T_):
        for k in range(1, r + 1):
            v_tk = v[t - k] if t - k >= 0 else 0.0
            w_tk = w[t - k] if t - k >= 0 else 0.0
            grad_b[k - 1] += err[t] * v_tk
            grad_a[k - 1] += err[t] * (-w_tk)
    return grad_a, grad_b, y


# ---------------------------------------------------------------------------
# Part E: machine-precision exactness vs BPTT/naive RTRL (state-space form)
# ---------------------------------------------------------------------------
def companion_from_coeffs(a, b):
    r = len(a)
    A = np.zeros((r, r))
    A[:-1, 1:] = np.eye(r - 1)
    A[-1, :] = -a[::-1]
    B = np.zeros((r, 1)); B[-1, 0] = 1.0
    C = b[::-1].reshape(1, r)
    return A, B, C


def bptt_gradients_siso_transfer(a, b, u, ytgt):
    """Reverse-mode adjoint on the companion-form state-space realization
    of the SAME (a,b), for cross-checking the two-filter closed form.

    Alignment note (found and fixed during verification): the
    companion-form state-space output C@x_t is one step AHEAD of the
    two-filter convention's y_t = sum_k b_k v_{t-k} (which uses only
    PAST v's, a genuine one-step delay relative to v_t itself) --
    confirmed directly by comparing raw output sequences before
    trusting any gradient comparison (they matched exactly once
    shifted by one step, ruling out a deeper construction error).
    Delaying the state-space output by one step (y[t] = C@x_{t-1},
    y[0]=0) aligns the two conventions exactly."""
    A, B, C = companion_from_coeffs(a, b)
    r = len(a)
    T_ = len(u)
    x = np.zeros((T_, r))
    xp = np.zeros(r)
    for t in range(T_):
        xp = A @ xp + (B @ u[t:t + 1]).ravel()
        x[t] = xp
    x_delayed = np.concatenate([np.zeros((1, r)), x[:-1]], axis=0)  # x_{t-1}, x_{-1}=0
    y = x_delayed @ C.ravel()
    err = (y - ytgt) / T_

    # Adjoint for the SHIFTED relationship: lambda[t] := dL/dx[t] gets a
    # DIRECT contribution err[t+1]*C from y[t+1]=C@x[t] (not err[t] --
    # the shift moves this by one index too), plus lambda[t+1]@A from
    # x[t+1]=A@x[t]+B@u[t+1], both only for t+1<T_.
    lam = np.zeros((T_, r))
    lam_next = np.zeros(r)
    for t in reversed(range(T_)):
        if t + 1 < T_:
            lam_t = err[t + 1] * C.ravel() + lam_next @ A
        else:
            lam_t = np.zeros(r)
        lam[t] = lam_t
        lam_next = lam_t

    grad_A = np.zeros((r, r))
    grad_B = np.zeros(r)
    for t in range(T_):
        grad_A += np.outer(lam[t], x_delayed[t])  # x[t]=A@x[t-1]+B@u[t]
        grad_B += lam[t] * u[t]
    grad_C = np.einsum("t,tn->n", err, x_delayed)  # y[t]=C@x[t-1]

    # convert companion-basis gradients back to (a,b)-coefficient
    # gradients: A's last row = -a[::-1], C = b[::-1]
    grad_a_from_state = -grad_A[-1, :][::-1]
    grad_b_from_state = grad_C[::-1]
    return grad_a_from_state, grad_b_from_state


# ---------------------------------------------------------------------------
# Part F: minimality (McMillan degree) check via Hankel rank.
# ---------------------------------------------------------------------------
def hankel_rank(a, b, K=None, tol=1e-9):
    r = len(a)
    A, B, C = companion_from_coeffs(a, b)
    K = K or 3 * r
    P = powers_of(A, 2 * K)
    markov = [(C @ P[k] @ B).item() for k in range(2 * K)]
    H = np.array([[markov[i + j] for j in range(K)] for i in range(K)])
    S = np.linalg.svd(H, compute_uv=False)
    return int(np.sum(S > tol * S[0])), S


def main() -> None:
    print("=" * 90)
    print("Phase B22.1: reconciliation + constructive closure")
    print("=" * 90)
    rng = np.random.RandomState(0)

    print("\nPart A: true Markov-parameter Jacobian ranks")
    for r in (2, 4, 8, 16):
        mag = (0.6, 0.9) if r < 16 else (0.92, 0.97)  # slower decay at r=16
        A, B, C = make_dense_siso(r, rng, mag_range=mag)
        rA, SA, rJ, SJ = markov_jacobian_ranks(A, B, C)
        print(f"  r={r:3d}: A-only rank={rA:3d} (pred {2*r-1})  "
             f"joint rank={rJ:3d} (pred {2*r})")
        if r == 4:
            print(f"    A-only spectrum near cutoff: {SA[max(0,rA-2):rA+3]}")

    print("\nPart B: locating the old B22 object")
    A, B, C = make_dense_siso(4, rng)
    r_old, _ = old_b22_object_rank(A, B, C)
    r_hidden, _ = hidden_state_sensitivity_rank(A, B, C)
    r_ext, _ = external_output_sensitivity_rank(A, B, C)
    x0 = rng.randn(4, 1)
    r_fixed, _ = nonzero_state_held_fixed_rank(A, B, C, x0)
    print(f"  old B22 object (full a,b indexed):     rank={r_old}")
    print(f"  B1 hidden-state sensitivity dx_t/dA:    rank={r_hidden}")
    print(f"  B2 external output sensitivity dy_t/dA: rank={r_ext}")
    print(f"  B4 nonzero-fixed-state sensitivity:     rank={r_fixed}")

    print("\nPart C: gauge null directions")
    A, B, C = make_dense_siso(4, rng)
    err = verify_joint_gauge_exact(A, B, C)
    print(f"  joint gauge exactness (max abs err, should be 0): {err:.2e}")
    dim_null, rank_constraint = residual_gauge_dimension(A, B, C)
    print(f"  residual gauge dim {{X: XB=0,CX=0}} = {dim_null}  predicted (r-1)^2={(4-1)**2}")
    err2 = verify_residual_gauge_kills_Aonly_response(A, B, C)
    print(f"  residual-gauge A-only Markov response (should be 0): {err2}")

    print("\nPart D/E: two-filter (V,W) eligibility vs BPTT, machine precision")
    for r in (2, 3, 4, 6):
        rng2 = np.random.RandomState(r + 100)
        a, b = make_transfer_coeffs(r, rng2)
        T_ = 30
        u = rng2.randn(T_)
        ytgt = rng2.randn(T_)
        grad_a_vw, grad_b_vw, y_vw = two_filter_eligibility_gradients(a, b, u, ytgt)
        grad_a_bptt, grad_b_bptt = bptt_gradients_siso_transfer(a, b, u, ytgt)
        err_a = np.max(np.abs(grad_a_vw - grad_a_bptt))
        err_b = np.max(np.abs(grad_b_vw - grad_b_bptt))
        print(f"  r={r}: max err grad_a={err_a:.2e}  grad_b={err_b:.2e}")

    print("\nPart F: minimality (Hankel rank vs r)")
    for r in (2, 4, 8):
        rng3 = np.random.RandomState(r + 200)
        a, b = make_transfer_coeffs(r, rng3)
        rank, S = hankel_rank(a, b)
        print(f"  r={r}: Hankel rank={rank}  (predicted {r})")


if __name__ == "__main__":
    main()
