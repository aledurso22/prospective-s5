"""Phase B22 -- function class vs prospective credit. Tests whether a
restricted source/query interface (m inputs, p outputs, both possibly
<< r) collapses the exact prospective-credit dimension toward O(r),
even when the ambient tangent bimodule dim(A P A) is the full r².

Two regimes, both measured directly rather than assumed:

  - A ALONE (B,C held fixed): d_credit stays the FULL r² -- no
    collapse. Confirmed not a bug via a direct commutator-perturbation
    check (a "gauge-shaped" dA=[T,A] genuinely changes the transfer
    function when B,C don't co-vary).
  - A,B,C JOINTLY trainable: d_credit = r*(m+p) EXACTLY, at every
    (r,m,p) tested -- the classical minimal-realization identifiable-
    parameter count, recovered from a fully dense, unconstrained A.
    Verified both via the abstract rank computation (coordinate-free
    by construction) and via real gradients on an actual simulated
    trajectory (gauge-direction gradient = exactly 0.0).

Run:  python -m credit_memory.b22_interface_credit
"""
from __future__ import annotations

import numpy as np

from credit_memory.b19_tangent_module import matrix_powers, subspace_basis


# ---------------------------------------------------------------------------
# Part A: system constructors
# ---------------------------------------------------------------------------
def make_dense_system(r, m, p, rng, mag_range=(0.6, 0.9)):
    A = rng.randn(r, r) / np.sqrt(r)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (rng.uniform(*mag_range) / (rho + 1e-9))
    B = rng.randn(r, m) / np.sqrt(m)
    C = rng.randn(p, r) / np.sqrt(r)
    E_list = [np.outer(np.eye(r)[j], np.eye(r)[k]) for j in range(r) for k in range(r)]
    return A, B, C, E_list


def make_companion_system(r, m, p, rng, mag_range=(0.6, 0.9)):
    """Controllable companion form: A is a shift matrix with a single
    trainable last row (r denominator coefficients) -- P is r-dim by
    CONSTRUCTION, not r^2."""
    roots = rng.uniform(mag_range[0], mag_range[1], r) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, r))
    coeffs = np.poly(roots)
    denom = np.real(coeffs[1:])
    A = np.zeros((r, r))
    A[:-1, 1:] = np.eye(r - 1)
    A[-1, :] = -denom[::-1]
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
        A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
    B = np.zeros((r, m)); B[-1, :] = rng.randn(m)
    C = rng.randn(p, r) / np.sqrt(r)
    E_list = [np.outer(np.eye(r)[-1], np.eye(r)[k]) for k in range(r)]
    return A, B, C, E_list


def make_block_companion_system(r, b, m, p, rng, mag_range=(0.6, 0.9)):
    """Block size b: r/b blocks, each internally dense b x b; blocks
    chained via a fixed companion-like shift. Not exercised in this
    pass's headline measurements -- kept for a future Part G."""
    assert r % b == 0
    nblk = r // b
    A = np.zeros((r, r))
    E_list = []
    for blk in range(nblk):
        sl = slice(blk * b, (blk + 1) * b)
        A[sl, sl] = rng.randn(b, b) / np.sqrt(b) * 0.8
        for j in range(b):
            for k in range(b):
                E = np.zeros((r, r)); E[blk * b + j, blk * b + k] = 1.0
                E_list.append(E)
        if blk < nblk - 1:
            A[blk * b + b - 1, (blk + 1) * b] = 1.0
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 1.0:
        A *= 0.85 / rho
    B = np.zeros((r, m)); B[0, :] = rng.randn(m)
    C = rng.randn(p, r) / np.sqrt(r)
    return A, B, C, E_list


# ---------------------------------------------------------------------------
# Part C: the three dimensions -- A-alone case
# ---------------------------------------------------------------------------
def d_ambient(A, E_list, tol=1e-9):
    _, _, dim = subspace_basis(
        [Aa @ E @ Ab for E in E_list
         for Aa in matrix_powers(A) for Ab in matrix_powers(A)], tol=tol)
    return dim


def d_credit_A_alone(A, B, C, E_list, tol=1e-9):
    """rank of the space of possible gradient-response FUNCTIONS across
    trainable A-directions: each E_k's full (a,b)-indexed response array
    is ONE row (all time-shift/output/input combinations for that single
    parameter); rank is computed ACROSS k, not across (k,a,b) jointly
    (which trivially collapses to <=1 for SISO -- an earlier, incorrect
    version of this function did exactly that and was caught before use)."""
    r = A.shape[0]
    powers = matrix_powers(A)
    rows = []
    for E in E_list:
        H = np.zeros((r, r, C.shape[0], B.shape[1]))
        for ai, Aa in enumerate(powers):
            for bi, Ab in enumerate(powers):
                H[ai, bi] = C @ Aa @ E @ Ab @ B
        rows.append(H.ravel())
    M = np.stack(rows, axis=0)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rank = _gap_rank(S, tol=tol)
    return rank, S


# ---------------------------------------------------------------------------
# Part B/C/D/E: the JOINT (A,B,C) case -- the practically relevant one,
# since routing (B,C's role) is trainable in the actual deep learner.
# Uses the actual impulse-response perturbation (needs matrix powers of
# A well past r-1, NOT the Cayley-Hamilton-truncated list used for the
# rank computation above -- an earlier version of this test truncated
# incorrectly and showed spurious "leakage" at t>r; fixed here).
# ---------------------------------------------------------------------------
def joint_credit(A, B, C, T_=None, tol=1e-9):
    r, m, p = A.shape[0], B.shape[1], C.shape[0]
    # generous default margin: a tight T_ (e.g. 3r+4) can leave the true
    # rank boundary numerically ambiguous for a badly-conditioned random
    # draw, caught directly at r=16 (T_=52 gave a spurious rank of 24
    # instead of the true 32 for one specific random system -- resolved
    # cleanly once T_ was increased, confirmed not a structural issue)
    T_ = T_ or (10 * r + 20)
    full_powers = [np.eye(r)]
    for _ in range(T_):
        full_powers.append(full_powers[-1] @ A)

    def impulse_response(dA, dB, dC):
        h = np.zeros((T_, p, m))
        for t in range(1, T_ + 1):
            val = full_powers[t - 1] @ dB
            val = C @ val
            for k in range(t - 1):
                val = val + C @ full_powers[k] @ dA @ full_powers[t - 2 - k] @ B
            val = val + dC @ full_powers[t - 1] @ B
            h[t - 1] = val
        return h.ravel()

    rows = []
    for j in range(r):
        for k in range(r):
            dA = np.zeros((r, r)); dA[j, k] = 1.0
            rows.append(impulse_response(dA, np.zeros_like(B), np.zeros_like(C)))
    for j in range(r):
        for mi in range(m):
            dB = np.zeros_like(B); dB[j, mi] = 1.0
            rows.append(impulse_response(np.zeros((r, r)), dB, np.zeros_like(C)))
    for j in range(r):
        for pi in range(p):
            dC = np.zeros_like(C); dC[pi, j] = 1.0
            rows.append(impulse_response(np.zeros((r, r)), np.zeros_like(B), dC))
    M = np.stack(rows, axis=0)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rank = _gap_rank(S, tol=tol)
    return rank, len(rows)


def _gap_rank(S, tol=1e-9):
    """Rank via the largest RELATIVE gap in the (log) singular-value
    spectrum, not a fixed threshold on S/S[0]. A fixed relative
    threshold fails once the true signal itself spans many orders of
    magnitude (e.g. from a well-decaying A's own spectral radius) --
    caught directly at r=16 in this phase, where tol=1e-9 cut off
    genuine signal (down to ~1e-12) before the true, unambiguous
    ~1e8-magnitude gap at the actual rank boundary. Falls back to the
    fixed-threshold count if no gap clearly exceeds it (e.g. rank-0/
    rank-full cases)."""
    if len(S) == 0 or S[0] <= 0:
        return 0
    logS = np.log10(np.maximum(S, 1e-300))
    gaps = logS[:-1] - logS[1:]
    if len(gaps) == 0:
        return 1
    best = int(np.argmax(gaps))
    if gaps[best] > 3.0:  # >= 3 orders of magnitude: unambiguous
        return best + 1
    return int(np.sum(S > tol * S[0]))


# ---------------------------------------------------------------------------
# Part F: gauge invariance -- real-gradient verification on an actual
# simulated trajectory (not just the abstract rank computation).
# ---------------------------------------------------------------------------
def simulate_loss(A, B, C, u, ytgt):
    r = A.shape[0]
    T_ = u.shape[0]
    h = np.zeros(r)
    ys = []
    for t in range(T_):
        h = A @ h + (B @ u[t])
        ys.append((C @ h).item())
    ys = np.array(ys)
    return 0.5 * float(np.mean((ys - ytgt) ** 2))


def verify_gauge_invariance(A, B, C, seed=0, T_=25, eps=1e-6):
    r = A.shape[0]
    rng = np.random.RandomState(seed)
    T_mat = rng.randn(r, r)
    dA = T_mat @ A - A @ T_mat
    dB = T_mat @ B
    dC = -C @ T_mat
    u = rng.randn(T_, B.shape[1])
    ytgt = rng.randn(T_)
    Lp = simulate_loss(A + eps * dA, B + eps * dB, C + eps * dC, u, ytgt)
    Lm = simulate_loss(A - eps * dA, B - eps * dB, C - eps * dC, u, ytgt)
    grad_gauge = (Lp - Lm) / (2 * eps)

    dA2 = rng.randn(r, r); dA2 *= np.linalg.norm(dA) / np.linalg.norm(dA2)
    Lp2 = simulate_loss(A + eps * dA2, B, C, u, ytgt)
    Lm2 = simulate_loss(A - eps * dA2, B, C, u, ytgt)
    grad_generic = (Lp2 - Lm2) / (2 * eps)
    return grad_gauge, grad_generic


def check_commutator_not_first_order_invariant_A_alone(A, B, C, seed=0):
    """Sanity check (Section 1 of PHASE_B22.md): confirms dA=[T,A] ALONE
    (B,C fixed) is NOT first-order invariant -- rules out d_credit_A_alone
    == r^2 being a bug in the rank computation."""
    r = A.shape[0]
    rng = np.random.RandomState(seed)
    T_mat = rng.randn(r, r)
    dA = T_mat @ A - A @ T_mat
    powers = matrix_powers(A)
    resp = [(C @ Aa @ dA @ Ab @ B).item() for Aa in powers for Ab in powers]
    return float(np.linalg.norm(resp))


def main() -> None:
    print("=" * 90)
    print("Phase B22: function class vs prospective credit")
    print("=" * 90)
    rng = np.random.RandomState(0)

    print("\nPart 1: A-alone (B,C fixed) -- dense vs companion, SISO")
    for r in (2, 4, 8, 16):
        A, B, C, E = make_dense_system(r, 1, 1, rng)
        rank, _ = d_credit_A_alone(A, B, C, E)
        print(f"  DENSE  r={r:3d}: dim(P)={len(E):4d}  d_credit(A-alone)={rank:4d}")
        A2, B2, C2, E2 = make_companion_system(r, 1, 1, rng)
        rank2, _ = d_credit_A_alone(A2, B2, C2, E2)
        print(f"  COMP   r={r:3d}: dim(P)={len(E2):4d}  d_credit(A-alone)={rank2:4d}")

    print("\nSanity check: commutator direction NOT first-order invariant, A-alone")
    A, B, C, E = make_dense_system(4, 1, 1, rng)
    print("  response norm:", check_commutator_not_first_order_invariant_A_alone(A, B, C))

    print("\nPart 2: joint (A,B,C), SISO -- exact 2r law")
    for r in (2, 4, 8, 16):
        A, B, C, E = make_dense_system(r, 1, 1, rng)
        rank, dimP = joint_credit(A, B, C)
        print(f"  r={r:3d}: dim(P)={dimP:4d}  d_credit={rank:4d}  predicted 2r={2*r}")

    print("\nPart 3: joint (A,B,C), MIMO -- exact r*(m+p) law")
    r = 8
    for m, p in [(1, 1), (2, 1), (1, 2), (2, 2), (4, 4), (8, 8)]:
        A, B, C, E = make_dense_system(r, m, p, rng)
        rank, dimP = joint_credit(A, B, C)
        print(f"  m={m} p={p}: dim(P)={dimP:4d}  d_credit={rank:4d}  r(m+p)={r*(m+p)}")

    print("\nPart 4: gauge invariance, real gradient on an actual trajectory")
    A, B, C, E = make_dense_system(6, 1, 1, rng)
    g_gauge, g_generic = verify_gauge_invariance(A, B, C)
    print(f"  gradient along gauge direction: {g_gauge}  (generic direction: {g_generic})")


if __name__ == "__main__":
    main()
