"""HiPPO-LegS construction and its (complex) diagonalization, S5-style.

Two routes are provided:

1. ``make_hippo_legs`` / ``hippo_eig_direct`` — the raw HiPPO-LegS matrix
        A[n, k] = -sqrt(2n+1) sqrt(2k+1)   if n > k
                  -(n+1)                   if n == k
                  0                        if n < k
   with B_hippo[n] = sqrt(2n+1), and its direct eigendecomposition.
   NOTE: the eigenvalues of this matrix are exactly -1, -2, ..., -N (all
   real) and its eigenvector matrix is EXPONENTIALLY ill-conditioned
   (cond(V) ~ 1e10 at N=16, ~1e20 at N=64), so V^{-1} — and hence
   B~ = V^{-1} B — cannot be computed in float64/float32. This route is
   kept for reference/diagnostics only.

2. ``make_dplr_hippo`` — the route S5 actually uses (s5/ssm_init.py):
   the HiPPO-LegS matrix has the NPLR structure  A = S - P P^T  with
   P[n] = sqrt(n + 1/2), where  S = A + P P^T = -0.5 I + K  is NORMAL
   (K real skew-symmetric). S is diagonalized by a UNITARY matrix V:
        S = V diag(Lambda) V*,   Lambda = -0.5 + i * eig(K)  (complex),
   computed stably via ``eigh(S * -1j)`` (Hermitian). Inputs are
   transformed to the diagonal basis as  B~ = V* B_hippo  (= V^{-1} B since
   V is unitary, cond(V) = 1), and outputs map back with Re(V .) — i.e. we
   work in the diagonal (complex) basis exactly like S5, which keeps only
   the diagonal (D) of the DPLR and treats (Lambda, B~, C) as trainable.
"""

from __future__ import annotations

import numpy as np


def make_hippo_legs(state_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct the HiPPO-LegS matrix and input vector.

    Args:
        state_size: N, the SSM state dimension.

    Returns:
        A: (N, N) real HiPPO-LegS matrix.
        B: (N,) real input vector, B[n] = sqrt(2n + 1).
    """
    n = np.arange(state_size)
    q = np.sqrt(2 * n + 1.0)
    # A[n, k] = -q[n] q[k] if n > k, -(n+1) if n == k, 0 otherwise.
    A = -q[:, None] * q[None, :]
    A = np.tril(A, k=-1) + np.diag(-(n + 1.0))
    B = q.copy()
    return A.astype(np.float64), B.astype(np.float64)


def hippo_eig_direct(state_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Direct eigendecomposition of HiPPO-LegS (REFERENCE ONLY — see above).

    The eigenvalues are -1, ..., -N (real) and V is exponentially
    ill-conditioned, so this must not be used for V^{-1} transforms at
    N >~ 16.
    """
    A, _ = make_hippo_legs(state_size)
    Lambda, V = np.linalg.eig(A)
    order = np.argsort(Lambda.real)
    return Lambda[order].astype(np.complex64), V[:, order].astype(np.complex64)


def make_nplr_hippo(state_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NPLR representation of HiPPO-LegS (from annotated-S4 / S5).

    Returns:
        A: (N, N) HiPPO-LegS matrix.
        P: (N,) low-rank factor, P[n] = sqrt(n + 1/2);  A + P P^T is normal.
        B: (N,) HiPPO input vector, B[n] = sqrt(2n + 1).
    """
    A, B = make_hippo_legs(state_size)
    P = np.sqrt(np.arange(state_size) + 0.5)
    return A, P, B


def make_dplr_hippo(state_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize the NORMAL part of HiPPO-LegS (the S5 initialization).

    S = A + P P^T = -0.5 I + K with K skew-symmetric, so S = V diag(Lambda) V*
    with V unitary and Lambda = -0.5 + i * eig(K).

    Args:
        state_size: N, the SSM state dimension.

    Returns:
        Lambda: (N,) complex64 eigenvalues of the normal part (Re = -0.5).
        V:      (N, N) complex64 UNITARY eigenvector matrix.
        B_tilde: (N,) complex64 input vector in the diagonal basis,
                 B_tilde = V* B_hippo = V^{-1} B_hippo.
    """
    A, P, B = make_nplr_hippo(state_size)
    S = A + P[:, None] * P[None, :]
    Lambda_real = np.mean(np.diagonal(S)) * np.ones(state_size)  # = -0.5
    # S * (-1j) is Hermitian; eigh is stable and returns a unitary V.
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    B_tilde = V.conj().T @ B
    Lambda = Lambda_real + 1j * Lambda_imag
    return (Lambda.astype(np.complex64), V.astype(np.complex64),
            B_tilde.astype(np.complex64))


def hippo_init(state_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HiPPO initialization bundle used by the SSM layers (S5/DPLR route).

    Returns:
        Lambda: (N,) complex64 eigenvalues (diagonal init).
        V:      (N, N) complex64 unitary eigenvector matrix.
        B_tilde: (N, 1) complex64 input vector in the diagonal basis.
    """
    Lambda, V, B_tilde = make_dplr_hippo(state_size)
    return Lambda, V, B_tilde[:, None].astype(np.complex64)


if __name__ == "__main__":
    N = 64
    Lam, V, Bt = hippo_init(N)
    print(f"N={N}  cond(V) = {np.linalg.cond(V.astype(np.complex128)):.3f}")
    print(f"Re(Lambda) = {np.unique(np.round(Lam.real, 8))}  "
          f"|Lambda| max = {np.abs(Lam).max():.2f}")
    print(f"|B_tilde| range: [{np.abs(Bt).min():.3f}, {np.abs(Bt).max():.3f}]")
    Ld, Vd = hippo_eig_direct(N)
    print(f"direct eig: eigenvalues in [{Ld.real.min():.1f}, {Ld.real.max():.1f}]"
          f" (real), cond(V_direct) = {np.linalg.cond(Vd.astype(np.complex128)):.2e}"
          "  <- why the DPLR route is used")
