"""B3 core machinery: lagged eligibility-readout cross-correlation
decomposition of the exact causal-dual gradient, and the R1/R2/R3
relevance constructions built from it.

Convention (matches credit_memory/hankel.py exactly): x_t = F x_{t-1} +
d u_t, G = sum_t c_t^dagger x_t, F diagonal (f_diag), d = ones(2N).

Lag decomposition (B3A):
  x_t = sum_{k>=0} F^k d u_{t-k}
  G   = sum_{k>=0} r_k @ (F^k @ d),   r_k[p] := sum_t conj(c_t[p]) u_{t-k}

Since F is diagonal, F is stored as its diagonal f_diag (2N,) throughout,
and F^k @ d = f_diag**k (elementwise, since d is all-ones).

Per-coordinate exact contribution (used by R1):
  x_t[p] = d[p] * sum_k f_p^k u_{t-k}
  g_p := sum_t conj(c_t[p]) x_t[p],     sum_p g_p == G exactly

Cross-Gramian (used by R2):
  M_cross := sum_{k=0}^{Kmax} (F^k d) (x) r_k    (outer product, 2N x 2N)
  trace(M_cross) == G exactly (for Kmax = T-1, i.e. no truncation)
  Eigen-decomposition M_cross = V diag(lambda) V^{-1} gives an EXACT
  modal split of G = sum_i lambda_i; a rank-r reduction keeps the r
  eigenpairs with largest |lambda_i| and Galerkin-projects (F, d) onto
  that (generally non-orthogonal) subspace.
"""
from __future__ import annotations

import numpy as np


def lagged_r_k(c_t, u_t, K):
    """c_t: (T,BATCH,2N) complex, u_t: (T,BATCH) complex -> r: (K+1,2N).
    r_k[p] = sum_{t,b} conj(c_t[t,b,p]) * u_t[t-k,b]  (u_{<0}:=0)."""
    Tn = c_t.shape[0]
    N2 = c_t.shape[-1]
    r = np.zeros((K + 1, N2), np.complex128)
    for k in range(K + 1):
        if k == 0:
            r[k] = np.einsum("tbp,tb->p", np.conj(c_t), u_t)
        else:
            r[k] = np.einsum("tbp,tb->p", np.conj(c_t[k:]), u_t[:-k])
    return r


def lag_decomposition_gradient(f_diag, r):
    """r: (K+1, 2N) -> partial sums G_K for K=0..len(r)-1, and full G."""
    K1, N2 = r.shape
    fk = np.ones(N2, np.complex128)          # f_diag^0
    partial = np.zeros(K1, np.complex128)
    acc = 0.0 + 0.0j
    for k in range(K1):
        acc = acc + np.sum(r[k] * fk)
        partial[k] = acc
        fk = fk * f_diag
    return partial


def per_coordinate_contribution(f_diag, d, c_t, u_t):
    """Exact per-coordinate decomposition g_p (R1's relevance score),
    computed directly (not via the lag sum) for numerical independence:
    x_t[p] = d[p] * xi_p_t, xi_p_t = f_p xi_{p,t-1} + u_t (per-batch)."""
    Tn, BATCH = u_t.shape
    N2 = f_diag.shape[0]
    xi = np.zeros((Tn, BATCH, N2), np.complex128)
    prev = np.zeros((BATCH, N2), np.complex128)
    for t in range(Tn):
        prev = f_diag[None, :] * prev + u_t[t][:, None]
        xi[t] = prev
    x = d[None, None, :] * xi                                  # (T,B,2N)
    g_p = np.einsum("tbp,tbp->p", np.conj(c_t), x)
    return g_p, x


def cross_gramian(f_diag, d, r, Kmax=None):
    """r: (K+1,2N) -> M_cross (2N,2N) = sum_k outer(F^k d, r_k)."""
    K1 = r.shape[0] if Kmax is None else min(Kmax + 1, r.shape[0])
    N2 = f_diag.shape[0]
    fk_d = np.array(d, dtype=np.complex128)
    M = np.zeros((N2, N2), np.complex128)
    for k in range(K1):
        M += np.outer(fk_d, r[k])
        fk_d = fk_d * f_diag
    return M


def top_r_eigen_reduction(M, F, d, r, eps=1e-10):
    """Eigendecompose M (any square matrix); return the rank-r Galerkin
    reduction (F_r, d_r, V_r) built from the r eigenpairs with largest
    |eigenvalue|. F: (2N,2N) dense (diag(f_diag) here), d: (2N,)."""
    N2 = M.shape[0]
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))[:r]
    V_r = V[:, order]                          # (2N, r)
    Vinv = np.linalg.pinv(V_r) if r < N2 else np.linalg.inv(V)
    if r < N2:
        # biorthogonal projector onto the top-r eigenspace: use the
        # pseudo-inverse of V_r directly as the left projection (exact
        # when V_r's columns are linearly independent, standard
        # oblique/Galerkin projection otherwise)
        W_r = Vinv                              # (r, 2N)
    else:
        W_r = Vinv[order, :]
    F_r = W_r @ F @ V_r
    d_r = W_r @ d
    return F_r, d_r, V_r, w[order]


def propagate_general(F_r, d_r, drive):
    """drive: (T,BATCH) -> z: (T,BATCH,r), z_t = F_r z_{t-1} + d_r u_t
    (F_r possibly dense, not diagonal)."""
    Tn, BATCH = drive.shape
    r = d_r.shape[0]
    z = np.zeros((Tn, BATCH, r), np.complex128)
    z_prev = np.zeros((BATCH, r), np.complex128)
    for t in range(Tn):
        z_prev = z_prev @ F_r.T + drive[t][:, None] * d_r[None, :]
        z[t] = z_prev
    return z


def reduced_gradient_general(F_r, d_r, V_r, drive, c_t):
    """Full pipeline for a general (non-balanced) rank-r reduction:
    z = propagate; readout c_r_t = V_r^dagger c_t; g_t = c_r_t^dagger z_t."""
    z = propagate_general(F_r, d_r, drive)
    c_r = c_t @ np.conj(V_r)            # (T,BATCH,r) = (V_r^dagger c)^T pointwise
    g_t = np.sum(np.conj(c_r) * z, axis=-1)
    return g_t, g_t.sum()


def freq_domain_g_p(f_diag, d, u_t, c_t, n_fft=None):
    """R3: frequency-domain recomputation of per-coordinate g_p via
    circular cross-spectrum and the known transfer function
    H_p(w) = d[p] / (1 - f_p e^{-iw}). Pooled over batch. Returns g_p
    (2N,) -- compare to per_coordinate_contribution's time-domain g_p."""
    Tn, BATCH = u_t.shape
    n = n_fft or Tn
    U = np.fft.fft(u_t, n=n, axis=0)                 # (n, BATCH)
    Cc = np.fft.fft(c_t, n=n, axis=0)                 # (n, BATCH, 2N)
    omega = 2 * np.pi * np.arange(n) / n
    z = np.exp(-1j * omega)                            # (n,)
    H = d[None, :] / (1.0 - f_diag[None, :] * z[:, None])   # (n, 2N)
    # cross-spectrum S_uc[w,p] = conj(C[w,p]) * U[w] summed over batch,
    # Parseval: sum_t conj(c_t[p]) x_t[p] = (1/n) sum_w conj(S_uc)[w,p]...
    # implemented directly via the standard DFT Parseval identity below.
    Ubar = np.conj(U)[:, :, None]                      # (n,BATCH,1)
    cross = np.sum(Cc * Ubar, axis=1)                   # (n, 2N) pooled batch
    g_p = np.sum(np.conj(cross) * H, axis=0) / n
    return g_p
