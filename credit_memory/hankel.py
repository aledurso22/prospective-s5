"""B2 core machinery: exact state-space form of the Phase-A causal-dual
system, its Gramians, Hankel spectrum, and square-root balanced
truncation. Per lower-layer mode m; L=2 only.

State-space form (B2A), per mode m, stacking the P-channel (N upper
modes) then the Q-channel (N upper modes) into one 2N-dim complex state:

  x_u = F x_{u-1} + d * Sa0_u[m],     F = diag(A, conj(A)),  A = diag(a1)
                                       d = ones(2N)

  g_t[m] = c_t^dagger x_t,             c_t = 0.5 * [ conj(B1[:,m]) * q1_t ;
                                                       B1[:,m] * conj(q1_t) ]

This is exactly Phase-A's (E2): the P-block of x is P[:,m], the Q-block
is Q[:,m], and c_t^dagger x_t reduces to
  0.5 * sum_j B1[j,m] conj(q1_t[j]) P_t[j,m]
  + 0.5 * sum_j conj(B1[j,m]) q1_t[j] Q_t[j,m]
by construction (see PHASE_A.md and PHASE_B2.md for the derivation of
c_t from (E2)).

Gramians:
  Wc solves Wc = F Wc F^dagger + d d^dagger        (architecture only,
                                                      d treated as a
                                                      unit-variance white
                                                      input -- a
                                                      normalization
                                                      choice, not derived
                                                      from data)
  Wo solves Wo = F^dagger Wo F + S,   S = E[c_t c_t^dagger]  (estimated
                                                      from a calibration
                                                      portion, OR S=I as
                                                      an isotropic prior
                                                      for the
                                                      architecture-only
                                                      level)

Both are closed-form since F is diagonal:
  Wc[p,q] = 1 / (1 - f_p * conj(f_q))
  Wo[p,q] = S[p,q] / (1 - conj(f_p) * f_q)
"""
from __future__ import annotations

import numpy as np


def build_F(a1):
    """a1: (N,) upper-layer poles -> f_diag: (2N,) = [A, conj(A)]."""
    return np.concatenate([a1, np.conj(a1)])


def build_c_t(q1_t, B1_col):
    """q1_t: (..., N) complex, B1_col = B1[:, m]: (N,) -> c_t: (..., 2N)."""
    top = 0.5 * np.conj(B1_col) * q1_t
    bot = 0.5 * B1_col * np.conj(q1_t)
    return np.concatenate([top, bot], axis=-1)


def analytic_Wc(f_diag):
    fp = f_diag[:, None]
    fq = f_diag[None, :]
    return 1.0 / (1.0 - fp * np.conj(fq))


def solve_Wo(f_diag, S):
    fp = f_diag[:, None]
    fq = f_diag[None, :]
    return S / (1.0 - np.conj(fp) * fq)


def estimate_S(q1_cal, B1_col):
    """q1_cal: (n_samples, N) -> S: (2N, 2N), mean outer product of c."""
    c = build_c_t(q1_cal, B1_col)          # (n_samples, 2N)
    return np.einsum("si,sj->ij", c, np.conj(c)) / c.shape[0]


def _herm_sqrt(M, eps=1e-13):
    """Hermitian PSD square root via eigh, clipping tiny/negative modes
    (numerical noise, not a modeling choice)."""
    w, V = np.linalg.eigh(M)
    w = np.clip(w.real, 0.0, None)
    return V @ np.diag(np.sqrt(w + eps)) @ V.conj().T, w, V


def hankel_singular_values(Wc, Wo):
    """sigma_i = sqrt(eig(Wc Wo)), via the numerically stable
    Wc^{1/2} Wo Wc^{1/2} route (same nonzero spectrum as Wc Wo)."""
    Rc, _, _ = _herm_sqrt(Wc)
    M = Rc.conj().T @ Wo @ Rc
    w, _ = np.linalg.eigh(M)
    w = np.clip(w.real, 0.0, None)
    sigma = np.sqrt(w)[::-1]               # descending
    return sigma


def balanced_transform(Wc, Wo, eps=1e-13):
    """Square-root balancing algorithm. Returns T, Tinv, sigma (descending)
    with Wc_bal = Wo_bal = diag(sigma) in the transformed (balanced)
    coordinates z = Tinv @ x."""
    Rc, _, _ = _herm_sqrt(Wc, eps=eps)
    M = Rc.conj().T @ Wo @ Rc
    w, U = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    w = np.clip(w.real, 0.0, None)
    sigma = np.sqrt(w + eps)
    T = Rc @ U @ np.diag(1.0 / sigma)
    Rc_inv = np.linalg.pinv(Rc)
    Tinv = np.diag(sigma) @ U.conj().T @ Rc_inv
    return T, Tinv, sigma


def reduced_system(f_diag, d, T, Tinv, r):
    """Petrov-Galerkin truncation to the top-r balanced coordinates.
    Returns F_r (r,r, generally dense), d_r (r,)."""
    F = np.diag(f_diag)
    F_bal = Tinv @ F @ T
    d_bal = Tinv @ d
    return F_bal[:r, :r], d_bal[:r]


def reduced_readout_vec(c_t, T, r):
    """c_t: (..., 2N) -> c_bal_t[:r]: (..., r), c_bal_t = T^dagger c_t."""
    return c_t @ np.conj(T)[:, :r]   # (T^dagger c)_k = sum_p conj(T_pk) c_p


def propagate(F_r, d_r, drive):
    """drive: (T, BATCH) complex (= Sa0[:, :, m]) -> z: (T, BATCH, r)."""
    Tn, BATCH = drive.shape
    r = d_r.shape[0]
    z = np.zeros((Tn, BATCH, r), np.complex128)
    z_prev = np.zeros((BATCH, r), np.complex128)
    for t in range(Tn):
        z_prev = z_prev @ F_r.T + drive[t][:, None] * d_r[None, :]
        z[t] = z_prev
    return z


def reduced_gradient(F_r, d_r, T, Sa0_m, q1, B1_col, r):
    """Full pipeline: reduced state + reduced readout -> per-step and
    summed gradient contribution for one mode m, one trajectory batch.
    Sa0_m: (T,BATCH); q1: (T,BATCH,N); B1_col: (N,)."""
    z = propagate(F_r, d_r, Sa0_m)                       # (T,BATCH,r)
    c_t = build_c_t(q1, B1_col)                          # (T,BATCH,2N)
    c_bal = reduced_readout_vec(c_t, T, r)                # (T,BATCH,r)
    g_t = np.sum(np.conj(c_bal) * z, axis=-1)             # (T,BATCH)
    return g_t, g_t.sum()
