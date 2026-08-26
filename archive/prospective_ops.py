"""Shared modal prospective operators for the Stage A-D mechanism program.

THEORY (sequence-time; NOT optimizer-time — these clocks stay separate).

For one isolated modal credit pathway with pole a = r e^{i theta}, the
exact future filter is lam_t = sum_k conj(a)^k q_{t+k}. Demodulating
q'_t = e^{-i theta t} q_t turns it into a REAL-pole noncausal smoother
lam'_t = sum_k r^k q'_{t+k}. The kernel's zeroth and first moments give
the prospective (lead) expansion

    lam' ~= c0* q' + c1* d_t q',   c0* = 1/(1 - r),  c1* = r/(1 - r)^2

— the sequence-time D^dagger ~ (1 - a)(I + a dt/(1-a) d_t) reading,
made per-mode and causal-implementable. This is THEORY ONLY where the
sequence-time signal is not smooth (our own noise measurements apply);
the operators here are fixed analytic arms, not theorems.

Operators (all causal; applied in the demodulated frame; remodulated
before use):

  base      q'                                  (identity)
  gain      c0* q'                              (analytic gain only)
  raw       c0* q'_t + c1* (q'_t - q'_{t-1})    (raw lead difference)
  ema       m'_t = rho m'_{t-1} + (1-rho) q'_{t-1};
            qv'   = c0* q'_t + c1*(1-rho)(q'_t - m'_t)
            (stable lead; rho global in {0.5, 0.9, 0.99})
  matched   ema with per-mode rho_j = r_j = |a_j|  (theoretically
            motivated arm — NOT claimed globally optimal)
  oppphase  y_t = a y_{t-1} + q_t in the ORIGINAL frame — causal filter
            with the conjugate response of the exact future filter
            (same magnitude, opposite phase; the lead-sign control)

Placement rule (from phase_probes): the top recurrent layer's online
gradient is EXACT (RTRL identity), so operators apply only at layers
0..L-2 — the site of the cross-layer instantaneous-error approximation
— and layer L-1 keeps its raw q.
"""
from __future__ import annotations

import numpy as np


def demod(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    """q (T, B, N) -> e^{-i theta_j t} q, per mode."""
    T = q.shape[0]
    theta = np.angle(a)
    return q * np.exp(-1j * np.outer(np.arange(T), theta))[:, None, :]


def remod(qp: np.ndarray, a: np.ndarray) -> np.ndarray:
    T = qp.shape[0]
    theta = np.angle(a)
    return qp * np.exp(1j * np.outer(np.arange(T), theta))[:, None, :]


def c0c1(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.clip(r, 0.0, 1.0 - 1e-9)
    return 1.0 / (1.0 - r), r / (1.0 - r) ** 2


def apply_operator(q: np.ndarray, a: np.ndarray, arm: str,
                   rho: float | None = None) -> np.ndarray:
    """Apply one prospective arm to one layer's q ((T, B, N), possibly
    real) with poles a (N,). Returns the corrected signal in the
    ORIGINAL frame (complex)."""
    if arm == "base":
        return q.astype(np.complex128)
    if arm == "oppphase":
        y = np.zeros_like(q, dtype=np.complex128)
        acc = np.zeros(q.shape[1:], np.complex128)
        for t in range(q.shape[0]):
            acc = a[None, :] * acc + q[t]
            y[t] = acc
        return y
    qp = demod(np.asarray(q, np.complex128), a)
    r = np.abs(a)
    c0, c1 = c0c1(r)
    if arm == "gain":
        out = c0[None, None, :] * qp
    elif arm == "raw":
        d = np.diff(qp, axis=0, prepend=qp[:1])
        out = c0[None, None, :] * qp + c1[None, None, :] * d
    elif arm in ("ema", "matched"):
        rh = (np.abs(a) if arm == "matched"
              else np.full(q.shape[2], float(rho)))
        m = np.zeros_like(qp)
        for t in range(qp.shape[0]):
            if t > 0:
                m[t] = rh[None, :] * m[t - 1] + (1 - rh[None, :]) * qp[t - 1]
        out = (c0[None, None, :] * qp
               + c1[None, None, :] * (1 - rh[None, :]) * (qp - m))
    else:
        raise ValueError(f"unknown arm: {arm}")
    return remod(out, a)


def build_err(q_list: list, a_list: list, arm: str,
              rho: float | None = None) -> list:
    """Apply the operator at layers 0..L-2; leave the top layer's q
    untouched (its online gradient is already exact)."""
    L = len(q_list)
    out = [apply_operator(np.asarray(q_list[l], np.complex128), a_list[l],
                          arm, rho) for l in range(L - 1)]
    out.append(np.asarray(q_list[L - 1], np.complex128))
    return out


ARMS = ["base", "gain", "raw", "ema", "matched", "oppphase"]
