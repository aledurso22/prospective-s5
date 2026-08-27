"""Shared exact-teacher machinery for Phase B, L=2 only.

This module re-states the Phase-A equations (`credit_memory/PHASE_A.md`,
verified in `credit_memory/phase_a_causal_dual.py`) in a reusable form so
Phase B code can probe/compress them without editing the already-frozen
Phase-A verification script. The equations themselves are NOT changed --
this is the same (E2) two-channel causal-dual identity, same conjugation
convention (`conj(a_1)` adjoint pole, `B_1[j,m]` routing, `Re(.)`
inter-layer coupling), against the same trusted BPTT reference
(`toyrig.ssm_rig.exact_lambda` + `assemble(..., direct=True)`).

All arrays are (T, BATCH, N) or (T, BATCH, N1, N0) as noted. Layer 0 =
lower (defective) layer, layer 1 = upper (top) layer, L is fixed to 2 by
the caller via `set_l2_config`.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from toyrig import ssm_rig as tcg


@contextmanager
def set_l2_config(N, T, BATCH, DELAY=0):
    keep = (tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY)
    tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY = 2, N, T, BATCH, DELAY
    try:
        yield
    finally:
        tcg.L, tcg.N, tcg.T, tcg.BATCH, tcg.DELAY = keep


def draw_trajectory(params, rng, T, BATCH):
    """One (x, r) trajectory batch: arbitrary input + arbitrary residual,
    matching the Phase 1 protocol ("generate arbitrary error sequences
    q_t") and credit_memory/phase_a_causal_dual.py's own convention."""
    x = rng.randn(T, BATCH)
    r = rng.randn(T, BATCH)
    return x, r


def compute_teacher(params, x, r):
    """Everything B1.0 asks to be exposed, for the fixed L=2 config
    already set via `set_l2_config`.

    Returns a dict with:
      q0, q1        : (T,B,N)   naive spatial errors (existing, online-
                                  available; NOT temporally exact above
                                  layer 0's own layer)
      Sa0, Sb0      : (T,B,N), (T,B,N,M)  existing within-layer eligibility
                                  (layer 0), unchanged from ssm_rig
      lam0, lam1    : (T,B,N)   exact BPTT adjoint (layer 0, layer 1)
      P, Q          : (T,B,N1,N0)  Phase-A two-channel causal-dual state
                                  (E1), per (upper mode j, lower mode m)
      g_t_online    : (T,B,N0)  per-step online contribution,
                                  conj(q0_t) * Sa0_t
      g_t_causal    : (T,B,N0)  per-step exact causal-dual contribution
                                  (E2)'s per-t summand, built from P,Q,
                                  q1 -- NO reverse-time pass used here
      g_t_bptt      : (T,B,N0)  per-step exact BPTT contribution,
                                  conj(lam0_t) * h0_{t-1}
      G_online      : (N0,)     sum_t,b g_t_online / B  (layer-0 "a" grad)
      G_causal      : (N0,)     sum_t,b g_t_causal / B  (must == G_bptt to
                                  machine precision -- Phase A's result)
      G_bptt        : (N0,)     sum_t,b g_t_bptt / B    (trusted reference,
                                  == assemble(direct=True)["a"][0])
    """
    L, N = tcg.L, tcg.N
    assert L == 2, "credit_memory.teacher is L=2 only (Phase B1 scope)"
    h, yhat = tcg.forward(params, x)
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    lam = tcg.exact_lambda(params, q)
    G_ex = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)

    a1 = params["a"][1]          # (N,) upper-layer poles
    B1 = params["b"][1]          # (N, N): B1[j, m]
    Sa0 = Sa[0]                  # (T, BATCH, N) existing eligibility
    q0, q1 = q[0], q[1]
    T, BATCH = Sa0.shape[0], Sa0.shape[1]

    P = np.zeros((T, BATCH, N, N), np.complex128)   # [t,b,j,m]
    Q = np.zeros((T, BATCH, N, N), np.complex128)
    runP = np.zeros((BATCH, N, N), np.complex128)
    runQ = np.zeros((BATCH, N, N), np.complex128)
    for t in range(T):
        runP = a1[None, :, None] * runP + Sa0[t][:, None, :]
        runQ = np.conj(a1)[None, :, None] * runQ + Sa0[t][:, None, :]
        P[t] = runP
        Q[t] = runQ

    g_t_causal = 0.5 * np.einsum("jm,tbj,tbjm->tbm", B1, np.conj(q1), P) \
        + 0.5 * np.einsum("jm,tbj,tbjm->tbm", np.conj(B1), q1, Q)
    g_t_online = np.conj(q0) * Sa0

    h0_prev = np.concatenate([np.zeros_like(h[0][:1]), h[0][:-1]], axis=0)
    g_t_bptt = np.conj(lam[0]) * h0_prev

    # raw sum over (t, batch) -- matches toyrig.ssm_rig.assemble's own
    # convention (no batch averaging) so this is directly comparable to
    # G_bptt = assemble(..., direct=True)["a"][0] below.
    G_causal = g_t_causal.sum(axis=(0, 1))
    G_online = g_t_online.sum(axis=(0, 1))
    G_bptt = G_ex["a"][0]

    return dict(q0=q0, q1=q1, Sa0=Sa0, Sb0=Sb[0], lam0=lam[0], lam1=lam[1],
               P=P, Q=Q, a1=a1, B1=B1,
               g_t_online=g_t_online, g_t_causal=g_t_causal,
               g_t_bptt=g_t_bptt,
               G_online=G_online, G_causal=G_causal, G_bptt=G_bptt,
               h0_prev=h0_prev)
