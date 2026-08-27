"""B5: deployable B4 rank-1 correction, generalized from the "a"-block
(verified throughout B3/B4) to ALSO cover the "b"-block, so a single
selected channel gives a complete, usable layer-0 parameter gradient
during actual training.

Phase A's LEMMA 1 (PHASE_A.md) is stated for an ARBITRARY driving
signal paired with an arbitrary local sensitivity -- it applies
identically whether that local sensitivity is Sa (paired with h_prev,
the "a"-gradient) or Sb (paired with the layer's own input x^0, the
"b"-gradient); only the driving signal Sa0 itself is shared between the
two (both P/Q channels are filtered versions of the SAME Sa0, per
Phase A -- the choice of parameter block only changes what gets
paired with the filtered channel at the readout step, not the channel
dynamics). This module implements exactly that: reuse the identical
selected-channel pole and readout weight (unchanged from B4C's verified
"a"-block construction) and apply it, via the same filter, to Sb0 as
well.

This is a NEW (not previously numerically verified) but structurally
identical extension; b5_train.py verifies it against BPTT at step 0 of
every run before using it for any training update, and reports that
check in every run's output.
"""
from __future__ import annotations

import numpy as np


def selected_channel_readout(f_diag, top_j, B1_col, N, q1):
    """Exactly credit_memory.phase_b4c_streaming_rank1.deploy_selected_
    channel's c_j construction, factored out for reuse (same formula, not
    reimplemented independently -- avoids a second chance to get the
    conjugation convention wrong)."""
    j_orig = top_j % N
    is_Q = top_j >= N
    q1_j = q1[:, :, j_orig]
    Bcoef = B1_col[j_orig]
    c_j = (0.5 * Bcoef * np.conj(q1_j) if is_Q
          else 0.5 * np.conj(Bcoef) * q1_j)
    lam = f_diag[top_j]
    return c_j, lam


def b4_layer0_gradient(f_diag, top_j_by_mode, B1, N, q1, Sa0, Sb0):
    """Sa0: (T,BATCH,N) existing layer-0 eligibility; Sb0: (T,BATCH,N,M)
    existing layer-0 input sensitivity; q1: (T,BATCH,N).
    Returns Ga (N,) complex, Gb (N,M) complex -- raw sums over (t,batch),
    same convention as toyrig.ssm_rig.assemble's own Ga/Gb."""
    Tn, Bn = Sa0.shape[0], Sa0.shape[1]
    M = Sb0.shape[-1]
    Ga = np.zeros(N, np.complex128)
    Gb = np.zeros((N, M), np.complex128)
    for m in range(N):
        top_j = top_j_by_mode[m]
        c_j, lam = selected_channel_readout(f_diag, top_j, B1[:, m], N, q1)

        xa = np.zeros((Tn, Bn), np.complex128)
        prevA = np.zeros(Bn, np.complex128)
        xb = np.zeros((Tn, Bn, M), np.complex128)
        prevB = np.zeros((Bn, M), np.complex128)
        for t in range(Tn):
            prevA = lam * prevA + Sa0[t, :, m]
            prevB = lam * prevB + Sb0[t, :, m, :]
            xa[t] = prevA
            xb[t] = prevB

        Ga[m] = np.sum(np.conj(c_j) * xa)
        Gb[m] = np.sum(np.conj(c_j)[:, :, None] * xb, axis=(0, 1))
    return Ga, Gb
