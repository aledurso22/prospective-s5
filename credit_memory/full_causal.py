"""B7: the FULL exact Phase-A causal (P/Q) forward-credit reconstruction
for layer 0's "a" AND "b" gradient blocks -- no compression, no channel
selection, all 2N candidate channels used exactly as derived. Uses the
exact repository Hermitian/conjugation convention established throughout
Phase A/B1-B6 (conj(c_t), conj(a) adjoint pole -- not a plain-transpose
convention).

  P_t[j,m] = a1[j]      P_{t-1}[j,m] + Sa0_t[m]
  Q_t[j,m] = conj(a1[j]) Q_{t-1}[j,m] + Sa0_t[m]

For the "b" block, the identical construction is applied with Sb0
(the layer's existing input-sensitivity) substituted for Sa0 as the
driving signal (Phase A's LEMMA 1 applies to an arbitrary paired local
sensitivity; credit_memory/b4_deploy.py already established this same
generalization for the rank-1/selected-channel case -- this module is
the FULL, uncompressed version of that same generalization):

  Ga[m]   = (1/2) sum_j B1[j,m]      sum_t conj(q1_t[j]) P_t[j,m]
          + (1/2) sum_j conj(B1[j,m]) sum_t q1_t[j]      Q_t[j,m]
  Gb[m,:] = identical, with P/Q built from Sb0 instead of Sa0

This is NOT a new theory or approximation -- it is exactly Phase A's
(E2), reused unmodified, generalized to "b" the same way B4 already was.
credit_memory/teacher.py's G_causal (already verified to machine
precision against BPTT) is the "a"-only special case of Ga below;
B7's own verification (credit_memory/b7_verify_exact.py) re-confirms
this and additionally verifies Gb.
"""
from __future__ import annotations

import numpy as np


def full_causal_gradient(a1, B1, N, q1, Sa0, Sb0):
    """a1: (N,) upper poles. B1: (N,N) routing [j,m]. q1: (T,BATCH,N).
    Sa0: (T,BATCH,N) existing eligibility. Sb0: (T,BATCH,N,M) existing
    input sensitivity. Returns Ga (N,) complex, Gb (N,M) complex --
    exact, uncompressed, all-channel reconstruction."""
    Tn, BATCH = Sa0.shape[0], Sa0.shape[1]
    M = Sb0.shape[-1]

    runPa = np.zeros((BATCH, N, N), np.complex128)
    runQa = np.zeros((BATCH, N, N), np.complex128)
    runPb = np.zeros((BATCH, N, N, M), np.complex128)
    runQb = np.zeros((BATCH, N, N, M), np.complex128)
    P_a = np.zeros((Tn, BATCH, N, N), np.complex128)
    Q_a = np.zeros((Tn, BATCH, N, N), np.complex128)
    P_b = np.zeros((Tn, BATCH, N, N, M), np.complex128)
    Q_b = np.zeros((Tn, BATCH, N, N, M), np.complex128)
    for t in range(Tn):
        runPa = a1[None, :, None] * runPa + Sa0[t][:, None, :]
        runQa = np.conj(a1)[None, :, None] * runQa + Sa0[t][:, None, :]
        runPb = a1[None, :, None, None] * runPb + Sb0[t][:, None, :, :]
        runQb = np.conj(a1)[None, :, None, None] * runQb \
            + Sb0[t][:, None, :, :]
        P_a[t], Q_a[t], P_b[t], Q_b[t] = runPa, runQa, runPb, runQb

    Ga = 0.5 * np.einsum("jm,tbj,tbjm->m", B1, np.conj(q1), P_a) \
        + 0.5 * np.einsum("jm,tbj,tbjm->m", np.conj(B1), q1, Q_a)
    Gb = 0.5 * np.einsum("jm,tbj,tbjmk->mk", B1, np.conj(q1), P_b) \
        + 0.5 * np.einsum("jm,tbj,tbjmk->mk", np.conj(B1), q1, Q_b)
    return Ga, Gb
