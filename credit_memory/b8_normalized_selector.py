"""B8: resource-normalized rank-1 CCM selector. No new theory of causal
credit (B7 settled that); this tests only whether the SELECTION
CRITERION used to pick a rank-1 channel is unnecessarily state-energy
biased.

B8E theory check (verified below, matching the repo's actual conjugation
convention): maximize |sum_t c_t^dagger (alpha x_{j,t})|^2 subject to
sum_t |alpha x_{j,t}|^2 <= 1, over complex scalar alpha, for fixed
candidate j:
  objective = |alpha|^2 |rho_j|^2,  constraint = |alpha|^2 E_j <= 1
  (rho_j := sum_t conj(c_t[j]) x_{j,t}, the repo's established Hermitian
   pairing, credit_memory/lagcorr.py's g_p; E_j := sum_t |x_{j,t}|^2)
Since the objective is increasing in |alpha|^2, the constraint saturates:
|alpha|^2 = 1/E_j, giving objective = |rho_j|^2 / E_j -- exactly the
boxed R_j. The phase of alpha is free (does not affect the objective),
consistent with alpha representing an unconstrained complex readout
gain. This is the SAME Hermitian pairing verified throughout B1-B7, not
a new convention.

B8B streaming implementation: extends credit_memory.streaming.
StreamingRelevance (not modified in place -- a subclass here) with one
extra real accumulator E_j per candidate channel. No BPTT, no full P/Q
teacher, no stored trajectory, no lag arrays, in the algorithm itself.

Run:  python -m credit_memory.b8_normalized_selector
"""
from __future__ import annotations

import numpy as np

from credit_memory.streaming import StreamingRelevance

EPS = 1e-9   # fixed, numerically-harmless floor; not tuned against BPTT


class StreamingRelevanceNormalized(StreamingRelevance):
    """Adds E_j = sum_t |x_{j,t}|^2 alongside the existing x_j, rho_j
    (unchanged base-class state/update)."""

    def __init__(self, f_diag, batch, mode="windowed", gamma=0.02):
        super().__init__(f_diag, batch, mode=mode, gamma=gamma)
        n2 = self.f_diag.shape[0]
        self.E = np.zeros(n2, dtype=np.float64)

    def step(self, u_t, c_t, record=False):
        super().step(u_t, c_t, record=record)
        # self.x was just updated (per-batch-element state); accumulate
        # its squared magnitude, summed over batch, matching the same
        # raw-sum-over-(t,batch) convention used everywhere else
        self.E = self.E + np.sum(np.abs(self.x) ** 2, axis=0)

    def R(self):
        return (np.abs(self.rho) ** 2) / (self.E + EPS)

    def top_channel_normalized(self, r=1):
        return np.argsort(-self.R())[:r]


def run_windowed_calibration_normalized(f_diag, cal_rows, m):
    """Same protocol as credit_memory.phase_b4c_streaming_rank1's
    run_windowed_calibration, extended to also return E."""
    from credit_memory.hankel import build_c_t
    batch = cal_rows[0]["Sa0"].shape[1]
    est = StreamingRelevanceNormalized(f_diag, batch, mode="windowed")
    for row in cal_rows:
        u_traj = row["Sa0"][:, :, m]
        c_traj = build_c_t(row["q1"], row["B1"][:, m])
        for t in range(u_traj.shape[0]):
            est.step(u_traj[t], c_traj[t])
    return est
