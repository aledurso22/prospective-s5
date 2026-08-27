"""B4B -- deployable streaming relevance estimator. No teacher state, no
BPTT, no future samples: receives only (u_t, c_t) one step at a time,
known architecture poles, and its own O(N) running statistics.

Per (candidate channel j, lower mode m), maintains exactly two complex
scalars:
  x_j       -- the channel's own causal filter state,
               x_j <- lambda_j x_j + u_t           (O(1) per channel)
  rho_j_acc -- the running cross-statistic accumulator,
               rho_j_acc <- rho_j_acc + conj(c_t[j]) x_j

This is the "direct pole-filter" implementation B4B calls for: no r_k
lag array, no stored trajectory, no reconstruction of the full P/Q
teacher. Memory per data stream (e.g. per batch element): O(2N) complex
during calibration/selection (one x_j per candidate channel, all 2N of
them, since which channel will win is not yet known), plus an O(2N)
accumulator shared across the stream; O(1) complex per stream once a
single channel is selected and frozen for deployment (only that
channel's own x_j is still needed; running BATCH streams in parallel
costs O(BATCH) copies of that O(1) state, not an O(N)-scaling cost).

Two accumulation modes:
  - WINDOWED (B4C): plain running sum over a fixed calibration window,
    frozen and used as-is afterward (matches B3's estimator exactly, but
    computed step-by-step here instead of batch numpy ops).
  - EMA (B4D): rho_j_acc <- (1-gamma) rho_j_acc + gamma * conj(c_t[j]) x_j,
    never stops adapting; gamma is the only new hyperparameter (a
    forgetting rate, distinct from any architecture pole).
"""
from __future__ import annotations

import numpy as np


class StreamingRelevance:
    """One instance per lower mode m. Call `step(u_t, c_t)` once per
    timestep (u_t, c_t[:] can be batched: u_t (BATCH,) complex, c_t
    (BATCH, 2N) complex -- batch elements are summed into the same
    accumulator, matching the raw-sum convention used throughout).
    """

    def __init__(self, f_diag, batch, mode="windowed", gamma=0.02):
        self.f_diag = np.asarray(f_diag, dtype=np.complex128)
        n2 = self.f_diag.shape[0]
        # per-(batch-element, channel) state: pooling batch BEFORE the
        # linear filter would still give the correct sum-of-states
        # (filtering is linear, sum(filter(u_b)) == filter(sum(u_b))),
        # but the CROSS TERM conj(c_t[b]) * x_t[b] is bilinear in
        # (c,u)-pairs that both vary independently per batch element b
        # (different random draws) -- pooling u before multiplying by
        # c would silently pair each c_t[b] with the WRONG (summed,
        # not that batch element's own) x. Caught before any B4C/D
        # number was trusted; fixed by keeping state per batch element.
        self.x = np.zeros((batch, n2), np.complex128)     # O(BATCH * 2N)
        self.rho = np.zeros(n2, np.complex128)             # accumulator O(2N)
        self.mode = mode
        self.gamma = gamma
        self.n_steps = 0
        self.history = []          # optional: rho snapshots over time
                                    # (diagnostic only, not part of the
                                    # deployed algorithm's memory)

    def step(self, u_t, c_t, record=False):
        """u_t: (BATCH,) complex, c_t: (BATCH, 2N) complex."""
        self.x = self.f_diag[None, :] * self.x + u_t[:, None]
        cross = np.sum(np.conj(c_t) * self.x, axis=0)     # (2N,), summed
                                                            # over batch
                                                            # AFTER pairing
        if self.mode == "windowed":
            self.rho = self.rho + cross
        elif self.mode == "ema":
            self.rho = (1 - self.gamma) * self.rho + self.gamma * cross
        else:
            raise ValueError(self.mode)
        self.n_steps += 1
        if record:
            self.history.append(self.rho.copy())

    def top_channel(self, r=1):
        return np.argsort(-np.abs(self.rho))[:r]

    def deployed_state_bytes(self, r=1):
        """Memory needed AFTER selection/freezing, for r deployed
        channels: r complex128 states (x_j for the selected channels
        only) -- no accumulator, no other channels' state needed."""
        return r * 16   # complex128 = 16 bytes


def run_windowed_calibration(f_diag, cal_rows, m):
    """Process calibration trajectories causally, one timestep at a
    time (genuinely streaming -- no vectorized whole-array reduction),
    return the frozen rho vector."""
    from credit_memory.hankel import build_c_t
    batch = cal_rows[0]["Sa0"].shape[1]
    est = StreamingRelevance(f_diag, batch, mode="windowed")
    for row in cal_rows:
        u_traj = row["Sa0"][:, :, m]                    # (T,BATCH)
        c_traj = build_c_t(row["q1"], row["B1"][:, m])   # (T,BATCH,2N)
        for t in range(u_traj.shape[0]):
            est.step(u_traj[t], c_traj[t])
    return est
