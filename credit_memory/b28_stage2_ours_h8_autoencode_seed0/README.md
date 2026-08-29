# Archive: OURS H=8 Autoencode seed-0, 1M frames — frozen, no further compute

See `PHASE_B28.md` §7 for the full diagnosis. This directory holds the
raw artifacts referenced there.

- `production_run.log` / `production_checkpoints.json` — the actual
  training run (task-tracked, checkpoints at 100k/250k/500k/1M), config
  r=16 k=2 n=24 nr=384 encoder-width=64 Phi-hidden=8, produced by
  `credit_memory/b28_ours_calibration_run.py`.
- `return_trajectory_500k_to_1M.json` — rolling recent500/recent50
  returns reconstructed from the per-episode list in
  `production_checkpoints.json` at 100k increments (no checkpoint was
  configured at those intermediate frames), plus the RTU seed-0
  reference values for the same two anchor frames.
- `shadow_r_projection_250k_plus5k.json` — read-only R-spectral-radius
  probe from a copy of the 250k checkpoint, 5k additional steps, no
  parameter updates.
- `shadow_stageb_250k.json` / `shadow_stageb_1M.json` — the read-only
  true-PLAY confusion-matrix / target-conditioned-probability /
  phase-conditioned-entropy / linear hidden-state target-decoding
  probe, run identically at the 250k and 1M checkpoints (both from
  frozen-parameter copies, no training).

**Known gaps, recorded rather than silently left out:**
- No 500k parameter checkpoint was preserved (the live run's single
  state file gets overwritten in place at each checkpoint; a copy was
  only made at 250k and at 1M). The 250k shadow diagnostic is the
  nearest available pre-collapse substitute — not literally 500k.
- No frozen RTU parameter checkpoint exists for any frame (the RTU
  driver, `b28_rtu_calibration_run_v2.py`, never pickled parameter
  state — it predates the resumability feature added for this run).
  Only RTU's aggregate JSON/log diagnostics exist, referenced above.
