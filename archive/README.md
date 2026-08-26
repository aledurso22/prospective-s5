# archive/ — concluded negatives and superseded lanes

Preserved for provenance. Every lane here is closed with a named
mechanism in `../EXPERIMENTS.md`. Do not reopen without new evidence;
do not delete — the ledger cites these files.

| path | lane | why archived |
|---|---|---|
| `solver/` | PESM/Newton solver (pesm_s5_spectrum, s5_state_inference, plds_*, registered_stiff_deq, registered_bissm) | a **positive but separate** program line (state-side curvature mass); stiff-DEQ training and BISSM killed inside it. Frozen. |
| `credit_filters/` | lane 2/4 (pac_probe v1, pac_deploy 1–4, orient_wiener, wiener_oracle, transfer_m, registered_oracle_b) | causal filter deployment is variance/staleness-limited (≤40% best); LTI deployment catastrophic; oracle-B result retracted. |
| `failed_self_bootstrap/` | F1 (bootstrap_teacher_f1.py) | self-EMA teacher self-amplifies (NaN most seeds); preregistered STOP — no observer stage built. |
| `forward_prospection/` | lane 0 (exact_failure, ghost_demo) | full prospection cancels the memory spectrum; two-step Euler adds a parasitic mode (the "ghost"). |
| `covariant_adam.py` | gauge reading | defect is architectural, not Adam's U(1) covariance. |
| `derive_phase.py` | derivation attempt | scalar phase not derivable from the operator alone (identically zero). |
| `recheck_curvature.py` / `recheck_curvature_matrix.py` | curvature readings | w is not a curvature object (anti-Newton corr −0.90). |
| `test_holonomy.py` | holonomy | shallow phase not additive down the stack (0/3). |
| `prospective_offline.py` / `prospective_ops.py` | Stage A v1 | INVALID placement (filtered the routed signal with the wrong-layer pole); superseded by `diagnostics/d1_exact_credit_factorization.py`. Kept because the ledger records the v1→v2 correction. |
