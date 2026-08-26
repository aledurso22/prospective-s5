# results/ — stored experiment artifacts

One directory per experiment, named after the **original** script name
(kept stable across the 2026-08 reorganization so stored paths still
resolve). Each holds a `summary.json` (config, git hash, per-seed
numbers, gates) and sometimes arrays/logs. Headline numbers with full
provenance: `../RESULTS_LEDGER.md`.

Directory → current script mapping highlights:

| results dir | produced by (current path) |
|---|---|
| `route_pc/` | `toyrig/routepc.py` (core PC0 protocol) |
| `core_routepc_reproduction/` | `core/package_core_routepc.py` |
| `control_2x2_normmatch/` | `controls/control_2x2_normmatch.py` |
| `prospective_offline2/` | `diagnostics/d1_exact_credit_factorization.py` |
| `oracle_real_vs_complex/` | `diagnostics/d2_modal_oracle.py` |
| `teacher_decompose/`, `prospective_kappa/`, `e1_e2_identification/`, `oracle_lagged_deficit/`, … | same-named `diagnostics/` scripts |
| `bootstrap_teacher_f1/` | `archive/failed_self_bootstrap/bootstrap_teacher_f1.py` |
| `bench/` | `train_bench.py` (cluster benchmark, when launched) |

Everything else maps by name to `diagnostics/`, `archive/`, or the solver
lane. Loose files at this level (`metrics_*.json`, `pesm_s5_spectrum.json`,
…) are early-lane artifacts kept for provenance.
