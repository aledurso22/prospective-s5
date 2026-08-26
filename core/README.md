# core/ — canonical deployable algorithms (toy)

Entry points for the frozen deployable algorithms. The implementations
live in `toyrig/` (frozen); these are the clean, documented runners.

| entry point | what it runs |
|---|---|
| `python -m core.train_routepc` | the full canonical protocol: online / routeA / PC arms, 5 paired seeds (PC0 = the deployable algorithm). Medians: online 0.0224, PC0 0.0073. |
| `python -m core.train_online` | online baseline only (5 seeds), the deployable reference. |
| `python -m core.package_core_routepc` | repackages stored artifacts into `results/core_routepc_reproduction/` (read-only packaging, no retraining). |

The S5/JAX deployable pipeline (online S5, routePC arm, benchmark arms)
is at the repo root: `train.py`, `train_bench.py`, `ssm/` — see
`../README_ROUTEPC.md` §3 and `../S5_ROUTEPC_INTEGRATION_PLAN.md`.
