# prospective-s5

Research codebase: **causal online learning for diagonal state-space
models**, built around the RoutePC/PC0 rule — a per-mode complex update
geometry learned **without any BPTT call** — plus the mechanism program
that explains it, and the JAX/S5 benchmark pipeline that tests it at
scale.

**Start here:** [`README_ROUTEPC.md`](README_ROUTEPC.md) (the algorithm,
six lines, canonical entry points, traps).
**Frozen numbers:** [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md).
**Full experiment history:** [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Layout

| path | what |
|---|---|
| `toyrig/` | frozen numpy toy rig (shared machinery; bitwise-pinned) |
| `core/` | canonical deployable entry points (`train_routepc`, `train_online`) |
| `controls/` | exact/oracle controls (2×2, norm-match, tbptt, LR) |
| `diagnostics/` | mechanism diagnostics (D1/D2, teacher deficit, κ, generality, …) |
| `archive/` | concluded negatives and superseded lanes (solver, credit filters, F1, forward prospection) |
| `external/` | independent second-agent validation material (see `INDEPENDENT_VALIDATION.md`) |
| `tests/` | regression/correctness gates (bitwise PC0, JAX meta FD, scans, external rig) |
| `results/` | stored artifacts, one dir per experiment |
| `docs/` | theory, handoff, and legacy documents |
| **S5/JAX pipeline** | `train.py`, `train_bench.py`, `bench_report.py`, `ssm/`, `scripts/` — the cluster benchmark (8 arms × 3 tasks, built, CPU-smoke-clean, unlaunched) |

## Quickstart

```bash
# environment: repo .venv (python 3.14, numpy + jax[cpu])
python -m tests.test_pc0_regression    # the freeze gate (~10 min, bitwise)
python -m tests.test_external_rig      # independent-rig identities (seconds)
python -m core.train_routepc           # reproduce online 0.0224 -> PC0 0.0073
```

Run convention: everything runs as `python -m <pkg>.<script>` **from the
repo root**. Results write to `results/<experiment>/`.

## Documents

| doc | content |
|---|---|
| `README_ROUTEPC.md` | the RoutePC algorithm, semantics, controls, traps |
| `RESULTS_LEDGER.md` | frozen headline results with provenance |
| `EXPERIMENTS.md` | complete ledger of everything tried |
| `INDEPENDENT_VALIDATION.md` | what the second agent's rig corroborates |
| `VERSION_CONTROL.md` | branches, tags, frozen states |
| `S5_ROUTEPC_INTEGRATION_PLAN.md` | how S5+RoutePC fits together (and what was NOT imported) |
| `docs/THEORY.md`, `docs/CLOSED_LOOP.md` | theory notes (stale relative to the PC0 arc — the ledger is current) |
| `docs/README_LEGACY_prospective.md` | the original baseline-vs-prospective program README |

## Status

Toy mechanism program: **complete and frozen** (tags
`routepc-pc0-frozen`, `routepc-mechanism-controls-frozen`).
S5 cluster benchmark: built and validated on CPU; **not yet launched** —
see `S5_ROUTEPC_INTEGRATION_PLAN.md` and `scripts/`.
