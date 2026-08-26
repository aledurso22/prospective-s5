# VERSION_CONTROL — branches, tags, frozen states

## Active branches

| branch | purpose |
|---|---|
| `main` | cleaned, stable, documented code + frozen reproducible experiments. Fast-forwarded to the consolidation tip on 2026-08-26. |
| `s5-routepc` | **active S5+RoutePC integration / cluster-development branch.** Branched from `main` at `8af6b8e`. Use this for the cluster benchmark. |
| `research/pesm-s5-spectrum` | historical lane branch (lanes 0–8 of the mechanism program). Fully merged into `main`; kept for provenance of the development history. |
| `research/prospective-credit-s5` | older lane-2 (credit filters) branch. Its surviving conclusions are archived under `archive/credit_filters/`; the branch is NOT merged (its lane-2 scripts were superseded on the pesm-s5-spectrum line). Keep for history only. |

## Frozen tags (scientific states)

| tag | commit | state |
|---|---|---|
| `routepc-pc0-frozen` | `f7c5671` | the PC0 core result: online 0.0224 → PC0 0.0073 median, 4/5 paired seeds, zero BPTT. Regression-gated bitwise by `tests/test_pc0_regression.py`. |
| `routepc-mechanism-controls-frozen` | `f7c5671` | the mechanism/controls state: D1 (rel 2.4e-15), D2 (complex 0.901 / real 0.765), E1/E2, κ-sweep, oracle lagged deficit (81%), F1 negative, 2×2 control. |

Both tags point at the provenance commit immediately after the last
science commit (`d402f78`); the tree at the tag contains all frozen
scripts at their ORIGINAL paths (pre-reorganization). The same frozen
code lives at the new paths on `main`/`s5-routepc` (verified bitwise
post-move).

## Important commits (recent)

| commit | content |
|---|---|
| `d402f78` | last science commit: 2×2 control — PC0 is credit repair, not generic preconditioning |
| `f7c5671` | provenance: other-agent material as received (+ the two frozen tags) |
| `8cdc4d6` | repository reorganization (toyrig/core/controls/diagnostics/archive/tests/docs); PC0 bitwise-verified post-move |
| `8af6b8e` | documentation: READMEs, RESULTS_LEDGER, INDEPENDENT_VALIDATION |

## Rules going forward

- The frozen toy implementation (`toyrig/`) does not change. If a genuine
  bug is found, fix on a branch, re-run `tests/test_pc0_regression`, and
  re-tag with a dated tag — never move the existing tags.
- Cluster development happens on `s5-routepc` and merges into `main`
  (fast-forward) when validated. `main` should always pass all
  `tests/` gates.
- Do not squash or rewrite history: experiment provenance lives in the
  commit chain (ledger cites commit hashes).
