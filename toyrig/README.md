# toyrig/ — the frozen numpy toy rig (shared machinery)

The delayed-copy SSM rig that every toy experiment builds on. **Frozen** —
behavior pinned bitwise by `tests/test_pc0_regression.py`. Do not modify
these files; build new diagnostics as new scripts that import them.

| module | formerly | provides |
|---|---|---|
| `ssm_rig.py` | `trained_credit_gains.py` | diagonal complex SSM: init, forward, sensitivities (RTRL S-slots), `spatial_q`, exact `exact_lambda`/`flat_grads`, Adam, clip, `init_params`, data packing. Global config via module attributes (`L, N, T, DELAY, M_IN, BATCH`). |
| `route_a.py` | `co_variational_metric.py` | routeA: exact-teacher modal-geometry learning (`train_route`), `batch_grad`, `scale_by_w` (the conj(w) convention), `exact_grad`, `clip`/`adam` on flat vectors, constants `LR, LR_M, CLIP`. Runnable: `python -m toyrig.route_a`. |
| `probes.py` | `decompose_w_final.py` | `make_data` (paired delayed-copy batches), `probe_blocks` (gradient block extraction). |
| `train_cell.py` | `depth_law.py` | `train_cell` (generic arm trainer), `STEPS` (1500). |
| `routepc.py` | `route_pc.py` | **the canonical PC0 implementation** (`train_pc`, beta=0) + the 5-seed paired protocol (`main`), BPTT counting wrappers. See `../README_ROUTEPC.md`. Runnable: `python -m toyrig.routepc` (same as `python -m core.train_routepc`). |

Conventions (FD-gated, do not change): flat layout `(Re G, −Im G)`;
`scale_by_w` applies `conj(w)` to the `(a, B)` blocks only; θ optimized by
Adam on the clipped flat gradient; w by plain SGD through the simplified
meta chain.
