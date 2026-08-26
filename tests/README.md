# tests/ — regression and correctness gates

Run from repo root: `python -m tests.<name>`. All must PASS before any
cluster launch or after touching `toyrig/`, `ssm/`, or `train_bench.py`.

| test | cost | what it gates |
|---|---|---|
| `test_pc0_regression.py` | ~10 min CPU | **the freeze**: fresh PC0 runs reproduce stored finals BITWISE (5/5 seeds), paired RNG streams identical, stop-gradient semantics (h_n w-invariant bitwise), zero BPTT calls. |
| `test_routepc_jax_meta.py` | ~1 min | JAX meta machinery: rotation vs numpy rig 1e-16, w=1 bitwise, no-meta fallback bitwise, teacher remap 4e-8, nested meta-gradient vs float64 FD 1.6e-6. |
| `test_online_s5_jax.py` | ~1 min | JAX online S5 gradient vs the numpy rig (~1e-7). |
| `test_scan.py` | ~1 min | all scan implementations (S5, prospective 2×2, conv) vs sequential references. |
| `test_modal_geometry_convention.py` | ~1 min | S5 modal geometry: w=1 == no meta bitwise, conj(w) leaf convention, per-mode independence, C/D untouched, jit/eager agreement. |
| `test_external_rig.py` | seconds | independent second-agent rig: FD-correct gradients, D1 restoration, top-layer exactness (see `../INDEPENDENT_VALIDATION.md`). |

Former names: `check_route_pc.py`, `check_routeA_meta.py`,
`check_online_s5.py` (renamed in the 2026-08 consolidation).
