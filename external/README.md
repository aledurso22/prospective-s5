# external/ — independent second-agent validation material

Self-contained validation package written by an **independent agent**
(received as `files_from_the_other_agent/`, committed verbatim in
`f7c5671`). **Not canonical project code — do not import from here.**
See `../INDEPENDENT_VALIDATION.md` for what reproduced and what is
toy-specific.

Contents:

- `ssm.py` — independent 2-layer diagonal complex SSM numpy rig
  (forward, BPTT, forward-eligibility online gradients with the
  Zucchet-style instantaneous-error approximation, oracle-w helpers).
- `verify.py` — BPTT vs finite differences; exact-adjoint restoration
  (D1); per-layer online-vs-BPTT alignment.
- `exp1_ceiling.py` — modal-oracle geometry ceilings (identity / global /
  per-mode real / per-mode complex), in-window vs held-out.
- `exp2_train.py <eta>` — 5-arm training race (bptt / online / routePC /
  routeA / oracle_w), plain-SGD base.
- `exp3_control.py` — w on online vs exact gradients, SGD base.
- `exp4_adam.py` — same 2×2 with Adam base (preconditioner crossover).

Run from inside this directory only (scripts do `from ssm import ...`,
which must resolve to the local file, not the repo's `ssm/` package):

```bash
cd external
../.venv/bin/python verify.py
```

A convention-checked adaptation of `verify.py` lives in the test suite at
`tests/test_external_rig.py` (run from repo root:
`python -m tests.test_external_rig`).
