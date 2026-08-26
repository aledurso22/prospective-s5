# S5_ROUTEPC_INTEGRATION_PLAN

How S5 + RoutePC fit together in this repository, what was reused from
where, and what was deliberately NOT imported.

## The two prospectiveness concepts — kept separate

| concept | object | code |
|---|---|---|
| **state prospectiveness** | forward/state dynamics: τ ṡ = −s + f + τ ḟ discretized as a 2×2 companion recurrence | `ssm/prospective/` (enabled by `model_type="prospective"` / `--state-prospective`; diverges at HiPPO init by design — the "ghost" result, lane 0) |
| **RoutePC prospectiveness** | optimizer/meta-time: the learned credit geometry w and its delayed causal correction | `toyrig/routepc.py` (toy) and `train_bench.py --arm routePC` + `ssm/online_s5/` (S5) |

They are independent config axes in `train_bench.py`:
`learning_rule ∈ {bptt, tbptt, online, routepc, oracle_exact_teacher,
online_frozen_geometry}` (derived from `--arm`, recorded in every metrics
JSON) × `state_prospective ∈ {false, true}` (`--state-prospective`).
**No flag means both at once.** Currently supported crossing:
state-prospective × BPTT only — there is no online/RTRL VJP for the 2×2
prospective recurrence, so `online`/`routepc` × `state-prospective`
raises `NotImplementedError` loudly instead of silently mixing concepts.
Do not run the full factorial until that VJP exists and this gate is
revisited.

## Code responsibilities (what comes from where)

| piece | source | why |
|---|---|---|
| S5 architecture, discretization, blocks, HiPPO init | this repo, `ssm/` (`baseline_s5/`, `shared/`, `model.py`) | existing, tested (`tests/test_scan.py`) |
| datasets / training infrastructure / SLURM | this repo (`train.py`, `train_bench.py`, `bench_report.py`, `scripts/`) | existing benchmark harness, paired streams, headroom gate |
| **online eligibility machinery** | this repo, `ssm/online_s5/` — home-grown custom VJPs | validated to ~1e-7 against the numpy rig (`tests/test_online_s5_jax.py`); conventions already FD-gated |
| **Zucchet/RTRL codebase** | **NOT imported — none exists in the tree and none is needed** | searched the whole workspace: no vendored/cloned copy. The only reference is a citation (docs/PROSPECTIVE_SSM_RESEARCH_HANDOFF.md → github.com/NicolasZucchet/Online-learning-LR-dependencies). Our online gradient is the same *rule class* (forward sensitivity × instantaneous error), implemented natively as a custom VJP; importing external code would add a convention-conversion risk for zero capability gain. |
| modal geometry + delayed causal meta update | `train_bench.py` (routePC step, lines ~415–444) + `ssm/online_s5/layer.py` (w injection via the `"meta"` flax collection) | ported from the frozen toy (`toyrig/routepc.py`); meta machinery FD-gated by `tests/test_routepc_jax_meta.py` |

## Conventions (verified, do not drift)

- Complex convention: `conj(w)` on the recurrence-mode blocks (Ga, Gb)
  inside the VJP; readout (C, D, dx) untouched. Leaf cotangents returned
  as `(Re G, −Im G)` — in leaf coordinates the rotation appears as
  `[[u,−v],[v,u]]`. Pinned by `tests/test_modal_geometry_convention.py`.
- Meta tree `w_re/w_im (H, N)` mirrors only SSM nodes
  (`train_bench.make_meta`); absent meta ⇒ w=1, bitwise equal to plain
  online (startup self-check + gate [0]).
- Meta optimizer: plain SGD at `--lr-m`; the meta chain does not
  differentiate through Adam/clip (documented simplification, carried
  over from the toy verbatim).
- routePC builds **no exact-gradient path** (no teacher model); the
  zero-BPTT invariant is structural and recorded in the metrics `audit`
  block.

## Memory / computation expectations

- online/routePC arms: O(1)-in-T eligibility state (analytic estimate
  `2·L·H·N·8 B`, reported per run as `eligibility_state_bytes`) instead of
  BPTT's O(T) activation tape; routePC costs ~3 online-gradient
  evaluations per step (h_n, g_prev inside the correction, applied
  update) — no backward pass over time.
- peak device memory reported per run (`peak_device_memory_bytes`,
  GPU only); sequence length is a first-class config (`--seq-len` for
  copy; T=784/196 for sMNIST via `--downsample`).

## Pre-cluster checklist status

- [x] canonical algorithm frozen (toy, tag `routepc-pc0-frozen`)
- [x] JAX meta machinery FD-gated (`tests/test_routepc_jax_meta.py`)
- [x] modal-geometry convention test on S5 modes
- [x] config exposes learning_rule × state_prospective independently
- [x] per-run provenance + instrumentation in metrics JSON
- [x] CPU smoke runs (baseline / online / routePC / state-prospective)
- [ ] GPU smoke runs on the cluster node
- [ ] benchmark grid launch (see `scripts/README.md`) — **NOT launched**
