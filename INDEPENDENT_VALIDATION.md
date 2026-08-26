# INDEPENDENT_VALIDATION — material from the second agent

**Provenance.** The directory `external/` (received as
`files_from_the_other_agent/`, committed verbatim in `f7c5671`, moved in
the 2026-08 consolidation) contains a self-contained validation package
written by an **independent second agent**: its own 2-layer diagonal
complex SSM numpy rig (`external/ssm.py`, 164 lines, no imports from this
repository) plus five small scripts. We ran every script on 2026-08-26
with the repo `.venv` (python 3.14); outputs summarized below.

**Status:** independent reference material — **not** canonical project
code. Do not import from `external/` in project code (its `ssm.py` would
also shadow the repo's `ssm/` package; run its scripts from inside
`external/` only). One adapted regression test lives at
`tests/test_external_rig.py`.

## What independently corroborates our main results

| their check (script) | their number | our corresponding result | verdict |
|---|---|---|---|
| BPTT vs finite differences (`verify.py`) | rel ≤ 4.4e-09 on every complex parameter | `toyrig` chains FD-gated throughout | rig gradients correct |
| exact adjoint × eligibility == BPTT (`verify.py` §2) | cos 1.000000000000000, rel **4.8e-16** | **D1**: rel 2.4e-15 (`diagnostics/d1_exact_credit_factorization.py`) | **D1 factorization independently reproduced** |
| top recurrent layer online == BPTT (`verify.py` §3) | cos 1.000000, rel **4.9e-16** | top-layer exactness (`diagnostics/phase_probes.py`) | **independently reproduced** |
| lower-layer online defect (`verify.py` §3) | cos 0.606, rel 0.93 | shallow-layer misalignment (phase_probes) | **independently reproduced** |
| modal-oracle ceiling, 8 seeds (`exp1_ceiling.py`) | held-out: identity 0.632 < global 0.679 < mode_real 0.751 < **mode_complex 0.966** | **D2**: identity 0.596 < real 0.765 < **complex 0.901** (`diagnostics/d2_modal_oracle.py`) | **same ordering, same winner**, on a different rig (P=12, T=120, lag=40, IIR-free task) |
| SGD-vs-Adam preconditioner crossover (`exp3_control.py` vs `exp4_adam.py`) | SGD: w helps on online AND bptt (0.08/0.05 vs 0.54/0.55 best-curve); **Adam: advantage vanishes** (online_w 0.0146 ≈ online 0.0155; bptt_w 0.0105 ≈ bptt 0.0115) | norm-matched PC0 == PC0 at ‖M_w g‖/‖g‖ up to 564 (`controls/control_2x2_normmatch.py`) | **"Adam absorbs the gain" independently reproduced** — and their SGD cell shows what w looks like when it IS partly a preconditioner |

## What did NOT independently reproduce (and why that is expected)

- **Training-level RoutePC/routeA win.** Their `exp2_train.py` regime has
  essentially **no BPTT-vs-online headroom** (median bptt 0.6848 ≈ online
  0.7092; bptt worse than online on 2/5 seeds), so by our own registered
  headroom discipline (the D6 lesson) it cannot measure the credit-repair
  effect. Their "routePC closes 252%" figure is a ratio over a ~0.02 gap —
  noise. Their per-seed routePC (2.50 → 0.074) and routeA basin-flip
  (7.91 on seed 4) are consistent with the bistability we document for
  these rigs (`README_ROUTEPC.md` §7), not with a clean refutation or
  confirmation. **Our canonical PC0 result rests on our own frozen,
  headroom-positive, paired protocol** (`RESULTS_LEDGER.md` §1).

## Convention note (checked before adapting)

- `external/verify.py` applies the complex Wirtinger factor-2
  (`dL/du = 2 Re g`) uniformly, including to the **real** bias `d` — hence
  its single "worst relative error 1.00" line (analytic exactly 2× FD).
  The rig's gradients are correct; the factor is a slip in their check.
  `tests/test_external_rig.py` corrects it (factor 2 only for complex
  parameters) and asserts rel < 1e-8 on all parameters.

## What is toy-specific only

Their rig (2 layers, P=10–12, poles ~U[0.9, 0.999], IIR-cascade and
fixed-lag tasks, plain-SGD or Adam base, no Adam/clip interplay matching
ours) shares no constants with the canonical rig — numeric values are not
comparable across the two, only the structural identities are.

Adapted into the test suite: `tests/test_external_rig.py` (origin:
`external/verify.py`, factor-2 slip fixed). Not adapted: `exp1–exp4`
(kept as runnable provenance in `external/`; their conclusions are
recorded above).
