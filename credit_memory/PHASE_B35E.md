# Phase B35e — moving-parameter trace diagnostic

Branch `S5-CCM-scale-validation`. Diagnostic only: does not modify
B35d (`credit_memory/b35d_streaming_sysid.py`, untouched), does not
redesign RegularBlock, does not optimize performance. Code:
`credit_memory/b35e_staleness_diagnostic.py` (`/tmp/b35e_diagnostic.log`).
Reimplements B35d's exact update equations purely to RECORD the
parameter/state history B35d's own scanned loop discards — same
primitives (`alg_mult_blockwise`, `transpose_mult_blockwise`,
`project_local_tails`, `adam_step`), imported unmodified.

## Setup

At checkpoint t (after t realized continual-update steps): realized
historical parameters theta_0..theta_{t-1}, checkpoint state h_t,
carried reduced eligibility s_t, current parameter theta_t, and C_out
FROZEN at its checkpoint value throughout (isolates theta's staleness
specifically). Grid: LR in {0.005, 0.02, 0.05}, update interval (steps
between parameter updates) in {1, 5}, checkpoints t in {50, 150, 300},
one representative B35d evaluation seed (11), C=64. Both RegularBlock
and GenericBlock tested (GenericBlock's analogue was technically
straightforward — same three-quantity construction with its own exact
module-wise sensitivity).

- **A. Carried**: `g_carried = E_t^T q_t` (transpose_mult(s_t,q_t) for
  RegularBlock; the analogous exact-module contraction for GenericBlock).
- **B. Path-diagonal AD**: replay theta_0..theta_{t-1} exactly as
  realized, with a SHARED additive perturbation alpha added to every
  one of them; `g_path = d(l_t(alpha))/d(alpha)` at alpha=0, via autodiff
  through a `jax.lax.scan` replay (not the original scanned run).
- **C. Frozen-current counterfactual**: replay the same input prefix
  from h_0 using theta_t (current, single fixed value) at every
  historical step; `g_frozen` = ordinary single-fixed-parameter gradient.

## Results

**A vs B (carried-vs-path): machine precision in EVERY cell tested**
(both architectures, all 3 LRs, both intervals, all 3 checkpoints —
18 conditions x 2 architectures = 36 checkpoints total): relative error
1.6e-16 to 7.7e-16. **The prediction `g_carried = g_path` is confirmed
exactly, with no exceptions.**

**A vs C (carried-vs-frozen): mean eps_frozen by (lr, interval), RegularBlock**

| lr | interval=1 | interval=5 |
|---|---|---|
| 0.005 | 0.0224 | 0.0002 |
| 0.02 | 0.1378 | 0.0037 |
| 0.05 | 0.6814 | 0.0510 |

- **Increases sharply and monotonically with learning rate** at fixed
  interval (0.0224 -> 0.1378 -> 0.6814, a ~30x range from lr=0.005 to
  0.05 at interval=1).
- **Decreases sharply when parameters are updated less often**
  (interval=5 vs interval=1 cuts eps_frozen by roughly 10-100x at every
  LR tested).
- One striking individual case: lr=0.05, interval=1, t=50: eps_frozen=
  1.245, **cos(g_carried,g_frozen)=-0.65** — the carried and frozen
  gradients point in NEARLY OPPOSITE directions, a clear instance of
  staleness corrupting gradient direction, not just magnitude, at high
  LR / frequent updates.
- GenericBlock shows the SAME qualitative pattern (e.g. at interval=1:
  eps_frozen grows from ~0.005-0.01 at lr=0.005 to ~0.27-0.34 at
  lr=0.05; at interval=5 it shrinks back down) — **this staleness
  mechanism is general to continual RTRL, not specific to
  RegularBlock's compressed representation**, exactly as the
  mathematical argument in B35d predicted (the closure/compression is
  orthogonal to the staleness).

**Loss-volatility correlation**: `corr(eps_frozen, 20-step trailing
loss volatility)` across all 18 RegularBlock (lr, interval, t) rows =
**0.43** — a real, positive, but moderate correlation, not an
overwhelming one.

## Answering the main questions directly

1. Carried-vs-path error stays at numerical precision? **Yes, always**
   (confirms the closure claim is unaffected by continual updates —
   B35d section 0, claim 2).
2. Carried-vs-frozen discrepancy increase with LR? **Yes, clearly and
   monotonically** (~30x range).
3. Decrease with less frequent updates? **Yes, clearly** (~10-100x).
4. Correlated with RegularBlock's B35d loss volatility? **Moderately**
   (r=0.43) — real, but not dominant.
5. Same mismatch diagnostic for GenericBlock? **Done — same
   qualitative pattern**, confirming the mechanism is shared, not
   RegularBlock-specific.

## Interpretation

The moving-parameter staleness mechanism is real, precisely
characterized (exact match to the path-diagonal counterfactual,
growing with LR, shrinking with update sparsity), and **is a plausible
contributing factor** to RegularBlock's B35d loss volatility (moderate
positive correlation) — but at r=0.43, it is **not sufficient on its
own to explain the full magnitude of B35d's failure**. Per instruction,
this is stated plainly rather than stretched: **the evidence supports
staleness as A contributing mechanism, not as the dominant explanation**
— B35d's optimization-noise/parameterization-sensitivity hypothesis
(from PHASE_B35D.md section 7) remains a live, likely co-occurring
cause, especially since GenericBlock exhibits the identical staleness
pattern yet did not show RegularBlock's degree of loss-curve
oscillation in B35d itself.

## Commit hash

See the commit introducing this file.
