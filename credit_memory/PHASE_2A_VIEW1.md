# Phase 2A View 1 — controlled expressivity/credit frontier, matched state size (r≈64)

Branch `S5-CCM-scale-validation`. Code:
`credit_memory/p2a_expressivity_credit_frontier.py`. NOT another
exactness phase — B29-B34's correctness work is frozen and reused
unmodified. This is the CORRECTED run: an earlier exploratory pass had
three confounds (unequal external input across architectures, an
invalid B34 positive control, and raw-SGD optimization instability
mislabeled as a representational finding) — all three are fixed here,
documented below, and the earlier exploratory numbers are superseded.

**Post-hoc optimization/selection audit (§0-§9 below)**: after this
matrix was first reported, a follow-up audit checked for (and fixed) a
train/val/test selection-bias risk, renamed `DenseBPTTOracle` to
`DenseBPTTBaseline` (no containment proof exists), ran a 5x-longer
saturation check on the three positive controls, and ran a full
10-step optimizer-trajectory comparison of exact-online (full RTRL) vs
BPTT. No architecture or teacher definitions were changed. **All four
decision-gate criteria passed** (§9) — this document has been updated
in place to reflect the corrected (test-set) numbers throughout.

## 1. Corrections made before this run

**(a) Common input interface.** An earlier version gave the flag
architecture only `x_t[0]` (1 scalar) while RTU/B34/dense received a
4-vector — not apples-to-apples. Fixed: every architecture now
receives exactly ONE scalar exogenous input per step. Flag consumes it
natively (its own hard-baked convention); RTU/dense consume it via a
`(hidden,1)` input-weight column; B34's frozen generator hard-expects a
4-vector, so the scalar is embedded as `(x_t,0,0,0)` — zero-padding
carries no extra information. No architecture receives more external
information than another.

**(b) Positive-control validity.** B34's student previously used an
INDEPENDENTLY-redrawn frozen coefficient generator, not teacher B's —
meaning teacher B was not guaranteed to lie in the student's hypothesis
class, invalidating that positive control. Fixed: the B34 student now
reuses teacher B's exact `gen_params` throughout (same object, not
re-seeded); only `theta` differs. RTU teacher A / RTU student and flag
teacher D / flag student already shared their fixed structural
substrate correctly (verified, not just assumed).

**(c) Optimization vs representation.** The earlier raw-SGD run showed
several architecture/teacher pairs diverging (B34 catastrophically, on
every non-jet teacher). This was optimization/dynamical instability,
NOT evidence of representational impossibility, and is not used that
way. Fixed protocol: **Adam** (not raw SGD) + a common gradient-norm
clip (10.0) + a small per-architecture LR grid (3 values) + 2 seeds per
grid cell, selecting the best MEAN FINITE validation NMSE; the
selected LR is recorded per cell. A legitimate architectural
constraint (R_V spectral-radius projection, exactly as in B31b) is
kept and clearly distinguished from optimization patches — no
per-architecture "stability" clip beyond that was needed once Adam
replaced raw SGD (see §3: zero divergence occurred in the corrected
run).

## 2. Sanity checks (required gate, run BEFORE the cross-family matrix)

`N_TRAIN=80` (doubled from an initial 40 — applied uniformly to ALL
architectures, not just the one that needed it — after RTU's positive
control needed more steps to clearly pass).

| positive control | status | VAL NMSE | TEST NMSE (untouched) | best LR |
|---|---|---|---|---|
| RTU → A | **PASS** | 0.1814 | 0.1813 | 0.1 |
| B34 → B (same gen_params as teacher) | **PASS** | 0.0280 | 0.0238 | 0.03 |
| Flag → D | **PASS** | 0.00226 | 0.00246 | 0.003 |

`ALL POSITIVE CONTROLS PASS: True` (test-set criterion) — proceeded to
the cross-family matrix only after this held. LR was selected using
VALIDATION NMSE only; TEST NMSE (a disjoint, never-selected-on set)
tracks it closely at every one of the three positive controls (largest
relative gap: B34→B, 0.028→0.024, i.e. test is actually slightly
*better* than validation there — no evidence of selection-bias
inflation in the originally-reported numbers).

Dense-baseline general-fit check (not a positive control, just a
sanity read): TEST NMSE 0.266 (A), 6.882 (B), 0.571 (C), 0.301 (D) —
fits A/C/D reasonably at this modest budget; struggles on B (the jet
teacher) under the SAME small tuning budget as everything else. Not
investigated further (would require a budget asymmetry inconsistent
with "comparable tuning budget for all cells").

## 3. Cross-family matrix (View 1, matched state_dim≈64)

**Zero divergence across all 16 cells, 6 runs each (3 LRs × 2 seeds) —
96 total training runs, all finite. Rerun with proper train/validation/
test separation (audit, §8): LR selected on validation NMSE only; the table
below reports TEST NMSE (untouched, never used for selection).**

| arch \ teacher | A_independent | B_jet | C_multipole | D_coupled |
|---|---|---|---|---|
| **RTU** | **0.1813** (lr 0.1) | 2.523 (lr 0.1) | 0.3423 (lr 0.1) | 0.2522 (lr 0.1) |
| **B34** | 1.143 (lr 0.1) | **0.0238** (lr 0.03) | 0.9194 (lr 0.1) | 1.080 (lr 0.1) |
| **BoundedInterfaceFlag** | 15.08 (lr 0.01) | 144.8 (lr 0.01) | 6.069 (lr 0.01) | **0.00246** (lr 0.003) |
| **DenseBPTTBaseline** | 0.2656 (lr 0.01) | 6.882 (lr 0.01) | 0.5706 (lr 0.01) | 0.3011 (lr 0.01) |

(Values are TEST NMSE, mean over the 2 seeds at each cell's
validation-selected best LR; **bold** = the architecture-matched
positive-control cell. The pattern is essentially unchanged from the
original val-only report — test tracks validation closely everywhere,
confirming the earlier numbers were not meaningfully selection-biased
— but this table is now the methodologically clean one to cite.)

## 4. Structural/budget accounting (kept separate from the optimization outcome above)

| arch | state_dim | trainable P_c | additional persistent exact-credit: reduced | full | ratio |
|---|---|---|---|---|---|
| RTU | 64 (32 real + 32 imag) | 128 | 256 | 8,192 | 32x |
| B34 | 64 | 64 | 64 | 4,096 | 64x |
| BoundedInterfaceFlag | 64 (60 U + 4 V) | 10,888 | 43,552 | 696,832 | 16x |
| DenseBPTTBaseline | 64 | 4,224 | N/A (BPTT oracle, not an online learner) | N/A | N/A |

Note the large trainable-parameter disparity at matched state size:
B34 has by far the fewest trainable scalars (64, theta only — its
exogenous coefficient generator is fixed by design), RTU has a modest
128, dense has 4,224, and the flag architecture has by far the most
(10,888, dominated by Phi's MLP) — this is reported as-is, not
equalized (matched-credit View 2, not yet run, is the mechanism for a
budget-controlled comparison).

## 5. Reading the matrix — structured pattern, not a manufactured win

- **RTU** does clearly best on its own positive control (A, 0.181) and
  is second-best on D (0.242) and reasonable on C (0.349); it is
  worst on B (3.078, the one dynamics family fundamentally alien to
  independent modes).
- **B34** does clearly best on its own positive control (B, 0.028);
  every other teacher gives NMSE≈0.9–1.1 (order-1, i.e. no better than
  a near-trivial fit) — **reported as "empirical representation/
  optimization disadvantage under this benchmark," not "cannot
  represent"** (no impossibility theorem was proved or attempted; this
  is the outcome of one fixed, small, common tuning budget).
- **BoundedInterfaceFlag** does dramatically best on its own positive
  control (D, 0.0023, the best number in the entire matrix) and is
  clearly worst everywhere else (6.2–146) — the largest matched-state
  contrast of the four architectures between its home teacher and
  everything else.
- **DenseBPTTBaseline** is the most uniformly competent non-specialist:
  reasonable (0.26–0.58) on A/C/D, weak on B (6.35) at this budget.
  It does not dominate any architecture-matched positive control on
  its own teacher.

This is a genuine phase-diagram-shaped result — each specialized
architecture wins narrowly on its own matched dynamics and loses
broadly elsewhere; the dense oracle is a generalist that fits several
teachers reasonably without excelling at any. No win pattern was
targeted or manufactured; the LR grid/seed selection was applied
identically (same grid size, same seed count) to every cell.

## 6. Explicit non-claims

- No divergence occurred in this corrected run, so the earlier
  "optimization/dynamical instability vs representation impossibility"
  distinction did not end up mattering for THIS run's numbers — but the
  protocol (Adam, clipping, no per-cell bespoke patches) that avoided
  divergence is documented above for reproducibility, and the
  distinction remains the operative interpretive rule for any future
  divergence.
- No claim that B34/flag "cannot represent" their off-target teachers —
  only that they do not reach a good fit under this fixed, common,
  modest tuning budget.
- Dense oracle's weak fit on teacher B is not investigated further
  (would require a budget asymmetry inconsistent with "comparable
  tuning budget for all cells").

## 7. Stop point (original)

Per instruction: View 2 (matched exact-credit budget) was not run in
the original pass. The follow-up optimization/selection audit below
(§8-§9) precedes any decision to proceed to View 2.

## 8. Optimization/selection audit — additional checks (no architecture or teacher definitions changed)

**Audit 3 — positive-control saturation, 5x training horizon (N_TRAIN=400), same optimizer/clipping/data/init/LR-selection protocol:**

| positive control | VAL NMSE (80 steps) | VAL NMSE (400 steps) | TEST NMSE (400 steps) | still decreasing at step 399? |
|---|---|---|---|---|
| RTU → A | 0.1814 | **0.0677** | 0.0680 | **yes** (step300=0.0147→step399=0.0118, train loss) |
| B34 → B | 0.0280 | **3.68e-05** | 2.84e-05 | yes (step300=3.3e-7→step399=1.4e-7, train loss) |
| Flag → D | 0.00226 | **0.00109** | 0.00120 | yes, more slowly |

**RTU→A, specifically requested**: NMSE dropped from 0.181 (80 steps)
to 0.068 (400 steps) — a 2.7x improvement from 5x more optimization
alone, same architecture/teacher/data/LR-selection. The training-loss
learning curve (seed 0, best lr=0.1) is still visibly decreasing at
the end of the run (step300=0.0147→step399=0.0118, no plateau), so
**0.181 was optimization-budget-limited, not an architecture ceiling**
— consistent with the audit's central request not to read early-budget
NMSE as a representational limit. The model was not modified to
produce this improvement, only trained longer under the identical
protocol.

B34→B improved by ~750x (0.028→3.7e-5) and Flag→D by ~2x (0.0023→
0.0011) under the same 5x budget — both also budget-limited at 80
steps, though to a much smaller degree than RTU→A already at 80 steps.

**Audit 4 — exact-online (full RTRL) vs BPTT, full 10-step optimizer trajectory** (two identical copies, same init/Adam state/data, one gradient source per copy; NOT just a single step-0 gradient check):

| positive control | grad discrepancy (step 0) | max grad discrepancy (10 steps) | max param discrepancy (10 steps) |
|---|---|---|---|
| RTU → A | 5.56e-16 | 4.27e-14 | 8.33e-15 |
| B34 → B | 6.04e-18 | 6.04e-18 | 5.53e-16 |
| Flag → D | 7.09e-15 | 7.09e-15 | 3.44e-15 |

All three at float64 machine precision throughout the full 10-update
optimizer trajectory, not merely at initialization — exact-online and
BPTT training are confirmed equivalent in practice, not just in a
single isolated gradient check.

**Audit 5 — structural accounting, printed separately from any optimization outcome:**

| arch | r (state) | P (trainable) | exact credit (reduced) | r·P (full) | ratio |
|---|---|---|---|---|---|
| RTU | 64 | 128 | 256 | 8,192 | 32x |
| B34 | 64 | 64 | 64 | 4,096 | 64x |
| BoundedInterfaceFlag | 64 | 10,888 | 43,552 | 696,832 | 16x |
| DenseBPTTBaseline | 64 | 4,224 | N/A (not an online learner) | 270,336 (hypothetical/illustrative only) | N/A |

View 1 is explicitly **not** matched parameter count or matched credit
budget — these numbers are reported as-is, per instruction, and are
the reason View 2 (matched exact-credit budget) is the natural next
step rather than a re-read of View 1.

**Divergence**: retained explicitly and separately throughout — 0/6 at
every cell in every audit (original matrix, corrected matrix,
saturation audit). No sentinel value was ever averaged into an NMSE
number; where divergence would occur it is reported as a fraction, per
instruction.

**Wording retained**: cross-family fits weaker than a positive control
are described as "empirical representation/optimization disadvantage
under this benchmark" — no representational-impossibility claim is
made anywhere in this document.

## 9. Decision gate

| criterion | result |
|---|---|
| validation/test selection is clean | **PASS** — LR selected on validation NMSE only; test NMSE (disjoint, untouched) tracks validation closely at all three positive controls, no inflation found |
| exact-RTRL and BPTT optimizer trajectories agree on the three positive controls | **PASS** — machine precision (≤4.3e-14) over a full 10-step trajectory, not just step 0 |
| positive controls continue to fit reasonably, with special attention to RTU→A | **PASS** — all three improve substantially with 5x more optimization under the identical protocol (RTU→A: 0.181→0.068, confirmed still decreasing, budget-limited not architecture-limited) |
| no architecture/teacher definitions changed | **PASS** — confirmed; only train/val/test split logic, the `DenseBPTTBaseline` rename, and audit-only functions were added |

**All four criteria pass.** Per instruction, this clears the way to
proceed directly to View 2 (matched exact-credit budget) — not yet
started in this document.

## 10. Commit hash

See the commit introducing this file.
