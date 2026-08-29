# Phase 2A View 1 — controlled expressivity/credit frontier, matched state size (r≈64)

Branch `S5-CCM-scale-validation`. Code:
`credit_memory/p2a_expressivity_credit_frontier.py`. NOT another
exactness phase — B29-B34's correctness work is frozen and reused
unmodified. This is the CORRECTED run: an earlier exploratory pass had
three confounds (unequal external input across architectures, an
invalid B34 positive control, and raw-SGD optimization instability
mislabeled as a representational finding) — all three are fixed here,
documented below, and the earlier exploratory numbers are superseded.

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

| positive control | status | NMSE (mean, 2 seeds) | best LR |
|---|---|---|---|
| RTU → A | **PASS** | 0.181 | 0.1 |
| B34 → B (same gen_params as teacher) | **PASS** | 0.028 | 0.03 |
| Flag → D | **PASS** | 0.0023 | 0.003 |

`ALL POSITIVE CONTROLS PASS: True` — proceeded to the cross-family
matrix only after this held.

Dense oracle general-fit check (not a positive control, just a sanity
read): NMSE 0.26 (A), 6.35 (B), 0.58 (C), 0.30 (D) — fits A/C/D
reasonably at this modest budget; struggles on B (the jet teacher)
under the SAME small tuning budget as everything else. Not
investigated further (would need a larger budget specifically for
dense-on-B, which is out of scope for "comparable tuning budget").

## 3. Cross-family matrix (View 1, matched state_dim≈64)

**Zero divergence across all 16 cells, 6 runs each (3 LRs × 2 seeds) —
96 total training runs, all finite.**

| arch \ teacher | A_independent | B_jet | C_multipole | D_coupled |
|---|---|---|---|---|
| **RTU** | **0.181** (lr 0.1) | 3.078 (lr 0.1) | 0.349 (lr 0.1) | 0.242 (lr 0.1) |
| **B34** | 1.145 (lr 0.1) | **0.028** (lr 0.03) | 0.937 (lr 0.1) | 1.095 (lr 0.1) |
| **BoundedInterfaceFlag** | 15.04 (lr 0.01) | 145.7 (lr 0.01) | 6.236 (lr 0.01) | **0.0023** (lr 0.003) |
| **DenseBPTTOracle** | 0.258 (lr 0.01) | 6.354 (lr 0.01) | 0.576 (lr 0.01) | 0.302 (lr 0.01) |

(Values are NMSE, mean over the 2 seeds at each cell's selected best
LR; **bold** = the architecture-matched positive-control cell.)

## 4. Structural/budget accounting (kept separate from the optimization outcome above)

| arch | state_dim | trainable P_c | additional persistent exact-credit: reduced | full | ratio |
|---|---|---|---|---|---|
| RTU | 64 (32 real + 32 imag) | 128 | 256 | 8,192 | 32x |
| B34 | 64 | 64 | 64 | 4,096 | 64x |
| BoundedInterfaceFlag | 64 (60 U + 4 V) | 10,888 | 43,552 | 696,832 | 16x |
| DenseBPTTOracle | 64 | 4,224 | N/A (BPTT oracle, not an online learner) | N/A | N/A |

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
- **DenseBPTTOracle** is the most uniformly competent non-specialist:
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

## 7. Stop point

Per instruction: View 2 (matched exact-credit budget) is NOT run yet.
This document reports View 1 in full — corrected positive controls,
corrected common input interface, corrected (Adam-based) optimization
protocol, the full 4×4 matrix, and divergence fractions (all zero) —
for review before deciding whether the observed differences justify
View 2.

## 8. Commit hash

See the commit introducing this file.
