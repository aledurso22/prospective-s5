# Phase B6 — prospective CCM tracking validation

Branch `credit-memory-repair`. Does not change the underlying CCM
temporal-credit mechanism (Phase A's (E1)/(E2), untouched); does not use
BPTT/exact adjoint/exact P/Q teacher state in any of T0-T3's training
algorithms (BPTT is evaluation-only). Code:
`credit_memory/b6_prospective_tracking.py`. Artifact:
`results/credit_memory/b6/b6_prospective_tracking_summary.json`. Same 8
seeds, `L=2, N=6, T=60, DELAY=20, BATCH=8, STEPS=600` as B5/B5.1;
`clip=0` primary (full T0-T3 x 8-seed matrix plus a `K=50` secondary for
T1), `clip=1` secondary (T0/T2/T3 only, T1 skipped for budget since its
mechanism does not depend on clip).

**Headline: Reactive suffices.** T1 (periodic recalibration), T2
(reactive EMA+hysteresis), and T3 (prospective predict-correct
+hysteresis) all show a consistent, though individually non-significant
at `n=8`, *directional* improvement in late-training gradient alignment
over the frozen T0 baseline. But **T3's prediction is not measurably
better than simply carrying the old relevance forward** (better on only
`51.2%` of logged steps — statistically indistinguishable from a coin
flip), and **T2 performs as well as T3 on every metric measured**
(final task loss, alignment maintenance) while switching channels less
often. Per the task's own decision rubric: **do not claim prospective
coding is needed here; the simpler reactive tracker is sufficient in
this regime.**

## B6A — arms (fixed hyperparameters, no sweep)

- **T0 frozen**: exactly B5's A2 protocol (one causal calibration
  prefix at init, frozen for all 600 steps). Verified to reproduce B5's
  `b4_causal` arm bitwise (seed 0 final loss `0.1857` in both).
- **T1 periodic recalibration**: identical calibration protocol
  (`causal_prefix_selection`, reused verbatim from B5), re-run every
  `K=100` steps (primary) using a dedicated calibration RNG stream
  disjoint from training data; `K=50` run as the permitted cheap
  secondary, clip=0 only.
- **T2 reactive EMA + hysteresis**: `rho_n` updated every training step
  via EMA (`gamma=0.08`, B4D's best-performing rate) on the raw
  per-batch cross-statistic (`credit_memory.lagcorr.
  per_coordinate_contribution`, unmodified, reset fresh each batch —
  reuses data the training step already computes, no extra forward
  passes).
- **T3 prospective predict-correct + hysteresis**: same per-step
  cadence and same raw observation `r_{n+1}` as T2, but
  `rho^-_{n+1} = rho_n + beta(rho_n - rho_{n-1})` (`beta=0.5`), then
  `rho_{n+1} = rho^-_{n+1} + K(r_{n+1} - rho^-_{n+1})` (`K=0.3`).
  Bootstrapped at the first step (`rho_prev=None`) by treating the
  prediction as "no change."

## B6B — hysteresis (implemented; hard/soft factorial skipped per instruction's own priority list)

T2 and T3 both use **hard selection with hysteresis**: stay on the
current channel unless a competitor's `|rho|` exceeds it by a fixed
relative margin (`0.15`). The soft top-2 variant was **not implemented**
— the instruction explicitly permitted dropping it ("if that makes the
matrix too large, prioritize T0/T1/T2+hysteresis/T3+hysteresis"), and
the 4-arm matrix was already the full scope taken on.

## B6C — does alignment stay repaired? (staleness curve, clip=0)

**Methodological note, stated up front**: unlike B5.1 (which measured a
*counterfactual* one-step action from an **online**-only trained
trajectory), B6 measures alignment **within actual CCM-guided training**
— T0/T1/T2/T3 each train their own parameter trajectory using their own
correction throughout. The two are not numerically comparable
checkpoint-by-checkpoint; B6's own T0-vs-T1/T2/T3 comparison, all
measured the same way, is the valid apples-to-apples signal here.

Median `a0` cosine to BPTT, by checkpoint:

| step | T0 (frozen) | T1 (K=100) | T2 (EMA) | T3 (prospective) | T1 (K=50) |
|---|---|---|---|---|---|
| 100 | 0.649 | 0.649 | 0.775 | 0.706 | 0.703 |
| 300 | 0.711 | 0.714 | 0.647 | 0.629 | 0.529 |
| **600** | **0.431** | **0.713** | **0.697** | **0.736** | **0.765** |

Median `b0` cosine to BPTT, by checkpoint:

| step | T0 | T1 | T2 | T3 | T1 (K=50) |
|---|---|---|---|---|---|
| 100 | 0.619 | 0.619 | 0.731 | 0.792 | 0.768 |
| 300 | 0.693 | 0.666 | 0.653 | 0.651 | 0.679 |
| 600 | 0.736 | 0.687 | 0.735 | 0.805 | 0.682 |

**At the late checkpoint (step 600), every adaptive arm (T1/T2/T3/T1-K50)
holds a clearly higher `a0` alignment than the frozen baseline** (`0.71`-
`0.77` vs T0's `0.43`) — directionally consistent with the staleness
hypothesis (a frozen channel selection loses alignment as parameters
drift; refreshing it, reactively or periodically, recovers some of that
loss). Paired significance at `n=8` (sign test, `cos_a0@600`, vs T0):
T1 `5/8` wins (`p=0.73`), T2 `6/8` (`p=0.29`), T3 `6/8` (`p=0.29`),
T1-K50 `6/8` (`p=0.29`) — **directionally uniform across all four
adaptive variants but not individually significant at this seed count**
(consistent with B5's own established pattern of real-but-small,
noisy-at-n=8 effects). `b0` is noisier (T0 itself recovers to `0.736` by
step 600 here, unlike B5.1's counterfactual measurement where it fell to
`0.303` — a direct illustration of the methodological note above: T0's
*own* training trajectory does not follow the same path as B5.1's
online-then-one-step counterfactual).

## B6D — prospective-tracking diagnostics

**Switch counts** (median total over 6 modes, 600 steps, clip=0):

| arm | median total switches |
|---|---|
| T2 (reactive) | **87** |
| T3 (prospective) | **185** |

T3 switches channels **more than twice as often** as T2 despite
identical hysteresis. All runs remained finite (no divergence from
either), so switching is not *pathological* in the sense of causing
training failure, but it is clearly **less stable**, not more — the
opposite of what a successful "smooths out reactive noise" prospective
mechanism would be expected to show.

**Prediction quality — the decisive check**: for T3, compare
`|rho^-_{n+1} - r_{n+1}|` (prediction error) against
`|rho_n - r_{n+1}|` (persistence/no-prediction baseline), over all
logged steps, all seeds, clip=0:

| | median error |
|---|---|
| prediction (`rho^-`) | 16.666 |
| persistence (`rho_n`) | 16.898 |

Prediction beats persistence on only **`51.2%`** of logged steps — not
distinguishable from chance. **The secant extrapolation is not earning
its keep**: it produces a barely-smaller error than simply assuming
"nothing changed since last time," while (per B6D above) leaving the
selector measurably less stable. This is the single clearest, most
decisive number in this phase.

## B6E — task/action outcome (clip=0, 8 seeds; clip=1 confirms the same pattern, not tabulated separately)

| arm | median final loss | vs T0 sign test |
|---|---|---|
| T0 | 0.1336 | -- |
| T1 (K=100) | 0.1326 | 3/8 wins, `p=0.73` |
| T2 | 0.1322 | 4/8 wins, `p=1.00` |
| T3 | 0.1320 | 4/8 wins, `p=1.00` |
| T1 (K=50) | 0.1316 | 4/8 wins, `p=1.00` |

**No significant task-loss difference between any arm and frozen T0** —
all differences are within noise at `n=8`, consistent with B5/B5.1's own
finding that the gradient-mechanism benefit here is real but too small
to reliably move task loss at this scale. T3 vs T2 head-to-head: median
final-loss difference `-0.0002` (T3 negligibly better), median
`cos_a0@600` difference `-0.0016` (essentially zero) — **T2 and T3 are
statistically indistinguishable from each other** on every outcome
measured.

## B6F — decision

Checking the four gates in order:

- **Strong prospective pass** requires all of: (1) periodic
  recalibration restores alignment — **partially true** (T1 shows a
  consistent, if non-significant, improvement at step 600); (2)
  prospective T3 matches that without explicit recalibration — **true**
  (T3's step-600 numbers are comparable to or better than T1's); (3)
  prospective prediction error is lower than persistence — **FALSE**
  (`51.2%`, not distinguishable from chance); (4) switching not
  pathological — **true in the narrow "does not cause divergence"
  sense**, but T3 switches over twice as often as T2. **Gate 3 fails
  outright, so "Strong prospective pass" is not concluded.**
- **Tracking problem confirmed, prospective law fails** requires T1 to
  work while T3 does not. T1 *does* show supportive evidence for the
  staleness hypothesis, but **T3 is not worse than T1** on the alignment
  metric (T3's step-600 `a0` cosine, `0.736`, is the highest of any
  adaptive arm) — so this is not the right classification either; T3
  does not fail to track, it simply does not add anything *beyond* what
  reactive tracking already provides.
- **No staleness rescue** requires T1 to fail to restore alignment. It
  does not fail — T1 shows the same directional recovery pattern as
  T2/T3. Not this case.
- **Reactive suffices**: "If T2 performs as well as T3 and prediction
  gives no measurable advantage: do not claim prospective coding is
  needed. Use the simpler reactive tracker for S5 and report prospective
  as unnecessary in this regime." **This is the best-supported
  classification**: T2 matches T3 on every outcome (B6E), T3's
  prediction does not beat persistence (B6D), and T3 switches channels
  more, not less, than T2.

**Combined reading**: B5.1's staleness diagnosis is corroborated in
direction (T1/T2/T3 all show better late-training alignment than frozen
T0, though not individually significant at `n=8`), supporting that
moving-relevance tracking is a real, addressable factor. But the specific
mechanism this phase was asked to validate — **secant prospective
prediction-correction** — adds no measurable benefit over simple
reactive EMA tracking in this regime, and introduces more channel
instability for that zero benefit. **Recommendation: if a tracking
mechanism is carried into the L=2 S5 validation, use T2's simpler
reactive EMA+hysteresis form, not T3's prospective predict-correct
form** — consistent with "Reactive suffices"'s own guidance, not a new
recommendation invented here.

## B6G — theory bookkeeping (sequence time vs. optimizer/meta time, kept separate)

**Sequence time** (per training-batch, `T=60` timesteps, the UNCHANGED
CCM mechanism from Phase A/B1-B4):
```
x_{j,t} = lambda_j x_{j,t-1} + u_t
```

**Optimizer/meta time** (this phase's new layer, one update per training
step `n`, `600` steps total — a slower, separate index from sequence
time `t`):
```
rho^-_{n+1} = rho_n + beta (rho_n - rho_{n-1})          (T3 predict)
rho_{n+1}   = rho^-_{n+1} + K (r_{n+1} - rho^-_{n+1})    (T3 correct)
```
`r_{n+1}` is a *fresh, single-batch* observation of the ordinary B4
statistic (reset every batch, no state carried in sequence time across
optimizer steps) — this keeps sequence time and optimizer time
genuinely separate: nothing about the per-timestep channel filter
`x_{j,t}` changed; only the *choice* of which channel `j` to deploy is
now tracked at the slower optimizer-time cadence. `D^{-1}` is not used
as a name for anything in this document. The old RoutePC mechanism is
not reinterpreted here. No least-action experiment was run; the
variational interpretation remains theory only.

## Artifacts

- `results/credit_memory/b6/b6_prospective_tracking_summary.json` —
  git hash, config (all fixed hyperparameters), full per-(arm, seed,
  clip) runs: loss curves, per-checkpoint whole/a0/b0 gradient cosine
  and relative error, T2/T3 switch counts, T3's full tracking log
  (`rho_n`, prediction, observation, residual, margin, dwell time per
  logged step).

## Not done in Phase B6 (by design, per scope)

No soft top-2 selector variant; no hyperparameter sweep over `beta, K,
gamma`, or the hysteresis margin; no S5 launch; no reinterpretation of
the old RoutePC mechanism; no least-action experiment.
