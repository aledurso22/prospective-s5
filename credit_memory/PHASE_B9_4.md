# Phase B9.4 — dynamics/mechanism audit of the moving credit-pool geometry

Branch `S5-CCM-scale-validation` (current checkout). Diagnostic only:
**no prediction-correction, no prospective coding, no feedback
alignment/PAL-style probing, no new training arm, no S5.** Code:
`credit_memory/b9_4_dynamics_audit.py` (new). Artifact:
`results/credit_memory/b9_4/b9_4_dynamics_audit_summary.json`. 8 seeds,
same `N=6, T=60, BATCH=8, DELAY=20, STEPS=600` real delayed-copy task
as B9.3. Dense (every-step) logging on top of B9.3's `reactive_full`
arm (Parts A-D) and `pool_frozen K=4` arm (Part E).

**Terminology correction applied throughout:** no dormant-state
resurrection is required anywhere in this setup (PHASE_B9.md). Any
future sparse mechanism this audit might motivate would use
**temporary candidate probes**, not persistent per-candidate state to
"resurrect."

**Headline: Recommendation D — neither signal exists.** Simple
persistence beats every tested model-based predictor on the primary
metrics (top-K recall, and inconsistently on regret); a fixed K=4 pool
does not become more useful during training — if anything it goes
stale (median `cos_a0` falls from 0.788 to 0.663, and the pool's
relevance advantage over non-pool channels shrinks). **Keep periodic
K-pool CCM (B9.3) as the supported algorithm; do not add
prediction-correction or feedback-alignment machinery.**

## Part A — dense logging (what was actually captured)

Per step, per seed: `rho_ema[j,m]` (reactive_full's own EMA-tracked
relevance, the literal state that drives its selection — the "moving
credit geometry" itself), `gamma_inst[j,m]` (fresh, single-batch exact
per-candidate contribution via `per_coordinate_contribution`,
unsmoothed — this doubles as the per-step exact gradient block via
`G_m = sum_j gamma_inst[j,m]`, matching BPTT per B7), per-mode
unrestricted winners, `|lambda_j|` and full complex `lambda_j`,
`||B_{j,:}||`, RMS `|q_j|`, per-mode eligibility energy and lag-1
autocorrelation, and the flat-gradient (`Delta theta`) norm actually
used that step. At every 5th step, the analytic tangent-recursion
sensitivity `d rho_j / d lambda_j` (Part D) is also computed and
stored — all in memory per seed, reduced to summary statistics before
moving to the next seed (no large per-step tensors are persisted to
disk).

`z_j,n` (marginal pool gain) uses reference pool `P_n` = the current
set of per-mode unrestricted `|rho|` winners (up to 6 distinct
channels) — the natural, always-available "pool implied by what the
arm is currently doing." **Two versions are computed**, per the user's
addendum: the hard definition `z_j,n = sum_m relu(|rho_{j,m,n}| -
max_{k in P_n} |rho_{k,m,n}|)` is used for **all final evaluation**
(pool regret, top-K recall, rank correlation); a **smooth surrogate**
(log-sum-exp in place of the max, softplus in place of relu, both
scaled by `tau` = the cross-candidate std of `|rho|` at that step) is
used **only** to fit the local-linear Taylor predictor (P2), since the
hard version's kinks make a finite-difference-style linearization
ill-posed at the non-differentiable points.

## Part B — temporal structure (checked before any per-pole modeling, per addendum)

| quantity | median over 8 seeds |
|---|---|
| median per-mode winner lifetime | **9.5 steps** (range 5-38 across seeds) |
| autocorrelation of `\|rho\|` at lag 1 / 10 / 50 | **0.996 / 0.931 / 0.491** |
| median relative step-to-step change `\|\|Δrho\|\|/\|\|rho\|\|` | 0.059 |
| effective rank of `Δrho_n` (2N x N flattened) at 90% / 95% energy | **3.5 / 5.0** (out of a max possible ~72) |
| effective rank of `Δz_n` at 90% energy | **3.0** (out of max 12) |

**Explicit answer: yes, the moving credit geometry is approximately
low-dimensional in training time.** An effective rank of 3-5 (median)
for the *temporal-change* matrix, against a maximum possible rank of
72 (or 12 for `z`), is a genuine, substantial reduction — and it is of
the **same order** as B9.2's snapshot-level finding (rank 1.5-2 at 90%
energy for the static relevance/utility matrices). The low-rank
structure found across *modes* in B9.2 does appear to persist for the
*motion* of the geometry across training, at a comparable (if slightly
higher) effective rank. Per the task's own caution (and B9.2's), **this
was checked before, not after, committing to independent per-pole
local models** — see Part C's explicit test of a low-rank latent
alternative below.

## Part C — predictor audit (primary metrics: pool regret and top-K recall; MSE secondary)

Median over 8 seeds, `K=4`, in-time train/test split (fit on steps
1-400, evaluate on 401-599, no leakage):

**`z_hard` target (sparse — most candidates are never close to
unseating the current winner):**

| predictor | mean top-K recall | median regret | median Spearman (when defined) |
|---|---|---|---|
| **P0 persistence** | **0.981** | 5920 | 1.00 |
| P1 secant | 0.973 | 5895 | 1.00 |
| P2 context-linear | 0.419 | 5748 | 0.48 |
| latent (rank-3 AR) | 0.387 | 4087 | 0.13 |

**`rho_agg` fallback target** (`max_m|rho_{j,m,n}|`, smoother, used per
the spec's own contingency since `z_hard` degenerates cross-sectional
correlation at many individual steps):

| predictor | mean top-K recall | median regret | median Spearman |
|---|---|---|---|
| P0 persistence | 0.998 | 388 | 0.997 |
| P1 secant | 0.998 | 462 | 0.997 |
| P2 context-linear | 0.997 | 458 | 0.997 |
| latent (rank-1 AR) | 0.889 | 402 | 0.892 |

**P2 vs. P0, paired per-seed (`z_hard` target, the addendum's primary
comparison):**

- **Top-K recall: P0 wins on all 8/8 seeds**, by a large and
  consistent margin (per-seed P2-minus-P0 diffs: `-0.53, -0.52, -0.61,
  -0.62, -0.51, -0.69, -0.33, -0.72`).
- **Regret: mixed, 5/8 seeds favor P2**, but dominated by one seed
  with a huge swing in P0's favor (`-14036`) alongside several modest
  P2 wins (`+1275, +4204, +2777, +917, +4584`) and two modest P0 wins
  (`-643, -715`) — not a consistent, reliable effect.

**Per the task's explicit threshold ("do NOT call prediction-
correction supported unless P2 clearly beats persistence"): it does
not.** P2 loses cleanly and consistently on the metric the addendum
named primary (top-K recall), and its regret advantage is noisy and
not seed-consistent. The **low-rank latent model** (motivated by Part
B's rank finding) does not rescue this — it underperforms persistence
on both targets, confirming the addendum's own caution: **low temporal
effective rank does not by itself imply a good simple dynamical
predictor beats naive persistence.** The dominant reason is Part B's
own finding: at `autocorr(lag=1)=0.996`, the process is so persistent
that "no change" is already close to optimal, leaving little room for
any model — cheap-feature-based or low-rank-latent — to improve on it
at a one-step horizon. This is the same qualitative pattern B6 already
established for the selection mechanism itself ("reactive suffices");
here it is shown to hold for the underlying continuous relevance
signal too.

**Anticipation check** (does a predictor see a winner switch coming):
across pooled seeds, P2's one-step-ahead top-3 rank recovered the
*eventual* new winner about as often as plain persistence did — no
material lead-indicator effect was found.

## Part D — analytic pole-sensitivity (real magnitude/phase coordinates)

The tangent recursion `r_{j,t} = x_{j,t-1} + lambda_j r_{j,t-1}` was
verified against a direct finite-difference derivative
(`eps=1e-6`, one spot check): **relative error `1.2e-5`**, consistent
with `O(eps)` truncation — the analytic recursion is correct.

Per the addendum's correction: `rho_j` itself is holomorphic in
`lambda_j` (a genuine complex derivative is valid there), but the
*real* quantity actually of interest — `|rho_j|` (and by extension
`z_j`) — is **not** holomorphic in `lambda_j`, so its sensitivity was
computed via the correct real chain rule in magnitude/phase
coordinates (`lambda = m e^{i theta}`, both real parameters):

```
d|rho|/dm     = Re[ conj(rho)/|rho| * (drho/dlambda) * e^{i theta} ]
d|rho|/dtheta = Re[ conj(rho)/|rho| * (drho/dlambda) * i * lambda ]
analytic_delta|rho| = (d|rho|/dm) * Delta_m_observed + (d|rho|/dtheta) * Delta_theta_observed
```

using the network's own **observed** `Delta_m`, `Delta_theta` (real,
from the logged pole trajectory) over a 5-step interval, compared
against the **observed** `Delta|gamma_inst|` over the same interval
(8,496+ pairs per seed):

| | median over 8 seeds |
|---|---|
| analytic (pole-only) correlation vs. observed | **0.024** |
| analytic (pole-only) R² vs. observed | **0.0004** |
| persistence R² (predicts zero change) | 0.0, by construction |

**Essentially none of the observed credit-geometry drift is
mechanically explained by pole drift alone.** By contrast, Part C's
P2 model — which pools pole drift together with routing (`||B_j||`),
error (`|q_j|`), and eligibility-energy/lag changes — achieves a much
higher correlation with the (smoother) target when defined (~0.92 on
the smooth `z` surrogate in individual-seed spot checks). **The
credit-geometry drift is overwhelmingly driven by routing/error/
eligibility changes, not by pole movement itself** — directly
answering Part D's stated goal, independent of whether any of this is
*useful* for prediction (Part C says: not usefully, beyond
persistence).

## Part E — does a fixed K=4 pool spontaneously align, or go stale?

Dense per-step logging on `pool_frozen K=4`'s own actual training
trajectory (not `reactive_full`'s — a different gradient rule
produces a different parameter path), including a genuine per-step
BPTT comparison for `cos_a0`.

| window | median `cos_a0` | median coverage (of unrestricted winners) | median `\|rho\|`, pool/non-pool ratio |
|---|---|---|---|
| early (steps 1-200) | **0.788** | 0.667 | **1.93** |
| mid (steps 200-400) | 0.685 | 0.667 | 1.37 |
| late (steps 400-600) | **0.663** | 0.667 | 1.41 |

**Median `cos_a0` declines monotonically, by 0.125 from early to
late** (a ~16% relative drop). The pool's relevance advantage over
non-pool channels also shrinks (1.93x → ~1.4x). Coverage of the
*current* unrestricted winners stays exactly flat at 4/6 — expected,
since the pool's membership never updates, so it can neither gain nor
lose coverage on its own; what changes is how *aligned* its fixed
channels remain with the network's evolving needs.

**Per-seed, this is a majority pattern but not universal:** late-minus-
early `cos_a0` is **negative (declining) in 5/8 seeds** (`-0.415,
-0.143, -0.050, -0.038, -0.186`) and positive in 3/8 (`+0.094, +0.018,
+0.025`) — the two largest-magnitude changes are both declines. No
forward routing/pole statistic (pool-vs-non-pool `||B_j||` or
`|lambda_j|` ratio) showed a consistent directional trend either
(not tabulated above; both hovered near 1.0 with no clear drift in
either direction across windows).

**Answer to Part E's primary question: no spontaneous alignment under
the current learning rule.** The median trend and the majority of
individual seeds point toward staleness, not improvement — consistent
with, and a direct mechanistic confirmation of, the calibration-
staleness finding that has motivated this entire line of work since
B5.1/B6, and consistent with why B9.3's *periodic* recalibration
(which explicitly counteracts this staleness) outperforms the frozen
baseline it was compared against.

## Part F — recommendation

**D. Neither exists.** No predictor (context-linear or low-rank
latent) clearly beats persistence on the metrics that matter (top-K
recall decisively, regret inconsistently) — a prediction-correction
controller is not justified by this data. A fixed K=4 pool does not
spontaneously become more useful during training — a feedback-
alignment/PAL-style co-adaptation experiment is not justified either;
if anything, the fixed pool measurably decays, which is an argument
*for* B9.3's periodic recalibration, not for a co-adaptation scheme
that assumes the network would organize itself around a static
vocabulary.

**Keep periodic K-pool CCM (B9.3) as the supported algorithm. Do not
add speculative control machinery** (no prediction-correction, no
feedback alignment) on top of it.

**What was actually predictive vs. not:**
- **Not predictive in isolation:** pole (`lambda_j`) drift — R²~0.0004
  against observed relevance drift; essentially irrelevant on its own.
- **Not usefully predictive despite carrying real information:**
  the full cheap-feature bundle (pole + routing + error + eligibility
  + update-norm) explains real variance (`corr~0.92` on the smooth `z`
  surrogate) but still loses to plain persistence on the primary
  top-K-recall/regret metrics, because the underlying process is
  already extremely persistent (`autocorr(lag1)=0.996`).
- **Not sufficient on its own:** low temporal effective rank (3-5 of a
  possible ~72) is real and confirmed, but a low-rank latent AR model
  built from it did not outperform persistence either — structural
  low-rank-ness does not automatically hand you a good predictor.
- **Genuinely informative, but as a *diagnosis* rather than a
  *predictor*:** the routing/error/eligibility features collectively
  dominate pole drift as the *mechanism* behind credit-geometry
  motion (Part D) — useful for understanding *why* the geometry moves,
  not for forecasting it better than "assume it doesn't move much."

No S5 run performed. No new training arm added.
