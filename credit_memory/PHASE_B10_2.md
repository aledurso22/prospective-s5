# Phase B10.2 — closing two theory-to-algorithm gaps

Branch `S5-CCM-scale-validation` (current checkout). Theory/mechanism
audit only: **no prediction-correction, no event triggers, no
feedback alignment, no new persistent training arm, no S5.** Code:
`credit_memory/b10_2_selector_and_calibration_theory.py` (new, reuses
B10/B10.1 machinery throughout). Artifact: `results/credit_memory/
b10_2/b10_2_summary.json`. Same 8 seeds, `N=6, T=60, BATCH=8,
N_CAL_TRAJ=4` protocol.

**Two independent verdicts: S-B (selector theory, PARTIAL) and C-B
(calibration theory, PARTIAL).** Both parts found genuine, confirmed
structure — but in each case the strongest, most decisive version of
the hypothesis (a validated conditional ranking bound; genuinely
subquadratic cheap calibration) did not survive contact with the data.

## Setup note

`D_m` and `gamma[j,m]` are the **scalar** quantities already
established throughout B9.1-B10.1 (`D_m = G[m]`, the exact full-bank
gradient; `gamma[j,m]` = the leak-fixed per-candidate contribution,
literally the same object as `rho[j,m]`) — this matches the "empirically
near-oracle" premise the task states as already established (B9.1's
own result), which is a fact about this *scalar* framework. An initial
attempt to instead treat `D_m`/`gamma[j,m]` as unsummed per-timestep
**vectors** (using `teacher.py`'s `g_t_bptt`) was tried first and
discarded: it produces a well-defined but much harsher, differently-
scaled quantity (97% of candidates showed negative "ideal utility"),
inconsistent with the near-oracle premise the task builds on — reported
here for transparency, not used in the final analysis. Under the
scalar framework, `\|rho[j,m]\| = \|\|gamma[j,m]\|\|` **by
construction** (literally the same number) — `cos_theta[j,m]` is
therefore a **phase-alignment** between two complex scalars, `Re[conj(D_m)
gamma[j,m]] / (\|D_m\|\|gamma[j,m]\|)`, not a high-dimensional vector
angle.

# PART I — why does `|rho|` select good physical poles?

## A. Direct relation between `|rho|` and ideal utility

Median over 8 seeds x 6 modes:

| quantity pair | value |
|---|---|
| Spearman(`\|rho\|`, `U`) | **0.20** (range -0.03 to 0.36 across seeds) |
| Pearson(`\|rho\|`, `U`) | **0.55** (notably stronger than Spearman) |
| Spearman(`\|rho\|`, `\|\|gamma\|\|`) | 1.00 (identical by construction, see above) |
| Spearman(`\|rho\|`, `cos_theta`) | 0.10 (weak) |
| top-1 agreement (`\|rho\|` vs. `U`) | **72.9%** (matches B9.1's own established finding exactly) |
| mean top-4 recall | 0.55 |
| median `\|rho\|`-guided pool regret | 2.26 (small, absolute units) |

The **Pearson correlation is meaningfully stronger than the Spearman
correlation** — the *linear/magnitude* relationship between `|rho|`
and `U` is more reliable than the *exact rank order*, echoing B9.1's
finding that the oracle utility landscape is dominated by a few large
values over a long, noisy tail (Spearman is sensitive to tail-order
noise that doesn't affect decisions).

## B. The "narrow positive cone" hypothesis — partially supported, not narrow

| | mean `cos_theta` |
|---|---|
| top-25% by `\|rho\|` | **0.372** |
| bottom-25% by `\|rho\|` | 0.209 |

Top candidates *are* shifted toward higher alignment than bottom
candidates — the direction of the hypothesis is correct — but **the
cone is not narrow**: both groups have standard deviations exceeding
their means (`std > mean` in both), and **42% of all (candidate,
mode) pairs have negative ideal utility** (a single candidate used
alone would make things *worse* than doing nothing) — overshoot is
common, not universal. **The hypothesis is falsified in its strict
form** (`cos_theta_j approx c_m > 0` with limited variation does not
hold — variation is large) but confirmed in its weaker, directional
form (favorable alignment is more common, not universal, among
high-`|rho|` candidates).

## C. Decomposing ranking failures

Across 8 seeds, a median of 167 of 396 possible `(j,k)` pairs per seed
(~42%) are ranked differently by `|rho|` than by `U`. Decomposition of
these misranked pairs:

- **85.4%** show *opposing signs* between the `cos_theta` difference
  and the `\|\|gamma\|\|` difference — i.e., **misranking predominantly
  happens exactly when alignment and magnitude disagree** (one
  candidate wins on angle, the other on magnitude, and `|rho|`'s
  single coherent-sum number can't cleanly resolve the tradeoff).
- Only **9.2%** of misranked pairs are the P/Q branch-pair of the same
  underlying pole — **P/Q cancellation is not the dominant failure
  mode.**
- Routing-factor differences were not isolated as a separate dominant
  cause beyond what the magnitude term already captures (routing
  enters through `\|\|gamma\|\|` itself, per B10's own factorization).

**This is the clearest mechanistic finding in Part I**: `|rho|` works
by collapsing alignment and magnitude into one number; it fails
specifically, and predictably, when those two dimensions pull in
opposite directions.

## D. Conditional ranking theorem — derived, tested, and found too conservative as stated

Derivation (bounded angular spread `cos_theta in [c-delta, c+delta]`,
bounded magnitude `\|\|gamma\|\| <= g_max`):

```
U_j - U_k = (||gamma_j||-||gamma_k||)[2||D||c - (||gamma_j||+||gamma_k||)]
           + 2||D||(delta_j ||gamma_j|| - delta_k ||gamma_k||)
```

If `2||D||c - 2 g_max > 0` (a "pre-overshoot" magnitude regime) and the
bracket-weighted magnitude gap exceeds the angular-spread error term
`4||D|| delta g_max`, then `||gamma_j|| > ||gamma_k|| => U_j > U_k`.

**Tested empirically with global (dataset-wide) percentile-based
`c`, `delta`, `g_max`, `D_typ` per seed: the theorem applies to ZERO of
2,017 checked pairs across all 8 seeds.** Only 1 of 8 seeds even
satisfies the basic pre-overshoot precondition at the global-parameter
level (`frac_seeds_pre_overshoot_regime = 0.125`) — **most seeds are
already in an overshoot-dominated regime globally**, so a single
global `(c, delta, g_max)` characterization is fundamentally mismatched
to this data's heterogeneity (angular alignment and magnitude both
vary too widely across candidates/modes for one global regime
description to certify anything). **Per the task's own instruction to
falsify if necessary: the theorem as derived is algebraically correct
but, parameterized globally, is vacuous on this toy's actual
statistics** — a per-mode-conditional (not global) parameterization
would likely be needed to produce a non-trivial certificate, which was
not pursued further given scope.

# PART II — can low-dimensional credit be accessed without full `MxN` calibration?

## A. Oracle sample-complexity curve

Median minimum `(r_rows, r_cols)` over 8 seeds, using **full-matrix**
QR pivoting (a geometry characterization, not a claimed cheap method):

| criterion | `r_rows` | `r_cols` | state-cost fraction |
|---|---|---|---|
| 95% Frobenius | 4.0 | 4.0 | **1.00** |
| 99% Frobenius | 5.5 | 5.5 | 1.38 |
| 95% winner preservation | 5.0 | 5.0 | 1.29 |
| 95% top-K recall | 5.5 | 5.5 | 1.38 |
| near-zero pool regret | 4.5 | 4.0 | 1.04 |

**At this toy's scale (`M=2N=12, N_lower=6`), no criterion is met
below 100% of the full calibration-state cost, even with oracle
(full-matrix-informed) pivot selection.** The additive row+column cost
model (`r_rows N_lower + r_cols M`) only pays off once `M, N_lower`
are large relative to the required rank — a small matrix like this
toy's genuinely does not have room for a *sub-quadratic* saving to
show up, independent of how good the pivots are. This is a scale
limitation of the diagnostic, not evidence against the underlying
low-rank geometry (Parts I-III of this and prior phases establish that
geometry clearly).

## B. Cheap, strictly no-leakage pivot discovery

`r=4`, median over 8 seeds (state-cost fraction is `1.0` for every
method at this `r`/scale, per II.A's finding — the comparison here is
about *decision quality per unit of the same cost*, not cost itself):

| method | Frobenius rel. err | winner preserved | pool regret | certified fraction |
|---|---|---|---|---|
| B0 oracle QR (uses full matrix — reference, not "cheap") | 0.034 | 0.833 | **0.29** | 0.667 |
| B1 random | 1.29 | 0.667 | 0.62 | 0.667 |
| **B3 adaptive cross approximation, no-leakage** | **0.077** | **0.75** | **3.62** | 0.667 |
| B4 deployed-K-pool-seeded rows | 0.130 | 0.667 | 0.65 | 0.667 |
| B5 U-QR-seeded columns | 0.181 | 0.667 | 2.28 | 0.667 |

B3 (a genuine, no-leakage adaptive cross approximation — each pivot
choice uses only entries from rows/columns already revealed, verified
by construction: `cur_reconstruct` at every step only reads previously
revealed rows/columns) reaches **Frobenius error within ~2x of the
oracle** and comparable winner preservation, but its **pool regret is
~12x worse than the oracle's** (3.62 vs. 0.29) — Frobenius accuracy
and pool-objective quality diverge here, a reminder (echoing B10's own
finding) that matrix error is not the right proxy for decision
quality. B5 (cheap, architecture/eligibility-only column seeding from
`U`'s own QR pivots, no credit propagation needed) does **not**
reliably beat random on regret (2.28 vs. random's 0.62) — eligibility
structure alone is not a good guide to which lower modes to prioritize.

## C. Cost accounting

At this toy's scale, `sampled_state_cost / full_state_cost >= 1.0` for
every tested method and every decision-quality criterion (Part II.A)
— **no genuine sub-quadratic saving was demonstrated here**; the
reported comparisons are about relative decision quality at matched
(roughly full) cost, which is still informative about *which rows/
columns matter*, but does not itself establish a cheaper algorithm at
this scale.

## D. Decision-aware / certified reconstruction

Certified fraction (via the margin theorem, `Delta_m > 2 epsilon_m`)
is **0.667 uniformly across every pivot method tested** in Part II.B,
including random — at `r=4`, the achieved `epsilon` is small enough
relative to typical margins that certification doesn't discriminate
between methods here; it does, however, confirm none of the tested
reconstructions produce spurious near-tie decisions the margin theorem
would catch.

## E. Matrix completion baseline — structure clearly matters

Same observed entries (union of B3's revealed rows/columns), generic
ALS matrix completion vs. CUR:

| | Frobenius rel. err | winner preserved | pool regret |
|---|---|---|---|
| generic matrix completion (ALS) | 0.786 | 0.333 | 7.16 |
| **CUR (same entries)** | **0.077** | **0.750** | **3.62** |

**Exploiting the row/column (CUR) structure specific to this problem
clearly and substantially beats generic entrywise matrix completion at
the identical information budget** — a decisive, unambiguous result
for Part II.E's stated question.

## F. Temporal reuse of pivots (warm start, not prediction)

4 seeds, two calibration events at steps 0 and 100 of an unmodified
online-arm training run: fresh ACA discovery (`fro=0.188`) vs.
warm-started ACA (initialized from the previous event's row pivot,
`fro=0.202`) — **no clear benefit from reuse** (warm start is
marginally, not significantly, worse on Frobenius error; winner
preservation is identical, `0.833`, for both). Pivot overlap between
the two calibration events is `62.5%` (moderate — consistent with
B9.2's finding that the optimal pool drifts but not completely, over
100 steps). **This diagnostic does not support temporal pivot reuse as
currently implemented**; it is reported as a null result, not a
negative one that rules out smarter reuse strategies.

# PART III — connecting `r_tc` to `K_epsilon`

`K_epsilon(r)`, median over 8 seeds, extending B10.1's single-threshold
curve to three regret tolerances:

| `r` | K at 2% regret | K at 5% regret | K at 10% regret |
|---|---|---|---|
| 1 | 10.0 | 7.0 | 5.5 |
| **2** | **6.0** | **2.5** | **2.0** |
| 3 | 3.5 | 2.5 | 2.0 |
| 4-6 | 3.5 | 2.5 | 2.0 |

**Confirms and sharpens B10.1's bridge**: the sharp transition at
`r=2` holds across all three tolerance levels, and the curve plateaus
from `r=3` onward at every tolerance — retaining more than 2-3
temporal coupling modes buys essentially nothing further in terms of
the physical pool size needed. At the tightest tolerance (2% regret),
`r=1` still needs a large pool (`K=10`, nearly the full `2N=12`); by
`r=2` this collapses to `K=6`, and by `r=3` to `K=3.5` — a genuine,
reproducible, multi-threshold confirmation of the `r_tc -> K_epsilon`
bridge.

# Verdicts

## Selector theory: **S-B — PARTIAL SUPPORT**

Clear, real structure was found (72.9% top-1 agreement matching B9.1;
Pearson correlation 0.55; a directional, if not narrow, alignment cone;
a clean mechanistic account of *when* `|rho|` fails — angle/magnitude
disagreement, not P/Q cancellation or routing). But the attempt to
formalize this into a **useful conditional ranking bound failed**: the
derived theorem, tested honestly with empirically-measured global
parameters, certified zero of 2,017 pairs. `|rho|`'s practical success
is real and partially explained, but **still contains task-specific
information not captured by the tested geometric account** (global
magnitude/alignment regime) — this is exactly S-B's definition, not
S-A (no useful bound was produced) and not S-C (the correlational and
mechanistic evidence is real, not absent).

## Calibration theory: **C-B — PARTIAL SUPPORT**

Oracle CUR remains strong and clearly beats generic matrix completion
at matched information (Part II.E, decisive). But **cheap, no-leakage
pivot discovery (B3 ACA) loses meaningful quality relative to the
oracle** — a 12x regret gap despite similar Frobenius error — and **no
method demonstrated genuinely sub-quadratic calibration-state savings
at this toy's scale** (state-cost fraction never dropped below 1.0 for
any decision-quality criterion). This is squarely C-B, not C-A (no
sub-quadratic saving shown) and not C-C (decision-preserving
reconstruction clearly does *not* require the full matrix — `r=4` of
`6` CUR reconstructions work reasonably well, just not "cheaply" in
the state-cost sense at this specific small scale).

No new persistent training arm added. No S5 run performed.

Commit: see this phase's commit hash in the repository log.
