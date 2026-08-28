# Phase B14 — finite-size/random-subspace null theory and width scaling

Branch `S5-CCM-scale-validation` (current checkout). Theory/mechanism
audit only: **no new training algorithm, no S5 benchmark suite.**
Code: `credit_memory/b14_finite_size_null_theory.py` (new). Artifact:
`results/credit_memory/b14/b14_summary.json`. 8 seeds for Parts A/B/E,
3 seeds x 4 widths for Part G (cost-limited), 300 Monte Carlo null
draws per comparison (100 for Part G given the width x seed cost).

**Headline: D — MIXED, with a major practical caveat.** The
*unweighted* pure-orientation overlap `C_tc` shows genuine,
statistically significant excess concentration beyond a matched
Haar-random-subspace null (top-2 energy at the ~100th percentile,
`r90` below the null's 5th percentile in most seeds) — a real,
reproducible anomaly, resolving B11-B13's mystery of *why* nothing
seemed to move the rank: the anomaly lives in temporal *orientation*,
which none of those interventions actually randomized in the way this
phase's Haar-rotation null does. But the *practically relevant*,
amplitude-weighted `K_tc` (and hence the deployed pool/`K_epsilon`
behavior) is **statistically indistinguishable from a matched
random-orientation null at every width tested (`N=6,12,24,48`)** —
and **critically, both real and null `K_epsilon/M` grow toward 1.0 as
width increases**, meaning the clean small-`K` compression found at
this toy's `N=6` scale is itself consistent with — and does not
obviously survive beyond — small-finite-size random-subspace geometry.

## Part A — dimension/rank bookkeeping

| quantity | value (median, `N=6` toy) |
|---|---|
| `T` (ambient temporal dimension, `= N_CAL_TRAJ*T*BATCH`) | 1920 |
| `M` (candidate credit channels, `=2N`) | 12 |
| `N_lower` | 6 |
| algebraic rank(`U`), rank(`V`) | 6, 6 (full — small continuous-data matrices are generically full algebraic rank) |
| effective rank(`U`) 90/95/99% | 3 / 4 / 5 |
| effective rank(`V`) 90/95/99% | 4 / 5 / 6 |
| `p` (dim of `V`'s temporal subspace used in whitening), `q` (dim of `U`'s) | 6, 6 (full algebraic support — all prior B12/B13 whitening used the *full* algebraic rank, not a tolerance- or effective-rank-truncated subspace) |

`r_tc = 2` throughout B10-B13 was measured on `VU` directly (equivalently
`K_tc`, per the exact isometry identity) — the amplitude-*weighted*
quantity. This phase separates that from the unweighted `C_tc`
explicitly for the first time.

## Part B — exact matched Haar-random subspace null (unweighted `C_tc`)

300 Monte Carlo draws of independent Haar-random `p=6`, `q=6`
dimensional subspaces of `C^1920`, per seed. Median over 8 seeds:

| metric | real | null median | null 5-95% | percentile of real |
|---|---|---|---|---|
| `r90` | **2.0** | 4.0 | [3, 4] | **0.0** (below the null's 5th percentile in 6/8 seeds) |
| top-2 energy fraction | **1.00** | ~0.73 | [0.68, 0.82] | **1.0** (above the null's 95th percentile in 7/8 seeds) |

`r_excess_95` (sequential, magnitude-based: does the `i`-th real
singular value exceed *its own rank's* null 95% quantile) is **0 or 1**
across all 8 seeds — the raw *magnitudes* of individual canonical
correlations are not exceptional. But the *shape* metrics (`r90`,
top-2 fraction) are unambiguously extreme. **This is the key nuance
the task anticipated in Part D**: raw `r_excess` (magnitude-based) is
essentially 0, while shape-based excess (concentration relative to
null) is unmistakable — neither the naive "genuine outlier
singular values" story nor the naive "purely generic, r_excess=0"
story is complete on its own.

## Part E — weighted (`K_tc`-style) matched null

`K_null = diag(Sigma_V) C_null diag(Sigma_U)`, using the **real**
`Sigma_U`, `Sigma_V` (U's and V's own measured singular-value spectra)
with an independent Haar-random temporal orientation. Median over 8
seeds:

| metric | real `K_tc` | null median | percentile |
|---|---|---|---|
| `r90` | **2.0** | **2.0** | **0.73** (not extreme) |
| `r_excess_95` | 0 (every seed) | — | — |

**This is the reconciling result.** Once the real, individually
moderately-decaying amplitude spectra of `U` and `V` are incorporated,
even a *randomly oriented* null already reproduces `r90~2` — the
amplitude-weighted rank collapse is **statistically typical of
matched-marginal random geometry**, even though the *unweighted*
orientation itself (Part B) is genuinely anomalous. The anomaly in
`C_tc` is real but gets "used up"/dominated by the marginal amplitude
decay once the practically-relevant weighting is applied — explaining
why B11-B13's extensive battery of interventions (which mostly
preserved marginal amplitude structure) never moved the *weighted*
rank: **the weighted rank was never anomalous to begin with, at this
toy's scale.**

## Part F — small-matrix `(p,q)` geometry map

Pure random-matrix simulation (no real model), `T=1920` (matching the
toy), 50 draws per cell:

| `(p,q)` | median `r90` | median top-2 frac |
|---|---|---|
| `(2,2)` | 2.0 | 1.00 |
| `(4,4)` | 3.0 | 0.89 |
| `(6,6)` | 4.0 | 0.71 |
| `(8,8)` | 5.0 | 0.62 |

Confirms the null generator's calibration and shows the expected
qualitative trend: larger `(p,q)` at fixed large `T` gives a *higher*
null `r90` and *lower* null concentration — i.e., **the toy's small
`p=q=6` is itself why the unweighted null's `r90=4` is already fairly
concentrated relative to the maximum possible rank of 6** — random
geometry at these small dimensions is not "flat"; it already has real
structure to compare against, which is exactly why the Monte Carlo
null (not a naive uniform-spectrum assumption) is essential here.

## Part G — width scaling: the decisive experiment

`N_lower in {6, 12, 24, 48}` (`M = 2N`), 3 seeds each, matched null
computed at each width's own measured `p, q, T`:

| `N` | median `r_tc` (raw `C_tc`) | median `r_tc` (weighted `K_tc`) | median null `r90` (`K_tc`) | median `K_epsilon` (5%) | median `K_epsilon`/`M` |
|---|---|---|---|---|---|
| 6 | 2.0 | **2.0** | 2.0 | 3.0 | **0.25** |
| 12 | 4.0 | **2.0** | 3.0 | 12.0 | **0.50** |
| 24 | 10.0 | **4.0** | 4.0 | 48.0 | **1.00** |
| 48 | 21.0 | **6.0** | 7.0 | 85.0 | **0.89** |

**Two critical findings.** First, `r_tc` (weighted) grows with width —
**not flat** as the toy-scale (`N=6`) evidence from B9-B13 might have
suggested — going from 2 to 6 as `N` grows 8x (sub-linear but
unambiguously growing). Second, and more consequential:
**`K_epsilon/M` — the fraction of the full candidate bank needed for
5% pool regret — grows from 0.25 at `N=6` toward ~0.9-1.0 by `N=24-48`.
The clean small-`K` compression this whole research program has relied
on since B9.2 is itself a small-`N` phenomenon at this toy's scale and
does not obviously persist as width grows.** (`K_epsilon` searches at
`N=24,48` fall back to the documented greedy heuristic once exact
combinatorial search becomes infeasible — greedy is not guaranteed
optimal, but needing `~90-100%` of the candidate bank even under a
practical heuristic is itself the meaningful signal, not an artifact
of exhaustive-search cost.)

Critically, **at every width, the real weighted `r_tc` stays close to
the matched null's own `r90`** (`2.0` vs `2.0`; `2.0` vs `3.0`; `4.0`
vs `4.0`; `6.0` vs `7.0`) — the growth in `r_tc` (and hence in
`K_epsilon`) with width is **consistent with, not in excess of, what
matched random-subspace geometry predicts at each width** — this is
not evidence of hidden, growing task complexity; it is what generic
finite-dimensional geometry does as dimensions grow, and the practical
compression benefit erodes accordingly regardless of *why*.

## Part H — T scaling

Not run as a separate dedicated pipeline in this pass, given the time
budget prioritized Part G (explicitly "the decisive experiment") — an
explicit scope reduction. B12's earlier T-scaling result (effective
rank stable at ~2 across `T=30,60,120`, no bulk/coherent separation
found) remains the standing evidence on this axis; it was not revisited
with the null-comparison machinery developed here.

## Part I — revisiting `r_tc -> K_epsilon` with null correction

The `r_tc -> K_epsilon` bridge established in B10.1/B10.2 (`r_tc~2 ->
K~2-4`) **holds at the toy's own `N=6` scale but does not extend
cleanly to larger widths**: the *ratio* `K_epsilon / r_tc` grows sharply
(`~1.5x` at `N=6`, `~6x` at `N=12`, `~12x` at `N=24`, `~14x` at `N=48`)
— even though `r_tc` itself only grows modestly. **Null-corrected
`r_excess` (weighted) is ~0 at every width tested** (real `r_tc`
tracks, does not exceed, the matched null) — so the "genuine
task-credit dimension beyond random geometry" interpretation of the
bridge is **not supported once null-corrected**: the bridge documented
in B10.1/B10.2 is better read as an empirical fact about *this specific
small-scale toy's* random-subspace geometry than as a scale-invariant
law connecting an intrinsic task dimension to a small physical
dictionary size.

## Part J — verdict: **D, MIXED**

- The unweighted, pure temporal-orientation overlap `C_tc` contains
  **genuine, reproducible excess concentration** beyond the matched
  Haar-random null (Part B) — not a finite-size artifact in this
  specific, narrow sense.
- The **practically relevant**, amplitude-weighted `K_tc` — and
  therefore the deployed relevance matrix, `K_epsilon`, and the whole
  K-pool compression story — is **statistically indistinguishable from
  a matched random-orientation null at every scale tested** (Parts E,
  G). The amplitude-weighted rank collapse is explained by marginal
  singular-value decay plus generic (not special) orientation, not by
  a genuine low-dimensional task-credit manifold exceeding random
  geometry.
- This directly explains why B11-B13's extensive intervention battery
  (which mostly left marginal amplitude structure intact) never moved
  the weighted rank: **there was no weighted-rank anomaly to destroy.**
  Their true target — the unweighted orientation anomaly — was never
  actually tested by rotation/whitening/pole-ablation in the way this
  phase's Haar-random-subspace null does.
- **Practically consequential finding, independent of the "why":**
  `K_epsilon/M` grows from `0.25` to `~0.9-1.0` across `N=6` to
  `N=48`, tracking what matched random geometry predicts rather than
  staying flat. **The K-pool compression benefit established in B9.2/
  B9.3 should not be assumed to persist at larger widths without
  direct verification** — this is the single most important actionable
  finding of Phase B14 for the broader research program.

No new persistent training arm added. No S5 benchmark suite run.
