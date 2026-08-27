# Phase B9.2 — shared candidate-pool diagnostic

Branch `S5-CCM-scale-validation` (current checkout). Diagnostic only:
**`|rho_j|` stays the per-lower-mode selector (B9.1's conclusion); no
prediction-correction/resurrection, no training-algorithm change, no
S5 run.** Code: `credit_memory/b9_2_shared_pool.py` (new). Artifact:
`results/credit_memory/b9_2_shared_pool_summary.json`. 8 seeds x 2
checkpoints (step 0 = init, step 600 = end of an unmodified "online"-arm
training run — b5_train.py's existing update, inlined only because that
script doesn't expose intermediate params; not a new algorithm), same
`N=6, T=60, BATCH=8, N_CAL_TRAJ=4, N_TEST_TRAJ=4` as every prior phase.
`K_LIST=[1,2,4,8,12=all]`. `2N=12` keeps every combinatorial search here
exact (max `C(12,6)=924` subsets) — no greedy fallback was needed, one
is implemented and documented for when `2N` grows too large to brute
force.

**Headline: yes, with caveats.** A pool of `K=4` (of `2N=12` candidates)
constructed from calibration data (rho-guided or oracle-utility-guided
exact search, or even the much cheaper "most-frequently-winning
channels" heuristic) already recovers **median cos = 0.879-0.892**
against a `K=12` (unrestricted) ceiling of **0.888**, with **zero**
median regret against unrestricted `|rho|` selection for `rho_exact`
and `most_frequent`. Architecture-only pools (largest pole magnitude,
uniform timescale coverage) and random pools do **not** work at small
`K` (`0.44-0.68` at `K<=4`) and need `K~=8-12` to catch up — **the pool
must be built from data, but a small, data-built pool is enough.**

## Part 1 — per-mode winners, stability, cross-seed overlap

| | checkpoint 0 | checkpoint 600 |
|---|---|---|
| median unique winning channels (of `N=6` modes) | 4.5 | 5.0 |

Winner frequency, pooled over all 16 (seed, checkpoint) draws x 6 modes
= 96 picks, across the 12 candidates: `{0:11, 1:4, 2:6, 3:6, 4:2, 5:5,
6:13, 7:8, 8:12, 9:13, 10:7, 11:9}` — no dominant channel, but not
uniform either (candidate 4 wins 2x, candidates 6/9 win 13x each).

**Stability across training is weak:** between step 0 and step 600,
**67% of modes' winners change** (median), and the median Jaccard
overlap of the two checkpoints' winner *sets* is only **0.38**. A pool
frozen at init would not reliably match the winners that matter after
training — consistent with B5.1/B6's established "calibration
staleness" finding, now confirmed at the level of individual candidate
identity, not just aggregate selector accuracy.

**Cross-seed overlap is also weak:** median pairwise Jaccard overlap of
winner sets across the 28 seed pairs is **0.14** at init, **0.27** after
training. Since only pole *magnitude* is architecture-fixed across
seeds (`u0=linspace(0.90,0.995,N)`) while phase and routing (`B1`) are
seed-specific, this is expected — **a pool is a property of one
model's own trained parameters, not a universal constant reusable
across independently-initialized models.**

## Part 2/3 — global candidate-pool oracle and construction methods

Median held-out gradient cosine vs. BPTT, by method and pool size `K`
(pooled over 16 (seed, checkpoint) draws; per-mode selection *within*
the pool always uses `|rho|`, matching what would actually be
deployed — only pool *construction* differs by method):

| method | K=1 | K=2 | K=4 | K=8 | K=12 (=all) |
|---|---|---|---|---|---|
| oracle_exact (exact search, maximize `sum_m max_{j in P} U_{j,m}`) | 0.785 | 0.872 | **0.892** | 0.888 | 0.888 |
| rho_exact (exact search, maximize `sum_m max_{j in P} \|rho_{j,m}\|`) | 0.806 | 0.859 | **0.879** | 0.888 | 0.888 |
| most_frequent (cheapest: top-K most-often-winning channels) | 0.749 | 0.811 | 0.879 | 0.888 | 0.888 |
| largest_lambda (architecture only) | 0.568 | 0.680 | 0.658 | 0.748 | 0.888 |
| uniform_coverage (architecture only) | 0.443 | 0.636 | 0.736 | 0.877 | 0.888 |
| random (median of 20 draws) | 0.591 | 0.670 | 0.714 | 0.811 | 0.888 |

`K=12` (the full candidate set) is a built-in sanity check: every
method's pool degenerates to "all candidates," and every method
correctly reproduces the unrestricted-`|rho|` baseline of 0.888 exactly
there.

**Regret vs. unrestricted `|rho|` selection** (utility units, median):
`rho_exact` and `most_frequent` reach **zero median regret by K=4**;
`oracle_exact` is *negative* at K=2/K=4 (`-403`, `-2293`) — a
pool built to maximize true oracle utility, combined with the
practical `|rho|`-within-pool pick, can occasionally *exclude* a
candidate that would otherwise mislead `|rho|`'s unrestricted argmax,
so restricting to a well-chosen pool sometimes slightly *helps* rather
than only costing. Architecture-only and random pools carry regrets
`4,600-155,000x` larger at `K<=4`, only reaching zero at `K=12`.

**Coverage** (mean number of the 6 modes' *unrestricted* `|rho|`
winners that land inside the pool): `rho_exact` covers 5.25/6 modes
already at `K=4` and all 6 by `K=8`; architecture-only/random methods
cover only 1.4-3.9/6 at the same `K`. Notably, `rho_exact` at `K=4`
achieves near-ceiling *cosine* despite missing ~1 mode's true winner —
reinforcing B9.1's finding that a near-miss selection costs little in
aggregate gradient fidelity, now at the pool level too.

## Part 4 — relevance/utility matrix structure

Effective rank (fraction of Frobenius energy) of `R[j,m]=rho_{j,m}`
(`2N x N = 12 x 6`) and the oracle-utility matrix `U[j,m]`, median over
all 16 (seed, checkpoint) draws:

| matrix | 90% | 95% | 99% | full rank |
|---|---|---|---|---|
| `R` (rho) | **2.0** | 3.0 | 4.0 | 6 |
| `U` (oracle utility) | **1.5** | 2.0 | 2.0 | 6 |

Both matrices are substantially low-rank relative to their `N=6` full
rank — most of the *energy* in "how relevant is candidate `j` to mode
`m`" is explained by 1-2 shared directions. **This is a structural
echo of, not identical to, Part 1's winner diversity:** low Frobenius
rank means a few directions dominate the *magnitude* pattern, but
individual per-mode *argmax* identity can still differ across those
few directions (as Part 1 showed with 4.5-5 unique winners out of 6
modes) — rank and "does a small discrete pool of individual channels
work" are related but distinct questions, and Part 2/3's direct pool
search is the one that actually answers the operative question here.
**Per the task's own caution: this low rank does not by itself imply a
cheap online algorithm** — extracting it online would require either
the discrete pool search already run here (needs the full `R`/`U`
matrices, i.e. full calibration) or an SVD/low-rank online estimator,
neither of which is free, and building either is explicitly out of
scope for this diagnostic.

## Part 5 — complexity projection (with a correction to B9.1)

**Correction:** B9.1's Part 5 stated a single shared channel gives
"`O(1)` persistent state instead of `O(N)`" for deployment. On closer
inspection this is imprecise: the candidate filter
`x_{j,m}(t) = lambda_j x_{j,m}(t-1) + Sa0_t[:,m]` depends on mode `m`
through its *input* even when the pole `lambda_j` is shared across
modes, so **deployment always needs `O(N_lower)` persistent filter
states, whether the channel is shared or chosen per mode** — sharing a
channel saves bookkeeping (one index instead of `N_lower`), not
deployed state or compute. The real saving from pooling is in
*adaptive scoring*, not deployment:

| stage | persistent state | compute per step |
|---|---|---|
| full adaptive scoring (current, all `2N_upper` candidates per mode) | `O(2 N_upper N_lower)` | `O(2 N_upper N_lower)` |
| **K-pool adaptive scoring** (this diagnostic's target) | `O(K N_lower)` | `O(K N_lower)` |
| current rank-1 deployment (any of B4-B9) | `O(N_lower)` | `O(N_lower)` |
| B9.1's one-shared-channel deployment (corrected) | `O(N_lower)` | `O(N_lower)` |

A `K`-channel shared dictionary reduces the *ongoing adaptive/scoring*
cost from `O(2N_upper N_lower)` to `O(K N_lower)` — a `2N_upper/K`-fold
reduction (`3x` at `K=4, N_upper=6` in this toy; would be far larger at
S5 scale, where `N_upper` is tens to low hundreds) — while leaving
deployment cost unchanged at `O(N_lower)`, matching the task's own
target characterization ("`O(K N_lower)` adaptive credit-state cost,
with occasional `O(N_upper N_lower)` pool recalibration" for periodic
re-examination of the full candidate set, needed because Part 1 showed
the optimal pool is not static across training).

## Part 6 — answer

**"Is there a small global vocabulary of upper SSM credit poles that
preserves the near-oracle per-lower-mode `|rho|` performance?"**

**Yes, conditionally.** A pool of `K~=4` out of `2N=12` candidates
(roughly two-thirds of `N_lower=6`), built from calibration data
(exact search on `|rho|` or oracle utility, or even the far cheaper
"most-frequently-winning channels" heuristic), recovers **essentially
all** of the achievable held-out gradient fidelity (`0.879-0.892` vs. a
`0.888` unrestricted ceiling) with **zero** median regret against the
current unrestricted `|rho|` selector. This is a genuine, actionable
`2N_upper/K`-fold reduction in ongoing adaptive-scoring cost.

The conditions that matter: **(1)** the pool must be built from data —
architecture-only construction (largest pole magnitude, uniform
timescale coverage) and random pools need `K~=8-12` (i.e., most or all
of the candidate set) to reach comparable fidelity, so there is no free
lunch from a fixed, data-independent dictionary; **(2)** the optimal
pool is not stable — 67% of per-mode winners change between init and
step 600 of training, and cross-seed overlap is low (Jaccard `0.14-
0.27`), so any deployed version needs the "occasional `O(N_upper
N_lower)` pool recalibration" the target architecture already
anticipates, not a one-shot calibration; **(3)** low matrix rank in
`R`/`U` is suggestive but not sufficient evidence on its own — the
actual discrete pool search (Part 2/3) is what establishes the result,
not the rank number by itself.

**Not yet implemented, not yet run on S5**, per the task's scope: this
diagnostic establishes that a `K`-pool architecture is worth building,
not what its recalibration schedule or online pool-selection mechanism
should be.
