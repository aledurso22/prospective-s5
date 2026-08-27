# Phase B9.3 — periodically recalibrated K-pool CCM (first real training arm)

Branch `S5-CCM-scale-validation` (current checkout). Toy system only,
per the task's scope: **no S5 run.** Code: `credit_memory/
b9_3_pool_training.py` (new). Artifact: `results/credit_memory/b9_3/
b9_3_pool_training_summary.json`. 8 seeds, `N=6, T=60, BATCH=8,
DELAY=20, STEPS=600`, `N_CAL_TRAJ=4`, real delayed-copy task (same
task/config as B5-B8), Adam `LR=1e-3`, `clip=0`.

**Verdict: PARTIAL PASS.** The core mechanism works — a periodically
recalibrated `K=4` pool matches the expensive unrestricted-reactive
ceiling in gradient fidelity and clearly beats both a frozen `K=4` pool
and a random `K=4` pool — but the margin over the plain "online"
baseline and the task-loss differences across all arms are not cleanly
resolved at this seed count. Recommend more seeds before treating any
specific refresh interval as validated, and before this configuration
is a candidate for S5.

## Algorithm implemented (exactly per spec)

- **Pool construction** (calibration events): full `O(2N)` calibration
  via `credit_memory/b6_prospective_tracking.py::causal_prefix_selection`
  (unmodified, already B9.1 leak-fixed), then the deployable pool is
  built with **`pool_most_frequent`** (B9.2's cheapest method that
  matched the exact pool well). An **oracle-utility pool is computed at
  every calibration event purely for logging** (coverage vs. oracle,
  Jaccard vs. oracle) — never used to build a deployable arm's gradient.
- **Between calibration events**: relevance is tracked reactively (EMA,
  `T2_GAMMA=0.08`, matching B6's T2 exactly) for **only the `K`
  candidates in the pool** — `f_diag[P]`-sized state, genuinely `O(K)`
  per mode, not `O(2N)`.
- **Per-mode selection**: each of the `N=6` lower modes runs its own
  `hysteretic_select` (`HYSTERESIS_MARGIN=0.15`) within the pool every
  step — no forced shared channel across modes.
- **No dormant-state resurrection**: the deployed gradient
  (`credit_memory/b4_deploy.py::b4_layer0_gradient`, unmodified) resets
  its own filter state fresh every step regardless of which pool
  channel is selected, exactly as PHASE_B9.md Part 1 established.
- **No prospective/prediction-correction** anywhere in this arm.

**Leakage-fix verification:** the full calibration events reuse
`causal_prefix_selection` unmodified, which already calls
`estimators[m].reset_filter()` at each calibration-trajectory boundary
(B9.1's fix). The between-calibration reactive observation
(`pool_batch_observation`, a K-length reimplementation of
`single_batch_observation` since that function hardcodes `d=ones(2N)`
via a module-global) calls `per_coordinate_contribution` fresh every
step on a single batch — no cross-call state to leak, matching
PHASE_B9.md Part 1(c)'s established stateless-per-step observation
design. No new leakage introduced.

## Arms

| arm | meaning | candidate states |
|---|---|---|
| A0 `online` | existing rule, unchanged | 0 |
| A1 `reactive_full` | continuous `O(2N)` reactive EMA over all candidates (=B7's `a1_rank1`) | `2N x N = 72` |
| A2 `pool_frozen` | pool + per-mode pick built once at step 0; per-mode pick *inside* the frozen pool still reacts continuously (pool membership never refreshed = `pool_periodic` with refresh=infinity, verified bit-identical to `refresh=600` below) | `K x N` |
| A3 `pool_periodic` | this phase's new arm | `K x N` |
| A4 `full_causal` | exact, uncompressed causal P/Q (`credit_memory/full_causal.py`, unmodified) | `72` |
| Bref `bptt` | exact reference | 0 |

## Results (median over 8 seeds)

`cos_whole` is the full flattened-gradient cosine vs. BPTT at the final
diagnostic checkpoint (step 600) — diluted by the layers/readout that
are identical across all arms. `cos_a0` isolates the layer-0 "a" block,
the only block that actually differs by arm, and is reported both at
the final checkpoint and averaged over the late window (steps 300+600)
to reduce single-diagnostic-batch noise.

| config | cos_whole | cos_a0 (late-avg) | late task loss | states |
|---|---|---|---|---|
| online (A0) | 0.848 | 0.591 | 0.1331 | 0 |
| bptt (Bref) | 1.000 | 1.000 | 0.1147 | 0 |
| full_causal (A4) | 1.000 | 1.000 | 0.1147 | 72 |
| reactive_full (A1, ceiling) | 0.821 | **0.710** | 0.1208 | 72 |
| pool_frozen K=1 | 0.920 | 0.602 | 0.1252 | 6 |
| pool_frozen K=2 | 0.803 | 0.579 | 0.1248 | 12 |
| pool_frozen K=4 | 0.830 | 0.484 | 0.1282 | 24 |
| pool_frozen K=8 | 0.816 | 0.735 | 0.1216 | 48 |
| pool_periodic K=4, refresh=50 | 0.827 | 0.620 | 0.1202 | 24 |
| **pool_periodic K=4, refresh=100** | 0.837 | **0.731** | 0.1212 | 24 |
| pool_periodic K=4, refresh=200 | 0.888 | 0.625 | 0.1227 | 24 |
| pool_periodic K=4, refresh=600 (=frozen) | 0.830 | 0.484 | 0.1282 | 24 |
| pool_periodic K=1, refresh=100 | 0.914 | 0.591 | 0.1225 | 6 |
| pool_periodic K=2, refresh=100 | 0.851 | 0.503 | 0.1238 | 12 |
| pool_periodic K=8, refresh=100 | 0.881 | 0.696 | 0.1189 | 48 |
| pool_periodic K=4, refresh=100, **random pool** | 0.804 | **0.442** | 0.1362 | 24 |

`refresh=600` reproduces `pool_frozen K=4` bit-for-bit (0.484/0.1282
exactly), as it must (no recalibration occurs within 600 steps for
either) — a useful internal consistency check.

## Paired ablations (8-seed sign test, `cos_a0` late-avg, per spec)

| comparison | median diff | sign-test wins/losses |
|---|---|---|
| **periodic K=4 vs. frozen K=4** | **+0.198** | **7 / 1** |
| **periodic K=4 (data-built) vs. periodic K=4 (random)** | **+0.256** | **7 / 1** |
| periodic K=4 vs. reactive_full (unrestricted ceiling) | +0.021 | 5 / 3 |
| periodic K=4 vs. online baseline | +0.141 | 4 / 4 |

**Periodic recalibration clearly beats both frozen and random pools**
(7/8 seeds each, consistent with a real effect, though `n=8` keeps
these individually short of conventional significance — `p~=0.07`
two-tailed for 7/8 under the null). **Periodic K=4 is statistically
indistinguishable from the full unrestricted-reactive ceiling** (5/3,
essentially a coin flip) **while using `72/24 = 3x` fewer candidate
states.** The comparison against plain "online" is genuinely noisy at
this seed count (4/4) — consistent with B5/B7's own established
finding that this single-diagnostic-batch `cos` metric can make
"online" look deceptively good on individual seeds depending on how
the real task's structure happens to align with its biased gradient
direction; task-loss differences across *all* arms here are similarly
modest, echoing B5.1's finding that task loss is a much less sensitive
signal than gradient cosine at this scale.

**Refresh-interval sweep is not sharply resolved.** `refresh=100` gave
the best `cos_a0` among `{50,100,200,600}` in this run, but with only
8 seeds and a single per-checkpoint diagnostic batch, the exact optimum
should not be treated as established — only the qualitative result
("periodic clearly beats frozen; too-infrequent recalibration decays
back toward frozen's number") is well-supported.

## Pool membership, turnover, and oracle-coverage diagnostic

Representative trace (seed 0, `K=4, refresh=100`):

| step | pool | coverage (of unrestricted winners) | Jaccard vs. prev pool | Jaccard vs. oracle pool |
|---|---|---|---|---|
| 0 | {0,5,6,9} | 0.83 | — | 0.60 |
| 101 | {3,4,6,9} | 1.00 | 0.33 | 0.60 |
| 201 | {3,4,6,9} | 1.00 | 1.00 | 1.00 |
| 301 | {4,5,6,9} | 1.00 | 0.60 | 0.60 |
| 401 | {0,4,5,9} | 1.00 | 0.60 | 0.33 |
| 501 | {3,4,6,9} | 1.00 | 0.33 | 0.60 |

Real, non-trivial turnover is happening (Jaccard vs. previous pool
ranges 0.33-1.0) — periodic recalibration is doing genuine adaptive
work, not a no-op, consistent with B9.2's finding that the optimal
pool is not stable across training. Aggregated over all 8 seeds and 6
recalibration events: median coverage (fraction of the `N=6` modes'
*unrestricted* `|rho|` winners that land inside the current pool) is
**1.00 at K>=4** (K=8: 1.00, K=2: not separately logged, K=1: **0.33**
— a `K=1` pool, being forced to serve every mode from one shared
channel, predictably covers only a third of modes' true winners).
**Median Jaccard of the deployable (`most_frequent`) pool vs. the
diagnostic-only oracle-utility pool is 0.60 at K=4** — the two
construction criteria agree on most, not all, of the pool, consistent
with B9.2's finding that `most_frequent` is a good but imperfect proxy
for the true oracle pool.

## Cost accounting (measured, 8-seed mean, 600 steps)

| | calibration (per event, amortized over the run) | between-event reactive tracking (measured) | deployment (measured, same for every arm) | candidate states |
|---|---|---|---|---|
| A1 reactive_full | 0.008s (one-time) | 0.457s total (`O(2N)`/step) | 0.478s total (`O(N)`/step) | 72 |
| A2 pool_frozen K=4 | 0.009s (one-time) | 0.408s total (`O(K)`/step) | 0.484s total | 24 |
| A3 pool_periodic K=4, refresh=100 | 0.050s total (6 events, amortized ~0.08ms/step) | 0.406s total (`O(K)`/step) | 0.481s total | 24 |
| A3 pool_periodic K=4, refresh=50 | 0.099s total (12 events) | 0.401s total | 0.477s total | 24 |
| A3 pool_periodic K=8, refresh=100 | 0.053s total | 0.477s total | 0.509s total | 48 |
| A4 full_causal | 0 (no calibration, always exact) | 0 | 0 (single vectorized call) | 72 |

Deployment cost is dominated by the always-present, arm-independent
`tcg.assemble()` call and is essentially flat across arms (~0.48-0.51s),
as expected (Part 5 of B9.2 already established deployment cost does
not depend on `K`). **The between-event reactive-tracking cost scales
with `K`** (0.401-0.408s at `K=4`, 0.477-0.509s at `K=8`/unrestricted)
but only mildly at this toy's `N=6, T=60` scale — Python-loop/function-
call overhead dominates the raw FLOP count here, exactly as B9.1/B9.2
already flagged; the asymptotic `O(K N_lower)` vs. `O(2N_upper
N_lower)` advantage (a genuine `3x` FLOP reduction at `K=4,
N_upper=6`) would widen sharply at production scale. The amortized
periodic-recalibration cost itself is negligible relative to per-step
cost even at the most frequent tested schedule (`refresh=50`: ~0.17ms/
step amortized vs. ~0.7ms/step reactive tracking).

## Answer to the primary question

**"Can periodic K~4 pool recalibration preserve most of the
unrestricted `|rho|` gradient/task benefit while reducing continuous
adaptive scoring from `O(2N_upper N_lower)` to `O(K N_lower)`?"**

**Yes, on the core mechanism — with a caveat on statistical power.**
Periodic `K=4` recalibration (1) matches the unrestricted-reactive
ceiling's gradient fidelity (median diff `+0.021`, 5/3 — a tie, not a
loss), (2) clearly and consistently beats both a frozen `K=4` pool
(7/8 seeds) and a random `K=4` pool at the identical schedule (7/8
seeds), and (3) does so at a measured `3x` reduction in candidate
states (24 vs. 72) with negligible amortized recalibration overhead.
This directly supports the target architecture.

The caveat: at `n=8` seeds and a noisy single-diagnostic-batch `cos`
metric, the margin over the plain **online** baseline specifically is
not a clean, consistent win (4/4), and task-loss differences across
*all* arms (including BPTT itself) are modest — both are consistent
with, not contradicted by, the established B5/B5.1/B7 pattern that this
metric is noisy and that task loss is a weak discriminator at this
scale, but they mean the *practical, deployable* benefit over doing
nothing is not yet demonstrated as robustly as the *mechanism*
comparison (periodic vs. frozen vs. random) is.

## Recommendation

**PARTIAL PASS.** Keep the periodic K-pool architecture as the leading
candidate mechanism — the internal comparisons (periodic vs. frozen,
data-built vs. random) that isolate what B9.3 is actually testing are
consistent and well-supported. Before treating this as validated for
an S5 run: (1) expand beyond 8 seeds (or run a B5.1-style action-utility
audit) to resolve whether the online-baseline margin and the
refresh-interval ordering are real or single-diagnostic-batch noise;
(2) do not yet commit to `refresh=100` specifically as an optimal
schedule. **No S5 run performed or recommended yet**, per the task's
scope.
