# Phase B9.5 — staleness DETECTION (event-triggered recalibration), not prediction

Branch `S5-CCM-scale-validation` (current checkout). Diagnostic only,
explicitly framed as an **event-triggered controller test, not
prospective coding or prediction-correction.** No new training
*algorithm* (gradient rule / pool-construction method) — this tests
different *recalibration-timing* policies plugged into B9.3's
already-accepted `pool_periodic` mechanism. Code: `credit_memory/
b9_5_staleness_detection.py` (new). Artifact: `results/credit_memory/
b9_5/b9_5_staleness_detection_summary.json`. 8 seeds, same task/config
as B9.3/B9.4, `K=4`.

**Verdict: C — FAIL.** No tested cheap, active-pool-only trigger
matches or beats B0's best periodic schedule (`refresh=100`) on the
quality-vs-calibration Pareto frontier. Individual signals show
non-trivial AUROC in isolation, but that did **not** translate into a
useful deployed policy — the actual closed-loop training runs (the
decisive test) are unambiguous. **Stop staleness-detection work here.
Keep periodic K-pool CCM (B9.3) as the supported algorithm.**

## Method

**Labeled aging-pool trajectory** (8 seeds): reuses B9.3/B9.4's
`pool_frozen K=4` mechanism (pool built once, never refreshed) so the
pool ages monotonically across a full 600-step real-task training run,
giving a clean staleness-onset trajectory. At every step this logs
**both** (a) full-bank/BPTT oracle labels — `D_n=cos_a0` (deployed vs.
BPTT), `R_n` = regret of the current pool vs. the true best `K`-pool at
that instant (via `best_pool_exact` on the instantaneous oracle
utility, same `S`=empty convention as B9.1/B9.4), coverage — used
**only** to define staleness events, never fed to any trigger — and
(b) the cheap, deployable, **active-pool-only** feature set: winner
margin (top-1 minus top-2 `|rho|` within the pool), in-pool relevance
drift `||rho_pool,n - rho_pool,n-1||`, entropy/concentration of
relevance over the `K` pool candidates, a within-pool switch flag,
mean `|q_j|` and `||B_{j,:}||` for active (in-pool) channels only,
eligibility energy/lag-1 statistics, optimizer-update norm, and pool
age. No full-bank, dormant-candidate, or oracle quantity ever enters a
feature or a trigger.

**Staleness events**: `stale_n = 1{D_n < tau_cos}` for `tau_cos in
{0.85, 0.75, 0.65}`; "stale within the next `h` steps" for `h in {1,
5, 10, 25}`, a genuinely forward-looking *evaluation* label only ever
used to score a trigger after the fact, never seen online.

**Generalized event-triggered trainer**: identical to B9.3's
`pool_periodic`, except the fixed-interval check is replaced by a
`trigger_fn(cheap_state) -> bool` callback. **Verified against B9.3
directly**: a fixed `age>=100` trigger reproduces B9.3's own
`refresh=100, seed=0` run to within run-to-run RNG-path equivalence
(`cos_a0` late-avg 0.859 here vs. 0.861 from B9.3's own `train()` on
the identical seed/config — confirms the reimplementation is faithful,
not a new mechanism).

## Part 1-2: individual cheap-signal AUROC (secondary diagnostic, per the task's own instruction not to over-weight this)

Median AUROC over 8 seeds, horizon `h=10`:

| feature | tau=0.85 | tau=0.75 | tau=0.65 |
|---|---|---|---|
| margin | 0.957 | 0.690 | 0.625 |
| drift | 0.484 | 0.623 | 0.588 |
| entropy | 0.661 | 0.656 | 0.587 |
| switch_flag | 0.501 | 0.499 | 0.511 |
| q_mag_pool | 0.468 | 0.401 | 0.530 |
| **Brow_pool** | **0.977** | **0.894** | 0.671 |
| elig_energy | 0.756 | 0.693 | 0.548 |
| elig_ac1 | 0.394 | 0.546 | 0.529 |
| upd_norm | 0.130 | 0.209 | 0.436 |
| **age** | **0.976** | **0.903** | 0.669 |

Several individual signals — especially `age` and `Brow_pool` — show
strong AUROC, particularly at the least-strict threshold (`tau=0.85`,
where ~53% of steps are "stale," making this close to detecting a
one-time, roughly-monotonic regime shift rather than a genuinely
recurring event). **This is a trap the task explicitly warned about**:
`age` is exactly what a fixed-period schedule already uses, so its
high AUROC does not represent new information beyond periodic
scheduling; `switch_flag` and `upd_norm` are near or below chance and
carry no signal. **AUROC here is a poor proxy for actual policy
quality** — see the decisive comparison below.

## B3: fitted logistic trigger

A pooled logistic model (all 9 cheap features + age, standardized,
`tau_cos=0.75, h=10`, in-time train/test split) was fit on one seed's
aging trajectory and evaluated held-out (same seed) and via a
median-over-8-seeds refit: **median held-out AUROC = 0.404 — at or
below chance.** Most per-seed test windows returned an undefined AUROC
(the held-out portion had only one label class present) or a
below-chance value (0.08, 0.48, 0.40). **The pooled linear combination
does not generalize even within a single seed's own held-out window,**
let alone across seeds — a direct illustration of why classifier-style
accuracy/AUROC was deprioritized as a primary metric.

## Part 5 — the decisive comparison: actual event-triggered training

Real 8-seed, 600-step training runs (the generalized trainer above),
`K=4`, `min_age=20` guard on every trigger, thresholds set from the
labeled trajectories' own pooled 25th/40th (margin) and 75th/90th
(drift) percentiles:

| policy | median cos_a0 (late-avg) | median # recalibrations |
|---|---|---|
| B0 refresh=50 | 0.620 | 11 |
| **B0 refresh=100 (best periodic, from B9.3)** | **0.731** | **5** |
| B0 refresh=200 | 0.625 | 2 |
| B0 refresh=600 (frozen) | 0.484 | 0 |
| B2a margin<p25 | 0.579 | 1.0 |
| B2a margin<p40 | 0.485 | 3.0 |
| B2b drift>p75 | 0.767 | 12.5 |
| B2b drift>p90 | 0.656 | 4.5 |
| **B3 logistic** | **0.484** | **0.0** (per-seed: `[28,0,0,0,0,0,0,0]`) |

**No policy beats `refresh=100` on the Pareto frontier.** B2a
(margin-triggered) underperforms even the comparably-cheap
`refresh=200` baseline at similar or higher calibration counts. B2b
(drift-triggered) at `p75` reaches a higher `cos_a0` (0.767) than any
periodic schedule, but only by paying for **12.5 median
recalibrations — 2.5x `refresh=100`'s cost** — not a genuine
efficiency gain, just a more expensive point on a worse curve (B2b at
`p90`, matched to `refresh=100`'s ~5-recalibration budget, scores
0.656, clearly below `refresh=100`'s 0.731). **B3 fails outright**: the
offline-fit logistic trigger stays silent (0 recalibrations) for 7 of
8 seeds — reproducing the frozen baseline's poor 0.484 — and fires
pathologically 28 times for the one seed (seed 0) it was fit on,
consistent with the held-out AUROC's own verdict that this trigger
does not generalize.

**Answer to the decisive question: no.** An event-triggered rule using
only active-pool quantities does not achieve `refresh=100`'s gradient
quality at fewer recalibrations, nor does it improve quality at a
matched calibration budget. The one candidate that improves quality
(B2b at `p75`) does so only by spending substantially more, not by
being smarter about *when* to spend.

## Control-theoretic framing (explicitly not prediction-correction)

This was tested as a pure event-triggered/hazard-style **detection**
problem — "is the current representation still acceptable" — not as
forecasting where the optimum moves (B9.4 already closed that
question). The result is a genuine, informative negative: **the weaker
hypothesis (detecting staleness is easier than predicting it) is not
supported by these active-pool-only signals either**, at least not
enough to beat a simple fixed schedule. The individual-signal AUROC
numbers show *some* structure exists (this is not "cheap signals carry
zero information") — but not enough, combined either simply (B2a/B2b
single-threshold) or via a fitted linear combination (B3), to produce
a controller that outperforms `refresh=100`.

## Answer / stop rule

**C. FAIL.** Cheap active-pool quantities do not reliably identify
staleness in a way that translates into a triggering policy beating
fixed periodic scheduling — individually promising AUROC numbers
(`age`, `Brow_pool`, `margin` at lenient thresholds) did not survive
contact with the actual closed-loop training comparison, and the one
learned combination (B3) failed to generalize even within-seed.
**Stop predictability/detection work on this line. Keep periodic
K-pool CCM (`refresh~=100`, per B9.3) as the supported algorithm.**
No prediction-correction, no feedback-alignment/PAL-style machinery,
and no event-triggered controller are recommended for implementation.
No new training arm was added as a persistent part of the codebase;
`train_event_triggered` exists only as this diagnostic's test harness.
No S5 run performed.
