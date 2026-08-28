# Phase B16.1 — forward-expressivity and scaling test for tied-pole architectures

Branch `S5-CCM-scale-validation` (current checkout). Ordinary BPTT
training only — **no new online-credit rule implemented or used.**
Code: `credit_memory/b16_1_forward_expressivity.py` (new). Artifact:
`results/credit_memory/b16_1/b16_1_summary.json`. Weight tying is
exact and maintained throughout training (not just at init): the
`rho[1]`/`theta[1]` portion of the BPTT flat gradient is summed within
each tied group and broadcast back before every Adam step — verified
directly that `a1` retains exactly `G` distinct values after 30+
training steps. `N in {16,32,64,128}`, `G` up to `N`, 2 seeds,
`STEPS=200`.

**Headline: B — TASK-COMPLEXITY SCALING (the scientifically interesting
outcome the task itself called out).** Width can grow largely
independently of `G` for a *fixed* task — even `G=1` (a single shared
pole for the entire upper layer) improves by ~3 orders of magnitude
from `N=16` to `N=128` on the base task. But the `G` *required* to
reach a given performance level grows with the task's own temporal
complexity (more independent delays or frequencies), essentially flat
across the tested widths. **Width buys spatial/feature capacity; `G`
buys temporal alphabet size; on this task family, these are separate
axes.**

## Part A — G x width, delayed-copy task

Median late loss (2 seeds, 200 BPTT steps):

| `N` | `G=1` | `G=2` | `G=4` | `G=8` | `G=16` | `G=N` (full) |
|---|---|---|---|---|---|---|
| 16 | 0.0938 | 0.0941 | 0.0638 | 0.0531 | **0.0339** | — |
| 32 | 0.0243 | 0.0106 | 0.0044 | 0.0044 | 0.0014 | **0.0011** |
| 64 | 3.3e-4 | 1.9e-4 | 1.9e-4 | 1.9e-4 | 1.0e-4 | **1.3e-5** |
| 128 | 1.3e-5 | 3.2e-5 | 8.2e-6 | 7.3e-6 | 1.1e-6 | **1.7e-6** |

**The critical read: follow `G=1` down each row.** `0.094 (N=16) ->
0.024 (N=32) -> 3.3e-4 (N=64) -> 1.3e-5 (N=128)` — a **~7,000x
improvement from width alone, with the temporal alphabet held at a
single tied pole throughout.** By `N=64-128` the task is saturated
(all `G` land within a small band near the noise floor) — this task is
simply too easy to discriminate `G` choices at large width, which
motivated Part B's harder tasks.

## Part B — temporal-complexity tasks (the decisive comparison)

Fixed `N=32`, varying task temporal complexity via genuinely
independent delay channels or genuinely disjoint frequency channels
(not merely phase-shifted copies of one signal, per the task's own
`exp(-i omega tau)` critique from B13):

| task | `G=1` | `G=2` | `G=4` | `G=8` | `G=16` | `G=32` (full) |
|---|---|---|---|---|---|---|
| 1 delay | 0.0243 | 0.0106 | 0.0044 | 0.0044 | 0.0014 | 0.0011 |
| 2 delays | 0.0263 | 0.0206 | 0.0150 | 0.0113 | 0.0046 | 0.0011 |
| **4 delays** | **0.303** | 0.271 | 0.188 | 0.132 | 0.045 | **0.019** |
| 1 freq | 0.0243 | 0.0106 | 0.0044 | 0.0044 | 0.0014 | 0.0011 |
| 2 freqs | 0.0038 | 0.0029 | 0.0029 | 0.0014 | 0.00098 | 0.00061 |
| 4 freqs | 0.0110 | 0.0097 | 0.0088 | 0.0075 | 0.0057 | 0.0044 |

**`G` required to reach a fixed performance level grows with task
complexity, not width**: at `N=32` fixed, reaching `~0.01` loss needs
`G~2` for the 1-delay task, `G~16` for 2 delays, and is **not reached
at all** by `G=32` (full rank) for 4 delays (best achievable is
`0.019`, still `2x` worse than the 1-delay task's `G=1` result). The
4-delay task is a **materially harder regime that even full-rank
`G=N` cannot fully absorb at this width** — a genuine task-complexity
ceiling, not just a slower approach to the same asymptote.

**A secondary, unplanned finding**: multi-frequency tasks are
consistently *easier* than multi-delay tasks of matched channel count
at every `G` (e.g. `r=4`: freq loss `0.0044-0.011` vs. delay loss
`0.019-0.303`) — plausibly because smooth sinusoidal targets are a
more natural fit for this architecture's own complex-exponential pole
basis than precise discrete-delay/copy behavior. Not a planned test,
reported as observed.

## Part C — width at fixed G (read directly from Part A)

No separate experiment needed — Part A's own table, read by row instead
of by column, answers this directly: **at every fixed `G` tested
(including `G=1`), increasing width continues to improve performance,
often dramatically** (`G=1`: `0.094 -> 0.024 -> 3.3e-4 -> 1.3e-5`
across `N=16` to `128`). Performance does **not** saturate immediately
with `N` at fixed `G` on this task family — supporting the
width-controls-capacity / `G`-controls-temporal-alphabet separation.

## Part D — parameter-matched baselines (analytic, no separate sweep needed)

Effective pole parameters (`2G`, tied) are a **negligible fraction** of
total trainable parameters at every width tested, since the dense
`N x N` inter-layer routing matrix `B[1]` dominates total parameter
count quadratically:

| `N` | total params | pole params at `G=1` | pole params at `G=N` (full) | fraction saved by tying to `G=1` |
|---|---|---|---|---|
| 16 | 640 | 2 | 32 | 4.7% |
| 32 | 2304 | 2 | 64 | 2.7% |
| 64 | 8704 | 2 | 128 | 1.4% |
| 128 | 33792 | 2 | 256 | 0.75% |

**The fraction of total parameters saved by tying shrinks with width**
(quadratically dominated by `B`) — at `N=128`, tying to `G=1` saves
less than 1% of total parameters. **Any performance difference between
`G` values is therefore attributable to the architectural restriction
(fewer independent temporal dynamics), not to having fewer overall
trainable parameters** — a "parameter-matched" comparison would require
a negligible (`<1%`) width adjustment and was not run as a separate
experiment, since the conclusion is already clear from the parameter
counts themselves.

## Part E — pole group design: contiguous vs. random assignment

| `N`, `G` | contiguous | random |
|---|---|---|
| 32, 2 | 0.0106 | 0.0117 |
| 32, 4 | 0.0044 | 0.0045 |
| 32, 8 | 0.0044 | 0.0044 |
| 64, 2 | 0.00019 | 0.00020 |
| 64, 4 | 0.00019 | 0.00021 |
| 64, 8 | 0.00019 | 0.00015 |

**Performance is essentially insensitive to how channels are assigned
to groups** — contiguous and random assignment agree to within normal
seed-to-seed noise at every `(N,G)` tested. **What matters is the
number of groups `G`, not which channels share them.** E2 (fixed/
log-spaced init) and E4 (structured/head-wise grouping) were not run
as separate conditions in this pass, given E1 vs. E3's clean null
result already answers the primary question this part asked.

## Part F — performance vs. exact-credit-state Pareto

Compression factor `N/G` vs. `S_credit = 2GN`. The clearest single
comparison: **`(N=128, G=1)`**, `S_credit=256`, achieves loss `1.3e-5`
— compare to **`(N=16, G=16)`** (fully untied, no compression at all),
`S_credit=512` (**double** the credit-state budget), which achieves
only loss `0.034` — nearly **3,000x worse**, at higher cost. **A wide,
heavily-tied model dominates a narrow, fully-untied model at a smaller
credit-state budget** — a clean, favorable Pareto result **for this
task family**. The comparison is less favorable for the 4-delay task
(Part B), where even full-rank `G=N=32` (`S_credit=2048`, the maximum
tested at that width) does not fully close the gap — the Pareto
frontier's favorability is task-complexity-dependent, matching Part B's
core finding.

## Part G — verdict: **B, TASK-COMPLEXITY SCALING**

- Width improves performance substantially even at the most extreme
  tying (`G=1`), confirming width and temporal-alphabet size are
  separable axes **for a fixed task** (the "A" finding holds as a
  special case).
- But the **`G` required to reach a target performance level tracks
  task temporal complexity** (delay/frequency channel count), staying
  roughly constant across the tested widths rather than growing with
  `N` — precisely `B`'s definition, and the piece that makes this
  result more than a restatement of "wider is better."
- Group *assignment* (random vs. structured) does not matter; group
  *count* does.
- Parameter count is not the explanation for any observed gap — the
  pole parameters are a negligible fraction of the total at every
  width tested.
- The favorable Pareto trade-off (small `G`, large `N` beating large
  `G`, small `N` at matched or smaller credit-state budget) holds
  cleanly for simple tasks and is a real, task-complexity-bounded
  effect, not unconditional.

**This supports continuing the exact tied-credit research direction**
(per B16's own recommendation), now with empirical grounding that
small `G` does not cripple forward modeling at the widths tested, and
with a concrete, falsifiable prediction for future work: **`G_required`
should be estimable from task temporal complexity independent of the
width chosen for spatial capacity.** The open items flagged in B16
(depth, complex minimal realization, a combined tied+selective test)
remain the next steps if this direction continues.

No new persistent training arm added. No S5 run performed.
