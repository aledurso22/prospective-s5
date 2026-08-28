# Phase B16.2 — full task-complexity x width x credit-complexity law, and depth/selectivity survival of the exact tied-credit closure

Branch `S5-CCM-scale-validation`. Ordinary BPTT for Parts A-E; Parts F-H
are standalone exact-algebra constructions (no tcg dependence, no new
persistent online-credit training rule). No S5. Code: `credit_memory/
b16_2_phase_diagram.py` (new). Artifact: `results/credit_memory/b16_2/
b16_2_summary.json`. Widths `N in {16,32,64,128}` full grid (256 as a
reduced-steps spot-check only, flagged below), `G` log-spaced
`{1,2,4,8,16,N}`, 2 seeds (3 for the N=128/delay-r8 boundary point),
200 BPTT steps main grid, 2000 for the Part E long-training control.

**Headline: the practically-relevant "solve the task to a useful
quality" criterion shows `G_required` collapsing to `G=1` at N=128 even
for the hardest task tested — a clean, sublinear-in-width result. The
strict "match the full untied optimum" criterion still needs `G~N`
almost everywhere, but Part E's long-training control shows at least
part of that gap is a training-duration artifact, not a true
expressivity wall, for at least one task family. Deep tied exact
credit closure (Part F) is confirmed at machine precision through
depth 8, with true state growth `O(2L)` (Part G), not the feared
exponential. Selectivity (Part H) gives an inconclusive, caveated
result. Verdict: C, EXPRESSIVITY-CREDIT TRADEOFF (see Part 9).**

## 1. Full N x G x task-complexity phase diagram

Median late loss (2 seeds unless noted), delay tasks (`r` independent
delayed channels, delays `5,10,...,5r`) and frequency tasks (`r`
disjoint sinusoid channels):

**delay_r1**

| N | G=1 | G=2 | G=4 | G=8 | G=16 | G=N |
|---|---|---|---|---|---|---|
| 16 | 0.0229 | 0.0187 | 0.0154 | 0.0109 | **0.0029** | — |
| 32 | 0.00149 | 0.00138 | 0.00089 | 0.00069 | 0.00052 | **0.00043** |
| 64 | 1.16e-4 | 1.51e-4 | 1.60e-4 | 8.52e-5 | 8.67e-5 | **2.02e-5** |
| 128 | 2.14e-5 | 2.88e-5 | 8.09e-6 | 6.11e-6 | 1.72e-6 | **1.69e-6** |

**delay_r8** (hardest task in the main grid)

| N | G=1 | G=2 | G=4 | G=8 | G=16 | G=N |
|---|---|---|---|---|---|---|
| 16 | 1.654 | 1.586 | 1.435 | 1.196 | **0.963** | — |
| 32 | 1.151 | 1.084 | 0.926 | 0.786 | 0.509 | **0.351** |
| 64 | 0.579 | 0.582 | 0.458 | 0.335 | 0.194 | **0.0406** |
| 128 | 0.223 | 0.221 | 0.161 | 0.118 | 0.0514 | **0.00173** |

**freq_r1 / freq_r8** (full curves in the artifact) show the same
qualitative pattern as B16.1: consistently lower absolute loss than
matched-count delay tasks at every `(N,G)`, and less G-sensitive.

**N=256 spot-check** (delay_r1, delay_r8; 1 seed, **100 steps, half the
main grid's 200** — an explicit, reduced-confidence data point, not
directly comparable in absolute scale to the N<=128 rows):

| task | G=1 | G=256 |
|---|---|---|
| delay_r1 | 0.00341 | 0.00140 |
| delay_r8 | 0.263 | 0.00749 |

Full curves for every `(N, task, G)` combination are in the JSON
artifact (`part_a`).

## 2. G_required: three framings, not one

`G_rel(x%)` (match within `x%` of the full untied loss) is the
framing used in B16.1 and in this phase's own preliminary checks; it
turned out to be **systematically misleading** once the full-model
loss itself gets very small, because matching a tiny number closely is
a much harder bar than solving the task well. Per the mid-phase
guidance, three complementary quantities are reported:

- **`G_rel20`**: min G with `L_G <= 1.2 L_full` (near-full fidelity).
- **`G_abs(eps)`**: min G with `L_G <= eps`, fixed eps per task family
  (delay_r1: 0.01, delay_r2: 0.05, delay_r4: 0.3, delay_r8: 1.0,
  freq_r1: 0.001, freq_r8: 0.05 — chosen from that family's own loss
  range, held fixed across N).
- **`G_S(x%)`**: min G with `(L_base - L_G)/(L_base - L_full) >= x`,
  where `L_base` is a trivial zero-predictor baseline (`0.5*mean(y^2)`,
  computed per task family: delay_r1=0.457, delay_r2=0.870,
  delay_r4=1.583, delay_r8=2.509, freq_r1=0.242, freq_r8=1.973) — the
  fraction of the achievable improvement over "predict nothing"
  recovered by a given `G`. This is the framing that turns out to be
  the cleanest width-scaling diagnostic.

| task | N | G_rel20 | G_rel20/N | G_S80 | **G_S80/N** | G_S95 |
|---|---|---|---|---|---|---|
| delay_r1 | 16 | 16 | 1.00 | 1 | 0.062 | 1 |
| delay_r1 | 32 | 32 | 1.00 | 1 | 0.031 | 1 |
| delay_r1 | 64 | 64 | 1.00 | 1 | 0.016 | 1 |
| delay_r1 | 128 | 16 | 0.125 | 1 | 0.008 | 1 |
| delay_r4 | 16 | 16 | 1.00 | 8 | 0.500 | 16 |
| delay_r4 | 32 | 32 | 1.00 | 1 | 0.031 | 16 |
| delay_r4 | 64 | 64 | 1.00 | 1 | 0.016 | 4 |
| delay_r4 | 128 | 128 | 1.00 | 1 | 0.008 | 1 |
| **delay_r8** | 16 | 16 | 1.00 | 8 | **0.500** | 16 |
| **delay_r8** | 32 | 32 | 1.00 | 16 | **0.500** | 32 |
| **delay_r8** | 64 | 64 | 1.00 | 4 | **0.062** | 64 |
| **delay_r8** | 128 | 128 | 1.00 | **1** | **0.008** | 8 |
| freq_r1 | 16-128 | mostly N | ~1.0 (32: 0.25) | 1 | 0.008-0.062 | 1 |
| freq_r8 | 16-128 | N,16,16,N | 0.25-1.0 | 1 | 0.008-0.062 | 1 |

**Reading this correctly, per the explicit caution against overclaiming
at N<=32**: at N=16/32, `delay_r8`'s `G_S80/N` sits at 0.5 — genuinely
proportional to N, not "tiny G." The decisive test is what happens at
larger N, and it happens cleanly: **`delay_r8`'s `G_S80/N` drops
0.5 -> 0.5 -> 0.062 -> 0.008 from N=16 to N=128 -- collapsing to
`G=1` at the largest width tested, for the single hardest task in the
grid.** Every other task family shows the same pattern, most already
flat at `G_S80=1` from N=32 onward. **`G_rel20`, in contrast, stays
pinned near `G=N` for delay tasks at every width tested** — a real,
separate phenomenon (matching an already-tiny residual loss to within
20% is a much stricter bar than solving the task usefully), and, per
Part 5 below, partly a training-duration artifact rather than a pure
expressivity limit.

## 3. Rational/McMillan-complexity task study — and a major correction from Part 5

Two new controlled families at N=64, single scalar input channel (so
`G_required` isn't confounded by input channel count the way delay/freq
`r` is): **K-mode damped-exponential sums** (`K` real poles, target =
mean of `K` parallel first-order filters of the same input) and **pure
delays** of increasing horizon `D`.

| task | L(G=1) | L(full,G=64) | G_rel20 | G1-recover-80% |
|---|---|---|---|---|
| kexp K=1 | 1.72e-4 | 8.50e-5 | 64 | 64 |
| kexp K=2 | 6.12e-4 | 1.29e-4 | 64 | 64 |
| kexp K=4 | 5.98e-4 | 8.74e-5 | 64 | 64 |
| kexp K=8 | 5.00e-4 | 9.00e-5 | 64 | 64 |
| puredelay D=5 | 1.16e-4 | 2.02e-5 | 64 | 64 |
| puredelay D=10 | 1.81e-4 | 2.65e-5 | 64 | 8 |
| puredelay D=20 | 2.62e-4 | 1.32e-5 | 64 | 16 (rel20: 64) |
| puredelay D=40 | 1.66e-2 | 6.32e-3 | 8 | 8 |

At face value (200-step training) this looks like a clean negative
result for the "`G_required ~ K`" hypothesis: **all four `K` values give
essentially the same `G=1` loss (1.7e-4 to 6.1e-4) and all need the
full `G=64` under `G_rel20`** — no visible `K`-dependence at all, and
no support for "`G_required` tracks McMillan degree."

**But Part 5's long-training control overturns this reading for
`kexp_K8` specifically** (the hardest `K` case): at 2000 steps, `G=1`
reaches `2.26e-7` — *better* than the full `G=64` model's `9.24e-5` at
the same step count (Part 5 table below). **The apparent "`G=1` can't
solve `K`-mode tasks" finding from the 200-step main grid is, at least
for this task family, a training-duration/optimization artifact, not a
true expressivity ceiling** — extreme tying may even be a *helpful*
inductive bias here (fewer free pole parameters, easier to identify),
not a constraint. **This means the Part C headline computed from the
main 200-step grid should not be trusted as a statement about
expressivity for the exponential-mode family; the `K`-vs-`G_required`
question is genuinely open and would require rerunning the full `K`
sweep at long training, which was not done in this pass (explicit
scope limitation).**

The pure-delay family is more consistent across step counts: `D=40`
(the longest delay, close to `T=60`) needs less `G` under `G_rel20`
(`G=8`, not full) than the shorter delays — but this is because `D=40`'s
own optimization is unstable at this width/step count (curve is
non-monotonic: `G=64`'s loss, `0.0063`, is *worse* than `G=16`'s,
`0.0016`), not because the task is intrinsically easier. **No clean
support either way for "`G_required` grows with delay horizon" from
this data** — the D40 anomaly needs a longer/more-seeded rerun before
drawing a conclusion, not attempted here.

## 4. Performance-vs-exact-credit-budget Pareto

`S_credit = 2GN`. Using `delay_r8` (the hardest task, richest budget
range) as the clearest case, best loss achievable at each matched
credit budget across all `(N,G)` combinations tested:

| S_credit | best (N,G) -> loss | next best |
|---|---|---|
| 128 | (64,1) -> 0.579 | (32,2) -> 1.084, (16,4) -> 1.435 |
| 256 | **(128,1) -> 0.223** | (64,2) -> 0.582, (32,4) -> 0.926, (16,8) -> 1.196 |
| 512 | (128,2) -> 0.222 | (256,1)* -> 0.263, (64,4) -> 0.458, (32,8) -> 0.786 |
| 1024 | (128,4) -> 0.161 | (64,8) -> 0.335, (32,16) -> 0.509 |
| 2048 | (128,8) -> 0.118 | (64,16) -> 0.194, (32,32) -> 0.351 |

*(256,1) uses the reduced-step N=256 spot-check, not directly
comparable in absolute scale — included for reference only.*

**At every matched budget, the widest available model with the
smallest `G` dominates every narrower, less-tied alternative** — e.g.
at `S_credit=256`, `(N=128,G=1)` beats `(N=16,G=8)` by ~5.4x despite
identical credit-state cost. This Pareto ordering is monotonic and
clean across the entire tested range: **spend a fixed credit budget on
width first, tying second.** The one exception worth flagging: going
from `(128,G=1)` to `(128,G=2)` (doubling the budget) buys almost
nothing (`0.223 -> 0.222`), while the N=256 spot-check at a *matching*
budget doesn't clearly beat it either — consistent with `delay_r8`
already being close to saturated at `N=128` for this training budget,
not a breakdown of the Pareto ordering itself.

## 5. Longer-training control (2000 steps, 10x the main grid)

| task | N | G | steps | median late loss | full/G=1 ratio |
|---|---|---|---|---|---|
| delay_r4 | 64 | 1 | 2000 | 8.16e-4 | |
| delay_r4 | 64 | 64 (full) | 2000 | 7.76e-6 | **~105x gap persists** |
| delay_r8 | 64 | 1 | 2000 | 4.67e-3 | |
| delay_r8 | 64 | 64 (full) | 2000 | 1.23e-3 | gap **shrinks** 14x->3.8x vs. 200 steps |
| kexp_K8 | 64 | 1 | 2000 | 2.26e-7 | |
| kexp_K8 | 64 | 64 (full) | 2000 | 9.24e-5 | **G=1 beats full G at matched steps** |

Three distinct outcomes, reported honestly rather than averaged into
one story:

- **`delay_r4`**: the `G=1` vs full-`G` gap survives 10x more training
  essentially unchanged in order of magnitude — this is a genuine
  expressivity limitation, not merely slower optimization.
- **`delay_r8`**: both conditions improve enormously with more
  training (`G=1`: 0.58 at 200 steps -> 0.0047 at 2000; full-`G`: 0.041
  -> 0.0012), and the *ratio* between them shrinks substantially
  (14x -> 3.8x) — most, but not all, of the apparent 200-step gap was
  an optimization-speed artifact, with a smaller genuine residual gap.
- **`kexp_K8`**: `G=1` fully overtakes full `G` with enough training —
  the 200-step-grid's "G=1 can't solve this" conclusion for this task
  family is **not an expressivity result at all**, per Part 3.

**This is the single most important methodological finding of this
phase**: the main grid's 200-step training budget systematically
*overstates* `G_required` for at least some task families, and the
degree of overstatement is task-specific and not predictable in
advance from the 200-step curves alone. Parts 1-4's numbers should be
read as an upper bound on `G_required`, not a precise measurement, for
any task not also covered by this long-training check.

## 6. Deep exact tied scalar recurrence (Part F)

For `h_t^l = a_l h_{t-1}^l + B_l h_t^{l-1}` (scalar tied `a_l` per
layer, real routing `B_l`), the claimed closed form `d h_t^l/d
theta_m = (B_l...B_1 v_m) z_{l,t}^m` with `z_{1,t}=a_1 z_{1,t-1}+u_t^m`,
`z_{l,t}=a_l z_{l,t-1}+z_{l-1,t}^m`, verified against (i) literal
full-vector RTRL forward-sensitivity accumulation, (ii) an
exact-by-linearity direct simulation, and (iii) an independently
derived reverse-mode BPTT adjoint recursion for a scalar readout loss,
depths `L in {2,3,4,6,8}`, widths `N in {6,10}`:

| L | N | err vs RTRL | err vs linear-sim | err vs BPTT adjoint |
|---|---|---|---|---|
| 2 | 6,10 | 5.3e-15, 8.9e-16 | 5.3e-15, 8.9e-16 | 4.4e-16, 1.4e-16 |
| 4 | 6,10 | 8.9e-15, 1.4e-14 | 8.9e-15, 1.4e-14 | 2.7e-15, 3.6e-15 |
| 6 | 6,10 | 3.1e-13, 1.4e-12 | 3.1e-13, 1.4e-12 | 0, 9.1e-13 |
| 8 | 6,10 | 6.8e-13, 9.1e-13 | 6.8e-13, 9.1e-13 | 2.3e-13, 2.3e-13 |

**Machine precision at every depth tested, including the independent
reverse-mode check.** Credit state per source mode is exactly `2L`
reals (the `z` chain), vs. `2N^2 L` for a naive per-unit BPTT
sensitivity — **the exact tied-credit closure survives depth exactly as
predicted, with no approximation and no width dependence.**

## 7. Complex depth minimal realization (Part G)

Revisiting the earlier symbolic conjugate-path count for complex tied
poles across depth: naive real/imaginary expansion of the product of
`L` complex tied scalars gives `2^(L-1)` distinct symbolic terms. The
actual minimal dynamical state was computed directly as the
controllability-matrix rank of the real widely-linear augmented
`z`-chain (`2` reals per layer, bidiagonal Jordan-chain structure):

| L | symbolic path count | minimal real dim | measured reachable rank |
|---|---|---|---|
| 2 | 2 | 4 | 4 |
| 3 | 4 | 6 | 6 |
| 4 | 8 | 8 | 8 |
| 6 | 32 | 12 | 12 |
| 8 | 128 | 16 | 16 |

**Measured rank exactly equals `2L` at every depth tested, while the
symbolic term count grows as `2^(L-1)`.** This directly confirms the
task's own suspicion: **the old exponential-depth claim is a
nonminimal-representation artifact** of naive real/imaginary expansion,
not a property of the actual dynamics — the true minimal state is
`O(L)` complex / `O(2L)` real, matching Part 6's `z`-chain exactly.

## 8. Combined tied + selective test (Part H)

`h_t = a_t h_{t-1} + x_t` (scalar `a_t` tied across all `N` units,
`x_t` genuinely `N`-dimensional iid drive), comparing three gates:
**const** (`a_t=a(theta_0)`, the true non-selective baseline this
project's exact-closure results actually apply to), **exogenous**
(`a_t=a(theta_0, fixed external signal_t)`, time-varying but
input-independent), and **selective** (`a_t=a(theta_0, x_t)`,
Mamba-style input-dependent gating, matching the literal H2 spec).
Effective (participation-ratio) rank of the sensitivity trajectory
`d h_t/d theta_0` over `t`:

| N | const | exogenous | selective |
|---|---|---|---|
| 8 | 4.65 | 3.54 | 2.79 |
| 32 | 7.61 | 6.42 | 6.67 |
| 128 | 8.66 | 7.41 | **11.91** |
| 512 | 9.52 | 8.52 | 9.82 |

**Result is inconclusive, reported honestly rather than forced into a
clean story**: all three conditions — including the fully
non-selective `const` baseline — show sensitivity rank *growing with
N*, confirming this specific toy (an unstructured, iid-noise-driven
`N`-dim state) does not preserve the `O(1)` closure once the pole
parameter's own sensitivity multiplies the full evolving state
`h_{t-1}` at every step, **regardless of whether the gate is
selective**. The marginal effect of true input-dependence beyond a
matched time-varying-but-exogenous baseline is small and
non-monotonic (`+4.5` at N=128, `+1.3` at N=512) — consistent with,
but not a stronger version of, B16 Part G's own finding.
**Methodological caveat, explicitly carried over from B16 Part G**:
this toy does not combine selectivity with Part A's actual structured,
routed (`B`-matrix) tied architecture — a fully faithful combined test
integrating the real routing structure was not built in this pass.
This question remains open, not resolved negatively or positively.

## 9. Verdict: **C, EXPRESSIVITY-CREDIT TRADEOFF**

Checking against the four offered options:

- **Task/width separation** (`G_required` task-dependent, weakly
  width-dependent): **strongly supported** under the baseline-relative
  `G_S80` framing — every task tested, including the hardest
  (`delay_r8`), collapses to `G_S80/N` well under 0.01 by `N=128`.
- **`G_required << N` surviving harder tasks**: **supported**, with the
  explicit caveat that `G_rel20` (near-full fidelity) does NOT show
  this at any width tested for delay tasks, and Part 5 shows part of
  that gap is a training-duration artifact rather than a hard ceiling.
- **Deep tied exact credit closure remains width-independent**:
  **confirmed at machine precision** (Part 6), with true state growth
  proven `O(2L)` not exponential (Part 7) — this pillar of "A" is fully
  satisfied.
- **Tied+selective structure retaining a small module**: **not
  established** (Part 8) — the toy test is inconclusive and has an
  acknowledged fidelity gap to the real architecture.

Three of "A"'s four criteria hold cleanly; the fourth is genuinely
open, not failing. That keeps this short of the full "A. STRONG
CO-DESIGN LAW" claim, and clearly rules out "D. WIDTH-COUPLED FAILURE"
(the width-scaling evidence is uniformly favorable) and "B.
NONSELECTIVE EXACT REGIME" (which requires selectivity to actively
*destroy* closure — Part 8 doesn't show that, just an unresolved
test). **"C. EXPRESSIVITY-CREDIT TRADEOFF" is the most defensible
verdict**: `G_required` (under a useful-quality criterion) clearly
correlates with task temporal complexity at fixed width, and is
substantially sublinear in width for every task tested so far — a
genuine, if not yet fully general, tradeoff law. The central open
items for a future phase: (a) rerun Part 3's `K`-mode family with
long training to get an unconfounded `K`-vs-`G_required` curve, (b) a
properly structured tied+selective combined test using the real
`B`-routed architecture, (c) resolve whether `G_rel20`'s persistence
at `G~N` is a true asymptotic ceiling or an artifact of the fixed
200-step budget used across the whole main grid.

No new persistent online-credit training rule implemented. No S5 run.

## 10. Commit hash

See the commit introducing this file.
