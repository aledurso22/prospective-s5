# Phase B35f — RegularBlock continual-optimization mechanism audit

Branch `S5-CCM-scale-validation`. Diagnostic only: B35d/B35e, the
architectures, optimizer, LR, clipping, and projections are all
unchanged. Code: `credit_memory/b35f_optimization_mechanism_audit.py`
(`/tmp/b35f_audit.log`). Reuses the frozen B35d LRs (regular=0.01/0.02,
generic=0.05/0.01 at C=128/64) and eval seeds (11,12,13), 9000
step-records per architecture per budget (3 seeds x 3000 steps).

## Hypothesis tested

RegularBlock's generalized/Jordan dynamics produce larger transient
eligibility/gradient amplification under single-sample learning than
GenericBlock.

## Key results, C=128 (where B35d's gap was largest)

| quantity | RegularBlock | GenericBlock |
|---|---|---|
| loss: median / p99 / max | 4.04e-3 / 1.03 / 10.7 | 1.13e-3 / 6.50 / 82.0 |
| unclipped grad norm: median / p99 / max | 0.038 / 1.66 / 13.5 | 0.037 / 37.9 / 500 |
| clipping activated | 17/9000 (0.19%) | 588/9000 (6.53%) |
| proj_correction: p90 / p99 / max | 1.2e-3 / 1.3e-2 / 5.3e-2 | 9.3e-3 / 1.3e-1 / 2.95e-1 |
| Gamma_H=50: median / p90 / p99 / max | 0.76 / 1.04 / 1.36 / **12.9** | **0.93** / **1.40** / **1.75** / 3.11 |
| rho(J): median / max | 0.51 / 0.92 | 0.57 / 0.95 |
| fraction of (checkpoint,factor) with Gamma_H>1 | 12.9% | **43.5%** |

**GenericBlock, not RegularBlock, has the higher TYPICAL transient gain**
at C=128 — higher median/p90/p99 Gamma_H, and 3.4x the fraction of
factors exceeding Gamma_H>1. GenericBlock's raw gradient tail is also
far more extreme (p99=37.9, max=500 vs RegularBlock's p99=1.66,
max=13.5) and it clips ~34x more often — yet still achieves the better
median loss. The one respect in which RegularBlock IS more extreme:
its worst single (checkpoint, factor) Gamma_H (12.9) exceeds
GenericBlock's worst (3.11) — a rare, severe outlier in a small number
of factors, not a systematic pattern (median/p90/p99 all favor Regular
being LOWER, not higher, transient gain).

C=64 shows a more balanced picture (Regular median Gamma_H=0.83 vs
Generic 0.80, fraction>1: 19.0% vs 18.3% — comparable), consistent
with C=64 being the budget where the B35d gap was smaller.

## Per-factor RegularBlock breakdown (C=128, n=9000 x Q=32 factors)

Base `|lambda_q|` median=0.517 (p99=0.777, capped at 0.95 by the base
clip); nilpotent-tail L1 norm median=0.571 (p99 saturates at 1.0, the
RHO_NIL cap — many factors sit AT the cap); per-factor eligibility norm
is heavy-tailed (median=0.63, p99=2.96, **max=53.1** — an 18x jump from
p99 to max, i.e. a rare extreme outlier factor, not a bulk effect);
per-factor gradient norm similarly heavy-tailed (median=4.3e-3,
p99=0.22, **max=13.5**, a 61x jump). So SOME individual factors do
occasionally develop very large eligibility/gradient norms — but this
is rare-outlier behavior, consistent with the Gamma_H tail (not the
median), not a systematic elevation of the whole population.

## Correlations with RegularBlock's own per-step loss (both budgets)

| quantity | corr(loss, ·), C=128 | corr(loss, ·), C=64 |
|---|---|---|
| elig_norm | **0.087** | 0.209 |
| unclipped_grad_norm | **0.667** | 0.663 |
| proj_correction | 0.318 | 0.491 |
| adam_relative_step | 0.571 | 0.570 |
| clipped_flag | 0.425 | 0.463 |

Eligibility norm itself is the WEAKEST correlate of loss at both
budgets. The strongest correlates are the raw (unclipped) gradient
norm and the resulting Adam step size — a generic "large update ->
next-step loss disturbance" optimizer signature, not something
specific to the eligibility trace's own magnitude. Projection
correction correlates moderately, meaning loss spikes do co-occur with
the discrete rescale-when-exceeding projection firing, but this is
downstream of (not independent from) the large-gradient events already
captured by the stronger correlates above.

## Diagnosis

**The evidence does not support "transient eligibility/gradient
amplification" as the primary mechanism.** The one purpose-built
diagnostic for that hypothesis (Gamma_H) shows GenericBlock with
HIGHER typical transient gain at the budget where RegularBlock's
failure was largest, and eligibility norm is the weakest of all tested
correlates of RegularBlock's own loss. The evidence instead points
toward **projection/optimizer interactions**: RegularBlock's discrete,
rescale-only-if-exceeding projection combined with Adam's own momentum
state (which has no knowledge that a hard correction just fired)
appears to be where the disruption concentrates — loss correlates most
with raw-gradient/Adam-step-size events and moderately with projection
firing, not with the eligibility trace's magnitude per se. A residual,
narrower version of the transient-amplification hypothesis survives
only for a small number of individual outlier factors (max Gamma_H=12.9,
max per-factor eligibility/gradient norms 18-60x their own p99) — rare
events, not the typical mechanism, and not ruled out as occasionally
contributing, but not the dominant explanation for the bulk of
RegularBlock's B35d loss volatility.

## Commit hash

See the commit introducing this file.
