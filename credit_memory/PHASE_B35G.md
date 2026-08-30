# Phase B35g — optimizer/projection interaction diagnostic

Branch `S5-CCM-scale-validation`. Architecture, credit recurrence
(`s_{t+1}=alg_mult(theta,s_t)+h_t`), B35d's result, and B35f's
interpretation are all unchanged. Code: `credit_memory/
b35g_optimizer_projection_diagnostic.py` (`/tmp/b35g_audit.log`).
C=128, EVAL_SEEDS=(11,12,13), same frozen data protocol as B35d/f.

## Three training-rule variants tested (theta's update rule only; C_out kept on Adam throughout)

- **A. Baseline**: Adam + existing hard per-factor projection (B35d
  as-is, frozen lr=0.01).
- **B. Adam + projection-aware moment reset**: whenever a factor's
  projection correction is nonzero, reset that factor's Adam m/v to
  zero immediately after, same lr=0.01.
- **C. Momentum-free normalized SGD** (`theta -= lr*g/||g||`) + the
  same projection, lr selected on TUNING_SEEDS=(100,101) only
  (grid {0.01,0.02,0.05} -> selected 0.01).

## Results

| | A: Adam baseline | B: Adam+moment-reset | C: normalized SGD |
|---|---|---|---|
| pre_nmse median | 0.265 | 0.238 | **0.839** |
| post_nmse median | 0.078 | 0.073 | **0.00062** |
| loss volatility (std, all seeds/steps) | 0.383 | 0.389 | **0.505** |
| projection-event frequency | 2706/9000 (30.1%) | **517/9000 (5.7%)** | 0/9000 |
| max consecutive-projection run | 134 | **19** | 0 |
| correction_ratio (median, when projected) | 0.102 | 0.210 | n/a |
| cos(delta_optimizer, c_t) (median / frac negative) | -0.099 / **93.3%** | -0.281 / 92.5% | n/a |
| corr(correction_ratio_t, loss_t) | 0.095 | 0.076 | n/a |
| corr(correction_ratio_t, loss_{t+1}) | 0.097 | 0.072 | n/a |

**A striking, unambiguous pattern in both A and B**: the projection
correction vector points in the OPPOSITE direction from the optimizer's
own step 92-93% of the time — Adam's update and the hard projection are
genuinely "fighting" each other at almost every projection event, exactly
the interaction pattern the hypothesis describes. Moment-reset (B)
successfully suppresses this: projection frequency drops 5.3x (30.1%
-> 5.7%) and the worst chattering run drops 7x (134 -> 19 consecutive
events).

**Yet loss volatility does not improve** — B's loss_std (0.389) is
statistically indistinguishable from A's (0.383), even slightly higher.
Pre/post NMSE for B are marginally better than A (within the seed-to-
seed spread already seen in B35d), not a substantial reduction.

**C (normalized SGD, no persistent momentum at all, zero projection
events)** shows the most dramatic change but not a clean win: post-change
NMSE improves ~126x (0.078 -> 0.00062), but pre-change NMSE gets
**3.2x worse** (0.265 -> 0.839) and overall loss volatility is the
**highest of the three** (0.505). Removing Adam's persistent state
entirely does not reduce oscillation — it trades a fast-but-noisy early
phase for a slow-but-eventually-precise later one, changing the shape
of the problem rather than curing the instability.

Correlation of the correction_ratio with same-step or next-step loss is
weak in both A and B (0.07-0.10) — a projection event happening does not
meaningfully predict a loss spike, consistent with B35f's own finding
that projection-related quantities were only moderately (not strongly)
tied to loss.

## Testing the falsifiable prediction

"If Adam/projection state mismatch drives B35d's instability, then
removing persistent adaptive momentum (SGD) or resetting the affected
Adam state after projection should substantially reduce RegularBlock's
oscillation... If neither intervention helps, reject this mechanism."

**Neither intervention reduces overall loss oscillation.** Moment-reset
(B) achieves its intended, verified effect on the mechanism it targets
(projection frequency and chattering both drop by 5-7x) without moving
loss volatility. Normalized SGD (C) eliminates projection events
entirely (0/9000) yet has the WORST loss volatility of the three. This
is the clean negative result the falsifiable design was built to
produce: **the intervention that should fix the hypothesized mechanism,
if it were the cause, worked exactly as intended on that mechanism and
still did not fix the symptom.**

## Diagnosis

**Reject Adam/projection state mismatch as the primary driver of
RegularBlock's B35d instability.** The interaction is real and
measurable (correction vectors oppose the optimizer's own step 92-93%
of the time — a genuine, verified "fighting" pattern) and moment-reset
correctly suppresses its frequency, but removing or suppressing it does
not substantially reduce loss volatility or meaningfully improve the
pre/post-change NMSE picture. Combined with B35f's earlier rejection of
typical transient eligibility/gradient amplification, this narrows the
live hypothesis space further: B35f's own tentative alternative
("projection/optimizer interactions dominate") is now also not
well-supported by this more direct causal test — the moderate
correlations B35f found (r=0.32-0.49 for proj_correction) appear to
reflect a downstream *symptom* that co-occurs with rough patches, not a
*cause* whose removal fixes them. The primary mechanism behind B35d's
RegularBlock loss volatility remains unidentified after B35e-g; per
instruction, no further mechanism (Hessian/staleness correction, new
parameterization) is implemented in this experiment.

## Commit hash

See the commit introducing this file.
