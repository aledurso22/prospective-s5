# Phase B36a — deconfounded generalized-mode representation efficiency

Branch `S5-CCM-scale-validation`. Leaves the B35 continual-learning
branch: pure BPTT / fixed-shared-parameter training throughout, no
per-sample updates, no Hessian transport, no K-sweeps, no moving-weight
diagnostics. Code: `credit_memory/b36a_generalized_mode_efficiency.py`
(`/tmp/b36a_full.log`, `/tmp/b36a_results.json`).

## Architectures (matched exactly)

RealDiagonal, ComplexLocal, DualLocal, all d=2, n factors, r=2n real
recurrent state/parameters, with an IDENTICAL trainable input vector
(r,) and readout vector (r,) for all three -- no architecture gets
extra width. Exact persistent eligibility count under online RTRL is
2n for ALL THREE (RealDiagonal: 1 scalar trace per independent
coordinate; Complex/Dual: 2 real coordinates per factor, established
in B35a-j) -- matched by construction, no discrepancy to report.

## Validity (before training)

Full-real-coordinate RTRL vs BPTT, and reduced (2n-scalar) exact RTRL
vs BPTT, fixed theta, for all 3 architectures x n in {1,2,4,8} (24
checks): every relative error in the 1.5e-16 to 6.5e-16 range, all
`<< 1e-10`. Teacher A's generalized-mode signature verified: c1=0.148
(nonzero), fit residual 6.66e-16.

## Capacity sweep, median NMSE(H) (H=64, 10 eval seeds; NMSE(2H)~NMSE(4H)
were numerically indistinguishable from NMSE(H) for both teachers --
see note below)

### Teacher A: generalized_mode

| n | RealDiagonal | ComplexLocal | DualLocal |
|---|---|---|---|
| 1 | 3.90e-3 | 4.07e-3 | 5.43e-3 |
| 2 | 2.66e-3 | 1.74e-3 | **4.72e-4** |
| 4 | 1.70e-3 | 9.12e-4 | 9.77e-4 |
| 8 | 9.81e-4 | **1.52e-4** | 1.78e-4 |

### Teacher B: oscillatory

| n | RealDiagonal | ComplexLocal | DualLocal |
|---|---|---|---|
| 1 | 0.149 | **1.11e-6** | 0.139 |
| 2 | 0.155 | **1.59e-5** | 0.138 |
| 4 | 0.0987 | **1.05e-4** | 0.0798 |
| 8 | 0.150 | **3.11e-5** | 0.0465 |

**ComplexLocal dominates the oscillatory teacher by 3-6 orders of
magnitude at every n** -- it can represent a rotation exactly; neither
RealDiagonal nor DualLocal has any rotational coupling. A clean,
unsurprising confirmation of the complementary control (Q4-style
prediction cleanly confirmed here in the pure-BPTT setting, unlike the
murkier online-training picture in B35j).

**Extrapolation note**: NMSE(2H) and NMSE(4H) were numerically almost
identical to NMSE(H) for both teachers at H=64 (e.g. DualLocal
n=2/generalized_mode: 4.7197e-4 vs 4.7199e-4). With lambda=0.85,
lambda^64 ~ 3e-5 -- the impulse response has essentially fully decayed
within the training horizon, so the "extrapolation" test is largely
moot for this H: there is very little residual signal left to
mis-extrapolate. This weakens that leg of the interpretation (the
horizons technically "agree" everywhere, but not because any
architecture is doing something impressive at long range -- there's
almost nothing left to get right or wrong by 2H/4H).

## Paired statistics (10 eval seeds, log-NMSE(H), bootstrap 95% CI)

### DualLocal vs ComplexLocal (generalized_mode)

| n | median %reduction | mean log-diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| 1 | -29.0% (worse) | -0.478 | [-2.045, 0.911] | No |
| 2 | **+72.9%** | -0.916 | [-1.861, -0.015] | **Yes** |
| 4 | -7.1% (worse) | -0.005 | [-0.807, 0.933] | No |
| 8 | -17.0% (worse) | 0.091 | [-0.703, 1.110] | No |

### DualLocal vs RealDiagonal (generalized_mode)

| n | median %reduction | mean log-diff | 95% CI | excludes 0 |
|---|---|---|---|---|
| 1 | -28.1% (worse) | 0.523 | [-1.425, 2.435] | No |
| 2 | **+82.3%** | -1.572 | [-2.424, -0.793] | **Yes** |
| 4 | **+42.5%** | -0.805 | [-1.436, -0.294] | **Yes** |
| 8 | **+81.9%** | -1.908 | [-2.711, -1.156] | **Yes** |

DualLocal's advantage over RealDiagonal is fairly consistent (3 of 4 n
significant, all favoring Dual). Its advantage over ComplexLocal is
**not** consistent -- significant at exactly one of four capacity
levels (n=2), where the win is real and large (73%, CI excludes zero);
at n=1 Dual is significantly worse (CI doesn't exclude 0 but the
direction is consistently unfavorable); at n=4, n=8 there is no
significant difference either way.

## Predeclared success criterion -- verdict

At n=2: DualLocal beats BOTH ComplexLocal (72.9%) and RealDiagonal
(82.3%) by >=25%, both paired 95% CIs exclude zero, and NMSE(H)~NMSE(2H)~
NMSE(4H) trivially agree (see extrapolation note -- the horizons
"persist" because there is almost no decay left to mis-extrapolate,
not because of a demonstrated extrapolation capability). **Read
literally, the criterion is satisfied at this one matched-resource
point.**

However, the capacity sweep was explicitly designed to test ROBUSTNESS
across n, not to certify a single favorable operating point. At the
other three of four capacity levels tested (n=1,4,8), DualLocal does
NOT show a significant advantage over ComplexLocal specifically (worse
at n=1, statistically tied at n=4 and n=8). Only the comparison
against RealDiagonal is robust across capacities.

**Verdict: the predeclared criterion is technically met at n=2, but
this is a narrow, single-capacity-level result, not a robust finding
across the sweep.** Following the spirit of the interpretation rule
for asymmetric wins ("if Dual beats Complex but not RealDiagonal,
conclude the claim is not established"), the analogous honest
conclusion here is: **DualLocal's advantage over RealDiagonal is real
and fairly robust; its advantage over ComplexLocal specifically is not
robustly established across the capacity sweep** -- it holds at one of
four tested widths, which is evidence of A REAL EFFECT AT THAT WIDTH,
not a general practical-efficiency claim for the nonsemisimple
structure over the semisimple (Complex) alternative.

## Pole-inspection diagnostic (explanatory only, not a success criterion)

RealDiagonal's learned poles on the generalized-mode teacher:
- n=2: [-0.176, 0.896, -0.375, 0.560] -- one mode close to the true
  lambda=0.85 (0.896), the rest spread out; no tight nearby pair.
- n=4: one mode at 0.891 close to the teacher; others spread, no tight
  pair near lambda.
- n=8: three modes clustered near 0.85-0.90 (0.894, 0.889, 0.901,
  within 0.012 of each other) -- weak support for a "near-degenerate
  cluster" mechanism, though not an isolated clean lambda-eps/lambda+eps
  pair as hypothesized, and only visible at the largest n tested.

RealDiagonal does not cleanly learn an isolated lambda +/- epsilon pair
at small n; at large n there is a loose cluster of near-degenerate
poles near the true value, giving partial, not clean, support for the
hypothesized mechanism.

## Interpretation

Per the predeclared rules: this is not a clean positive result for the
"nonsemisimple gives a useful distinct temporal inductive bias"
hypothesis relative to Complex specifically (only 1 of 4 capacities
shows a significant Dual-over-Complex win), though it IS a fairly
robust result relative to RealDiagonal. The algebraic closure theorem
(exact O(P) credit, verified again here to machine precision) is
unaffected either way. Universal superiority is not claimed in any
direction. Per instruction: not proceeding to matched-credit frontier
or SCIFAR; stopping after this phase.

## Commit hash

See the commit introducing this file.
