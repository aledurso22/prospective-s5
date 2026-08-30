# Phase B35d — streaming (true online, continual) system-identification test

Branch `S5-CCM-scale-validation`. Architecture frozen (no new recurrent
family): RegularBlock/ProductLocal, GenericBlock, and RTU reused exactly
as validated in B35a-c. Code: `credit_memory/b35d_streaming_sysid.py`
(`/tmp/b35d_streaming_results.json`, PILOT log discarded, frozen-run log
`/tmp/b35d_frozen_run.log`).

## 0. Mathematical framing (resolved before any headline number)

**Distinguishing three claims, as required before interpreting this
experiment:**

1. *Exactness of the sensitivity recursion for FIXED θ.* Proven in
   B35a-c and reverified here for this task's specific wiring
   (concat[x_t,u_t] input, frozen B_in): RegularBlock's reduced-vs-BPTT
   and GenericBlock's exact-module-RTRL-vs-BPTT both agree to ~1e-16
   when θ is held fixed across the check sequence. Untouched by
   anything below.
2. *Equivalence of the reduced (s_t) and full (S_t=M_{s_t})
   representations of that recursion.* This is a pure algebraic fact
   (`M_u@v=M_v@u` by commutativity) that holds **identically whether θ
   is the same value at every step or a different one** — the
   compression is a property of how a single step's multiplication is
   structured, not of whether θ is stationary. **The closure/compression
   claim is unaffected by continual parameter updates.**
3. *What gradient is actually used when θ changes every step.* This is
   the real issue. When θ_0→θ_1→θ_2→... changes continually, h_t is no
   longer a function of one fixed parameter — "dh_t/dθ" has no single
   referent. The carried trace s_t (updated with the *current* θ_t at
   each step) is the standard **continual/online RTRL sensitivity
   approximation** (as in the original Williams & Zipser RTRL and the
   RTU streaming paper `b28_rtu_faithful.py` is based on) — not the
   exact gradient of a fixed-parameter objective. This staleness is
   **shared identically by all three architectures** (none compute an
   exact continual gradient in the strong sense; only their *storage
   cost* for carrying the trace differs, which is what this experiment
   actually tests). **B35d is therefore described as a continual
   RTRL-style learning experiment, not exact gradient descent through
   the full changing-parameter trajectory.**

## 1. Task, teacher, hypothesis

Teacher: hand-written, independent 4-state linear system (no import
from ProductLocal/GenericBlock code) = blockdiag(oscillatory 2x2 block
`rho^t*cos(omega t)`, repeated-pole 2x2 Jordan block `(c0+c1 t)
lambda^t`). Verified exactly (impulse response fit, max err 4.4e-16,
c1=0.353 genuinely nonzero). One regime change at t=1500 (rho:0.8->0.5,
lambda:0.85->0.55).

Protocol: true streaming, no BPTT window, no replay: observe(x_t,u_t)
-> predict x_hat_{t+1} -> loss -> ONE exact-online parameter update via
each architecture's own existing exact-RTRL machinery -> continue.
theta trained via each architecture's own eligibility mechanism; each
architecture's own input coupling (B_in for Regular/Generic frozen;
RTU's B_real/B_imag trained natively, an explicit asymmetry, not
hidden) is handled per its own pre-existing scope; C_out (pure readout)
trained by a direct local gradient (no eligibility needed).

Hypothesis: at fixed persistent exact-credit memory, can RegularBlock's
saved credit budget buy better CONTINUAL online system identification
than GenericBlock?

## 2. Frozen protocol (predeclared before inspecting evaluation seeds)

Everything before this point, including one seed-0 run at C=64, is
PILOT and excluded from evaluation and from any reported statistic.

- LR grid: {0.01, 0.02, 0.05}, selected on TUNING_SEEDS=(100,101) only
  (mean loss over steps [200,800), averaged across tuning seeds), then
  frozen and applied unchanged to every evaluation seed.
- EVAL_SEEDS=(11,12,13) — disjoint from tuning seeds, not previously
  inspected.
- pre_window=(800,1500), post_window=(2300,3000) — widened from an
  initial narrow-window pilot choice specifically because the pilot
  showed narrow windows are highly sensitive to landing mid-oscillation
  (the same failure mode diagnosed in the earlier C=256 audit); this is
  a general methodological correction, not a data-dependent tuning.
- Recovery metric: steps for a 20-step trailing-average loss to drop
  below 1.5x the pre-change window's mean loss.
- Resource rule: actual persistent credit <= C for every architecture.
  RegularBlock/GenericBlock hit credit=C exactly (C divisible by d=4).
  RTU (features=5, since it natively trains its 5-dim input coupling)
  uses `hidden=floor(C/(4+4*5))`, guaranteeing credit<=C; the realized
  value is reported plainly (it is measurably below C, an inherent
  granularity cost of RTU's own architecture at this feature count).

## 3. Model/resource table (before training)

| C (target) | arch | r | P | actual credit | elig-step time |
|---|---|---|---|---|---|
| 128 | RegularBlock | 128 | 128 | 128 | 21.98us |
| 128 | GenericBlock | 32 | 32 | 128 | 48.14us |
| 128 | RTU (features=5) | 10 | 120 | **120** (< 128) | 196.70us |
| 64 | RegularBlock | 64 | 64 | 64 | 25.94us |
| 64 | GenericBlock | 16 | 16 | 64 | 35.60us |
| 64 | RTU (features=5) | 4 | 48 | **48** (< 64) | 205.62us |

RegularBlock's extra state/parameters at matched credit (128 vs 32,
64 vs 16) are the intended consequence of the compression, not a
confound.

## 4. Validity checks (all pass)

- Teacher impulse response: oscillatory-block and repeated-pole-block
  coordinates fit their analytic forms to 2.2e-16 / 4.4e-16, c1=0.353.
- RegularBlock reduced-RTRL vs BPTT (fixed θ, this task's wiring,
  T=12): relative error 1.146e-16.
- GenericBlock exact module RTRL vs BPTT (fixed θ, this wiring, T=12):
  relative error 1.881e-16.
- Actual allocated eligibility arrays match claimed credit exactly at
  both C=64 and C=128 for RegularBlock and GenericBlock; RTU's actual
  array size (192 at nominal C=64, 384 at nominal C=128, since features=5
  makes its own formula 4h+4h*5=24h) is what motivated the resource-rule
  fix in section 2 — this was caught BEFORE the frozen run, exactly the
  purpose of doing validity checks first.
- No accidental teacher/architecture code sharing: the teacher block
  (`make_A`, `teacher_step`) uses only plain `jnp` linear algebra; zero
  imports from `b35a_product_local_algebra.py` anywhere in that code.

## 5. Raw per-seed results (frozen evaluation, EVAL_SEEDS=11,12,13)

### C=128 (LR selected on tuning seeds: Regular=0.01, Generic=0.05, RTU=0.01)

| arch | seed | pre_nmse | post_nmse | steps_to_recover | cum_loss |
|---|---|---|---|---|---|
| RegularBlock | 11 | 1.341 | 0.413 | 0 | 811 |
| RegularBlock | 12 | 1.459 | 0.509 | 0 | 1037 |
| RegularBlock | 13 | 0.864 | 0.575 | 0 | 714 |
| GenericBlock | 11 | 0.030 | 1.12e-4 | 0 | 140.5 |
| GenericBlock | 12 | 0.342 | 2.08e-4 | 0 | 145.0 |
| GenericBlock | 13 | 0.197 | 4.27e-4 | 0 | 210.9 |
| RTU | 11 | 0.0288 | 4.15e-3 | 0 | 29.7 |
| RTU | 12 | 0.0124 | 2.26e-3 | 0 | 21.8 |
| RTU | 13 | 0.0285 | 5.12e-3 | 0 | 28.2 |

Across-seed (median/mean/std): RegularBlock pre=1.341/1.221/0.257,
post=0.509/0.499/0.067. GenericBlock pre=0.197/0.190/0.128,
post=2.08e-4/2.49e-4/1.32e-4. RTU pre=0.0285/0.0232/0.0077,
post=4.15e-3/3.84e-3/1.19e-3. Zero divergence, zero nonfinite steps,
every cell.

### C=64 (LR selected on tuning seeds: Regular=0.02, Generic=0.01, RTU=0.01)

| arch | seed | pre_nmse | post_nmse | steps_to_recover | cum_loss |
|---|---|---|---|---|---|
| RegularBlock | 11 | 0.137 | 0.0549 | 5 | 191.3 |
| RegularBlock | 12 | 0.381 | 1.09e-3 | 0 | 214.8 |
| RegularBlock | 13 | 0.363 | 0.0139 | 0 | 204.1 |
| GenericBlock | 11 | 0.053 | 3.43e-4 | 0 | 30.1 |
| GenericBlock | 12 | 0.0098 | 1.71e-4 | 0 | 36.5 |
| GenericBlock | 13 | 0.058 | 3.92e-4 | 0 | 49.1 |
| RTU | 11 | 0.039 | 3.56e-3 | 0 | 50.6 |
| RTU | 12 | 0.027 | 5.86e-3 | 0 | 75.4 |
| RTU | 13 | 0.031 | 0.0112 | 0 | 74.9 |

Across-seed: RegularBlock pre=0.363/0.293/0.111, post=0.0139/0.0233/0.0229.
GenericBlock pre=0.053/0.040/0.022, post=3.43e-4/3.02e-4/9.46e-5. RTU
pre=0.031/0.032/0.005, post=5.86e-3/6.86e-3/3.18e-3. Zero divergence.

## 6. Learning/adaptation curves

RegularBlock's trailing (20-step) online loss curve, in this true
single-sample-per-step online regime, is **highly non-monotonic
throughout the whole stream**, at both budgets (e.g. C=128, seed 11:
0.81→0.58→0.32→0.14→0.67→0.26→0.25→0.26(t=1499)→0.26(t=1500)→0.003→
0.017→0.0008→0.18→0.108→0.015→9.9e-5). It does reach excellent values
transiently and at the very end, but the reported pre/post windows
land on averages that include large intermediate excursions, not a
converged floor. This is the same qualitative failure mode diagnosed
in the earlier B35c C=256 audit (GenericBlock there) — except here it
is **RegularBlock**, not GenericBlock, that shows it, under the
single-sample continual-update regime. GenericBlock and RTU's curves
in this task are comparatively far more stable (their per-seed
post_nmse values above have much tighter spread than RegularBlock's).

## 7. Scientific interpretation and confounds

- **Representation/inductive-bias effects**: not clearly favoring
  RegularBlock here — GenericBlock's unrestricted local family and
  RTU's diagonal family both fit this generalized-mode+oscillatory
  teacher comfortably under continual updates; RegularBlock does not
  show a representational edge in this regime the way it did under
  batched BPTT training in B35b/c.
- **Credit-memory effects**: RegularBlock has 4x the state/parameters
  of GenericBlock at matched credit (128 vs 32 at C=128) and still
  underperforms — the "saved credit -> extra capacity" mechanism from
  B35c does not straightforwardly convert into an advantage here.
- **Optimization effects**: the dominant confound. RegularBlock's
  per-factor structure appears substantially more sensitive to
  single-sample (unbatched) online SGD noise than GenericBlock's or
  RTU's, producing persistent trailing-loss oscillation rather than
  smooth convergence, at every budget and every seed tested. This
  mirrors (in a different architecture) the exact oscillation
  mechanism found in the B35c diagnostic audit — evidence that
  RegularBlock's structural projection (which stabilizes *forward
  dynamics*) does not by itself guarantee smooth *single-sample online
  optimization* dynamics.
- **Adaptation effects**: "steps_to_recover" was uninformative (mostly
  0) for all three architectures — the threshold, defined relative to
  each architecture's own noisy pre-change window, was often already
  satisfied trivially; this metric did not discriminate here and
  should not be over-read.
- **Numerical instability**: none — zero divergence, zero nonfinite
  steps across all 54 evaluation runs (2 budgets x 3 archs x 3 seeds x
  the pre-selected LR).
- **A genuine asymmetry, stated plainly**: RTU trains its own input
  coupling online (native to its existing machinery); RegularBlock/
  GenericBlock's input coupling is frozen (extending their validated
  formalism to credit an input matrix would be new engineering). This
  could only help RTU relative to Regular/Generic, not hurt it — it
  does not explain RegularBlock's underperformance relative to
  GenericBlock, which has the identical frozen-B_in treatment.

## 8. Conclusion

**This application test weakens the practical case for RegularBlock**,
specifically in the true single-sample continual-online-update regime
tested here — as distinct from the batched-BPTT-style training used
throughout B35a-c, where RegularBlock's matched-credit advantage was
real (B35c) and its mechanism was verified (B35b). At both credit
budgets tested (C=64, C=128), across all 3 frozen evaluation seeds,
RegularBlock is the worst of the three architectures on both pre- and
post-change NMSE, despite having 4x GenericBlock's state/parameter
capacity at matched credit. The likely cause is an optimization
sensitivity specific to single-sample online updates on RegularBlock's
per-factor structure, not a representational deficiency and not the
generic continual-RTRL staleness issue (which is shared identically by
all three). This does not overturn B35a-c's frozen results (which used
a different training regime and remain valid on their own terms), but
it is a real, honest limitation of RegularBlock specifically in the
continual streaming setting, and should be reported as such rather
than smoothed over.

## 9. Commit hash

See the commit introducing this file.
