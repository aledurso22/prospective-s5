# Phase B2 — exact state-space form, Hankel spectrum, balanced truncation

Branch `credit-memory-repair`. Implementation + falsification of the
supplied hypothesis. L=2 only, per scope. No training, no Stage 0, no
RoutePC/Meta-Adam/prospective-residual mechanism. Code:
`credit_memory/hankel.py` (Gramian/Hankel/balanced-truncation core),
`credit_memory/phase_b2a_state_space.py` (B2A),
`credit_memory/phase_b2bc_hankel_truncation.py` (B2B+B2C),
`credit_memory/phase_b2d_three_levels.py` (B2D). Artifacts:
`results/credit_memory/phase_b2{a,bc,d}_*_summary.json`.

**Headline**: a *principled* (Gramian/observability-weighted, no free
optimization, no BPTT used to build the reduction) truncation to **4
complex states** (out of the exact 12, `2N` with `N=6`) recovers median
held-out cosine **0.967** against BPTT (75% of the online→exact gap) at
r=4, and **0.995** (97.4% of the gap) at r=8 — a completely different
outcome from Phase B1's free-parameter cross-seed fit, which never beat
the online baseline. This directly satisfies the "2-4 states get to
~0.90+" gate.

## B2A — state-space verification

`F = diag(A, conj(A))`, `A = diag(a1)`, `d = ones(2N)`, state
`x_u = [P_u[:,m]; Q_u[:,m]]`, readout `g_t[m] = c_t^dagger x_t` with
`c_t` built from `q1_t` and `B1[:,m]` (see `credit_memory/hankel.py`
docstring for the exact derivation from Phase-A's (E2)). Verified against
both `credit_memory/teacher.py`'s P/Q implementation and the trusted
BPTT reference, 5 seeds, N=6, T=40, BATCH=8:

```
seed 0: vs P/Q rel_err=7.3e-16   vs BPTT rel_err=8.5e-16
seed 1: vs P/Q rel_err=5.0e-17   vs BPTT rel_err=3.2e-16
seed 2: vs P/Q rel_err=5.0e-16   vs BPTT rel_err=6.7e-16
seed 3: vs P/Q rel_err=3.7e-16   vs BPTT rel_err=8.4e-16
seed 4: vs P/Q rel_err=5.7e-16   vs BPTT rel_err=6.9e-16
```
All pass. Pure convention check, as expected.

## B2B — credit Hankel spectrum

**No BPTT gradient information used.** `Wc` solved analytically from
`(F,d)` alone (closed form for diagonal `F`: `Wc[p,q] = 1/(1-f_p
conj(f_q))`). `c_t` built from `q1` (the existing, already-causal,
BPTT-free naive spatial error) and `B1[:,m]`; `S = E[c_t c_t^dagger]`
estimated from **4 calibration trajectories** (T=60, BATCH=8 each, i.e.
1920 (t,b) samples) per (seed, mode); `Wo` solved analytically given `S`
(same closed form, transposed roles). Hankel singular values
`sigma_i = sqrt(eig(Wc Wo))`, computed two independent ways (direct
eigendecomposition and via the square-root balancing algorithm) and
cross-checked to agree on the non-negligible spectrum (`rtol=1e-5`; the
numerically-negligible tail, `<1e-3` of the top singular value, differs
at the `~1e-7` absolute level from `eps`-regularization in the matrix
inverse used only by the balancing algorithm — irrelevant to any `r<=8`
truncation).

**8 architecture seeds, 6 modes each (48 spectra). Dimensions needed for
cumulative squared Hankel-SV mass (median over the 48 spectra, exact
dimension = 12):**

| threshold | 80% | 90% | 95% | 99% |
|---|---|---|---|---|
| median dims (of 12) | 3 | 4 | 5 | 7 |

Per-seed spread (dims for 90% mass, one value per mode):
```
seed 0: [6, 4, 6, 2, 4, 4]   seed 1: [4, 4, 2, 4, 2, 4]
seed 2: [5, 5, 5, 4, 5, 3]   seed 3: [2, 2, 2, 2, 2, 2]
seed 4: [5, 4, 6, 5, 8, 2]   seed 5: [2, 2, 2, 2, 8, 2]
seed 6: [4, 6, 5, 4, 3, 4]   seed 7: [3, 4, 4, 4, 4, 4]
```

**This alone answers the question posed: yes, the exact 12-dimensional
causal state is genuinely low-rank in the gradient-relevant (Hankel)
metric** — half the modes need `<=4` of 12 dimensions for 90% of the
squared singular-value mass, though seed 4/mode 4 needs 8 (near-exact),
showing the effect is not universal.

## B2C — balanced truncation ladder

Square-root balanced truncation (`credit_memory/hankel.py:
balanced_transform`) to `r in {1,2,4,8,12(exact)}`, using the same
calibrated `Wc,Wo` as B2B. Evaluated on **4 disjoint test trajectories**
per seed (never touch calibration). Cosine/rel-err are computed on the
**full N=6-mode gradient vector** per test trajectory (a single complex
number's normalized self-inner-product is trivially 1 and was caught and
discarded as a bug during implementation — see Implementation notes).
`frac_gap_recovered = (cos_r - cos_online) / (1 - cos_online)` per
trajectory, then medianed.

| r | median cos | median rel.err | median norm ratio | median frac. gap recovered |
|---|---|---|---|---|
| 1 | 0.488 | 0.998 | 0.200 | **-0.334** |
| 2 | 0.883 | 0.825 | 0.492 | **0.073** |
| **4** | **0.967** | 0.554 | 0.843 | **0.750** |
| **8** | **0.995** | 0.126 | 0.999 | **0.974** |
| 12 (exact) | 1.000 | 8.8e-15 | 1.000 | 1.000 |

**Exact-dimension (r=12) regression check**: balanced truncation to the
full dimension is only a change of basis, so it must reproduce BPTT
exactly. Confirmed: all 8x6x4=192 (seed, mode, test-trajectory) rows
`rel_err < 1e-8` (max `6.5e-14`).

C0 online baseline on this same test set: median cos **0.628** (a
different config/test-set than B1's `0.821` — not directly comparable
across documents; see Implementation notes).

**r=4 clears the "2-4 states get to ~0.90+" gate** (0.967); r=2 is close
but under it (0.883); r=1 fails outright and is *worse* than online
(negative gap recovery — a single balanced-truncated dimension throws
away net-useful information relative to doing nothing).

## B2D — three levels of information

Same 8 seeds, same calibration/test trajectories, same r-ladder as B2C,
for direct comparison.

**L1 — architecture only** (`Wc` analytic as always; `Wo` from an
isotropic prior `S=I`, i.e. **no calibration data at all**):

| r | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| median cos | 0.634 | 0.765 | 0.919 | 0.965 | 1.000 |
| median frac. gap recovered | 0.048 | 0.093 | 0.324 | 0.887 | 1.000 |

**L2 — causal calibration** = B2C above (repeated for the row-by-row
comparison): r=1: 0.488, r=2: 0.883, r=4: 0.967, r=8: 0.995.

**L3 — exact-gradient oracle** (r free complex-linear channels per mode,
full `q1`-based readout — strictly richer functional family than the
balanced-truncated one, since it is not constrained to use only the
`{a1[j], conj(a1[j])}` pole set; fit by Adam against BPTT on the *same*
4 calibration trajectories, evaluated on the *same* 4 held-out test
trajectories; upper-bound capacity test, per-instructions optimizer
failure is not evidence against the representation):

| r | 1 | 2 | 4 |
|---|---|---|---|
| median cos | **0.944** | 0.964 | 0.957 |
| median frac. gap recovered | **0.819** | 0.854 | 0.860 |

### The unexpected finding: L1 vs L2 crossover at low r

**L2 < L1 only at r=1** (L1 0.634 vs L2 0.488 — the calibrated
construction is worse than the naive isotropic-prior one at the smallest
size); from r=2 onward L2 >= L1 and the gap grows in L2's favor (r=2:
0.883 vs 0.765; r=4: 0.967 vs 0.919; r=8: 0.995 vs 0.965). At r=1
there is only one direction to keep, and the calibration-estimated `S`
(from 4 trajectories) appears to occasionally misrank which single
direction is truly dominant for a *disjoint* test trajectory — sampling
noise in a low-data, low-redundancy regime dominating the modest
weighting benefit calibration should provide. From r=2 up, calibration's
benefit becomes clear and grows with r. This is a genuine, measured
effect, not asserted going in.

### The unexpected finding: L3 r=1 dramatically beats L2 r=1

`L3 r=1` (0.944) is **far above** `L2 r=1` (0.488) and even above `L2
r=4` (0.967 is close but L3 r=1 gets 82% of the gap with a SINGLE
channel vs L2's 75% with FOUR). Per the directive's own gate: **"If the
causal construction fails but the exact-gradient reduced-order oracle
succeeds: Compression exists; we haven't yet learned the correct causal
relevance metric."** This gate fires cleanly at r=1: a single complex
channel *can* carry most of the gradient-relevant information (proven by
the oracle), but the specific causal relevance metric used here
(Hankel/observability weighting restricted to the natural `{a1[j],
conj(a1[j])}` eigenbasis of the exact system) does not find it at r=1 —
the oracle's freedom to use an *effective* pole outside that fixed
2N-element pole set, plus full `q1`-based (not eigenbasis-restricted)
readout, is doing real work. By r=4, L2 (0.967, no BPTT in construction)
essentially catches up to L3 (0.957) and even nominally exceeds it —
L3's oracle fit degrades at higher r for at least one seed (see failure
examples below), consistent with overfitting on only 4 calibration
trajectories once parameter count grows (`r(2+4N)=104` complex/mode at
r=4).

## Reading against the pre-registered gates

- **"1 complex state gets ~0.90+ using only architecture + causal
  statistics"**: **NO** (L2 r=1 = 0.488, L1 r=1 = 0.634 — both fail).
- **"1 fails but 2-4 states get there"**: **YES, at r=4** (L2 = 0.967);
  r=2 is close but under (0.883).
- **"causal construction fails but exact-gradient oracle succeeds
  (compression exists, wrong relevance metric)"**: **YES, specifically
  at r=1** (L2=0.488 fails, L3=0.944 succeeds). At r>=4 this gate is
  moot since L2 itself already succeeds.
- **"even the oracle requires large r"**: **NO** — the oracle succeeds
  already at r=1 (0.944).

**Combined reading**: the exact causal-dual state is genuinely
compressible (B2B's low-rank Hankel spectrum, B2C's r=4 result), *and*
there is headroom beyond what Hankel/balanced-truncation-in-the-natural-
eigenbasis currently captures at the smallest sizes (B2D's L3-vs-L2
crossover at r=1) — pointing at a real, currently-unidentified better
relevance metric or basis for the r=1 case specifically, while r=4 is
already a strong, principled, non-oracle result on its own.

## Failure examples (not just medians)

- **L2 r=1**: every seed's r=1 truncation underperforms online on the
  majority of test trajectories (median frac_gap_recovered=-0.33); this
  is the headline failure at the smallest size.
- **L1 r=1, seed 5**: median cos 0.44 region (see per-seed printout in
  the script log) — architecture-only reduction with no calibration data
  is close to uninformative at r=1 for some seeds.
- **L3 r=1, seed 5's test trajectories**: individual cosines as low as
  **0.276** and **0.602** despite a median of 0.944 — the oracle is not
  uniformly reliable even though it wins on aggregate; one seed's fit
  clearly landed in a poor optimum.
- **L3 r=4, seed 6's test trajectories**: cosines **0.628, 0.659,
  0.685** — a case where *more* oracle capacity (r=4 vs r=1) produced a
  *worse* fit for that architecture, consistent with overfitting the
  104-complex-parameter-per-mode model to only 4 calibration
  trajectories.

## Implementation notes (bugs caught and fixed during this phase)

- **Scalar-cosine triviality bug**: an early draft of B2C computed
  `cos` per-mode as a scalar `|conj(a)b|/(|a||b|)`, which is
  *identically 1 for any nonzero complex numbers* (the absolute value
  discards all phase information, so a length-1 "vector" is trivially
  "aligned with itself"). Caught before any numbers were reported;
  fixed by assembling the full N-mode gradient vector across all modes
  before computing cosine, exactly as in B1.
- **Hankel-singular-value cross-check tolerance**: the two independent
  computations of `sigma` (direct `eig(Wc Wo)` vs. via the balancing
  algorithm) initially failed a strict `atol=1e-9` assertion; root cause
  was `eps`-regularization (`1e-13`) inside the balancing algorithm's
  matrix square-root/inverse, affecting only the numerically-negligible
  tail of the spectrum (confirmed by inspection: differences occur
  exclusively in the smallest 1-2 singular values, absolute scale
  `~1e-7`, irrelevant to any `r<=8` truncation). Relaxed to compare only
  singular values `>1e-3` of the top one, `rtol=1e-5` — all 48
  (8 seeds x 6 modes) spectra pass.
- **`n_raw` parameter-count bug in the L3 oracle**: `r*(1+4N)` should
  have been `r*(2+4N)` (one complex pole per channel needs 2 real
  parameters, magnitude+phase, not 1). Caught immediately via a JAX
  reshape shape-mismatch error before any fit ran.

## Comparability caveat

This document's C0/online numbers (median cos 0.628 on this experiment's
test set) are **not** the same figure as B1's C0 (0.821) or the
project-wide 0.596/0.901 figures — different N/T/BATCH, different random
trajectories, different (calibration vs. no-calibration) protocol. Every
number in this document is self-contained within its own table; no
cross-document numeric comparison is intended unless stated explicitly.

## Artifacts

- `results/credit_memory/phase_b2a_state_space_summary.json` — git hash,
  config, per-seed rel-err rows, `all_pass`.
- `results/credit_memory/phase_b2bc_hankel_truncation_summary.json` —
  git hash, config (seeds, N, T, BATCH, calibration/test trajectory
  counts, r-ladder), full per-(seed,mode) spectrum rows and
  `dims_for_mass`, full per-(seed,r,test-trajectory) truncation rows,
  `exact_regression_all_pass`, aggregates.
- `results/credit_memory/phase_b2d_three_levels_summary.json` — git
  hash, config, L1/L3 full per-row results, L3 fit-loss histories per
  seed/r.

## Not done in Phase B2 (by design, per scope)

No L=3+ testing; no prospective coding, Meta-SGD, Meta-Adam, RoutePC, or
optimizer-state adaptation; no task training or S5 Stage 0; L3's oracle
r-ladder was kept to `{1,2,4}` (not extended to 8) given fit cost scaling
and diminishing returns already visible by r=4.
