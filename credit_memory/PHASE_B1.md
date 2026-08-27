# Phase B1 — compressibility of the exact causal P/Q credit state

Branch `credit-memory-repair`. Implementation + falsification only, per
directive: theory classes were supplied, not developed here. L=2 only.
No training, no Stage 0, no RoutePC/Meta-Adam/prospective-residual/
least-action mechanism. Code: `credit_memory/teacher.py` (shared exact
machinery, does not edit Phase A), `credit_memory/phase_b1_0_probe.py`
(B1.0), `credit_memory/phase_b1_compression_ladder.py` (B1.1-B1.4).
Artifacts: `results/credit_memory/phase_b1_0_probe_summary.json`,
`results/credit_memory/phase_b1_compression_ladder_summary.json`.

## Phase-A equations used (pasted verbatim from `PHASE_A.md`)

```
G_exact^0[m] = (1/2) sum_j B_1[j,m]      sum_u conj(q^1_u[j]) P_u[j,m]
             + (1/2) sum_j conj(B_1[j,m]) sum_u q^1_u[j]      Q_u[j,m]        (E2)

P_u[j,m] = a_1[j]      P_{u-1}[j,m] + Sa^0_u[m],   P_{-1} := 0               (E1, channel P)
Q_u[j,m] = conj(a_1[j]) Q_{u-1}[j,m] + Sa^0_u[m],   Q_{-1} := 0               (E1, channel Q)
```

`Sa^0` is the existing, unmodified within-layer eligibility trace
(`toyrig/ssm_rig.py:115-131`); `q^1` is the existing, unmodified naive
spatial error at the top layer; `a_1[j]` is the upper layer's own pole;
`B_1[j,m]` is the existing routing weight from lower mode `m` to upper
mode `j`. `j` indexes upper-layer (top) modes, `m` indexes lower-layer
(defective) modes.

## B1.0 — exact-teacher probe (`credit_memory/teacher.py`)

Reimplements (E1)/(E2) independently of `phase_a_causal_dual.py` (that
file is unedited) and checks it against `toyrig.ssm_rig.assemble(...,
direct=True)`, the trusted BPTT reference. Config: L=2, N=6, T=40,
BATCH=8, 5 seeds, arbitrary (x, r).

| seed | causal-vs-BPTT rel. err | online cos vs BPTT | causal(exact) cos |
|---|---|---|---|
| 0 | 7.9e-16 | 0.410 | 1.000 |
| 1 | 7.1e-16 | 0.952 | 1.000 |
| 2 | 6.2e-16 | 0.974 | 1.000 |
| 3 | 2.4e-16 | 0.691 | 1.000 |
| 4 | 1.0e-15 | 0.856 | 1.000 |

**A bug was caught and fixed during this step**: the first draft divided
`G_causal`/`G_online` by `BATCH` while `G_bptt` (from `tcg.assemble`) is
a raw, unaveraged sum — this produced a constant `1/BATCH` scaling
mismatch (rel. err. `0.875` identically across all seeds, cosine still
`1.000`, correctly flagging a pure-scale bug rather than a direction
bug). Fixed by removing the averaging in `teacher.py`; all reruns below
use the corrected version.

Online cosine varies substantially by seed (0.41-0.97) at this small
N=6/T=40 arbitrary-`r` config — a different regime from the `~0.596`
project-wide figure (which used trained-task residuals, larger N, and a
different aggregation). This probe's own `C0` numbers (below) are the
correct baseline for this experiment; the `0.596`/`0.901` figures are
not directly comparable here and this document does not claim they are.

## B1.1-B1.4 — compression ladder

### Model classes and parameter counts

All classes are driven only by `Sa^0` (existing) and read out against
`q^0` (existing, C1/C2) or `q^1` (existing, C3) — the two signals already
available at the exact point Phase-A's P/Q system is driven. `gamma`
(the drive-signal scale in the handoff's schematic) is gauge-fixed to 1
for C1/C2: it is not separately identifiable from the readout scale `c`
given only the gradient objective (rescaling `z` by any constant and
`c` by its inverse leaves `G` unchanged), so fixing it removes a flat
direction rather than restricting the model.

| class | recursion | readout | free params/mode | real DOF/mode |
|---|---|---|---|---|
| C0 | none (`z_t := Sa^0_t`) | `G[m]=sum_t conj(q^0_t)Sa^0_t` | 0 | 0 |
| C1 | `z_t=alpha z_{t-1}+Sa^0_t` | `G[m]=sum_t c conj(q^0_t) z_t` | `alpha, c` | 4 |
| C2 | `z_t=alpha z_{t-1}+beta conj(z_{t-1})+Sa^0_t+delta conj(Sa^0_t)` | `G[m]=sum_t[c conj(q^0_t)z_t + d q^0_t conj(z_t)]` | `alpha,beta,delta,c,d` | 10 |
| C3 | per **(j,m) pair**: `z1,z2` exactly as `P,Q` (E1) | exactly (E2) | `pole1,pole2,w1,w2` per pair | 8/pair, `8*N^2` total |

`alpha` (C1) is parameterized `sigmoid(rho)*exp(i*theta)` (`|alpha|<1`,
stability); C2's `alpha,beta` share a magnitude budget
`|alpha|+|beta|=sigmoid(rho_mag)<1` split by `sigmoid(rho_split)` — a
sufficient stability condition for the widely-linear (improper) AR(1)
recursion. C3's `pole1,pole2` are independently `sigmoid`-bounded the
same way. All `c,d,w1,w2,delta` are unconstrained complex.

### Fitting protocol and split (B1.3)

- N=6, T=40, BATCH=8 (same config as B1.0).
- 19 total architecture seeds: **TRAIN**=`{0..7}` (8), **VAL**=`{8,9,10}`
  (3), **TEST**=`{11..18}` (8). Splits are by seed (independent
  `init_params(seed)` draws — independent `a_1, B_1, c` per seed), not
  merely by data.
- Per TRAIN seed: 2 trajectories used in the pooled fit loss, 2 more held
  out (never touch the fit loss) for a same-seed/held-out-trajectory
  diagnostic. Per VAL/TEST seed: 3 trajectories, all held out.
- **One parameterization per class** (`alpha[m],c[m],...`, shape `(N,)`)
  is fit once by gradient descent (Adam, `lr=3e-2`, 800 steps) on the
  mean `1 - cos(G_compressed, G_bptt)` loss pooled over all 16
  train-seed fit trajectories jointly. No re-fitting per test trajectory
  or per test seed. TEST-seed data never enters the fit loss.
- BPTT gradients are used only as the offline fitting/evaluation oracle;
  nothing here is deployed, and no exact gradient is proposed for any
  training arm.

### Headline result: C0 vs C1 vs C2, cross-seed generalization

| | TEST median cos | TEST median rel.err | frac. of (1-C0) gap recovered |
|---|---|---|---|
| C0 online | 0.821 | 0.887 | -- |
| C1 complex-linear | 0.773 | 0.993 | **-0.19** |
| C2 widely-linear | 0.584 | 1.050 | **-0.96** |
| C1 (state-MSE control, see below) | 0.713 | -- | -- |
| C3 free fit (informative only) | 0.597 | -- | -- |
| C3 **positive control** (exact params) | **1.000000** | **0.0** | 1.00 |

**Under this cross-seed pooled-fit protocol, neither C1 nor C2 beats the
online baseline on held-out architectures — both are worse, and C2 is
worse than C1.** Per-seed TEST detail (C1, 3 trajectories/seed; `cos_online`
in parens):

```
seed 11: 0.601 (0.850)  0.427 (0.347)  0.270 (0.586)
seed 12: 0.830 (0.793)  0.731 (0.703)  0.864 (0.882)
seed 13: 0.234 (0.301)  0.658 (0.879)  0.163 (0.360)
seed 14: 0.924 (0.938)  0.975 (0.990)  0.877 (0.927)
seed 15: 0.897 (0.946)  0.795 (0.928)  0.804 (0.919)
seed 16: 0.857 (0.871)  0.340 (0.503)  0.725 (0.683)
seed 17: 0.751 (0.787)  0.239 (0.679)  0.848 (0.766)
seed 18: 0.881 (0.880)  0.717 (0.725)  0.973 (0.962)
```

C1 is close to (sometimes marginally above, more often marginally below)
the online baseline almost everywhere — the pooled fit converges close
to a near-trivial solution (small `alpha`, `c` near 1), not a
seed-transferable correction. C2 (10 real DOF/mode vs 4) fits TRAIN
better (loss `0.046` vs C1's `0.108`, both from a `0.31` random-init
start) but generalizes worse — classic overfitting given only 16 pooled
fit trajectories for 60 real parameters.

### Diagnostic: does seed-generalization explain the gap? (within-seed fit)

Separately from the headline pooled-cross-seed fit, C1/C2 were also
fit **and** evaluated on trajectories from a single fixed architecture
(2 fit trajectories, 1 held-out trajectory, same seed — no cross-seed
transfer asked at all), for 4 representative TEST seeds:

| seed | C0 (online) | C1 (within-seed) | C2 (within-seed) |
|---|---|---|---|
| 11 | 0.815 | **0.906** | 0.626 |
| 12 | 0.875 | **0.969** | 0.748 |
| 13 | 0.769 | **0.428** (failure) | **0.880** |
| 14 | 0.936 | **0.999** | **0.995** |

Also, pooled-train-seeds' own held-out trajectories (same seeds as the
pooled fit, different data) reach median cos `0.832` (C1) / `0.604`
(C2) vs pooled TEST-seed median `0.773` (C1) / `0.584` (C2), and VAL
median `0.864` (C1) / `0.574` (C2) — consistent with a real, if modest,
seed-generalization gap on top of a much larger within-seed effect.

**Reading**: a single complex-linear causal state (C1) *can* recover
most of the missing gradient information for a *given, fixed*
architecture (seed 12: `0.875 -> 0.969`; seed 14: `0.936 -> 0.999`) —
this is direct evidence the compressibility hypothesis is often true at
the per-network level. But (a) it fails outright on at least one
architecture even within-seed (seed 13: `0.769 -> 0.428`, a genuine
regression, not just "no help"), and (b) a single filter fit once and
shared across independently-drawn architectures does **not** transfer —
the online baseline is at least as good on pooled TEST seeds. Mode index
`m` has no consistent cross-seed identity in this toy (each seed draws
`a_1, B_1` independently), so this is not surprising in retrospect, but
it was not assumed going in and is now measured, not asserted.

### State-MSE vs gradient-weighted fitting (B1.2)

C1 was additionally fit by minimizing `||z_t[m] - target_t[m]||^2`
(state reconstruction), where `target_t[m] := sum_j |B_1[j,m]|^2
P_t[j,m] / sum_j |B_1[j,m]|^2` — the coupling-weighted projection of the
*true* P channel onto a single aggregate channel (the natural "what
would a single channel look like if it tried to reproduce the dominant
part of the true multi-channel P" target; there is no unique canonical
choice here and this is flagged as a design decision, not a derived
quantity).

- State-MSE-fit C1, evaluated by gradient cosine on TEST seeds: median
  cos **0.713**.
- Gradient-weighted-fit C1 (headline): median cos **0.773**.

Gradient-weighted fitting wins, as the hypothesis predicted, though the
margin (`0.773` vs `0.713`) is modest relative to how different the two
loss landscapes are (loss values `1231 -> 1231` for MSE, effectively
flat/not converging much past init, vs `0.31 -> 0.11` for the gradient
loss, a real 65% reduction) — the state-MSE objective appears to be a
substantially harder, less-informative optimization landscape (very
slow loss decrease relative to gradient-weighted fitting) even though
its final gradient-cosine outcome is only moderately worse. This is
weak-to-moderate support for the state-complexity/gradient-information-
complexity distinction, not strong support — both final numbers are
below C0 here, so this comparison is best read as "which failure is
less bad," not "which succeeds."

### C1 vs C2 (key comparison 1)

C2 (widely-linear, 10 real DOF/mode) is worse than C1 (complex-linear, 4
real DOF/mode) on every held-out metric measured here: pooled TEST
median cos (`0.584` vs `0.773`), pooled TRAIN-seed-held-out-trajectory
median (`0.604` vs `0.832`), VAL median (`0.574` vs `0.864`), and 2 of 4
within-seed diagnostic seeds. **This does not connect cleanly to the
earlier observation that a general real `2x2` modal map has a higher
oracle *ceiling* than a plain complex multiplier** (`FINAL_MODAL_
GEOMETRY_AUDIT.md`, per-mode real `0.765` vs per-mode complex `0.901` vs
full `2x2` `0.922`) — that was a *static*, per-checkpoint, unconstrained
least-squares oracle fit (maximal information, no cross-seed
generalization asked, no recursive/dynamical structure). C2 here adds a
genuine extra *temporal degree of freedom* (a second pole via `beta`)
and more free readout parameters, which increases overfitting risk under
a small pooled fit set. At current sample sizes the wider model is worse
in both cross-seed and (for 2 of 4 seeds) within-seed settings. Whether
C2's extra capacity would win with a within-seed fit and more
trajectories is untested here.

### C2 vs exact P/Q (key comparison 2)

Exact P/Q (C3 positive control) is machine-precision (cos `1.000000`,
rel. err. `0.0`) by construction, since it *is* the (E2) formula. C2's
median TEST cos is `0.584`, i.e. compressing the true `4` real teacher-
state dimensions per `(j,m)` pair (`P,Q`, `N=6` upper modes each) down to
`2` real dynamical dimensions per lower mode `m` (aggregating away the
`j`-index entirely) **loses substantial learning-relevant information
under this fitting protocol** — C2 does not merely fall short of the
`0.901`-type ceiling, it falls below the `C0` online baseline on held-out
architectures.

### Free-fit C3 (informative only, not required)

An unconstrained optimization of the full `(j,m)`-pair-indexed C3 model
(same class as the exact teacher, `8*N^2=288` real parameters, fit the
same pooled-cross-seed way) reaches TEST median cos `0.597` — similar to
C2, and well below both C0 and the exact positive control. Per the
directive, this is **not** used as evidence against the representation:
the positive control already proves the exact solution is in the search
space and reachable analytically; this free fit only shows that
unconstrained gradient descent, pooled across independently-random
architectures with no seed-conditioning, does not find it — expected,
since (as with C1/C2) the true optimum is different per seed and no
single shared parameter set can be simultaneously optimal for all of
them.

## Does a one-complex-state causal representation preserve a substantial
## fraction of the exact credit correction?

**Not under a shared-across-architectures fit (the headline, most
stringent test asked for): no — C1 is statistically indistinguishable
from, and on aggregate slightly worse than, the online baseline on held-
out architectures (median cos `0.773` vs `0.821`, `frac_gap_recovered
= -0.19`).**

**Within a single fixed architecture: often yes, sometimes dramatically
(seed 12: `0.875->0.969`; seed 14: `0.936->0.999`), but not reliably
(seed 13: `0.769->0.428`, a real failure).** The representation-capacity
question ("does a tiny causal state exist that recovers most of the
gap, for a given network") is therefore not yet answered uniformly
negative or positive — it appears architecture-dependent, and no
mechanism tested here explains which architectures it will work for.

## Failure examples (not just medians)

- **Seed 13, C1 within-seed**: cos regresses from online's `0.769` to
  `0.428` after fitting on 2 trajectories of the *same* architecture,
  evaluated on a 3rd held-out trajectory of that same architecture. Not
  a generalization failure — a genuine within-seed representational/
  optimization failure.
- **Seed 11, C1 pooled-cross-seed, trajectory 3**: cos `0.586 -> 0.270`
  (online to C1), one of several TEST trajectories where the pooled
  filter actively hurts rather than merely failing to help.
- **C2 systematically**: worse than C1 on 3 of 4 within-seed diagnostic
  seeds and on every pooled aggregate metric — more capacity did not
  help at these sample sizes, and often actively overfit.

## Artifacts

- `results/credit_memory/phase_b1_0_probe_summary.json` — B1.0, git hash,
  config (N,T,BATCH), per-seed exact/online/causal numbers.
- `results/credit_memory/phase_b1_compression_ladder_summary.json` — full
  B1.1-B1.4 results: git hash, config (N,T,BATCH,steps,lr), explicit
  `train_seeds`/`val_seeds`/`test_seeds`/trajectory counts, per-class
  parameter counts, per-seed and aggregate cos/rel_err/norm_ratio/
  frac_gap_recovered for C0-C3 (test, train-seed-held-out-trajectory, and
  val splits separately), state-MSE-vs-gradient fit-loss histories, C3
  positive-control and free-fit results, and the within-seed diagnostic.

## Not done in Phase B1 (by design, per scope)

No L=3+ testing; no prospective coding, Meta-SGD, Meta-Adam, RoutePC, or
optimizer-state adaptation; no task training or S5 Stage 0; no online
learning of any compressor (all fitting is an offline representation-
capacity oracle, per B1.3).
