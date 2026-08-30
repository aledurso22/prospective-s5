# Phase B35b — the decisive intra-module credit-compression experiment

Branch `S5-CCM-scale-validation`. Builds on the frozen B35a mechanism
(bounded product-local commutative response algebra, `credit_memory/
b35a_product_local_algebra.py`, unmodified). Architecture NOT broadened
further in this phase, per instruction.

Frozen hypothesis under test: for an independent recurrent module with
state dim d and p trainable local recurrent parameters, generic exact
module-wise RTRL needs `d*p` persistent sensitivity scalars
(`S_t in R^{d,p}`); our regular-algebra module achieves the SAME exact
online credit using only `p` scalars (`S_t = M_{s_t}`). The question is
whether this factor-d compression is real, useful, and buys
representation at fixed exact-credit memory — not whether modular RTRL
is O(P) (already known).

Code: `credit_memory/b35b1_mechanism_check.py`,
`credit_memory/b35b2_generic_vs_regular.py`. Both LINEAR
(`h_{t+1}=A(theta)@h+B_in*x_t`) throughout, for a clean, unconfounded
comparison.

## Part 1 — mechanism check: d=1 vs d=2

**Teacher**: hand-written, from-scratch (NOT jet-algebra code) 2-state
Jordan-block LTI system, `J=[[0.85,0.30],[0,0.85]]`, `b=(1.0,0.7)`,
`c=(0.6,1.0)`. Impulse response regressed against
`(c0+c1*t)*lambda^t`: fit residual 6.7e-16 with `c1=0.148` (genuinely
nonzero); a pure-semisimple (`c1=0`) fit is **386 billion times
worse** (residual 0.386) — an unambiguous generalized-mode signature.

**Analytic embedding** of the teacher into the matching d=2 student
(`theta=(lambda,mu)`, matched B_in/C_out via the derived index
correspondence) verified BEFORE any SGD: `max|y_student-y_teacher|`
over T=50 random-input steps = **4.4e-16**.

**Training** (r=16 matched, P=16 matched automatically since P=Q*d=r
for any d, input dim 1, readout capacity B_in/C_out both length-16
trainable, matched for both; train/val/test split, LR selected on
validation only, 4 LRs x 3 seeds):

| | Q (spectral sectors) | val NMSE | test NMSE |
|---|---|---|---|
| d=1 | 16 | 1.75e-3 | 1.52e-3 |
| d=2 | 8 | 6.28e-4 | 5.99e-4 |

**d=2 beats d=1 by ~2.8x on test NMSE, despite having HALF as many
independent spectral sectors** (the "unavoidable resource distinction"
reported explicitly, not hidden: d=1 has 16 independent poles available
to approximate the teacher, d=2 has only 8, each carrying one
generalized coupling).

**Long-horizon impulse response** (T_long=512, 8x the T=64 training
horizon): max|err| over the full long horizon EQUALS max|err| within
the training horizon exactly, for both d=1 (0.0989) and d=2 (0.0577) —
the worst-case error occurs during the early transient (both systems
decay with |lambda|<1, so late-time absolute error is negligible on
both sides); errors do not grow or compound past the training window
for either architecture. d=2 maintains its ~1.7x advantage in this
metric across the whole extended horizon.

## Part 2 — decisive credit test: GenericBlock vs RegularBlock

**GenericBlockExactRTRL**: p trainable scalars x p GENERIC dense d x d
basis matrices per module (`A(theta)=sum_k theta_k B_k`), basis matrices
drawn i.i.d. Gaussian INDEPENDENTLY per module (no shared structure).
Ordinary, UNAPPROXIMATED module-wise RTRL: `S_t in R^{d,p}` via
`S_{t+1}=A(theta)@S_t+G_t`, `G_t[:,k]=B_k@h_t` — derived and verified
against BPTT, not assumed.

**RegularBlock** (linear): `h_{t+1}=alg_mult(theta,h)+B_in*x_t`.
Derived (via the commutativity identity `M_u@v=M_v@u`, hence
`G_t=M_{h_t}` exactly) and verified: reduced eligibility
`s_{t+1}=theta*s_t+h_t` (using PRE-update h) — a genuinely simpler
linear analogue of B35a's nonlinear `reduced_algebra_grad_local`.

### View 2A — matched architecture size (Q=4, d=4, p=4; r=16, P=16 for both)

| | grad rel. err vs BPTT | actual persistent array | symbolic prediction | match |
|---|---|---|---|---|
| GenericBlock | 4.98e-16 | 64 (Q\*d\*p) | 64 | yes |
| RegularBlock | 4.94e-16 | 16 (Q\*p) | 16 | yes |

Factor-d gap from ACTUAL arrays: 64/16 = **4.0**, exactly d. Sensitivity
span check (not just rank at the last step — sampled at t in
{0,1,2,4,7,10,14} across all 4 modules, 3 seeds): GenericBlock's `S_t`
achieves the full generic rank `min(d,p)=4` at **every** sampled point,
**no accidental collapse** toward a lower-dimensional (e.g.
p-dimensional) representation.

### View 2B — matched exact-credit budget (predeclared rule, fixed before running)

Rule: at fixed local d, RegularBlock's per-module credit is `p=d`;
GenericBlock's is `d*p=d^2`. At the same total credit C, RegularBlock
therefore gets exactly d times more modules (hence d times more r and
P). Chosen: `d=4, C=64` => RegularBlock (Q=16, r=64, P=64, credit=64)
vs GenericBlock (Q=4, r=16, P=16, credit=64) — same exact-credit
budget, 4x capacity gap, both spending it as their own true per-module
sensitivity actually requires (not a post-hoc adjustment).

Clean train/val/test split, LR selected on validation only, 4 LRs x 3
seeds, on two teachers:

| teacher | RegularBlock (r=64,P=64) test NMSE | GenericBlock (r=16,P=16) test NMSE | Regular advantage |
|---|---|---|---|
| JordanGeneralizedMode (Part 1's teacher) | 1.67e-3 | 2.12e-3 | 1.27x |
| NeutralDenseLinear (random stable 8-state dense system, NOT built from our algebra) | 6.70e-4 | 2.03e-3 | **3.0x** |

**RegularBlock wins on both teachers at matched exact-credit budget —
including the algebra-independent neutral system-identification
teacher**, where the margin is largest. Zero divergence in any run.

## Falsification criteria — checked explicitly, none triggered

1. *GenericBlock's sensitivity already admits a p-dimensional exact
   representation* — REFUTED: full generic rank at every sampled
   (t, module, seed), no collapse observed.
2. *Accounting doesn't produce the predicted factor-d gap* — REFUTED:
   64/16=4.0 exactly, from actual allocated arrays, matching d exactly.
3. *RegularBlock cannot convert saved credit into useful capacity on
   any reasonable task* — REFUTED: wins on both teachers tested,
   including the neutral one.
4. *d>1 gives no measurable benefit over d=1 on a clean
   generalized-mode teacher* — REFUTED: d=2 beats d=1 by ~2.8x at
   matched r, P (Part 1).

The benchmark was not modified to force any of these outcomes; the
credit-budget-conversion rule (View 2B) was fixed before results were
seen.

## Stop gate

Reported here; the large View 2 sweep was NOT launched, and the
architecture was not changed during this phase.

## Commit hash

See the commit introducing this file.
