# Phase B35a — bounded product-local commutative response algebra

Branch `S5-CCM-scale-validation`. Motivated by Phase-2A View 2's medium
tier, where B34's own positive control (B34->B) diverged 6/6 at r=500
with Adam+clipping — a symptom, not the cause. Investigation (preserved
as evidence, not further modified):
`credit_memory/p2a_b34_stability_audit.py` (`/tmp/p2a_view2_PARTIAL_AUDIT_ARTIFACT.log`)
showed the untrained TEACHER's own forward dynamics explode with r
(max|h_t|: 0.58 -> 128 -> 2.1e5 -> **1.4e16** at r=64/200/500/800).
`credit_memory/p2a_b34_dimension_norm_audit.py` traced this to an exact
algebra identity, `||M_a||_inf = ||a||_1`, verified numerically: every
r-length coefficient vector in a single big jet (theta, and every
generated coefficient a_t/b_t/kappa_t/c_t) participates in ONE
length-r convolution operator, so a fixed per-coefficient variance
gives an aggregate L1 gain that grows with r; 1/sqrt(r) rescaling still
drifts (confirmed empirically), only ~1/r-type control keeps it O(1)
for theta/kappa specifically, and even then a_t's own unscaled tail
left residual growth — establishing this as a construction-level
scaling defect of the single-long-jet parameterization, not an
optimizer-tuning problem. Full single-long-jet View 2 was killed after
its medium tier and never rerun; its partial log is preserved as
`/tmp/p2a_view2_PARTIAL_AUDIT_ARTIFACT.log`.

B35a's fix: bound the local factor size d structurally (a direct
product of small commutative rings, `A = product_{q=1}^Q R[eps_q]/(eps_q^d)`)
instead of rescaling coefficients as a function of total r.

Code: `credit_memory/b35a_product_local_algebra.py` (core blockwise
algebra + model + per-factor stability projection),
`b35a1_exactness.py`, `b35a2_credit_and_scaling.py`,
`b35a4_representation_teachers.py` (`/tmp/b35a4_results.json`).

## 1. Architecture

Element = flat `(r,)=(Q*d,)` vector, reshaped to `(Q,d)`. Multiplication
is BLOCKWISE: `(a*b)[q]` = length-d truncated convolution within
factor q (b34a's exact Toeplitz regular-rep, reused at size d);
cross-factor multiplication is EXACTLY ZERO (a direct product of rings
has componentwise multiplication — exact, not an approximation). The
whole-algebra regular representation `M_a` is BLOCK-DIAGONAL:
`blockdiag(M_{a_1},...,M_{a_Q})`. `d=1` is the semisimple/diagonal
(RTU-like) endpoint; `d=r` (Q=1) is the old single long jet; `d in
{2,4,8}` are the new bounded product-local candidates.

**Stability design (the actual fix):** every A-valued quantity that
enters the recurrent multiplier — theta (the sole trainable A-valued
parameter, m=1) AND every generated coefficient (a_t, b_t, kappa_t,
c_t) — splits per factor into a semisimple/base component (index 0,
clipped to a fixed range) and a nilpotent tail (index>=1, L1-capped to
`RHO_NIL=1.0` via a rescale-only-if-exceeding projection, same style as
the flag's `R_V` spectral projection). Applied identically regardless
of Q — adding factors never changes any per-factor constant. The
projection is applied to theta after every optimizer step, so the
constraint holds during training, not only at init.

## 2. B35a-1 — exactness (ALL PASS, machine precision)

Tested (Q,d) in {(4,1),(4,2),(4,4),(2,8),(1,8),(1,16)}:

- `alg_mult_blockwise == M_a @ b` (explicit block-diagonal matrix): 0
  to 1e-16.
- `M_a @ M_b == M_{a*b}`: 1e-16 to 1e-17.
- `J_t = M_{u_t}` (u_t = A_theta_t) and `G_t = M_{g_t}`
  (g_t = d_t*(kappa_t*h+b_t)): 1e-17 to 1e-16; **off-block entries of
  J_t, G_t verified exactly 0** (not just assumed from the theory).
- reduced RTRL == full RTRL == BPTT: relative error <= 7.97e-16.
- 10-step reduced-RTRL vs BPTT optimizer trajectory (Q=16,d=4,r=64):
  grad discrepancy ~1e-15, param discrepancy ~2e-16 to 2e-17.

## 3. B35a-2 — actual credit accounting (m=1: theta)

Verified from REAL allocated arrays (actual trainable-theta array size,
actual persistent-`s`-array size from `reduced_algebra_grad_local`),
not formulas alone, at r=64/200/500/800 (d=4 fixed): actual == symbolic
== r in every case; `P=r`, persistent credit=`r`=`P`, generic
sensitivity=`r*P`=`r^2`, ratio=`r`. **Identical accounting to the old
single jet** — product-local changes stability/compute, not the
compression ratio.

## 4. B35a-3 — scaling (d=4 fixed, r=64->800)

| r | max\|h_t\| | RMS\|h_t\| | max_q‖M_θq‖_inf | max_q‖M_Aθq‖_inf | nonfinite | reduced-step time |
|---|---|---|---|---|---|---|
| 64 | 0.101 | 0.024 | 1.121 | 0.769 | 0 | 6.9μs |
| 200 | 0.221 | 0.028 | 1.351 | 1.527 | 0 | 10.7μs |
| 500 | 0.171 | 0.028 | 1.388 | 1.276 | 0 | 22.0μs |
| 800 | 0.172 | 0.026 | 1.519 | 1.132 | 0 | 30.5μs |

Compare to the old single jet at the same r (from the dimension-norm
audit, condition A): max|h_t| = 0.58 -> 128 -> 2.1e5 -> 1.4e16. All
three scaling hypotheses hold: local operator gain stays
dimension-independent (~0.8-1.5, flat, not growing with Q); reduced
update runtime grows ~4.4x over a 12.5x range in r (sub-O(r), not
O(r^2)); persistent storage remains exactly O(P)=O(r) (section 3).

**Nuance, checked and reported precisely rather than glossed over:** an
isolated Q=1 (single big factor) test with the SAME per-factor L1 cap
applied stayed stable (max|h_t| ~0.05-0.10) even at d=500. So **the L1
cap is what fixes forward-dynamics stability; the bounded-d product
structure is what fixes compute cost** (confirmed O(r)-ish vs the old
O(r^2)) — two separate benefits of the construction, not one unified
mechanism, and the report should not conflate them.

## 5. B35a-4 — representation-teacher sanity (r=64, modest scale)

RTU / old single jet / product-local d=2,4,8, on B_long (old
single-jet teacher, `make_teacher_B_jet`), C_multi (old independent
multi-pole teacher, `make_teacher_C_multipole`), and a NEW `E_local`
teacher (product-local Q=16,d=4, four distinct semisimple base sectors
+ nonzero nilpotent tails within every sector — requires both spectral
diversity AND within-sector generalized coupling). Shared-h0/W
(View-1-style) training, light train/val split (no held-out test set),
small LR grid, 2 seeds — a sanity check, not a rigorous benchmark.

| | B_jet | C_multipole | E_local |
|---|---|---|---|
| RTU | 3.078 | **0.349** | 5.326 |
| OldSingleJet | **0.028** | 0.937 | 0.700 |
| ProductLocal d=2 | 1.573 | 2.968 | **0.387** |
| ProductLocal d=4 | 1.750 | 2.833 | 1.026 |
| ProductLocal d=8 | 2.288 | 3.343 | 2.836 |

Zero divergence anywhere (itself notable next to the old single jet's
View-2 medium-tier failure). All three hypothesized column-winners
hold: old single jet wins B_long (exact functional match, expected);
RTU wins C_multi; a product-local variant (d=2) wins E_local outright
(beats RTU by ~14x, the old jet by ~1.8x). Reported without smoothing
over two honest caveats: it is d=2, not d=4 (E_local's own d), that
wins E_local, and performance monotonically worsens with d across all
three teachers in this run — plausibly an untuned-LR-grid /
optimization-difficulty effect rather than a representational one (no
per-architecture LR tuning was done, matching View 1's early-run
caveat), not evidence the effect is architecture-forced.

## 6. Stop gate

Exactness passes; scaling measurements in hand; B/C/E sanity in hand.
View 2 was NOT restarted; S5/LRA untouched. `/tmp/p2a_view2_PARTIAL_AUDIT_ARTIFACT.log`
and the dimension-normalization audit are preserved as the evidence
motivating this architectural transition.

## 7. Commit hash

See the commit introducing this file.
