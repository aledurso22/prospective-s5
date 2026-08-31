# Phase B37a — universal quotient recurrence: exactness verification

Verification only. No training, no optimization, no FFT polynomial arithmetic
(direct O(r^2) convolution + synthetic division throughout).

Code: `credit_memory/b37a_universal_quotient.py` (main suite),
`b37a_exact_rational_check.py` (exact-arithmetic proof),
`b37a_isolate_smallest_failure.py` (isolation). Logs `/tmp/b37a_full.log`,
results `/tmp/b37a_results.json`.

## 1. Exact equations used

Algebra `A = R[x]/(q_a)`, `q_a(x) = x^r + sum_{j=0}^{r-1} a_j x^j` (monic),
`u_theta(x) = sum_{k=0}^{r-1} theta_k x^k`. Per step, with `x_t in R^m`,
`B in R^{r x m}`:

```
u_theta * z_t = q_a * v_t + r_t          (exact polynomial division)
z_{t+1}       = r_t + B x_t
```
`deg(u z) <= 2r-2` so `deg(v_t) <= r-2` (stored length-r, `v_t[r-1]=0`).

Reduced eligibilities — each a SINGLE algebra element (r scalars), all
initialised to 0:
```
s^theta_{t+1} = rem(u s^theta_t, q) + z_t
s^a_{t+1}     = rem(u s^a_t,     q) - v_t
s^{b_j}_{t+1} = rem(u s^{b_j}_t, q) + x_{j,t} * 1        (1 = e_0)
```
Reconstruction claim (what is under test):
`D_theta z_t = M_{s^theta_t}`, `D_a z_t = M_{s^a_t}`,
`D_{B[:,j]} z_t = M_{s^{b_j}_t}`, where `M_c` is multiplication-by-c
(column i = `rem(x^i c, q)`).

Derivation of the novel `a`-term: from `u z = q v + r`, differentiating at
fixed `u, z` and reducing mod q gives `dr/da_j = -rem(x^j v, q)`, i.e.
`G^a_t = -M_{v_t}`.

## 2. Parameter ordering / companion convention

- `theta[k]` = coefficient of `x^k` in u; `a[j]` = coefficient of `x^j` in q
  (leading `x^r` fixed at 1, NOT a parameter); `B` is `(r, m)`, column j is
  input channel j. Jacobians indexed `[output_coord, param_index]`.
- Coefficient vectors are ASCENDING (index k = coefficient of x^k).
- COLUMN companion: `C_q[m, m-1] = 1` (m=1..r-1), `C_q[:, r-1] = -a`.
  Eigenvalues of `C_q` = roots of q; eigenvalues of `M_u = u(C_q)` = `u(lambda_i)`.

## 3. Tolerances / precision

dtype float64 throughout (`jax_enable_x64`), machine eps = 2.220446e-16.
Preregistered PASS threshold `< 1e-10` on
`max_t ||S_t^reduced - S_t^ref||_F / (1 + ||S_t^ref||_F)`, reported separately
for theta, a, B against BOTH reference paths. Seeds 0 and 1 per case;
T = 40 steps, m = 2 inputs; r in {2,4,8,16,32}; 10 families x 5 r x 2 seeds
= 100 cases. Overflow limit 1e150 (no case overflowed).

Three paths compared: **A** reduced (analytic), **B** full RTRL with
jax.jacobian LOCAL Jacobians (independent of the derivation), **C** end-to-end
BPTT/autodiff (independent of derivation and of accumulation order).

## 4. Headline result — the derivation is EXACT

In **exact rational arithmetic** (float64 values are exactly rational; the
model needs only `+ - *` since q is monic), a minimal forward-mode dual-number
AD of the forward model was compared against the analytic reduced trace:

| r | T | d/dtheta | d/da | d/dB |
|---|---|---|---|---|
| 2 | 8 | **0** | **0** | **0** |
| 4 | 8 | **0** | **0** | **0** |
| 8 | 6 | **0** | **0** | **0** |
| 16 | 5 | **0** | **0** | **0** |
| 32 | 4 | **0** | **0** | **0** |

Differences are **identically zero as exact rationals** — not merely small.
The compressed r-scalar traces reproduce generic differentiation exactly,
including the novel sensitivity to the algebra structure `a` itself.

## 5. float64 behaviour and the preregistered threshold

Against the preregistered `<1e-10`: **70/100 pass vs full-RTRL reference,
73/100 vs autodiff reference; 30 cases FAIL.** No case overflowed.

The failures are floating-point conditioning, not derivation error. The
predictor is the **transient state amplification** `max|z_t|`, not `cond(M_u)`:

| predictor | log-log correlation with error |
|---|---|
| `max\|z\|` | **0.952** |
| `cond(M_u)` | 0.862 |

| `max\|z\|` bucket | pass rate | worst error |
|---|---|---|
| < 1e2 | **58/58** | 3.2e-11 |
| 1e2 – 1e4 | **8/8** | 2.2e-12 |
| 1e4 – 1e6 | 7/11 | 1.5e-09 |
| 1e6 – 1e9 | 0/13 | 1.5e-05 |
| >= 1e9 | 0/10 | 3.2e-02 |

Sharp boundary at `max|z| ~ 1e4–1e6`, exactly the float64 signature: one loses
about `log10(max|z|)` digits from eps = 2.2e-16, so `max|z| ~ 1e6` leaves ~1e-10.

Per family (worst vs autodiff): `lambda_I` 3.2e-11 PASS, `stiff` 1.7e-14 PASS,
`real_distinct` 7.4e-11 PASS (even at cond 6.6e18); `complex_conjugate` 3.6e-10,
`random_stable` 3.8e-08, `repeated_eigs` 2.5e-06, `nearly_defective` 1.5e-05,
`exact_jordan` 1.3e-05, `nonnormal` 1.6e-05, `multi_jordan_shared` 3.2e-02.
The failures are precisely the **defective / nearly-defective / nonnormal**
families — those with large transient amplification — which is the classical
(Wilkinson) weakness of companion/polynomial representations.

## 6. Smallest failing example, isolated against EXACT ground truth

`family=exact_jordan, r=8, seed=0` (`q=(x-lam)^r`, rho(M_u)=0.765054,
cond(M_u)=1.790e7, max|z|=9.14e6). Exact rational forward-mode AD as ground
truth, all three float64 paths measured against it:

| path | worst relative error vs EXACT |
|---|---|
| **REDUCED (this work)** | **1.651e-09** |
| BPTT/autodiff | 3.095e-09 |
| full RTRL (r x r matrix) | 2.962e-06 |

**The reduced trace is the MOST accurate of the three float64 paths** — the
full-RTRL *reference* is ~1800x less accurate than the quantity it was being
used to validate. Much of the apparent "failure" in section 5 is roundoff in
the reference, not in the reduced trace: full RTRL propagates an r x r matrix
through 40 multiplications by an ill-conditioned `M_u`, whereas the reduced
path propagates only r numbers.

## 7. Representation side

**Notation, disambiguated.** `AK = KC_q` and `A = K C_q K^{-1}` are the SAME
equation, not two assertions — the second is the first with `K` inverted. Both
are the `u(x) = x` SPECIAL CASE of the universal criterion `AT = T u(C_q)`,
with `T = K`. `K` is the controllability matrix of the target `A` itself (not
an auxiliary matrix standing in for `A`). The scope limitation is that
`AT = T C_q` (i.e. `u = x`) forces `A` to be **cyclic**, and R1's targets are
random dense matrices, which are generically cyclic — so **R1 alone does not
test universality**; it only confirms the cyclic special case. The derogatory
case is tested in section 7a with nontrivial `u`.

**R1** (`u(x)=x`, controllable canonical form, CYCLIC targets only; `a`
obtained by SOLVING `K a = -A^r b` from Cayley–Hamilton rather than via
ill-conditioned root-finding; `K` = controllability matrix of `A`):

| r | cond(K) | ‖AK−KC_q‖ rel | ‖A−KC_qK⁻¹‖ rel | Markov err |
|---|---|---|---|---|
| 2 | 7.3e0 | 2.3e-17 | 8.2e-17 | 9.3e-17 |
| 8 | 5.5e3 | 1.6e-16 | 7.9e-14 | 1.2e-16 |
| 16 | 1.4e5 | 2.8e-16 | 1.3e-12 | 3.7e-16 |
| 32 | 1.6e9 | 4.1e-16 | 2.1e-09 | 6.4e-16 |
| 32 (sd1) | 6.5e10 | 3.2e-16 | 9.2e-08 | 4.7e-16 |

The identity `AK = KC_q` (equivalently `A = K C_q K^{-1}`; the `u=x` case of
`AT = T u(C_q)`) holds at **machine precision at every r** including 32. Only the explicit inversion `K C_q K^{-1}` degrades, tracking
cond(K) as expected. **Markov parameters `c A^k b = (cK) C_q^k (K^{-1}b)`
match to ~1e-16 over k=0..59 at every r.**

**R2** (nontrivial u, `A := T u(C_q) T^{-1}`): reconstruction error exactly
0.0 at every r; `‖M_u − u(C_q)‖` (polynomial-division construction vs Horner
matrix powers, two independent code paths) 0 to 4.6e-14; Markov error 1.5e-16
at r=2 rising to 2.9e-06 at r=32 (same nonnormal-conditioning limit).

### 7a. CORRECTION — universality holds; the earlier caveat was FALSE

An earlier revision of this document claimed that "not every Jordan structure
is reachable from a single quotient", citing `J_2+J_1+J_1` as unreachable from
`q=(x-alpha)^4`, and concluded that products of quotients are required. **That
claim was wrong and is retracted.** The enumeration behind it was incomplete:
writing `u(C_q) = u(alpha)I + c_1 N + c_2 N^2 + c_3 N^3`, it considered
`c_1=0, c_2!=0` (giving `J_2+J_2`) and `c_1=c_2=c_3=0` (giving `lambda I`) but
**omitted `c_1=c_2=0, c_3!=0`**. That omitted case is exactly the counterexample:

```
q(x) = (x-alpha)^4 ,   u(x) = lambda + (x-alpha)^3   =>   u(C_q) = lambda I + N^3
```
`N^3` has rank 1 and square zero, so its nilpotent type is `J_2(0)+J_1(0)+J_1(0)`,
giving `u(C_q) ~ J_2(lambda)+J_1(lambda)+J_1(lambda)`. Verified numerically:
`rank(u(C_q)-lambda I) = 1`, `||(u(C_q)-lambda I)^2|| = 3.4e-16`, detected Jordan
type `[2,1,1]`.

**Corrected mathematical statement.** `C_q` is always cyclic/nonderogatory —
one Jordan block per distinct root — but `u(C_q)` **need not be**, and this is
the whole point. Every matrix `A` of size r (arbitrarily derogatory) satisfies
`A = T u(C_q) T^{-1}` for a suitable monic `q` of degree r and `deg u <= r-1`.
Constructive proof, with each step verified numerically below:

1. Let `A` have Jordan blocks `{(lambda_i, n_i)}_{i=1..p}`, `sum n_i = r`. Pick
   **distinct** `alpha_i` and set `q = prod_i (x-alpha_i)^{n_i}`.
2. With `g_{i,k} = q(x)/(x-alpha_i)^k` (k=1..n_i) one has
   `x*g_{i,k} = g_{i,k-1} + alpha_i g_{i,k}` in `A=R[x]/(q)` (with `g_{i,0}=q=0`),
   so `V = [... g_{i,k} ...]` gives `C_q V = V D`, `D = (+)_i J_{n_i}(alpha_i)`,
   hence `u(C_q) V = V u(D)` with `u(D) = (+)_i u(J_{n_i}(alpha_i))`.
3. Impose Hermite conditions `u(alpha_i) = lambda_i` for every block, plus
   `u'(alpha_i) = 1` only for blocks with `n_i >= 2`. The condition count is
   `p_1 + 2 p_2 <= p_1 + sum_{n_i>=2} n_i = r`, so Hermite interpolation returns
   `deg u <= r-1` — **inside the model**. This is the step the retracted claim
   missed: size-1 blocks cost only ONE condition, so the budget always fits.
4. Per block, `u(J_n(alpha)) = lambda I + P` with `P` nilpotent of rank `n-1`
   (because `u'(alpha) != 0`); `W = [P^{n-1}e | ... | Pe | e]`, `e = e_n`, gives
   `P W = W N` and hence `u(J_n(alpha)) W = W J_n(lambda)`.

`T = V * blockdiag(W_i)` then satisfies `u(C_q) T = T A` exactly.

**Re-audit results** (`credit_memory/b37a_representation_reaudit.py`), criterion
`||u(C_q)T - TA|| / (1 + ||A|| ||T||)`, plus Markov equality under transformed
ports `B_q = T b`, `C_q = c T^{-1}`:

| target | r | u(C_q) Jordan type | ‖u(C_q)T−TA‖ rel | Markov err | cond(T) | verdict |
|---|---|---|---|---|---|---|
| `lambda*I_4` | 4 | [1,1,1,1] | 5.34e-17 | 3.68e-15 | 2.9e2 | PASS |
| `lambda*I_6` | 6 | [1,1,1,1,1,1] | 1.69e-16 | 1.14e-13 | 2.9e4 | PASS |
| `J2+J1+J1` | 4 | [2,1,1] | 4.83e-16 | 1.88e-15 | 1.1e2 | PASS |
| `J2+J2` | 4 | [2,2] | 7.28e-16 | 3.02e-15 | 3.5e1 | PASS |
| `J3+J2` | 5 | [3,2] | 4.09e-16 | 1.76e-15 | 4.2e2 | PASS |

**ALL 65 equal-eigenvalue partitions for r = 2..8: 65/65 PASS.** Worst
`||u(C_q)T-TA||` rel = 2.15e-12, worst Markov error = 4.33e-09 (both at r=8,
tracking cond(T) up to ~4e8 — floating-point conditioning, not a structural
limit). The independent Jordan-type check confirmed `u(C_q)` has exactly the
target block structure in **65/65** cases.

**Conclusion: ONE quotient suffices for arbitrary derogatory structures.** A
product of quotients is NOT necessary for representation. (A product may still
be preferable for other reasons — e.g. the float64 conditioning of the
companion parameterisation documented in section 5, or bounded local factor
size — but that is a numerical/architectural argument, not a representational
one.)

## 8. Verdict

**The universal quotient model and its compressed trace match generic
differentiation EXACTLY** — proven in exact rational arithmetic, difference
identically zero for theta, a and B at every r in {2,...,32}, including the
novel sensitivity to the algebra structure `a`.

The preregistered float64 threshold `<1e-10` holds on **66/66 cases with
transient amplification max|z| < 1e4** and fails only where float64 cannot
deliver ten digits by ANY method (autodiff included); in the isolated smallest
failing case the reduced trace is the most accurate of the three paths tested.
The 30 nominal failures are a property of the float64 conditioning of the
quotient/companion parameterisation in defective and nonnormal regimes, not of
the reduced-eligibility derivation.

**Representation universality (corrected).** Every matrix of size r, including
arbitrarily derogatory ones, is `T u(C_q) T^{-1}` for a single monic `q` of
degree r and `deg u <= r-1`: verified on all 65 equal-eigenvalue Jordan
partitions for r=2..8 (65/65, worst residual 2.15e-12) plus the required
targets `lambda I`, `J2+J1+J1`, `J2+J2`, `J3+J2`. The earlier "products of
quotients are required" caveat was based on an incomplete case enumeration and
has been retracted — see section 7a.

Stopped here as instructed; no training experiments performed.
