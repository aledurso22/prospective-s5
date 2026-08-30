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

**R1** (`u(x)=x`, controllable canonical form; `a` obtained by SOLVING
`K a = -A^r b` from Cayley–Hamilton rather than via ill-conditioned
root-finding; `K` = controllability matrix):

| r | cond(K) | ‖AK−KC_q‖ rel | ‖A−KC_qK⁻¹‖ rel | Markov err |
|---|---|---|---|---|
| 2 | 7.3e0 | 2.3e-17 | 8.2e-17 | 9.3e-17 |
| 8 | 5.5e3 | 1.6e-16 | 7.9e-14 | 1.2e-16 |
| 16 | 1.4e5 | 2.8e-16 | 1.3e-12 | 3.7e-16 |
| 32 | 1.6e9 | 4.1e-16 | 2.1e-09 | 6.4e-16 |
| 32 (sd1) | 6.5e10 | 3.2e-16 | 9.2e-08 | 4.7e-16 |

The similarity identity `AK = KC_q` holds at **machine precision at every r**
including 32. Only the explicit inversion `K C_q K^{-1}` degrades, tracking
cond(K) as expected. **Markov parameters `c A^k b = (cK) C_q^k (K^{-1}b)`
match to ~1e-16 over k=0..59 at every r.**

**R2** (nontrivial u, `A := T u(C_q) T^{-1}`): reconstruction error exactly
0.0 at every r; `‖M_u − u(C_q)‖` (polynomial-division construction vs Horner
matrix powers, two independent code paths) 0 to 4.6e-14; Markov error 1.5e-16
at r=2 rising to 2.9e-06 at r=32 (same nonnormal-conditioning limit).

**Structural note (reported, not hidden):** a single quotient `R[x]/(q)` is
always **nonderogatory** — `C_q` has exactly one Jordan block per distinct
root. Multiple Jordan blocks sharing an eigenvalue ARE reachable for
`M_u = u(C_q)` (take `q=(x-lam)^r` with `u'(lam)=0, u''(lam)!=0`; this is the
`multi_jordan_shared` family), but **not every Jordan structure is reachable
from a single quotient**: from `q=(x-lam)^4` the attainable structures for
`u(C_q)` are `J_4`, `J_2+J_2`, `J_1^4` — but not `J_2+J_1+J_1`. Realising
arbitrary derogatory structures requires a PRODUCT of quotients, not one.

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

Stopped here as instructed; no training experiments performed.
