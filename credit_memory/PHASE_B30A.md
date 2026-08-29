# Phase B30a — general flag-SSM reduced-RTRL correctness/scaling test, r=64, d=4

Branch `S5-CCM-scale-validation`. Standalone, independent of B28/B29 —
no training, no runtime optimization, no LRU comparison. Code:
`credit_memory/b30a_flag_r64d4_test.py` (`main()` reproduces every
number below).

**Headline: the general (non-hand-derived) reduced-module algorithm
passes at machine precision on a genuinely larger, generically-coupled
64-dim system with 10,088 trainable parameters, giving an exact 16x
reduction in persistent sensitivity storage.**

## 1. Setup

`T = V ⊕ U`, `dim V = d = 4`, `dim U = 60`, `r = 64`:
```
u_{t+1} = R_U u_t + D_U x_t                        (U: pure linear, NO theta, NO v)
v_{t+1} = R_V v_t + K u_t + B_V Phi_theta(C_V v_t + C_U u_t, x_t)
```
`theta` = Phi's own MLP weights only (`W1,b1,W2,b2`, a 9→560→8 network,
`P_c=10,088`); `R_U,D_U,R_V,K,B_V,C_V,C_U` are all fixed/untrained.
Unlike B29, the flag here needs no eigenvector argument: `u_{t+1}`
never reads `v_t` or `theta`, so `D_theta u_t = 0` for all `t` by
induction — the full sensitivity is confined to V purely because of
the one-directional (U→V, never V→U) block-triangular coupling. The
reduced path restricts the SAME `jax.jacobian`-based RTRL machinery to
V's own dynamics (`u_t` injected as a precomputed, theta-free value) —
a general restricted-operator algorithm, not a hand-derived closed
form (contrast B29's scalar recursion).

Three independent gradient paths: BPTT (`jax.grad` through
`jax.lax.scan`), full exact RTRL (`S_{t+1}=J_tS_t+G_t`, full 64×64/
64×`P_c` Jacobians via autodiff), reduced exact RTRL (`E_{t+1}=J_{V,t}E_t+G_{V,t}`,
4×4/4×`P_c` Jacobians of `v_step` alone via autodiff).

## 2. Correctness results (5 seeds × T∈{1,5,20,50})

| quantity | value |
|---|---|
| r, d | **64, 4** |
| P_c | **10,088** |
| worst full-vs-BPTT relative error | **9.749e-16** |
| worst reduced-vs-BPTT relative error | **9.749e-16** |
| reconstructed-sensitivity error (S_recon vs full S_t) | **exactly 0** |
| full RTRL's U-rows of S_t, any sampled t | **exactly 0** |
| persistent sensitivity, full | **645,632 float64 scalars = 5.165 MB** |
| persistent sensitivity, reduced | **40,352 float64 scalars = 0.323 MB** |
| reduction ratio | **16.00x** (target 64/4=16x) |

All 20 individual settings land at ~1e-15–1e-16 relative error — the
reduced/full agreement here is exact to the bit (not merely close),
because both paths use the same underlying `jax.jacobian` machinery
restricted to a smaller domain, not two independently-derived formulas.
No T=100 point was run (T≤50 already decisive; T=100 would add compute
without changing the conclusion, per explicit instruction).

## 3. Structural diagnostics

| diagnostic | value |
|---|---|
| forward reachable rank from x (linear skeleton, seeded at `[D_U;0]`) | **64/64** |
| invariant residual `‖(I−PP⁺)J_tP‖` over 8 sampled states | **exactly 0** |
| nonzero commutator `‖[A_lin, Q]‖_F` (Q = nonlinear-correction contribution to J_t) | **0.3015** |
| evidence against a fixed direct sum: `‖K‖_F`, `‖C_U‖_F`, `‖d(v_10)/d(u_0)‖_F` | 0.294, 1.369, 0.0931 (all nonzero) |

**Common-invariant-complement diagnostic** (added before finalizing the
"no fixed direct-sum" claim, per explicit instruction not to overstate
it without this check): any invariant complement to V would be a graph
`{(u,Lu)}`, requiring one `L` (4×60) solving `L A_U − A_{V,t}L = B_t`
simultaneously across sampled states (`A_U=R_U`, time-invariant, since
`u_{t+1}` never reads theta or v). Stacked least-squares over n=8/16/32
sampled states:

| n_samples | normalized residual |
|---|---|
| 8 | 0.1052 |
| 16 | 0.1064 |
| 32 | 0.1118 |

Residual is clearly nonzero and stable as n grows — **no common
invariant complement was found for this sampled Jacobian family**.
This is a finite-sample finding, not elevated to a universal proof of
indecomposability (had the residual been near numerical zero, that
would have meant a common complement might exist and the "not a fixed
direct sum" language would have needed retracting instead).

## 4. Commit hash

See the commit introducing this file.
