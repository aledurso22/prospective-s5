# Phase B31a — correctness gate for jointly-trained recurrent-Jacobian families, r=64, d=4

Branch `S5-CCM-scale-validation`. Extends B30a/B30b's flag architecture:
`R_U, D_U` remain fixed/untrained; `R_V, K, B_V, C_V, C_U` (previously
fixed structural matrices) become PART of theta, trained jointly with
Phi. Code: `credit_memory/b31a_joint_family_correctness.py`
(`main()` reproduces every number below). No training here (B31b).

**Headline: every one of the six jointly-trained families (R_V, K,
B_V, C_V, C_U, Phi) is structurally V-valued — none appears in
`u_{t+1}`'s formula, so their direct sources have identically zero
U-rows for any parameter values, and the invariant subspace survives
arbitrary (not just initialization) parameter draws exactly.**

## 1. Per-family accounting

| family | shape | P_f | full (64×P_f) | reduced (4×P_f) | V-valued |
|---|---|---|---|---|---|
| R_V | (4,4) | 16 | 1,024 | 64 | YES (structural) |
| K | (4,60) | 240 | 15,360 | 960 | YES (structural) |
| B_V | (4,8) | 32 | 2,048 | 128 | YES (structural) |
| C_V | (8,4) | 32 | 2,048 | 128 | YES (structural) |
| C_U | (8,60) | 480 | 30,720 | 1,920 | YES (structural) |
| W1 | (560,9) | 5,040 | 322,560 | 20,160 | YES (structural) |
| b1 | (560,) | 560 | 35,840 | 2,240 | YES (structural) |
| W2 | (8,560) | 4,480 | 286,720 | 17,920 | YES (structural) |
| b2 | (8,) | 8 | 512 | 32 | YES (structural) |
| **D_V** | — | **0** | — | — | not present in this implementation (v_{t+1} has no direct x_t term); not fabricated |

**Total P_c = 10,888. M_full = 64·P_c = 696,832 floats (5.575 MB).
M_reduced = 4·P_c = 43,552 floats (0.348 MB). Ratio = 16.00x.**

"V-valued" is not just asserted — each family appears *only* inside
the `v_{t+1}` equation; none appears anywhere in `u_{t+1}=R_U u_t+D_U x_t`.
So the direct source `G_t=d(s_{t+1})/d(family)|_{s_t fixed}` has
identically zero U-rows for every value the family takes, a fact about
the functional form, not the specific numbers.

## 2. Correctness suite (5 seeds × T∈{1,5,20,50}, joint theta over all 9 leaves)

| quantity | value |
|---|---|
| worst full-vs-BPTT relative error | **1.983e-15** |
| worst reduced-vs-BPTT relative error | **1.983e-15** (identical to full's) |
| reconstructed-sensitivity error (S_recon vs full S_t) | **exactly 0** |
| full RTRL's U-rows of S_t, any sampled t | **exactly 0** |
| `ALL < 1e-8` | **True** |

**Per-family direct-source U-rows check** (numerical confirmation of
the analytical claim above, computed by isolating each family's own
Jacobian slice at 3 sampled timesteps): worst leak over all
(t, family) pairs = **exactly 0**.

## 3. Invariance survives arbitrary parameter values, not just initialization

12 independently-drawn "post-update-like" theta values (fresh random
draws spanning `R_V`'s spectral radius from 0.1 to 0.95, not merely
the initialization distribution), each with a freshly sampled state:
max `‖(I−PP⁺)J_tP‖` = **exactly 0** across all 12. This demonstrates
the invariance is a structural consequence of the parameterization
(`u_{t+1}` never reads `v_t` or any of R_V/K/B_V/C_V/C_U/theta) and
therefore survives arbitrary optimizer updates to these families — not
an incidental property of the specific initialization tested in B30a.

## 4. Commit hash

See the commit introducing this file.
