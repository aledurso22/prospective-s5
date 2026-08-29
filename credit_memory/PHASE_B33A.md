# Phase B33a — exact lifted eligibility recurrence / proof-of-principle (full ordinary rank, 2-d dynamic credit)

Branch `S5-CCM-scale-validation`. Code:
`credit_memory/b33a_lifted_eligibility.py` (`main()` reproduces every
number below). Standalone, no training.

**Scope label, stated explicitly per instruction: this is an ABSTRACT
eligibility-system test, not a trainable RNN.** `J_t` and `G_t` are
GIVEN directly as scalar combinations of a fixed involution `K` — they
are NOT derived as `D_h F_theta` / `D_theta F_theta` for any actual
recurrent parameterization. Do not read this as RTRL on a real model.

**Headline: tests a strictly more general claim than B29–B32 — exact
compression does NOT require `rank(S_t) < r`. The eligibility matrix
`S_t` is FULL ordinary matrix rank (64) at essentially every sampled
timestep, while its entire dynamic (time-varying) content lives in a
fixed 2-dimensional lifted subspace `span{I,K}`.**

Central result to preserve, verbatim:

```
rank(S_t) = 64 at essentially every sampled time,  while  S_t = a_t I + b_t K
```

so the exact dynamic eligibility state has only two coefficients
despite full ordinary sensitivity rank. This establishes:

```
rank(S_t) = r   does NOT imply   r*P independent dynamic credit degrees of freedom.
```

The significance of this result is **representational / minimal-state
complexity** — not yet a total-memory-footprint reduction (see §3).

## 1. Construction

`r=P=64`. `K = Q D Q^T`, `D=diag(+-1)` (32 of each sign), `Q` dense
orthogonal (via QR of a random Gaussian matrix) — confirmed `K^2=I` to
~1e-15. Per-step scalars `alpha_t∈U(0.30,0.60)`, `beta_t∈U(-0.20,0.20)`
(stable: eigenvalues `alpha±beta` of the lifted 2×2 transition stay
<1 in magnitude), `gamma_t,delta_t~N(0,0.3²)`. `J_t=alpha_t I+beta_t K`,
`G_t=gamma_t I+delta_t K`, `S_{t+1}=J_t S_t+G_t`, `S_0=0`.

Because `K^2=I`, `span{I,K}` is exactly invariant, giving the closed
2-d recursion `a_{t+1}=alpha_t a_t+beta_t b_t+gamma_t`,
`b_{t+1}=beta_t a_t+alpha_t b_t+delta_t`, reconstructed as
`S_hat_t=a_t I+b_t K`.

## 2. Correctness (5 seeds × T∈{1,5,20,100,1000})

| quantity | value |
|---|---|
| worst reconstruction error `‖S_t−S_hat_t‖` | **8.882e-16** |
| worst query error `‖q_t^T S_t − q_t^T S_hat_t‖` | **1.332e-15** |
| `ALL < 1e-8` | **True** |
| fraction of sampled timesteps with ordinary rank(S_t)=64 | **1.00, at every (T,seed) setting tested, including T=1000** |

`K^2=I` confirmed to ~1e-15 at every seed. The headline is directly
demonstrated: `rank(S_t)=64` (full) essentially always, while the
reconstruction from just 2 scalars (`a_t,b_t`) matches the full 64×64
matrix to machine precision at every one of the 5×5=25 settings tested
(including T=1000).

## 3. Storage accounting (explicit, per instruction — no blanket claim)

| | value |
|---|---|
| Persistent TIME-VARYING dynamic credit, full | r·P = 64·64 = **4,096 float64 scalars** (32,768 bytes) |
| Persistent TIME-VARYING dynamic credit, reduced | **2 float64 scalars** (16 bytes) |
| **Ratio on the dynamic/time-varying axis** | **2,048x** |
| Static/structural storage: K, stored densely | r² = **4,096 float64 scalars** (32,768 bytes) |

**K is NOT part of any forward model in this abstract test** — it
exists solely to enable the lifted-credit trick, so it is not free
overhead and must be counted if nothing else already requires storing
it. Total footprint keeping only the current step's persistent state:
full = 4,096 floats; lifted = 4,096 (K, static, one-time) + 2 (current
`a_t,b_t`) = **4,098 floats — NOT smaller than the full approach once K
is honestly counted**. The scientifically supportable claim is
specifically about persistent **dynamic** credit storage (4,096→2, a
2,048x reduction on that axis alone), explicitly **not** a "2048x total
memory reduction."

## 4. Falsification: `J_t^eps = alpha_t I + beta_t K + eps·R` (R generic, ~99.97% outside span{I,K})

| eps | span_dim{S_1..S_T}, T=100 | span_dim, T=500 | forced-recon rel error, T=500 |
|---|---|---|---|
| 0 | 2 | 2 | 5.173e-16 |
| 1e-8 | 6 | 6 | 5.243e-08 |
| 1e-6 | 6 | 6 | 5.243e-06 |
| 1e-4 | 14 | 14 | 5.241e-04 |
| 1e-2 | 96 | 158 | 5.107e-02 |
| 1e-1 | 35 | 28 | 1.000e+00 (saturated) |

**Clean falsification claim, restricted to eps≤1e-2 where the dynamics
remain numerically interpretable**: at **eps=0, the lifted span stays
exactly 2 at every T tested (5, 20, 100, 500)** — exact 2-d lifted
closure holds indefinitely. For **generic nonzero eps (1e-8 through
1e-2)**, the lifted span leaves dimension 2 and grows, and the forced
(deliberately-blind, B29-style) two-coefficient reconstruction becomes
systematically incorrect, growing with `eps`. **Full recurrence
remains exact throughout**, at every eps (computed correctly — no
approximation was ever applied to path A). No claim is made that span
dimension or relative error is monotonic in T; it is not required to
be, and it visibly is not (see below).

**eps=0.1 row: excluded from structural evidence, not a counter-
example**. At eps=0.1 the raw result is kept in the table above for
completeness, but it must **not** be read as structural evidence: `R`
was drawn with no stability constraint, so `eps·R` at this magnitude
pushes the full recurrence into severe numerical instability —
trajectory values reach ~1e86 by T=500 (`forced_recon_max|d|` literally
hits 9.1e86 at that setting), and the SVD-based numerical-rank estimate
becomes dominated by a few enormous directions, producing the
non-monotone span_dim(T=100→500: 35→28) visible in the table. This is a
numerical-conditioning artifact of an unconstrained large perturbation,
not a structural finding about the lifted-closure theory, and it is
**not** used to support (or undermine) the falsification claim above.
The clean claim — exact 2-d closure at eps=0, growing lifted span and
broken forced-reconstruction at generic eps — rests entirely on the
eps≤1e-2 rows, where the dynamics stay numerically interpretable.

## 5. Commit hash

See the commit introducing this file.
