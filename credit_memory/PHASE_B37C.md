# Phase B37c — native ProductLocal parameterization

Branch `S5-CCM-scale-validation`. B37b is frozen: its results are **read**
from `results/b37b/*.json`, never recomputed. Purpose is diagnostic —
**does removing the global quotient chart remove the B37b conditioning
failure?** — not another universality experiment.

Code: `b37c_productlocal_native.py` (algebra + model + training),
`b37c_exactness.py`, `b37c_sweep.py`, `b37c_intrinsic_conditioning.py`,
`b37c_analyze.py`. Logs `/tmp/b37c_{exactness,sweep,intrinsic,analysis}.log`,
results `results/b37c/rows.json`.

## 1. Fixed native algebra

Per teacher, π is chosen **once** from the teacher's real-Jordan **block
types only** (a real block of size n → factor `(R,n)`; a conjugate pair of
size-n blocks → factor `(C,n)`); no eigenvalue or eigenvector is used to
build π or to initialize the primary arm.

    A_π = ∏_{j=1}^{J} K_j[ε_j]/(ε_j^{d_j}),   K_j ∈ {R, C},   r = Σ_j d_j dim_R K_j

`r = Σ_j d_j dim_R K_j` verified = teacher dimension for all 18 cases.
Realized partitions: `R1×r` (diagonalizable families), `C1×(r/2)`
(complex_conjugate), `R4`/`R8` (exact_jordan), `R2×(r/2)`
(multi_jordan_shared). Local bases and multiplication tables are fixed
structural constants. **No `H(q)` CRT transform, no trainable global `q`,
no companion matrix, no root extraction, no Vandermonde or Hermite
system, no runtime basis change, and nothing of that kind is
differentiated through.** Complex factors are realified directly into
real coordinates.

## 2. Trainable model

`u ∈ A_π`, `b_c ∈ A_π` per input channel, dense `C_out`:

    z_{t+1} = u z_t + Σ_c b_c x_{c,t},   y_t = C_out z_{t+1}

(the `y`-from-`z_{t+1}` convention is B37b's, kept for comparability;
m = 1 here). Local multiplication `(a_j b_j)_n = Σ_{ℓ≤n} a_{j,ℓ} b_{j,n−ℓ}`.

A structural consequence worth stating: each `M_{u_j}` is lower-triangular
Toeplitz, so **ρ(M_u) = max_j |u_{j,0}|** — stability depends only on the
constant terms. Verified against `eigvals` to < 1e-12 on all 18 specs.
This is the structural reason the B37b coefficient→pole sensitivity
cannot arise here; it is a property of the parameterization, not a
projection or regularizer (none was added).

## 3. Exact eligibility verification (before training)

    e^u_{t+1}   = u e^u_t + z_t,              e^u_0 = 0
    e^{b_c}_{t+1} = u e^{b_c}_t + x_{c,t} 1_A,  e^{b_c}_0 = 0
    ∇_u ℓ_t = M_{e^u_t}^T q_t,  ∇_{b_c} ℓ_t = M_{e^{b_c}_t}^T q_t,  ∇_{C_out} ℓ_t = (∂ℓ/∂y_t) z_t^T

Three independent paths (reduced traces / dense full RTRL / autodiff),
all 18 family×r cases, T=25:

| check | worst relative error |
|---|---|
| `M_a M_b = M_{a*b}` | 3.92e-17 |
| `M_a v = a * v` | 5.10e-17 |
| reduced vs autodiff (u, b, C_out) | 9.98e-17 |
| reduced vs full RTRL | 1.21e-17 |
| **the reduction itself: `M_{e_t} = S_t`** | 1.07e-16 (exactly 0 for 14/18) |

**Worst across all paths 1.07e-16 — PASS** (preregistered 1e-10).

Measured storage `P_dyn = (m+1)r = 2r` (8 at r=4, 16 at r=8) against
`(m+1)r² ` for full RTRL — an **r×** reduction. Multiplication cost
`Σ_j d_j(d_j+1)/2 · (4 if C)` = 4–36 flops vs `r²` = 16–64 for a dense
`M @ S`. Reported as measured; no kernel optimization attempted.

## 4. Critical control — structural capability first

Before any trainability claim, each π was verified capable of the
teacher's Jordan type: with `u_j = λ_j + ε_j`, `M_{u_j}` **is** the real
Jordan block, and `‖M_u Φ − Φ J‖ = 0.00e+00` **exactly** for all 18
cases, with `‖AT − TM‖/(1+‖A‖‖T‖) ≤ 3.8e-14`. No failure below is a
partition-capability failure.

## 5. Result — the global-chart failure is gone

Divergence rate over all exact/near-exact initialization arms:

| | divergence rate |
|---|---|
| B37b global quotient chart | **0.509** |
| B37c native ProductLocal | **0.000** |

Zero divergence in **every** family, and zero in the primary generic arm
too. Held-out NMSE at ε=1e-2 perturbation of the exact realization
(the B37b catastrophe):

| family | r | B37b | B37c |
|---|---|---|---|
| random_stable_diag | 8 | nan (diverged) | **1.76e-08** |
| distinct_real | 8 | inf | **5.06e-08** |
| repeated_poles | 8 | nan | **6.32e-08** |
| multi_jordan_shared | 8 | inf | **1.93e-09** |
| stiff | 8 | nan | **1.66e-08** |
| nonnormal | 8 | 5.87e+31 | **3.76e-06** |

At ε=1e-6, where B37b already lost most families (up to 9.0e+106), B37c
stays at 1e-21…1e-12 everywhere. **The avoidable global-chart component
was real and has been removed.**

## 6. Result — nonnormal still fails, and fails differently

| r | arm | NMSE | Markov | ρ(M) | Γ_H | max\|z\| | ‖b‖ | ‖C_out‖ | div |
|---|---|---|---|---|---|---|---|---|---|
| 4 | generic (4000) | 9.69e-01 | 1.82e+02 | 0.958 | 0.96 | 1.05e+02 | 2.8e+01 | 2.6e+01 | 0.00 |
| 4 | exact | **7.00e-15** | 1.66e-06 | 0.900 | 0.90 | 2.39 | 1.39 | **5.40e+05** | 0.00 |
| 8 | generic (4000) | 8.47e-01 | 2.66e+02 | 0.990 | 0.99 | 1.11e+02 | 4.2e+01 | 4.2e+01 | 0.00 |
| 8 | exact | **1.17e-14** | 1.51e-06 | 0.900 | 0.90 | 0.86 | 0.54 | **6.02e+05** | 0.00 |

The failure mode has changed qualitatively. In B37b the exact
realization *drifted away and blew up*; in B37c it is **reached and
held** (1e-14, no divergence, no drift). What remains is purely a
search failure from a generic initialization: validation loss plateaus
at ~1e8–1e9, gradient norms stay at 1e5–1e6 (vs ~1e-2 for every
successful family), and `max|z|` sits at ~110 vs ≤ 20 elsewhere.

**Model-independent conditioning of the teachers** (`b37c_intrinsic_conditioning.py`,
nothing here depends on the parameterization):

| family | r | Γ_H = max_k‖A^k‖₂ | peak \|g_k\| | required ‖C‖/‖b‖ |
|---|---|---|---|---|
| all other families | 4, 8 | 0.41 – **153** | 0.3 – 176 | 0.02 – 4.2e4 |
| **nonnormal** | 4 | **7.42e+04** | **5.36e+04** | 3.89e+05 |
| **nonnormal** | 8 | **4.63e+04** | **1.55e+04** | 2.98e+06 |

nonnormal is the unique family whose *transfer-level* transient gain is
~1e4 despite ρ(A)=0.90 — a 300× gap to the next largest (exact_jordan
r=8, 153) — and it is the unique failure. The exact realization needs an
output port 5–6 orders larger than its input port (this is `cond(T)=1e6`,
i.e. the teacher's own similarity conditioning, surfacing as a port norm);
the generic arm never finds that asymmetry, settling at ‖b‖≈‖C‖≈30–40.

Honest statistics: rank correlations across the 18 cells are weak —
`ρ_S(log NMSE, log Γ_H) = +0.311`, `ρ_S(log NMSE, log required ‖C‖/‖b‖)
= +0.296` — because 16 of 18 cells succeed and the ranks among successes
are noise. The evidence is **separation, not correlation**: on Γ_H and
peak |g_k| nonnormal is an isolated outlier by 2–3 orders of magnitude.
Note also that large required port asymmetry alone does **not** predict
failure (`nearly_defective` needs ‖C‖/‖b‖ = 4.2e4 and trains to 5.5e-07);
the large *transfer-level transient gain* is what distinguishes nonnormal.

## 7. Ordinary families — negative results preserved

On the primary generic arm at 4000 steps B37c is a wash, not a uniform
win: better on 6 cells (stiff r=4 **473×**, exact_jordan r=8 **160×**,
multi_jordan_shared r=4 **137×**), comparable on 5, and **worse on 5**
(random_stable_diag r=4 8.9e-06 vs 2.2e-06 and r=8 3.2e-05 vs 3.8e-06;
distinct_real r=8 5.9e-04 vs 4.3e-05; repeated_poles r=4 5.0e-05 vs
7.1e-06; nearly_defective r=4 4.1e-06 vs 7.1e-07). All losses are
factors of 4–15× at NMSE levels of 1e-4…1e-6, with zero divergence
either way. Recorded as found; not tuned away.

## 8. Verdict

- **Exact credit**: verified to 1.07e-16 on three paths; `P_dyn=(m+1)r`.
- **Structural capability**: `M_uΦ − ΦJ = 0` exactly for all 18 cases.
- **Avoidable chart component removed**: divergence 0.509 → 0.000;
  ε=1e-2 perturbations go from inf/1e+31 to 1e-9…1e-6.
- **Residual failure**: nonnormal only, and only from generic
  initialization — the exact realization is now reachable and stable.
  It correlates with the model-independent transfer/port transient gain
  (Γ_H ≈ 5e4, peak |g_k| ≈ 1.5e4, required ‖C‖/‖b‖ ≈ 1e6).

This matches the second preregistered branch. Per §6 no universality is
claimed: for fixed π the model covers **one real-Jordan stratum plus its
degenerations, not all M_r(R)**; the universal one-quotient theorem
(B37a) remains a separate result. No stability projection, regularizer,
balancing, adaptive basis, alternative optimizer, or further
architectural fix was added, and no further intervention follows.

**PRODUCTLOCAL HELPS BUT INTRINSIC CONDITIONING REMAINS**

## Commit hash

See the commit introducing this file.
