# Phase B29 — reduced exact-sensitivity theorem test, explicit r=4, d=1 flag

Branch `S5-CCM-scale-validation`. Standalone theorem-validation
experiment, independent of B28's Autoencode work — no training, no
environment, no benchmark. Code: `credit_memory/b29_flag_r4d1_test.py`
(`main()` reproduces every number below).

**Headline: the theorem's smallest nontrivial instance passes at
machine precision, and the falsification test confirms exactness
disappears precisely when the invariant-module assumption is broken.**

## 1. Setup

Fixed `R` (upper bidiagonal, 0.7 on the diagonal, 0.2 on the
superdiagonal) and recurrence
`h_{t+1} = R h_t + e4 x_t + e1 tanh(w.h_t + beta*x_t + b)`, trainable
family `theta=(w1,w2,w3,w4,beta,b)` (6 scalars; R, e1, e4 fixed). Claim:
exact sensitivity `S_t=D_theta h_t` (in R^{4x6}) is confined to
`V=span(e1)`, because `R`'s first column is `R e1 = 0.7 e1` — e1 is an
eigenvector of R, which is exactly what keeps `span(e1)` invariant
under `J_t = R + phi'(z_t) e1 w^T`.

Three independent gradient paths:
1. **BPTT** — `jax.grad` through the whole unrolled sequence
   (`jax.lax.scan`).
2. **Full exact RTRL** — `S_{t+1}=J_t S_t+G_t`, with `J_t`/`G_t`
   computed via per-step `jax.jacobian` (autodiff) — independent of
   both BPTT's single reverse pass and the reduced path's hand-coded
   closed form.
3. **Reduced exact RTRL** — hand-derived closed-form scalar-row
   recursion `E_{t+1} = (0.7 + phi'(z_t) w1) E_t + phi'(z_t)[h_t^T,x_t,1]`,
   maintaining `1x6=6` floats instead of `4x6=24`.

Loss: `L = sum_t [sin(q_t.h_t + phi_t) + 0.5(q_t.h_t)^2]` with random
per-timestep readout `q_t` and phase `phi_t`, float64 throughout.
5 seeds × sequence lengths {1, 5, 20, 100}, random `h0`, random inputs,
random `theta` init in a numerically stable range.

## 2. Correctness suite results (R = R_BASE, flag holds)

Worst case over all 20 (seed, T) settings:

| quantity | value |
|---|---|
| worst full-vs-BPTT relative error | **9.27e-15** |
| worst reduced-vs-BPTT relative error | **9.38e-15** |
| worst reconstructed-sensitivity (`S_recon=e1⊗E_t` vs full `S_t`) max\|Δ\| | **1.42e-14** |

All ≪1e-10 (target ≲1e-12). Persistent sensitivity storage: full
`4x6=24` float64 scalars, reduced `1x6=6` float64 scalars — **exact 4x
reduction**.

## 3. Falsification suite (R_eps = R_BASE + eps·e2e1ᵀ, reduced path
frozen to the OLD d=1 recursion)

Sweep `eps in {0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1}` × T in {1,5,20,100}.
Full RTRL stays at machine precision (~1e-16 to 7e-15) for **every**
`eps` — it never assumed the invariant. Reduced-path error is
machine-precision at `eps=0` and becomes large once the invariant is
broken (e.g. relative error ~5e-7 to 2e-8 already visible at
`eps=1e-8`; 21–69% wrong by `eps=0.1`).

**Correct statement of the effect (correcting an overstatement made
in-session before archiving)**: reduced-gradient error grows strongly
with the leakage magnitude `eps`. Longer horizons *allow* leakage to
accumulate, but the printed relative-error values are **not monotonic
in T** — the theorem does not predict monotonicity in T for the
relative gradient error, and the raw numbers (e.g. at `eps=1e-2`:
T=5→0.316, T=20→0.0137, T=100→0.207) confirm this directly. Do not cite
this result as showing monotonic growth with sequence length.

## 4. Structural diagnostics (eps=0), reusing
`algebra_closure`/`krylov_subspace` from `b25_nonlinear_credit.py`

| diagnostic | value |
|---|---|
| forward controllability/reachability rank from e4 | **4/4** |
| generated algebra dim(span{R, e1 w^T}) for representative w | **7/16** |
| nonzero commutator `‖[R, e1 w^T]‖_F` | **0.2057** |
| commutant dimension of {R, e1 w^T} | **1/16** |
| source orbit (Krylov subspace of R seeded at e1) dimension | **1** (= span(e1), confirming the flag structurally) |

## 5. Commit hash

See the commit introducing this file.
