# Phase B37b — trainability of the universal quotient chart

Branch `S5-CCM-scale-validation`. B37a is frozen and untouched (its
primitives `mult_matrix`, `companion` are imported, not modified). This
phase asks only: **can `(q, u, B, C)` actually be learned from data?**
Offline BPTT/autodiff throughout — no online updates, no RTRL, so that
chart conditioning is isolated from the credit-assignment algorithm.

Code: `b37b_quotient_trainability.py` (sweep), `b37b_analyze.py`,
`b37b_root_sensitivity.py`, `b37b_conditioning_cause.py`,
`b37b_armC_long.py`. Logs `/tmp/b37b_{full,analysis,rootsens,cause,armC_long}.log`,
results `results/b37b/{rows,armC_long}.json`.

Nothing new was introduced: no new bases, no factorization, no
ProductLocal, no regularization, no FFT arithmetic, no stability
projection. The arms differ only in initial parameter *values*.

## Setup

Teachers `h_{t+1}=A h_t + B_* x_t`, `y_t = C_* h_t`, SISO, built as
`A = S J_real S^{-1}` with `S`, `J_real` **real** (so `A` is exactly
real; max imag part 0.00e+00) across nine structural families. The
complex similarity `S_c = S Π` to the complex Jordan form is retained
so the exact realization is constructible (`‖A S_c − S_c J‖ ≤ 1.7e-16`).

Student: `z_{t+1} = u_θ(C(a)) z_t + B x_t`, `ŷ_t = C z_t`, trainable
`(a, θ, B, C)`, `r ∈ {4, 8}`. `u_θ(C_q)` is built once per forward pass
with B37a's `mult_matrix` (identical to the per-step `rem(u·z, q)`).
32 train / 16 val / 16 test sequences of length 64, Adam, 400 steps,
LR ∈ {3e-4, 1e-3, 3e-3} **selected on validation only**, 3 eval seeds.
Scoring is input–output equivalence; the teacher's `A` is never a target.

Arms: **A** constructive (exact realization from known Jordan data),
**B** perturbed constructive at relative ε ∈ {1e-6, 1e-4, 1e-2},
**C** generic stable init (`u(x)=x`, `q` with random stable roots).

## Validity (before training)

All 18 (family × r) constructive realizations are exact: intertwiner
residual `‖AT − TM‖/(1+‖A‖‖T‖) ≤ 2.9e-10`, initial NMSE ≤ 1.9e-12,
initial Markov error ≤ 3.1e-6. **Representation is confirmed for every
family, including all defective ones.**

## Result 1 — the constructive chart is not usable under optimization

Held-out NMSE, median over 3 seeds (full tables in `/tmp/b37b_analysis.log`):

| family | r | A_constructive | B ε=1e-6 | B ε=1e-4 | C_generic |
|---|---|---|---|---|---|
| random_stable_diag | 8 | 1.5e-12 | 9.0e+106 | inf | 2.6e-04 |
| distinct_real | 8 | 4.4e-16 | 2.1e+00 | 2.8e+125 | 3.0e-03 |
| repeated_poles | 8 | 7.6e-12 | 6.2e+67 | inf | 3.4e-03 |
| multi_jordan_shared | 8 | 9.0e-16 | 7.6e-01 | 1.9e+204 | 5.4e-04 |
| stiff | 8 | 5.5e-14 | 6.7e+36 | 3.0e+221 | 1.4e-02 |

At r=8 the constructive arms **diverge in 100% of runs** for 7 of 9
families, while the generic arm **never diverges in any run**. Arm A
retains its exact init only because the best-validation checkpoint is
the init; its parameters after 400 Adam steps reach NMSE up to 1e+213.
Part of that drift is generic (Adam's normalized step is full-size at a
zero-gradient point), but the *consequence* is chart-specific: an
`O(lr)` step in `(a,θ)` sends `ρ(u(C_q))` above 1.

## Result 2 — mechanism: coefficient→pole conditioning

`b37b_root_sensitivity.py`, no training. Perturb `(a,θ)` by relative ε
and measure pole displacement `κ = max|Δpole|/ε` and `P(ρ>1)`:

| family | r | κ @ ε=1e-6 | P(unstable) @ 1e-6 |
|---|---|---|---|
| random_stable_diag | 8 | 3.6e+06 | 0.96 |
| repeated_poles | 8 | 2.5e+06 | 0.79 |
| stiff | 8 | 6.1e+05 | 0.62 |
| nearly_defective | 8 | **2.7e+00** | **0.00** |

A **1e-6** relative perturbation of the monomial coefficients moves
poles by up to 2.5 and destabilizes 96% of realizations at r=8. This is
Wilkinson coefficient→root ill-conditioning, appearing as a hard
trainability barrier. Consistent with B37a §5's float64 conditioning.

## Result 3 — cause isolation: it is node placement, not the chart

`nearly_defective` being the *best*-conditioned family (κ=2.7) is the
tell. The B37a-style construction places interpolation nodes `α_i` on a
fixed grid in [0.25, 0.85] regardless of the teacher spectrum, so `u`
must be a high-degree oscillatory interpolant carrying `α_i → λ_i`.
`nearly_defective` has all `λ_i` nearly equal, so its `u` is nearly
constant and benign. Placing `α_i` **at** `λ_i` instead (so `u ≈ x`) —
same architecture, same parameter count, only initial values change:

| family | r | ‖θ‖ grid → at-λ | κ grid → at-λ | P(unst) grid → at-λ |
|---|---|---|---|---|
| random_stable_diag | 8 | 1.7e6 → **1.00** | 4.2e6 → 3.2e4 | 0.96 → **0.00** |
| distinct_real | 8 | 2.5e4 → **1.00** | 3.3e5 → 1.1e2 | 0.12 → **0.00** |
| complex_conjugate | 8 | 2.1e4 → **1.00** | 2.2e4 → 6.0e2 | 0.00 → 0.00 |
| stiff | 8 | 1.2e5 → **1.00** | 7.6e5 → 1.0e5 | 0.62 → **0.00** |

and training from perturbed inits then succeeds where it had diverged
100% of the time (e.g. random_stable_diag r=8, B ε=1e-6: 9.0e+106 →
**1.7e-06**; distinct_real r=8: 2.1e+00 → **2.4e-08**).

**But the repair does not generalize.** `α` must stay distinct, so for
repeated/defective spectra one must either cluster the nodes (Hermite
system becomes singular: `multi_jordan_shared` r=8 and both
`nearly_defective` rows give a numerically singular intertwiner,
cond > 1e14) or spread them (‖θ‖ explodes: `multi_jordan_shared` r=4,
‖θ‖ 1.2e1 → 4.7e6, κ 2.2e3 → 1.0e8, P(unstable) 0 → 1). Neither
placement is well-conditioned for every family — a genuine structural
tension in the monomial chart, not merely a bad choice on my part.

## Result 4 — from a generic stable init the chart trains fine

Arm C's ~1e-3 plateau was optimization budget, not a floor. At 4000
steps (`b37b_armC_long.py`), same init, same LR grid:

| family | r=4 NMSE | r=8 NMSE | gain vs 400 steps | divergence |
|---|---|---|---|---|
| random_stable_diag | 2.2e-06 | 3.8e-06 | 293× / 68× | 0.00 |
| complex_conjugate | 2.0e-06 | 7.9e-06 | 448× / 19× | 0.00 |
| nearly_defective | 7.1e-07 | 9.4e-07 | 3272× / 534× | 0.00 |
| multi_jordan_shared | 6.3e-06 | 3.0e-06 | 213× / 180× | 0.00 |
| repeated_poles | 7.1e-06 | 3.4e-05 | 70× / 102× | 0.00 |
| stiff | 8.8e-05 | 1.8e-05 | 111× / 789× | 0.00 |
| exact_jordan | 6.0e-05 | 1.1e-03 | 254× / 549× | 0.00 |
| distinct_real | 2.6e-04 | 4.3e-05 | 12× / 69× | 0.00 |
| **nonnormal** | **9.6e-01** | **9.9e-01** | **1.0× / 1.0×** | 0.17 |

Still improving monotonically, `ρ(u(C_q))` stays inside the unit disk,
zero divergence for 8 of 9 families. `exact_jordan` r=8 is the weakest
of these: NMSE 1.1e-3 but Markov error 1.48 — it fits the training
input distribution while its impulse response is still wrong at some k.

## Result 5 — correlation with the B37a transient statistic

Spearman over all 270 runs: failure correlates with the B37a
transient-amplification statistic, but only moderately, and it predicts
*divergence* better than final accuracy:

- `ρ_S(diverged, log max|z| at init)` = **+0.511** (n=263)
- `ρ_S(diverged, log cond(T))` = +0.371
- `ρ_S(log NMSE, log max|z| at init)` = +0.277
- `ρ_S(log NMSE, log cond(T))` = +0.287
- `ρ_S(log NMSE, log cond(S) of teacher)` = +0.121
- `ρ_S(log NMSE, ρ(A) of teacher)` = +0.185

So the B37a statistic is a real but partial predictor: it flags the
blow-up risk, not the achievable fit.

## Verdict by teacher family

Classification: **LEARNABLE** = generic stable init reaches NMSE < 1e-2
at both r with zero divergence, and a near-exact init is also usable;
**LEARNABLE (init-fragile)** = generic init works but the constructive
realization is unusable under optimization; **CONDITIONING-LIMITED** =
fails from every init tested.

| family | verdict |
|---|---|
| complex_conjugate | LEARNABLE |
| exact_jordan | LEARNABLE (weakest; Markov error 1.48 at r=8) |
| random_stable_diag | LEARNABLE, init-fragile |
| distinct_real | LEARNABLE, init-fragile |
| repeated_poles | LEARNABLE, init-fragile |
| multi_jordan_shared | LEARNABLE, init-fragile |
| nearly_defective | LEARNABLE, init-fragile |
| stiff | LEARNABLE, init-fragile |
| **nonnormal** | **CONDITIONING-LIMITED** |

`nonnormal` (cond(S)=1e6, cond(T)=3e11) is the one genuine failure: its
exact realization exists (arm A NMSE 1.8e-12) but no initialization
tested learns it, at either r, at any budget — NMSE stays ≈ 1.0 with
Markov error ≈ 12.5. The representation exists and is not reachable.

## Interpretation

"Representation exists" and "representation is practically learnable"
come apart here, and they come apart in **both** directions, which was
the point of the phase:

1. The chart is **learnable from a generic stable init** for 8 of 9
   families — better than expected, with zero divergence in 108 runs.
2. It is **not usable from the constructive realization** at r=8: a
   1e-6 relative perturbation is already fatal for most families. If
   one ever wants to warm-start from a known realization, the naïve
   monomial/companion chart cannot carry it.
3. The dominant cause is **coefficient→pole conditioning driven by
   interpolation-node placement**, not the quotient recurrence. Placing
   nodes at the target eigenvalues repairs it completely for
   well-separated spectra and breaks it for repeated/defective ones, so
   no single fixed placement rule suffices.
4. One family (`nonnormal`) is conditioning-limited outright.

Per instruction the negative results are preserved as run, not patched:
no stability projection, regularization, rescaling, or alternative
parameterization was added. Result 3 identifies the cause of the
failure and is deliberately scoped as a diagnostic — it is not adopted
as a fix, and no conditioning remedy is proposed here.

## Commit hash

See the commit introducing this file.
