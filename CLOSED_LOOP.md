# Closed-loop temporal credit — the north-star program

**Candidate claim:** temporal credit assignment is a closed-loop problem.
Optimizing static approximation to the BPTT adjoint can produce unstable
learning (wiener_oracle: cosine 0.32→0.73 with horizon, deployment
−6.97); what matters is a trajectory-adaptive orientation of causal
credit that remains inside the stable learning region.

**Spotlight gates (need ≥3 of 4):**
A. theory separating static adjoint approximation from closed-loop stability;
B. mechanism: orientation-only/bounded correction explains routeA-survives-
   vs-Wiener-explodes;
C. generality beyond the S5 copy setup;
D. a practical causal rule approaching routeA without future information.

## Directive 1 — causality audit of routeA (settled from the code)

**Base update:** strictly causal within the sequence (S-slot: causal
sensitivities + instantaneous q); no cross-batch leakage (all signals
from batches ≤ n).

**Meta update (w):** exact BPTT adjoint over the full current batch —
reads future timesteps *within* the batch. Route A as a training
procedure is therefore **not strictly causal**: w is learned with a
BPTT teacher, second-order (analytic chain through the update), ~3×
online's per-step compute. w itself: L×N complex, O(1) memory.

**Closest strictly-causal variant (already measured):** deploy w
frozen — per-mode constant rotation, zero future information, zero
overhead — final loss 0.0053 (vs best fully-causal estimator 0.0178,
vs routeA live 0.0015). Decomposition: **offline BPTT-supervised
orientation learning + strictly causal free deployment.** Any
"causal credit" claim must be scoped to the deployment side; the
orientation teacher is offline exact credit.

## Directive 2 — orientation-only Wiener: RESOLVED

Arms (paired seeds {0,1,2}, frozen K-filters from trained params,
K ∈ {1,4,16,32,64,96}): err_t = e^{i(arg λ̂_t − arg q_t)}·q_t — the
Wiener orientation with online magnitude. Result: **every arm worse
than online** (orient1 −0.9 → orient96 −6.2, clipK64 −9.6), monotone
worse with horizon; **deployed cosine collapses to 0.05–0.23** (vs
static 0.32–0.73 at trained params). The orientation itself, not only
the gain, fails to transfer into the loop. Per-timestep orientation
inherits estimator noise (the raw-ρ failure mode). Measured stability
ladder: per-timestep orientation (destructive) < per-batch constant
derived phase (36%) < frozen-periodic (40%) < frozen learned phase
(100%+) < live co-adapted phase (routeA, 14×).

**Verdict (Spotlight B): routeA's advantage IS trajectory
co-adaptation** — no static object, at any horizon, survives the loop.
Directive 3's accuracy-vs-stability non-monotonicity exists in the
data (best static cosine ↔ worst deployment) — the paper's central
figure, no new runs needed.

## Directive 2b — matched-function-class control (`matched_phase.py`)

The agent's correction: D2 conflated horizon with estimator granularity
(per-timestep phase ≠ routeA's constant rotation), and frozen learned
phase already preserves ~86% of routeA's gain. Control: reduce each
Wiener K-filter to a single constant per-mode rotation
w^K = e^{i arg c^K}, deployed exactly like routeA, at three estimation
rates. Results (frac of online→routeA gap):

- frozen K=1–32: −0.9 (horizon irrelevant); frozen64: −0.7;
  **frozen96: +0.41** (horizon pays only at the task's predictability
  horizon K≈96); anchor: frozen-learned-phase 0.0053 still 3× better.
- **rate ladder is non-monotone**: frozen 0.0525 < refresh-200 0.0595
  (staleness+noise is the worst cell) < **per-batch 0.0111 (+0.65)**.
  The deployment barrier is STALENESS; variance is poison only when
  also stale.
- **algorithmic point**: per-batch closed-form optimal phase (teacher
  each step, no meta-gradient) captures 65% of routeA's gain;
  meta-learning buys the final 7× — refinement, not orientation.

## Method-gate diagnostics: `phase_track.py` + `optimum_track.py`

`phase_track` (learned-phase trajectory): arg w nearly static along
training (hold error 0.0017 rad/25 steps); momentum predictor 6% < 20%
bar. Record: *a predictor for the learned Route-A phase trajectory is
not useful.*

`optimum_track` (instantaneous credit-projection optimum): the object
measured is φ*_credit(θ) = arg(E[λq̄]/E|q|²) — the credit-MSE-optimal
scalar phase, NOT the instantaneous post-update learning optimum.
Result: **φ*_credit(θ) moves strongly along training** (Var_train ≈
0.34 vs Var_batch ≈ 0.0017, ratio ~196, 40–200× the noise floor on
every layer), while learned w is comparatively stable and sits
~0.5–0.8 rad away from it. So routeA does not track the moving
credit-projection optimum. Combined with D3/D4, the emerging
distinction:

  credit reconstruction optimum ≠ learning-useful update geometry.

Simonetto is removed from the main mechanism story (tracking the moving
credit-projection optimum is not what routeA does). NOT measured:
whether the training-optimal phase moves, or whether the learning
optimum is static — neither claim is made.

## Directive 4 — exact stability counterexample: LANDED (`d4_stability.py`)

Minimal exact model: single complex mode, quadratic loss, affine
gradients — every estimator's learning map is exactly M_E(b−b*) with
machine-precision M_E. At |a| = 0.95 (Hankel σ = 9.7), same step size:

| estimator | credit MSE | GD | ΔL at test step | P(ρ>1) |
|---|---|---|---|---|
| wiener64 | **0.540 (best)** | **diverges** | **+3.5e4** | 1.00 |
| exact BPTT | 0.000 | diverges | +6.9e4 | 0.97 |
| online | 0.899 | converges | −2.74e3 | 0.00 |
| phase-only | 0.899 | converges | −2.72e3 | 0.00 |

**Better approximation of backpropagated credit produces worse
learning** — credit-MSE minimization and next-step-loss minimization
are different objectives (Wiener's own η_opt would descend; its stable
step is ~3× smaller). Under Adam nothing diverges (all ≈ 0.8): gain
instability is absorbed by per-coordinate normalization — the exact
form of "gain is Adam's job." D5 fell out: at |a| = 0.995 phase-only GD
converges (2.9e-9) where online diverges (4.8e5) — **the orientation
correction's stability advantage emerges exactly as the Hankel floor
diverges**. Slow modes are where causal credit is hardest to
stabilize; orientation is the stabilizer. Measured, not asserted.

## Directives 3–7 — status

- D3: assemble the static-cosine vs deployed-improvement plot from
  existing arms (online / PAC phase / frozen learned / Wiener full /
  Wiener orient / routeA). No new runs.
- D4 (linear stability theory): exact learning-Jacobian spectral radius
  in the quadratic model; counterexample construction — NEXT BUILD.
- D5 (slow modes ↔ instability): from D4's machinery; Hankel floor
  σ = |a|/(1−|a|²) already measured (corr 0.99).
- D6 (generality): second recurrent family — deferred until A lands.
- D7 (routeA complexity): covered by the audit above.
