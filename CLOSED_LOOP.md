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
