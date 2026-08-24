# The prospective action — a unification, and the gaps it opens

**Status:** theory spine of the project. Statements marked [proven] are
verified in this repo (script in parentheses); [known] is prior art;
[open] is unfalsified.

---

## 1. The action

The prospective dynamics comes from a Rayleighian (dissipation
functional) evaluated *prospectively* — the potential is sampled at the
predicted future state, not the current one:

```
R_τ[q̇] = ½‖q̇‖² + ( Φ(q + τ q̇) − Φ(q) ) / τ
```

Stationarity `∂R/∂q̇ = 0` gives the implicit prospective flow, and
linearizing `∇Φ(q + τq̇) ≈ ∇Φ(q) + τH q̇` (H = ∇²Φ) gives its working form:

```
q̇ = −∇Φ(q + τq̇)            (implicit prospective flow)
(I + τH) q̇ = −∇Φ(q)   ⇒   q̇ = −(I + τH)⁻¹ ∇Φ      (linearized)
```

Three ingredients generate everything that follows:

- the **dissipation** `½‖q̇‖²` (first-order, no mass),
- the **prospective evaluation** `Φ(q + τq̇)` — the field at the predicted
  state — which becomes a **mass matrix** `M = (I + τH)⁻¹` multiplying
  the velocity,
- optionally, **constraints** handled by Lagrange multipliers, which turn
  the multiplier trajectory into the credit/adjoint variable.

## 2. The method dictionary — one action, a family of methods

Each row is a choice of: how implicit the evaluation is (τ = 0 →
explicit; linearized → prospective metric; fully implicit → proximal),
what the mass matrix is, and what space the action lives on.

| term in the action | generated method | status |
|---|---|---|
| bare dissipation, `τ = 0` | gradient flow → vanilla GD | [known] |
| same, on a **chain energy** `E(s) = Σ ½‖s_t − a s_{t−1} − b_t‖²` | the **linear SSM recurrence** itself (its equilibrium is the rollout) | [proven: `pesm_s5_spectrum.py`, gate G2 to 1.5e-15] |
| mass/inertia on `q̇` | heavy ball / momentum | [known] |
| prospective evaluation of the field **at the predicted state** | **Nesterov acceleration** (NAG evaluates the gradient at the lookahead point) | [known — the prospective reading of Nesterov] |
| fully implicit evaluation (no linearization) | **proximal point / implicit Euler** — unconditionally stable on convex problems | [known] |
| Hessian mass `(I + τH)⁻¹` | Newton / damped Newton / natural gradient (optimization); **mass matrix in Riemannian HMC** (sampling) | [known] |
| same mass, on a **stiff chain energy** | **the prospective structured Newton**: one Hermitian tridiagonal solve = 3 associative scans, exact, κ-independent | [proven: `pesm_s5_spectrum.py`, `s5_state_inference.py`, `plds_benchmark.py`, `plds_mcmaze.py`] |
| Lagrange multipliers on an action | equilibrium/DEQ models; the adjoint as a multiplier trajectory (EqProp/VLE lineage) | [known] |
| prospective term on the **memory state** of an SSM | spectrum cancellation `τṡ = −s` | [proven dead: `exact_failure.py`] |
| prospective term as an **error filter** in credit | matched filter of the credit operator — phase-exact, gain-inverted | [proven dead: `gradient_alignment.py`] |

The unification claim: gradient descent, Nesterov, proximal methods,
Newton, natural gradient, and Riemannian HMC are **discretizations and
metric choices of the same prospective action**. The prospective mass
`M = (I + τH)⁻¹` is the common object — it is what Nesterov's lookahead,
the proximal subproblem's inner solve, and HMC's mass matrix all are.

## 3. The multiplier block — the time reversal the action already contains

The right variational object is the **constrained** action — loss plus
Lagrange multipliers enforcing the recurrence (not a quadratic penalty
on the residual: with the loss attached, penalty stationarity gives
`r_t − A† r_{t+1} = q_t` on the residual, which is incompatible with
the exact rollout `r = 0` unless q = 0):

```
A[s, λ] = Σ_t ℓ_t(s_t) + Σ_t λ_t† (s_t − A s_{t−1} − B x_t)
```

Its two stationarity conditions are exactly the two blocks:

```
δ/δλ:  s_t = A s_{t−1} + B x_t      — the forward rollout
δ/δs:  λ_t = q_t + A† λ_{t+1}       — the exact credit recursion (BPTT adjoint), q = −∇ℓ
```

The multiplier *is* the credit variable; the time-reversed operator A†
is not an addition to the action — it is the action's second
stationarity block. (The loss-free quadratic chain is the degenerate
special case whose minimizer is the rollout [proven: 1.5e-15] — that
gate belongs to the solver line, where no loss competes.)

**Why the prospective approximation of this block is conj(D) — a
derivation, not a hindsight match.** Write the multiplier equation as
the operator equation `Dλ = q`, per mode `D(ω) = 1 − āe^{iω}`, and
give it the least-squares action `E(λ) = ½‖Dλ − q‖²`. Its gradient at
λ = 0 is `−D†q`: the adjoint (matched) operator applied to the signal
is the *first descent direction of the multiplier's own variational
problem* — the cheapest causal object in it. The exact solution
satisfies the normal equations `λ⋆ = (D†D)⁻¹ D†q`, so

```
λ⋆ = (D†D)⁻¹ λᵖʳᵒ,   λᵖʳᵒ = D† q
   ⇒  arg λ⋆ = arg λᵖʳᵒ  per frequency   (D†D = |D|² is positive real)
```

The prospective multiplier is phase-exact by construction, and its
entire defect is a positive modal gain [also verified directly:
`gradient_alignment.py` to machine precision]. Equivalently: exact
credit = (prospective direction) × (curvature of the multiplier
energy), and that curvature is a *positive scalar per mode*.

**Complementarity** [empirical: `factorize_w.py`]: the action's factor
D† is a rotation in each mode's (Re, Im) plane — the one object a
diagonal optimizer cannot express. The missing factor (D†D)⁻¹ is a
positive real gain — exactly what Adam supplies. Frozen phase-only
rotation closes 113% of the online→full gap; frozen gain adds nothing.
*The prospective principle derives the orientation of credit; the
optimizer derives the rest.* This is the precise, surviving form of
"the action derives the learning algorithm" — false in the metric
reading (§5, Cell 2), true in the multiplier reading.

**Boundary** [proven]: the action's output is a filter, not a scalar —
collapsing D† to one phase per mode under any symmetric weighting is
identically zero (`derive_phase.py`: odd-phase cancellation around
resonance; pole of D⁻¹ outside the unit circle). This kills the
*isolated-mode* scalar derivation only; it does **not** kill the full
stacked operator `D_full` (temporal recurrence ⊗ inter-layer
Jacobians), whose matched part carries cross-layer structure — the
natural candidate for the learned metric's load-bearing shallow-layer
phases [open]. Closure test [open]: `ψ_j = arg∫Ŵ_j D_full†` reduced
per mode, Ŵ measured causally from the online gradient stream.

## 4. The stability law

For the blended step `s ← s − [γH + (1−γ)/η · I]⁻¹ ∇E`, the prospective
flow has an exact stability boundary

```
dt ≤ 2γτ      (γ = 1 is the stiffness-free point)
```

[proven: `ghost_demo.py` part D, measured boundary]. This is the
prospective action's face of a general principle: **the admissible step
size is set by the mass of the metric, not by the curvature of the
problem** — explicit Euler pays `1/λmax`, the prospective metric pays
nothing (affine invariance for quadratics).

## 5. The grid, and which cells are empty

Reading the dictionary as a grid — *implicitness* × *metric quality* ×
*structure available* — exposes exactly two empty cells:

**Cell 1 (occupied by us): structured implicit methods.** A fully
implicit/prospective step is only as cheap as the Hessian's structure
allows. On chain/quasi-separable problems it is free (3 scans); on dense
problems it is unaffordable and its approximations fail — the diagonal
metric gives κ-reduction 1.00× on dense stiff maps [proven:
prospective-deq's own review finding]. So the cell "cheap implicit step"
is filled exactly where the structure exists and is provably closed
elsewhere. This is the solver habitat we measured.

**Cell 2 (tested, with a corrected derivation status): the metric as a
learned object.** Every member of the grid takes the mass matrix as
*given* — identity, Hessian, or a fixed schedule. We tested making it
learned. The verdict has three parts, and the middle one corrects an
earlier over-claim in this document:

1. **The action as written gives no learning rule for M.** In
   `R = ½q̇ᵀMq̇ + (Φ(q+τq̇)−Φ(q))/τ`, M enters only through the
   dissipation, so `∂R/∂M = ½q̇q̇ᵀ`, whose stationarity forces `q̇ = 0`.
   The M-variation is degenerate: there is **no Euler–Lagrange equation
   for the metric** in this functional. An earlier version of this file
   advertised "Euler–Lagrange equations for both the flow and the
   metric" — that was wrong.

2. **What actually works is a meta-gradient — and only its phase does
   work.** Route A: a per-mode complex metric `M = diag(w̄_j)` rescaling
   the online gradient, learned by the one-step lookahead objective
   `min_ϕ L(θ − ηM_ϕ g_online)` — MAML over the optimizer's geometry,
   not stationarity of the action. It works: final loss 0.0016 vs 0.0224
   online RTRL. The factorization (`factorize_w.py`) assigns the win to
   the phase: frozen `e^{i·arg w}` alone closes 113% of the online→full
   gap, while frozen `|w|` adds nothing beyond what Adam supplies.

3. **But the phase is learned, not derived — three independent kills.**
   (a) It is not the curvature mass: (I+τH)⁻¹ is real-symmetric and
   cannot hold W's rotational part (71% of W's energy in layer 0), and
   |w| is anti-correlated with the mass (to −0.90)
   (`recheck_curvature_matrix.py`). (b) It is not derivable from the
   credit operator alone: ψ_j = arg∫W_j conj(D_j) dω is *identically
   zero* — a symmetric real weighting of an odd-phase filter around its
   resonance cancels, equivalently the pole of D⁻¹ lies outside the
   unit circle for |a|<1 so the average returns the DC coefficient
   (`derive_phase.py`; the scalar phase lives in the signal spectrum,
   not the operator). (c) It is task-bound: frozen to a new
   delay/horizon it ties online (`transfer_phase.py`), though random
   phases *hurt* (+14%) — it is specific structure, not noise.

   What remains interpreted: the learned phase matches the fitted
   per-mode credit correction α in deep layers (0.03–0.05 rad) — but
   α-at-init fails as a deployed phase (11% of gap), so the load-bearing
   shallow-layer components are exactly the ones the credit-defect
   reading does not explain. Meta-learning discovered a real, specific
   mechanism; no closed form currently derives it.

The provenance, corrected after the audit: *the variational viewpoint
identified the descent-field geometry as the right object, and the
spectral analysis explains why the defect is phaseful (D⁻¹ =
conj(D)/|D|² — orientation vs gain); but the phase that repairs is
found by meta-learning, not computed.* "The prospective action derives
the successful learning algorithm" is **not** supported at any level.

A genuinely variational route to a metric-learning rule remains open but
is new theory: make the metric dynamical in the action, e.g. add
`κ/2‖Ṁ‖² + V(M,q,q̇)`, giving a real co-variational dynamics
`κM̈ − ∂V/∂M + ½q̇q̇ᵀ = 0`. Whether its solution resembles the
empirically successful metric is untested — and the transfer result
(task-specificity) is evidence against forcing it.

## 6. Where SSMs sit in the unification

The SSM recurrence is the **first-order quadratic member** of the family:
gradient flow on a quadratic chain energy, whose minimizer has a closed
form (the associative scan). Everything the unification can offer an SSM
starts where the closed form stops:

- quadratic + explicit scan: nothing to solve — the prospective machinery
  has no purchase ([proven]).
- non-quadratic anchors / couplings / non-Gaussian observations: the
  closed form dies, the problem becomes a stiff equilibrium, and the
  prospective-mass member becomes the solver ([proven]: the inference
  line).

That is the whole placement, now derivable from one principle: the action
only generates work where minimization is nontrivial, and the SSM is the
trivial member of its own family.
