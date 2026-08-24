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

## 3. The stability law

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

## 4. The grid, and which cells are empty

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

**Cell 2 (genuinely open): the metric as a co-variational object.** Every
member of the grid takes the mass matrix as *given* — identity, Hessian,
or a fixed schedule. The action does not require this: the mass can be
part of the variational problem itself,

```
min over (q, M ∈ ℳ)  R[q̇, M],
```

whose Euler–Lagrange equations produce *both* the flow and a learning
rule for the metric. That is a principled derivation of "learned
optimizers / learned preconditioners" from an action — a cell with no
occupant. The neuroscience lineage points at it directly (VLE's
"learnable backward couplings": the backward metric is learned, not
derived). Nothing in this repo falsifies it — it was never tested. It is
the one genuinely new question the unification opens:

> Can the mass matrix of the prospective action be learned jointly with
> the trajectory, so that *the geometry of learning is itself learned* —
> and does that learned metric beat the derived ones (Hessian, diagonal,
> per-mode) in regimes where the exact metric is unavailable?

The prospective-credit experiments delimit where it cannot work: as a
signal filter (matched ≠ inverse) and as a per-mode gain (regime-
dependent, training-unstable). A co-variational metric is neither of
those — it acts on the descent field, where the prospective mechanism
survives.

## 5. Where SSMs sit in the unification

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
