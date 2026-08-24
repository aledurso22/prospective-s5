# Prospective dynamics in sequence models — program state and evidence map

**Date:** 2026-08-24 · **Branches:** `main` (S5 no-gos), `research/prospective-credit-s5` (credit lane), `research/pesm-s5-spectrum` (solver + inference) · **Status:** solver slot positive with derivation + real data; route A positive as a *learned* (not derived) per-mode phase mechanism; all other slots closed with mechanisms.

The program: import prospective dynamics (Zucchet/Senn/Sacramento lineage —
NLA, GLE, VLE) into S5-style state-space models, and find where — if
anywhere — the mechanism helps. Answer, completed: **the only slot where
the prospective mechanism is *derived* is the solver metric; the credit
lane's positive is a learned per-mode rotation whose phase — not gain —
does the work, and it has no closed form.**

## The four-slot map

| slot | verdict | mechanism | evidence |
|---|---|---|---|
| memory dynamics | dead | `τ(I−A)ṡ = −(I−A)s + …` cancels ⇒ `τṡ = −s`: the memory spectrum disappears | derivation; `exact_failure.py` (main) |
| discretized recurrence | dead | two-step Euler ⇒ parasitic root, `μ₁μ₂ = A`; overflow ~step 14/784 at any ρ; even γ=0 explicit Euler unstable (130.33) | `exact_failure.py`, `ghost_demo.py` (main) |
| credit signal | dead as a filter; **alive as a per-mode gain** | prospective filter = matched filter of the credit operator; credit needs the inverse filter; gain error `|1−āe^{iω}|²` | `gradient_alignment.py`, `optimal_credit_filter.py` (credit branch) |
| continuation across training | dead | registered FAIL (v3): context motion dominates; warm starts switch branches | prospective-deq project (external) |
| **solver metric** | **positive** | `M = (I+τH)⁻¹` mass matrix; exact tridiagonal Newton via 3 associative scans; κ-independent | this branch |

## The positive (this branch)

**Solver level** (`pesm_s5_spectrum.py`): on the real bilinear HiPPO
spectrum (complex, oscillatory, κ up to 2.9e10), the prospective
Gauss-Newton step (one Hermitian tridiagonal solve = 3 associative scans)
is exact in 1 step for the quadratic chain and converges quadratically
with the tanh anchor; GD needs ~κ steps; Anderson stalls at ~0.7
residual; Broyden diverges. Gates: scan==dense (3e-16); one step == S5
rollout (1.5e-15) — *the equilibrium of the chain energy is the S5
forward pass*.

**Synthetic inference** (`s5_state_inference.py`, `plds_benchmark.py`):
Poisson PLDS with stiff AR(1) latents, convex posterior, exact Hessian.
Newton: 1–4 NFEs to 1e-8, flat across κ = 4e2→4e8. L-BFGS: hundreds of
evals and **false-converges** (stops at 51% residual, reports success) at
κ=4e8. Kalman gate: Newton MAP == exact RTS posterior mean (2.8e-15).
B4: at κ=4e8 the dynamics gradient ∂E/∂λ at a loose solve is **99%
corrupted** — solver quality is a training issue when the model is
implicit and stiff.

**Real data** (`plds_mcmaze.py`, NLB'21 MC_Maze, DANDI 000128): 40
trials × 124 bins × 137 channels, 20 ms bins, 8 stiff latent modes
(κ ≤ 4e8), Poisson + loading matrix (block-tridiagonal Hessian). Newton:
**2 NFEs, ~5 ms/trial, residual ~1e-9 every trial**. L-BFGS: median 2054
NFEs, never converges (res 26–63), false-converges on 4/40 trials. GD:
frozen. The synthetic story survives real neural data.

**Fit experiment** (`plds_mcmaze_fit.py`): alternating joint-MAP fit of
(λ, C, d), inner solve by arm. Newton: 51k inner NFEs / 121 s; L-BFGS:
1.32M NFEs / 319 s — same outer progress at ~26× fewer evaluations.
Held-out LL slightly favors the loose arm (−4812 vs −5297, single seed) —
the loose-tolerance/regularization effect, consistent with the registered
DEQ null: **learning is robust to loose solves; tight solves matter when
the latent trajectory itself is the deliverable.**

## The credit lane (branch `research/prospective-credit-s5`)

- `theory_checks.py`: the phase identity `arg H_pro = arg H_BPTT` holds
  to machine precision (Test A/B/C, 5/5).
- `gradient_alignment.py`: six estimators × L×|a| sweep, both regimes.
  The prospective filter never beats online RTRL; gain inversion and
  ±ω mixing identified; all four nulls gated and pass. Registered as a
  clean negative.
- `optimal_credit_filter.py`: closed-form optimal causal K-tap credit
  filter per mode + transfer check. Apparent positive: in the rig's
  dense-loss setting, per-mode gains beat online RTRL by +0.3-0.45 cosine
  and transfer across data realizations of the same task.
- `trained_credit_gains.py` (this branch): the training races. v1
  (unconstrained modes): gain arms explode; v2 (legal parametrization,
  a = sigmoid(rho) e^{i theta}): nothing explodes, registered bar NO WIN
  (oracle 0.042 vs online 0.024; bptt 0.00003). Two audit findings:
  (i) the rig's alignment win was regime-dependent (dense loss, per-mode
  median accounting) and reverses on the copy task at init (0.58 vs
  0.77); (ii) the instability mechanism was directly measured (gain arms
  push |a| past 1 by step ~50 in v1; BPTT approaches 0.999 and retreats).
  Exploratory signal, RETRACTED by its registered confirmation
  (`registered_oracle_b.py`, bar fixed before running): at 5 seeds the
  oracle_B advantage flips (copy: oracle_B 2.67x WORSE than online) —
  the v2 3-seed gap was seed variance. On the adding task all arms,
  including bptt, sit at the chance plateau (~0.083) — a degenerate
  discriminator. Credit lane closed with complete coverage — EXCEPT the
  co-variational metric below.

## The co-variational metric arc (`co_variational_metric.py` → `derive_phase.py`, this branch)

The prospective action's freed metric slot: the per-(layer, mode) metric
w is not given but *learned* online as a descent-field preconditioner
(conj(w) on the gradient blocks; never a filter on the error signal).

- **route A (meta-gradient)** — w descends the one-step-lookahead loss on
  the same batch through the analytic chain. **Registered positive**:
  median final loss 0.0016 vs online RTRL's 0.0224 (**14x better**, all 5
  seeds finite, no boundary-pushing — amax_end ~ online's). bptt 0.0001
  remains the exact-credit ceiling.
- **route B (consistency residual)** — diverges on every seed (nan).
  Clean negative.

(A stale-flat bug in the first version — adam's output never written
back — was fixed in e538eb6; all numbers above are post-fix.)

The mechanism audit, four experiments, each with a preregistered bar:

1. **Not the curvature mass** (`recheck_curvature.py`,
   `recheck_curvature_matrix.py`). Scalar: corr(|w|, 1/curv_a) = −0.03.
   Matrix: W_j = uI + vJ vs the action's mobility (I+τH_j)⁻¹, τ swept.
   The Hessian block is real-symmetric ⇒ the mobility has *exactly zero*
   rotational part, yet layer 0 carries 71% of W's energy in rotation —
   structurally unreachable at any τ. The gain profile is ANTI-
   correlated with the mass (corr down to −0.90). "The action derives
   the metric" is dead at the level of shape, scale, and fit.
2. **The phase is the mechanism** (`factorize_w.py`). Frozen-factor
   arms: phase-only e^{i·arg w} closes **113%** of the online→full gap
   (medians: online 0.0284, phase 0.0053, mag-only 0.0080, full-frozen
   0.0080) and beats the frozen full metric on every seed. The gain is
   redundant with Adam; the rotation is the part no diagonal optimizer
   can express.
3. **Specific, but task-bound** (`transfer_phase.py`, unseen
   D=200/T=256). Random phases HURT (+14% vs online) — the learned
   phase is structured, not a generic regularizer. But the frozen phase
   ties online (0.0935 vs 0.0932): the *number* does not transfer.
   Confound noted for any future protocol: D=200 may be capacity- rather
   than credit-limited; a BPTT headroom arm is mandatory on any new
   transfer task (headroom at D=50 was 0.996).
4. **Not derivable** (`derive_phase.py`). The zero-parameter spectral
   phase ψ_j = arg∫W_j(ω)·conj(D_j(ω)) dω, W_j = mode power response,
   is **identically zero**: a symmetric real weighting of an odd-phase
   filter around its resonance cancels (equivalently, the pole of D⁻¹
   lies outside the unit circle for |a|<1, so the average returns the
   DC coefficient, 1). The scalar phase does not live in the operator
   alone. Arms: phase_theory ≈ online (20% of gap); phase fitted from
   exact credit *at init* also fails (11%) — the useful phase is
   end-of-training structure. And the learned phase matches the fitted
   correction α only in deep layers (0.03–0.05 rad) while departing
   1–2 rad in shallow layers; since alpha_init fails and learned wins,
   **the load-bearing part of w is the part the credit-defect
   interpretation does not explain.**

Verdict: route A survives as a *learned* per-mode complex rotation of
the online gradient — phase-not-gain, specific, task-bound, only
partially interpreted. "The prospective action derives the algorithm"
is dead at every level: no Euler–Lagrange equation for M exists in the
action, the learned metric is not the curvature mass, and the phase has
no closed form in D(ω). What survives as theory is the diagnosis, not
the derivation: D⁻¹ = conj(D)/|D|² explains *why* causal credit has a
phaseful defect and *why* a rotation is the repair Adam cannot supply —
but the phase that repairs is learned, not computed.

## What the mechanism is, one sentence

Applied to signals (memory, recurrence, credit), the prospective operator
`1+τ∂ₜ` cancels poles, spawns parasites, or inverts gains — matched
filter where an inverse is needed. Applied to the descent field, the
derived mass `(I+τH)⁻¹` flattens curvature — the only placement where
the discretization is exact and the memory survives — while a *learned*
per-mode rotation repairs the phaseful half of causal credit's defect
(gain being Adam's job), a mechanism meta-learning discovered but no
closed form yet derives.

## Reproduce

```bash
python exact_failure.py          # SSM no-gos (main)
python ghost_demo.py
git checkout research/prospective-credit-s5
python theory_checks.py          # phase theorem
python gradient_alignment.py     # credit null + nulls
python optimal_credit_filter.py  # per-mode-gain positive + transfer
git checkout research/pesm-s5-spectrum
python pesm_s5_spectrum.py       # solver on the S5 spectrum + 4-solver showdown
python s5_state_inference.py     # synthetic PLDS
python plds_benchmark.py         # B1–B4 suite (Kalman gate, capability, B4)
python plds_mcmaze.py            # real-data figure (needs data/nlb/, DANDI 000128)
python plds_mcmaze_fit.py        # fit experiment
python co_variational_metric.py  # route A registered positive (14x)
python recheck_curvature.py && python recheck_curvature_matrix.py  # not the mass
python factorize_w.py            # phase is the mechanism (113%)
python transfer_phase.py         # specific, task-bound (random hurts)
python derive_phase.py           # not derivable (zero-phase theorem)
```

## Next steps (in order)

1. **Community baselines** for the inference paper: LFADS-style
   variational posterior and Pólya-Gamma augmentation arms on MC_Maze —
   the two names a reviewer will demand.
2. **Multi-seed fit comparison** (the loose-vs-tight learning effect),
   and held-out behavioral decoding on NLB (the "latents as deliverable"
   figure).
3. **The shallow-layer question** (credit lane's one open thread): the
   learned phase's load-bearing part lives in shallow layers and matches
   no per-mode credit defect — cross-layer coupling is the remaining
   candidate mechanism. Any transfer revisit requires a BPTT headroom
   arm on the target task first.
4. **Write-up**: four-slot theory map + solver + inference benchmark;
   venue assessment honestly scoped (main-track if 1–2 land; workshops
   otherwise).
