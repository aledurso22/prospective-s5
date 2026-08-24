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

## The PAC arc — what the learned phase IS (`pac_probe2.py`, `pac_deploy*.py`, this branch)

The prospective-adjoint-credit probe: measure the optimal scalar
projection of exact credit onto causal credit and compare with the
learned phase. Setup uses the rig's own validated objects: `spatial_q`
(Γ-routed instantaneous credit), `exact_lambda` (stacked adjoint with
the instantaneous cross-layer term, fd-gated). All 8 registered bars
PASS:

- **P5**: the error is not white at the modes (|ρ(1)| = 0.15–0.50) —
  the derive_phase zero was a symmetric-weighting artifact; the phase
  lives in the error's lag-1 autocorrelation.
- **P1 + REL**: `arg w` matches `arg c*` (c* = Σ āᵏρ(k), exact under
  stationarity) up to the learned phase's own cross-seed reliability
  (R 0.92–0.96 vs ceiling 0.995 top layer; shallow layers track their
  lower ceilings). **CTRL**: identical match at online-baseline params
  — the phase structure is task+architecture, not a trajectory
  artifact. This is the strongest result in the credit lane.
- **P2**: the shallow-layer attenuation is the stacked adjoint's
  instantaneous cross-layer term (identity residual ~1e-15 top, ~1.0
  shallow — the old "gate failure" was the cross-layer signal, not a
  bug).
- **P3l**: the AR(1) closure `K = 1/(1 − āρ(1))` predicts `arg w` as
  well as or better than the full optimal projection.
- Ceiling: best scalar leaves 90–99% of credit variance unexplained;
  3-tap FIRs recover nothing — w is a preconditioner aligning dominant
  directions, not a credit reconstruction (an information ceiling, not
  a method weakness).

Deployment (P4, `pac_deploy.py` full-K; `pac_deploy2.py` phase-primary
+ oracle-β control, arms amended per review and logged): phase-oracle
closes **36%** of the online→routeA gap (median 0.0188 vs online
0.0284, routeA 0.0015); phase-EMA ~0%; full-K worse than online. Verdict
per preregistered reading: **directionally right, not load-bearing** —
the causal one-statistic law recovers the orientation's sign but not
the learned phase's full content; meta-learning is the better estimator
of the object the theory identifies. Orientation-scarce / gain-Adam's-
job now holds at probe, frozen-arm, and live-deployment levels.

Analyses A & B (preregistered 50879e3, `pac_analysis.py`):
**A (pairing) FAIL** — the resolvent combination −arg(1−āρ(1)) does
NOT beat bare arg ρ(1) at L2/L3 (0.817 vs 0.856; 0.987 vs 0.998); per
the preregistered kill, the "resolvent of the adjoint generator"
framing is dropped as the explanation of w's value.
**B (horizon) PASS** — c(H) = Σ_{k≤H} āᵏρ(k) matches w best at H = 1
with monotone decline (3/4 layers, no rise-then-fall): the action's τ
reads as Route A's one-step lookahead horizon — the first structural
explanation of why that meta-objective worked.
Exploratory deploy3 (`pac_deploy3.py`, raw e^{i arg ρ(1)}): catastrophic
(0.17–0.27, fracs −5 to −9, ~10× worse than online). The static winner
is deployment poison: raw arg ρ(1) is unbounded and noisy per batch,
while arg(1 − āβ) is phase-bounded by construction. Lesson recorded:
the deployment barrier is stability — causal estimation trades lag
(EMA, worse) against variance (oracle, better but noisy) — and the
comb form, though it lost the static pairing, is the variance
stabilizer that makes deployment possible at all.

## The TBPTT baseline (`tbptt_baseline.py`, this branch)

Windows {1, 4, 16, 64} + full BPTT, paired seeds, same protocol.
Medians: online 0.0284; tbptt1 0.1776; tbptt4 0.1519; tbptt16 0.1180;
tbptt64 **0.0003**; bptt ~0.00003. Predeclared reading confirmed:
W = 16 fails, W = 64 works. Two structural findings:

1. **Key cell:** buffered 64-step exact credit beats the streaming rule
   on loss (tbptt64 0.0003 vs routeA 0.0015, ~5x). The streaming rule's
   remaining advantage is O(1) memory and no backward pass, not
   accuracy. This bounds the paper's claim honestly.
2. **Truncation is worse than online at every window below the delay**
   (0.118–0.188 vs 0.0284): the online rule is not "tbptt with W = 1" —
   its S-slot carries exact state sensitivities and only its error
   signal is truncated, so it beats 16-step exact-credit buffering.
   The streaming family has a real, measured edge in its regime.

(A complex-dtype bug in the first run — shallow-layer adjoints stored
real because spatial_q returns real arrays there — was caught by
numpy's ComplexWarning and fixed; committed numbers are the clean
rerun. G6 honored.)

## Directive-04 tests (`test_holonomy.py`)

Pontryagin identification accepted as framing (THEORY.md §3 names the
constrained action's blocks Hamilton's equations). Test A (rotational
depth law): naive additive phase-variance models fail the measured
fractions (0.71/0.37/0.06/0.01) by orders of magnitude — the growth is
structured, not diffusive. Test B (phase additivity/holonomy): **NO
HOLONOMY** (0/3 seeds; increment concentrations 0.20–0.89 < 0.7 bar) —
the shallow phase does not accumulate additively down the stack; the
connection reading is decoration. The symplectic/metriplectic
vocabulary survives as framing only; nothing predictive emerged.

## The field-theory round (`pac_deploy4.py`, `covariant_adam.py`)

Applying the taxonomy back to physics tools:

- **deploy4** (horizon-1 form c(1) = 1 + āρ(1), estimation rate,
  frozen-periodic): c(1)-oracle 28% (no better than the comb); slower
  EMA worse; **frozen-periodic (re-estimate every 200 steps) is the
  best derived law: 40%** of the gap (median 0.0178). STABILITY bar:
  frozen200 beats EMA ⇒ the deployment barrier is **variance, not
  lag** — the GENERIC-degeneracy reading (the orientation channel must
  not produce entropy) held.
- **covariant Adam** (the gauge prediction): if the phase defect is
  Adam's broken U(1) covariance, shared-modulus normalization should
  reproduce the phase arm with no learned w. **BAR FAIL** (median frac
  −0.15): covAdam rescues the pathological seed-0 plateau
  (0.0727→0.0028 — variance normalization, not covariance) but degrades
  both healthy seeds ~1.5x. Gauge reading rejected; routeA is robust
  to the optimizer (0.0027 median). Per the preregistered branch, the
  defect is **architectural** — the `.real` routing forced by
  real-valued targets on complex SSMs. That closes the locus: not the
  filter, not the optimizer, not the estimator — the causal-credit
  structure of real-output complex SSMs itself, with the learned phase
  as the right repair.
- Unifications kept (framing, no new predictions): PAC's c* = Σāᵏρ(k)
  IS the Mori–Zwanzig memory-kernel closure of the projected credit
  dynamics (ρ = the eliminated fluctuation's autocorrelation, FDT; the
  AR(1) closure = continued-fraction level 1; Analysis B's H=1 win =
  short-ranged kernel). The deployment barrier IS physical causality
  (MSR: exact credit = advanced propagator, causal = retarded,
  G^A = (G^R)† = the phase theorem; a causal system cannot realize an
  advanced propagator, only reconstruct its orientation from noise
  statistics). "Making the mass rotate" needs a Hamiltonian sector in
  the optimizer (Berry curvature lives in real-time dynamics), and
  would rotate on principle, not at the credit-required angle — wrong
  target.
- **LR control** (`lr_control.py`): standard Adam across LR
  {3e-4…1e-2} — best median 0.0136, nowhere near covAdam's seed-0
  0.0028; the rescue is structural (shared-v normalization), not rate.
  Protocol note: an inline re-implementation of cvm.adam reproduced
  registered seed values only up to ~1e-13 — over 1500 steps in this
  bistable landscape, float-operation ordering flips basins (seeds 1/2
  differed, seed 0 matched). Within-script pairing is the robust unit;
  cross-process bit-reproducibility is fragile at basin boundaries.

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
python pac_probe2.py             # what the phase IS (8/8 bars, CTRL)
python pac_deploy2.py            # causal law deploys 36% (directional)
python pac_analysis.py           # A: resolvent dropped; B: horizon 1
python tbptt_baseline.py         # buffered-64 beats streaming; W<delay fails
python test_holonomy.py          # no additive phase accumulation
```

## Next steps (in order)

1. **Community baselines** for the inference paper: LFADS-style
   variational posterior and Pólya-Gamma augmentation arms on MC_Maze —
   the two names a reviewer will demand.
2. **Multi-seed fit comparison** (the loose-vs-tight learning effect),
   and held-out behavioral decoding on NLB (the "latents as deliverable"
   figure).
3. **Credit lane — closed with mechanism, no open threads.** The
   learned phase is the optimal scalar credit projection (probe,
   CTRL-controlled, horizon 1); deployment is stability-limited (36%
   best); holonomy and resolvent framings tested and rejected;
   tbptt64 bounds the streaming claim to memory/compute, not accuracy.
   Remaining optional diagnostic: per-mode arg K vs arg w along the
   deployment trajectory (is the derived phase systematically off or
   merely noisy).
4. **Write-up**: four-slot theory map + solver + inference benchmark;
   venue assessment honestly scoped (main-track if 1–2 land; workshops
   otherwise).
