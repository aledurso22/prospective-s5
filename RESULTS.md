# Prospective dynamics in sequence models — program state and evidence map

**Date:** 2026-08-23 · **Branches:** `main` (S5 no-gos), `research/prospective-credit-s5` (credit lane), `research/pesm-s5-spectrum` (solver + inference) · **Status:** solver slot positive at solver level + real data; all other slots closed with mechanisms.

The program: import prospective dynamics (Zucchet/Senn/Sacramento lineage —
NLA, GLE, VLE) into S5-style state-space models, and find where — if
anywhere — the mechanism helps. Answer, completed: **exactly one slot
survives, and it is the solver metric.**

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
  filter per mode + transfer check. **Positive:** in the deep/slow regime
  (L≥4, |a|≥0.9), a per-mode complex scalar gain — fit once, transferred
  to fresh data — beats online RTRL by +0.3–0.45 cosine (e.g. narrowband
  L=8, |a|=0.99: 0.425 → 0.878 transferred). The prospective (1,−a) taps
  are far from optimal. Interpretation: the online rule's defect is
  mode-dependent and data-stable (the spectral defect law), so a fixed
  per-mode gain/rotation repairs it — the VLE "learnable gain" slot,
  validated. One-layer null intact (nothing beats exact RTRL at L=1).

## What the mechanism is, one sentence

Applied to signals (memory, recurrence, credit), the prospective operator
`1+τ∂ₜ` cancels poles, spawns parasites, or inverts gains — matched
filter where an inverse is needed. Applied to the descent field, it is
the metric `(I+τH)⁻¹` that flattens curvature — the only placement where
the discretization is exact and the memory survives.

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
```

## Next steps (in order)

1. **Community baselines** for the inference paper: LFADS-style
   variational posterior and Pólya-Gamma augmentation arms on MC_Maze —
   the two names a reviewer will demand.
2. **Multi-seed fit comparison** (the loose-vs-tight learning effect),
   and held-out behavioral decoding on NLB (the "latents as deliverable"
   figure).
3. **The trained-gain credit test**: learn the per-mode gains end-to-end
   on a long-credit task (the oracle gains transfer; can they be learned
   from the loss, not the exact gradient?). This is the remaining
   credit-side question, now with a concrete target (the transfer taps).
4. **Write-up**: four-slot theory map + solver + inference benchmark;
   venue assessment honestly scoped (main-track if 1–2 land; workshops
   otherwise).
