# FINAL_MODAL_GEOMETRY_AUDIT

Final small-scale geometry/action audit of RoutePC, per the G-program
(G0–G8 + refinements A–E). All runs on the frozen toy protocol (delayed
copy D=50/T=128, L=4, N=16, batch=32, 1500 steps, LR=LR_M=1e-3,
CLIP=1.0, Adam θ / SGD w unless stated); every replay gate vs stored
finals bitwise; causal arms make zero BPTT/exact calls (audited).
Artifacts: `results/geometry_audit/`; scripts: `controls/g*.py` on
branch `geometry/modal-audit`.

**Interpretation discipline:** oracle representation quality ≠ causal
learnability; D⁻¹ ≠ M_w; global norm ≠ relative modal gain; phase-only
≠ full directional geometry. p-values are used to avoid unsupported
superiority claims, not as the story.

---

## 1. Cartesian-conditioning diagnostic (G0) — self-annealing FALSIFIED

Hypothesis tested: large |w| self-anneals Cartesian-SGD meta-learning
(η_eff = η/ρ²), freezing the geometry into bad basins.

Measured (15 seeds × PC0 + pcPhase trajectories, all bitwise-gated):

- |Δα| and |Δφ| do **not** collapse as ρ⁻²; they **grow** with ρ
  (log-log slopes +0.6/+0.6 on failure seeds, +0.3/+0.2 on successes) —
  the causal chain signal itself scales with ρ (J†h involves
  G(w)-blocks ∝ w). ρ²-rescaling is anti-stationary (CV 27–41 vs 4–5
  raw).
- **Timing falsifies causality**: on 5/6 PC0 failure seeds ρ sits at
  ~1.0 until *after* bad-basin entry, growing to ~13 later; success
  seeds reach the *same* final ρ (median 12.8 vs 12.8) with no harm.
- **Radial drift is orthogonal to failure.** It is an engineering
  nuisance, not the failure mechanism.

Clipping: **fire rate = 1.000 on every step of every arm/seed** — the
global-norm clip is always active in this regime.

## 2. Coordinate/gauge arms (G1 + refinement A) — polar branch closed

- Free log-polar (B) at the original meta-LR: NaN 5/5 (runaway α by
  ~step 500). Gauge-fixed polar (C): NaN 5/5. The ρ⁻² annealing is real
  and **load-bearing for stability** of the Cartesian-SGD meta-learner.
- LR control (A): free polar at η·{1e-1, 1e-2, 1e-3, 1e-4} — finite at
  1e-3/1e-4, beats online 5/5, median 0.0137 at both; **not
  competitive** with PC0 (bar 1.5×PC0 = 0.0110). So the original NaNs
  were an LR mismatch, not intrinsic instability — and with the rate
  corrected, polar does not outperform simpler variants. **Polar branch
  closed** (no bounded/gauge expansion).

## 3. The clipping regime is where the benefit lives (G3 + GC + G3X)

- G3 (clip vs clean nonbinding clip, 4 arms × 5 seeds; relative benefit
  vs online, clip → noclip): online 0.0224 → 0.0251;
  **real +0.268 → +0.156** (keeps ~58%);
  **PC0 +0.674 → +0.294** (keeps ~44%);
  **pcPhase +0.620 → +0.132** (keeps ~21%).
  Removing the clip shrinks every geometry's benefit substantially —
  the phase channel most, the real positive gain least — but does not
  annihilate it. (An earlier interim reading of "entirely
  clip-dependent" came from a nonbinding-clip implementation that
  rescaled instead of passing through; corrected in the rerun, and the
  two pc0 noclip traces were cross-checked against G3X for
  consistency.)
- G3X (C3 crossover at nonbinding clip): PC0 ≈ online, BPTT+w ≈ BPTT,
  interaction medians ≈ 0 at every budget (Δ_credit > 0 on 5/5).
  **Without the normalization channel, M_w no longer preferentially
  repairs defective credit** — the credit-repair specificity measured
  in C3 is a clipped-Adam-regime phenomenon.
- GC: corr(ρ, ‖Δθ‖) is mostly **negative** (−0.91, −0.73 on healthy
  seeds) — actual parameter displacement is decoupled from the radius.
  Common-scale invariance probe: scaling all w by κ ≤ 100 changes the
  post-clip direction by cos ≥ 0.9990 and the Adam update by
  cos ≥ 0.99996 — **common radial scale is a near-exact gauge** (the
  untouched c blocks carry ~0.2% of the gradient norm). Gauge language
  is quantitatively supported; per-layer *relative* gain is not a gauge
  (see §4's R_A).

## 4. Action-Jacobian arms (GD/E1/E2 + GE2) — the residual is what pins ρ

- E1 (clip-aware residual): median 0.0175 (5 seeds), sane.
- E2 (clip+Adam-action-aware residual): median 0.0121 (5 seeds);
  **ρ max ≈ 1.0 on every seed** — respecting the one-step action
  Jacobian completely suppresses the radial runaway.
- E2 → 15 seeds (GE2, no 1.5× gate): median 0.0167, wins vs online
  9/15, fails 6/15 [4,6,7,8,10,11]. sd(log|w|) ≈ 0.000–0.005 per layer:
  E2's geometry is effectively phase-only with **no relative gain
  structure**.
- R_A (numeric JVP of the actual clip+Adam action): global radial/total
  sensitivity ratio **0.113** (early 0.082 → late 0.125) — the COMMON
  radial direction maps to near-zero parameter displacement. Per-layer
  relative-radial ratios 0.59–0.87 — relative gain does move the
  action. Distinction established: radial residual *exists* (share
  0.41–0.48), radial action sensitivity is null *only for the common
  direction*.
- GE2B decomposition of the radial residual (per-layer
  r_α = common uuᵀ + relative (I−uuᵀ)): common energy share **0.05–0.09
  (pooled 0.069)**, early/mid/late flat. E2's radial residual is
  ~93% RELATIVE-radial — not parked in the action-null common
  direction. Its sd(log|w|) ≈ 0 therefore is not a gauge artifact: the
  action-aware transform simply keeps the relative-radial residual too
  small to accumulate any gain structure. E2 discards the
  relative-gain channel that the best arms exploit — consistent with
  its weaker 15-seed profile. E2 is retained as a **mechanistic
  action-Jacobian control**, not the primary algorithm.

## 5. Residual × MetaOpt 2×2 (G1/GD/GB/GAA) — the full table

| | SGD MetaOpt | Adam MetaOpt |
|---|---|---|
| schematic residual | PC0: 0.0167, 6/15 fails | pc0_adam: **0.0120, 3/15 fails** |
| action-aware residual | E2: 0.0167, 6/15 fails | AA: 0.0191, 5/15 fails |

(15 paired seeds; medians and failure counts vs online.) Registered
prediction confirmed: AA does not beat E2/pc0_adam on median — both
modifications partly repair the same pathology, and combining them is
slightly worse than pc0_adam alone.

**pc0_adam (GB)**: beats online 12/15, median ratio 0.597, failures
[3,9,10] all marginal (ratios 1.25–1.7, no catastrophic basin); radius
bounded (median |w| 1.5–1.8) **with** relative modal gain retained
(sd(log|w|) 0.14–0.16, max/min ~1.6–1.9 per layer) — not collapsed to
phase-only.

## 6. Representation ceilings (G4/G5, oracle-only)

| geometry | held-out cos median | per-seed range |
|---|---|---|
| identity | 0.596 | 0.466–0.775 |
| per-mode real | 0.765 | 0.673–0.885 |
| per-mode complex | 0.901 | 0.813–0.977 |
| full 2×2 per mode | 0.922 | 0.883–0.977 |
| complex + rank-1 cross-mode | 0.908 | 0.868–0.981 |
| complex + rank-2 cross-mode | 0.931 | 0.925–0.980 |

Full 2×2 and rank-2 give small (+0.02, +0.03) held-out gains over the
complex ceiling; rank-2 is positive on 5/5 seeds but small. Per the
registered bar: **not material — no causal 2×2 or low-rank
implementation.** The complex modal family is representationally
sufficient; the remaining gap is not primarily cross-modal.

## 7. Failure taxonomy (G7 + GE-SIG)

- Every failure across every causal arm carries **RESIDUAL_SPIKE** as
  its primary label (a >10× meta-residual-norm explosion vs its own
  early median); two PC0 failures add GRAD_COLLAPSE.
- GE-SIG on bounded-radius failures (pc0_adam [3,9,10] — all marginal):
  no distinct signature; they are threshold-straddling seeds.
- **Teacher-alignment collapse REFUTED as the failure precursor**:
  cos(r_causal, r_exact) does not drop before failure (failures'
  early/mid cos is if anything higher: 0.67/0.73 vs 0.50/0.44), and
  sign flips are absent (negative-cos fraction 0–8%).
- Catastrophic failures (PC0/pcPhase) instead show a **persistent ~10×
  larger late meta-residual norm** (‖ε‖_late 3.6M vs 0.34M) with normal
  teacher alignment — the meta-learner keeps taking large corrections
  without converging. The basin failure is a *trajectory* phenomenon,
  not an information phenomenon.

## 8. Statistics (G8, 15 paired seeds)

| pair | ratio med | wins | sign p | Wilcoxon (log) p |
|---|---|---|---|---|
| PC0 vs online | 0.861 | 9/15 | 0.607 | 0.600 |
| pcPhase vs online | 0.718 | 11/15 | 0.119 | 0.107 |
| **pc0_adam vs online** | **0.597** | **12/15** | **0.035** | **0.008** |
| e2action vs online | 0.858 | 9/15 | 0.607 | 0.524 |
| AA vs online | 0.617 | 10/15 | 0.302 | 0.208 |
| pc0_adam vs pcPhase | 0.671 | 8/15 | 1.000 | 0.679 |

Only **pc0_adam** has a statistically supported win over online.
pcPhase is marginal; PC0/E2/AA are not separable from online at n=15.
pc0_adam vs pcPhase is not separable — pc0_adam's edge is the failure
profile (3 marginal failures vs 4 including catastrophic seed-3
rescues) and the median/geomean.

C3 extras (clipped regime): BPTT+w worse than BPTT on 0/5 seeds at
every early budget (sign p = 0.0625, the n=5 floor);
Spearman(repair, miscalibration) = −0.90 at K=129 — stronger credit
repair accompanies stronger miscalibration of exact credit.

## 9. The recommendation

**Primary S5 candidate: PC0 with Adam MetaOpt for w (`pc0_adam`).**

Evidence: only arm with a statistically supported 15-seed win over
online (sign p 0.035, Wilcoxon 0.008); best median (0.0120 vs online
0.0226, PC0 0.0167); best failure profile (3/15, all marginal);
bounded radius (|w| ~1.5–1.8) with the load-bearing relative modal
gain structure retained (sd(log|w|) ~0.14–0.16); zero BPTT.

**Controls retained (not primary):**
- PC0 full complex (frozen reference; radial runaway, 6/15 failures).
- phase-only `routePCphase` (0.0137, 4/15; the simplest defensible
  variant; only one that rescues catastrophic basins).
- E2 action-aware — **mechanistic action-Jacobian control**: pins
  ρ ≈ 1 exactly and establishes the action-null/gauge decomposition
  (R_A, GE2B), but loses the relative-gain structure (sd(log|w|) ≈ 0)
  and with it the performance edge; 6/15.
- AA (no added benefit; 5/15).

**Not carried forward:** free/gauge log-polar (stable only at reduced
LR, no advantage), full-2×2 and low-rank cross-mode geometries (oracle
gains +0.02/+0.03, below the materiality bar), block-metric MetaOpt
(preconditions not met: polar doesn't help, and Adam already solves
the conditioning issue).

**Mechanism one-liner for the paper:** The modal correction has strong
gradient-level credit representation, but its causal training benefit is
mediated by the clipped-Adam update geometry. Under clipping, relative
modal gain and phase reshape the normalized update direction; without
clipping the learned correction is nearly behaviorally inert. The common
radial scale is a near-exact gauge; Adam-MetaOpt on w keeps the geometry
bounded while preserving the relative modal gains; the benefit is
specific to defective (online) credit and vanishes on exact credit at
every budget; and the residual closed-loop failures are trajectory/basin
events with a persistent meta-residual-explosion signature, not an
information-collapse phenomenon.
