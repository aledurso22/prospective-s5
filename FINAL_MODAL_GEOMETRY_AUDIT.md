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
  ~step 500). Gauge-fixed polar (C): NaN 5/5. The formal Cartesian
  log-polar metric contains a ρ⁻² coordinate factor, but G0 shows that it
  does not describe the observed Cartesian trajectories and is not an
  accepted self-annealing explanation.
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
  **Without the normalization channel, M_w no longer has a preferential
  effect on the defective-online-credit arm** — the C3 interaction is a
  clipped-Adam-regime phenomenon, not optimizer-independent credit repair.
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
[3,9,10] with exact ratios **1.253, 1.767, 1.165**. These are not all
marginal, although none is the 8× catastrophic basin seen for PC0; radius
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
- GE-SIG on bounded-radius failures (pc0_adam [3,9,10]): no distinct
  signature was found. Their exact ratios span 1.165–1.767, so
  “threshold-straddling” is not an adequate description of all three.
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
pc0_adam vs pcPhase is not separable — pc0_adam's edge is the lower
failure count (3 vs 4) and the median/geomean, not a uniformly marginal
failure profile.

C3 extras (clipped regime): BPTT+w worse than BPTT on 0/5 seeds at
every early budget (sign p = 0.0625, the n=5 floor);
Spearman(repair, miscalibration) = −0.90 at K=129 — stronger credit
repair accompanies stronger miscalibration of exact credit.

## 9. The recommendation

**Primary S5 candidate: PC0 with Adam MetaOpt for w (`pc0_adam`).**

Evidence: only arm with a statistically supported 15-seed win over
online (sign p 0.035, Wilcoxon 0.008); best median (0.0120 vs online
0.0226, PC0 0.0167); lowest failure count (3/15; exact ratios above);
bounded radius (|w| ~1.5–1.8) with the load-bearing relative modal
gain structure retained (sd(log|w|) ~0.14–0.16); zero BPTT.

**Controls retained (not primary):**
- PC0 full complex (frozen reference; radial runaway, 6/15 failures).
- phase-only `routePCphase` (0.0137, 4/15; the simplest defensible
  variant; only one that rescues catastrophic basins).
- E2 action-aware — **mechanistic action-Jacobian control**: pins
  ρ ≈ 1 exactly and establishes the action-null/gauge decomposition
  (R_A, GE2B). Stated precisely: at the inherited registered MetaOpt
  settings, action-aware E2 does not improve performance and collapses
  learned relative gain (sd(log|w|) ≈ 0); this does not establish that
  exact action-aware hypergradients are intrinsically inferior — only
  that this registered configuration is not the better arm. 6/15.
- AA (no added benefit; 5/15).

**Not carried forward:** free/gauge log-polar (stable only at reduced
LR, no advantage), full-2×2 and low-rank cross-mode geometries (oracle
gains +0.02/+0.03, below the materiality bar), block-metric MetaOpt
(preconditions not met: polar doesn't help, and Adam already solves
the conditioning issue).

## 10. Bridge audit (B1–B8): is the learned geometry the exact-credit correction?

Post-hoc analysis on 15 bitwise-gated pc0_adam replays (training exact
calls 0/0; exact credit only in the probes), checkpoints K ∈ {500, 1000,
1500}, held-out probe batches, D2 conventions.

**B1 — learned-w exact-gradient alignment.** Median over seeds at
K=1500 (K=500/1000 similar): L0 C_id 0.448 → C_learned 0.466; L1 0.326
→ 0.322; L2 0.503 → 0.540; L3 (negative control) 1.000 → 0.987.
**ΔC ≈ 0 everywhere (±0.04)** while the per-mode complex oracle at the
same params reaches **0.82–0.99**. The learned geometry does NOT
statically approximate the exact-credit correction at its own
checkpoints, even though a modal scalar could. The closed-loop benefit
is not terminal static credit reconstruction — the D3/optimum_track
lesson, now established for the winning causal arm.

**B2 — analytic eligibility-credit correspondence (unshuffled).** Phase agreement
with the analytic gradient-level statistic c_g^stat is strong:
**MRL(arg c_g^stat, arg w) = 0.80–0.96** per layer at every checkpoint.
Relative log-gain correlation (common layerwise scale removed) ≈ 0.
This is a descriptive correspondence; B7 tests whether it is
mode-specific.

**B3 — cross-seed transplant (K=1500).** Lower layers: identity 0.361,
diagonal (self) 0.385, off-diagonal 0.366. B8 shows that differences of
these pooled medians are not supported by recipient-paired uncertainty.
Top layer: 1.000 → 0.987/0.989 (any rotation of an exact gradient only
degrades it ✓).

**B4 — mode shuffle.** Shuffled w gives C ≈ C_id ≈ C_learned — the
static effect is too small for mode assignment to matter there.

**B5 — exact pc0_adam failure ratios** (L_method/L_online): **s3
1.253, s9 1.767, s10 1.165**. They are not all marginal; none is the
8.16× catastrophic PC0 seed-3 basin.

**B6 — learned phase × analytic gain.** The hybrid
`|c_g^stat| exp(i arg w_learned)` does **not** uncover a hidden static
alignment benefit. At K=1500, medians `(C_id, C_learned, C_hybrid,
C_oracle)` are L0 `(0.448, 0.466, 0.421, 0.958)`, L1
`(0.326, 0.322, 0.291, 0.823)`, L2 `(0.503, 0.540, 0.467, 0.897)`,
and L3 `(1.000, 0.987, 0.897, 1.000)`. Recipient-paired median
`C_hybrid-C_learned` is `+0.002/-0.037/-0.068/-0.095`; the pattern is
similarly null-to-negative at K=500/1000. **The hypothesis that learned
phase is right while learned relative gain destroys static alignment is
not supported.**

**B7 — phase effect size and shuffle specificity.** The analytic phase is
not trivial: pooled median `|arg c_g^stat|` at K=1500 is
`0.291/0.270/0.386/0.384` rad for L0–L3 (IQRs
`[0.135,0.636]/[0.121,0.532]/[0.152,0.770]/[0.151,0.994]`). But the
high raw MRL is largely retained after 256 fixed within-layer mode
shuffles. At K=1500, median correct MRL is
`0.961/0.907/0.885/0.889`, shuffle-null median is
`0.916/0.887/0.864/0.837`, and recipient-paired correct-minus-shuffle
median is only `+0.039/+0.012/-0.020/+0.056`. Across checkpoints these
differences remain small. Thus the high B2 MRL is substantially explained
by common phase concentration and is **not strong evidence of a
mode-specific match**.

**B8 — transplant uncertainty and D2 reconciliation.** Using recipient
seed as the independent unit, self-minus-identity has median `+0.00375`
(bootstrap 95% interval `[-0.00281,+0.03597]`; 9/15 positive), while the
recipient-median off-diagonal-minus-identity effect is `+0.00018`
(`[-0.00198,+0.00485]`; 8/15 positive). The pooled `0.366-0.361`
difference is therefore not evidence of shared defect structure.

The D2 identity `0.596` and bridge identity `0.361` are not the same
aggregation. D2 evaluates the global vector (all four recurrent layers
plus readout) at RouteA final parameters on seeds 0–4; B3 evaluates only
defective lower recurrent layers L0–L2 at RoutePCAdam final parameters on
seeds 0–14. At the same RoutePCAdam parameters, changing lower-only to the
D2 global aggregation raises the first-five median from `0.342` to
`0.569` (paired increase `+0.183`, bootstrap interval
`[+0.113,+0.292]`). RouteA D2 global is `0.596`; the paired
RoutePCAdam-minus-RouteA global difference is heterogeneous (median
`+0.022`, interval `[-0.248,+0.115]`). The seed-set shift is small
(`-0.019` lower-only, first five minus all 15). The discrepancy is
therefore primarily the inclusion of the exact top/readout blocks, not a
checkpoint mismatch.

**Bridge reading.** Learned phase has high unshuffled correspondence with
the analytic eligibility-weighted statistic, but the shuffle audit shows
that much of this is shared phase concentration rather than mode-specific
matching. Neither learned gain nor analytic gain combined with learned
phase improves static exact-gradient alignment. Separately, G3/G3X show
that the causal training benefit is mediated by clipped-Adam update
geometry. **The mechanism linking the residual phase correspondence to
that training benefit remains open.**

## 11. Mechanism-first action audit (M1–M6)

`controls/m1_m6_action_mechanism.py` replays all 15 frozen RoutePCAdam
trajectories bitwise, with training exact calls `0/0`, and clones
`theta_n` plus the complete base-Adam state immediately before the update at
K = 500/1000/1500. Identity, learned `w`, unit-modulus learned phase, and
the static modal credit oracle `w_C` then take exactly one
corrected-gradient → global-clip → Adam action. Every candidate is evaluated
on the same next batch. The action-gradient implementation passes a central
finite-difference gate at relative error `3.0e-6`.

**M1 — actual next-batch utility.** Paired median
`F_{n+1}(w)-F_{n+1}(I)` is:

| K | learned | learned phase only | static credit oracle `w_C` |
|---:|---:|---:|---:|
| 500 | `-4.26e-6` (8/15 improve) | `-1.01e-5` (11/15) | `-1.202e-3` (15/15) |
| 1000 | `+1.22e-5` (6/15) | `-1.79e-5` (11/15) | `-1.356e-3` (15/15) |
| 1500 | `+2.40e-5` (4/15) | `-1.05e-5` (11/15) | `-4.11e-4` (13/15) |

Bootstrap median intervals for learned `w` are
`[-2.74e-5,+1.28e-5]`, `[-2.31e-5,+1.03e-4]`, and
`[+1.05e-5,+4.85e-5]`. Thus learned `w` has no actual one-step utility
under the accumulated optimizer state and is slightly harmful late. The
phase-only action has a small, consistent effect, but it is two orders of
magnitude below `w_C`; `w_C` demonstrates that this post-update test has
ample sensitivity to a useful modal correction.

**M2 — credit oracle is not the action oracle.** A bounded per-mode
log-gain/phase estimate of `w_F`, minimizing the actual next-batch objective
through clip+Adam, was fit from identity, learned, and `w_C` starts at 20
representative checkpoints (success seeds 0/1 and failure seeds 3/9/10;
K = 500/501/1499/1500). All 60 fits reached the registered 60-iteration
limit. More importantly, **0/20 estimates were stable across starts either
in geometry or optimizer-action space**. At K=500/1500, median pairwise
fit-action cosine is `0.960/0.935` (minimum `0.886/0.879`) and symmetric
relative action distance is `0.287/0.365`. Therefore these are denoted
`hat w_F`, not treated as identified oracles.

Action-space comparisons are nevertheless consistent. At K=500/1500,
learned-action cosine with the best fitted action is `0.699/0.651`,
essentially identity's `0.696/0.650`; learned objective regret is
`1.142e-3/0.985e-3`, likewise identity-level
(`1.129e-3/0.960e-3`). The static credit oracle is closer: action cosine
`0.856/0.797`, regret `0.424e-3/0.396e-3`. Secondary geometry comparisons
agree but are not the primary evidence under the near-gauge: learned versus
`hat w_F` has phase MRL `0.589/0.576` and relative-log-gain correlation
`-0.368/-0.490`; `w_C` versus `hat w_F` has `0.844/0.892` and
`+0.781/+0.800`. Because `hat w_F` itself is action-unstable, these values
are descriptive rather than a claim that `w_C = w_F`.

**M3 — optimizer-state dependence.** Learned-minus-identity paired medians
for actual Adam / reset-`m,v` Adam / clipped SGD are:

| K | accumulated Adam | `m=v=0`, optimizer time retained | clipped SGD |
|---:|---:|---:|---:|
| 500 | `-4.26e-6` | `-2.40e-4` (11/15 improve) | `-2.57e-6` |
| 1000 | `+1.22e-5` | `+2.43e-4` (6/15) | `-5.14e-6` |
| 1500 | `+2.40e-5` | `-8.50e-4` (12/15) | `+0.53e-6` |

The K=1500 reset-Adam median interval is
`[-1.95e-3,-1.15e-4]` (one-sided sign `p=0.0176`), whereas clipped SGD is
null and accumulated Adam is slightly worse. The correction therefore does
not act mainly through the immediate globally normalized-gradient direction.
Coordinatewise Adam normalization can expose a one-step benefit after
moment reset, but the actual accumulated state cancels it at these frozen
checkpoints. This does not explain the full-training win; it instead points
to optimizer-state/trajectory shaping over multiple updates.

**M4 — exact scope of the lag-one correlation identity.** Let optimizer
batch/update time be `n`; sequence time inside a batch remains `t`. In the
repo convention the positive drive is
`c_n = J_{n-1}^dagger g_n^on`, the descended meta-gradient is
`-LR*c_n`, and subtracting that meta-gradient moves `w` along `+c_n`.
For the corrected complex input-matrix block only,

`c^B_{n,j} = sum_k conj(G^B_{n,jk}) G^B_{n-1,jk}`,

an exact lag-one complex cross-correlation. Its maximum numerical relative
error is `3.33e-16`. The recurrence block is parameterized by
`a=u exp(i theta)`, so its exact contribution instead contains the derived
`(rho,theta)` Jacobian factors `u` and `u(1-u)`; it is **not** asserted to
equal the same raw cross-correlation. The readout `c` is uncorrected by
`w`.

EMA filtering increases raw-correlation phase concentration with learned
`w`: median MRL rises from `0.662–0.948` raw to `0.894–0.966` for the EMA
and is `0.838–0.916` after the componentwise Adam transform. EMA versus
sequence-time `c_g^stat` has MRL `0.792–0.883`; median circular offsets are
generally small (within about `0.25` rad). These are explicitly
optimizer-time correlations across minibatches, not future temporal-credit
operators at sequence time. Because B7 shows strong common phase
concentration, the M4 MRLs are descriptive and are not claimed to be a
mode-specific causal explanation.

**M5/M6.** B6/B7 supply the phase-specificity controls. In addition, the
K=1500 static-oracle median `|arg w_C|` for L0–L3 is
`1.013/0.933/0.736/~0` rad; median magnitudes are
`10.45/3.94/2.55/1.00`, with broad lower-layer dispersion. Since neither
geometry nor optimizer action yields a stable `w_F` estimate at any of the
20 M2 checkpoints, M6's target-motion/tracking-lag/curvature comparison is
**not feasible and was not run**. Raw `||w-hat w_F||` is not used as
evidence.

**M1/M4 specificity amendment and stopping rule.** The frozen M1 fit and
thresholds were left unchanged. A separate bitwise-gated audit
(`controls/m1_m4_specificity.py`) adds 128 deterministic null actions per
checkpoint: (i) within-layer mode shuffles of the learned complex values and
(ii) independent within-layer permutations of learned magnitude and phase,
which exactly preserve both empirical marginals while destroying mode and
magnitude/phase assignment.

Learned-minus-null-median `F_{n+1}` differences at K=500/1000/1500 are
`-1.23e-5/+0.64e-6/-0.76e-6` against shuffled learned `w`, and
`-1.24e-5/+4.93e-6/+3.03e-6` against the marginal-matched random null.
Learned `w` beats the per-snapshot shuffle/random median on only
`10/11`, `7/7`, and `8/7` of 15 seeds. At the primary K=1500 checkpoint,
null draws beat learned `w` at median fractions `0.484/0.539`. Thus the
small learned-versus-identity differences in M1 are **not
learned-controller-specific**.

For M4, define `K_forward = g_n^dagger g_{n+1}`. Because the deployed
gradient uses `conj(w)`, the repo's positive complex-B drive is
`K_drive = conj(K_forward)`. At K=1500, online B-block
`MRL(K_drive,c_g^stat)` is `0.773/0.797/0.757/0.867` for L0–L3, but
correct-minus-256-shuffle MRL is only
`+0.018/+0.039/-0.022/+0.044`. The corresponding exact-gradient MRL is
`0.889/0.901/0.758/0.867`, with shuffle-specificity
`+0.024/+0.030/+0.007/+0.039`. The lower online-gradient defect is large
(online-versus-exact relative errors `0.979/0.986/0.912`) and the top layer
is exact (`2.0e-15`), yet phase specificity is neither stronger for online
than exact lower-layer gradients nor larger in defective lower layers than
the exact top layer. The same conclusion holds for the diagnostic raw
complex `(a,B)` group; that statistic is not identified with the constrained
`(rho,theta)` Jacobian.

The pre-registered conjunction therefore fails both parts:
`M1_specific=false`, `M4_specific=false`. **STOP the mechanism search.**
Do not escalate automatically to a multi-step counterfactual, another
`M_w`, or an optimizer variant.

**Mechanism reading.** The winning learned geometry does not improve the
actual one-step objective under its own accumulated Adam state. It is also
not close to the best fitted action in either action or regret, while a
static credit oracle is detectably useful and closer. The RoutePC residual
does contain an exactly correlation-like B-block signal, and its filtered
phase is concentrated with learned `w` and `c_g^stat`, but this does not yet
connect that signal to the training win: learned actions do not beat the
shuffled/random M1 controls, and M4 correspondence does not beat the
shuffle/exact/top-layer nulls. Closed-loop path dependence remains a possible
description, not a supported mechanism from these diagnostics. Per the
registered rule, no follow-on mechanism diagnostic is authorized.

## 12. S5 instrumentation

Every S5 run now records the clipping covariates in its metrics JSON:
`p_clip` (fraction of steps with pre-clip global norm > clip), the
pre-clip norm distribution (p50/p90/max), and
`chi=||g_pre||/C` when C>0. The historical S5 default is restored:
**`--clip 0` (unclipped Adam)**. Clipping is an explicit experimental
factor; `--clip 1.0` selects the toy's mechanistic regime. CPU smoke at
the tiny clipped config: p_clip = 1.0,
pre-clip norm p50 ≈ 60× threshold — the S5 model sits in the same
always-clipped regime as the toy.

The dedicated Stage 0 launcher runs only the paired matrix
`{BPTT, Online} x {clip=0, clip=1.0}` with matched seeds/configs. Its
report requires at least three paired seeds, a positive clipped
Online→BPTT gap on every seed, and median relative headroom at least 0.2
before authorizing even the small correction pilot. It never authorizes
the large sweep.

**Mechanism one-liner for the paper:** The learned correction improves
closed-loop training through clipped-Adam geometry, but it neither improves
static exact-gradient alignment nor the actual one-step next-batch objective
under its accumulated Adam state. A static credit oracle improves that
one-step objective strongly, and the learned action is identity-like relative
to an optimizer-aware fitted optimum. The exact B-block meta-residual is a
lag-one optimizer-time gradient correlation whose filtered phase is
concentrated with learned `w` and sequence-time `c_g^stat`, but the learned
controller fails shuffled/random action nulls and the correlation fails
shuffle/exact/top-layer localization controls. The mechanism is therefore
unresolved and the registered program stops; neither temporal-credit repair
nor path-dependent controller optimization is established. Under clipping,
relative modal gain and phase reshape the
normalized update direction; without clipping the learned correction is
nearly behaviorally inert. The common
radial scale is a near-exact gauge; Adam-MetaOpt on w keeps the geometry
bounded while preserving the relative modal gains; the benefit is
specific to defective (online) credit and vanishes on exact credit at
every budget; and the residual closed-loop failures are trajectory/basin
events with a persistent meta-residual-explosion signature, not an
information-collapse phenomenon.
