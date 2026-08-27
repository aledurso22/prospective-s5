# RESULTS_LEDGER — frozen headline results with provenance

Canonical record of the important frozen results. Every row: experiment,
current script path, stored artifact, producing commit, seeds, per-seed
values, summary, and whether the arm is **deployable** (zero BPTT) or
**oracle/diagnostic** (BPTT allowed, audited).

Frozen tags: `routepc-pc0-frozen`, `routepc-mechanism-controls-frozen`
(both at the provenance commit `f7c5671`, one commit after the last
science commit `d402f78`; see `VERSION_CONTROL.md`).

Protocol unless stated: delayed copy D=50/T=128, L=4, N=16, batch=32,
STEPS=1500, LR=LR_M=1e-3, CLIP=1.0, Adam θ / plain-SGD w, seeds
{0,1,2,3,4} paired streams, final loss = mean of last 100 steps.

---

## 1. Core result — online vs RoutePC/PC0 (DEPLOYABLE)

Script: `toyrig/routepc.py` (run: `python -m core.train_routepc`)
Artifact: `results/route_pc/summary.json` (git `107a9d2`);
standalone artifact `results/core_routepc_reproduction/` (commit `ddc659a`).
Regression gate: `tests/test_pc0_regression.py` — fresh PC0 runs reproduce
stored finals **bitwise**, BPTT calls 0/0.

| arm | s0 | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|---|
| online | 0.0727 | 0.0224 | 0.0284 | 0.0109 | 0.0118 | **0.0224** |
| PC0 | 0.0167 | 0.0031 | 0.0025 | 0.0889 | 0.0073 | **0.0073** |
| routeA (oracle ref) | 0.0015 | 0.0009 | 0.0016 | 0.0248 | 0.0018 | 0.0016 |

4/5 paired wins (seed 3 is the bistable bad-basin reversal — see
`README_ROUTEPC.md` §7); relative median improvement **0.674**; PC0 median
R_gap **0.90** of the online→routeA gap.

## 2. BPTT control + interaction (ORACLE)

Script: `controls/control_2x2_normmatch.py` · artifact:
`results/control_2x2_normmatch/summary.json` (git `225d503`).

| arm | s0 | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|---|
| BPTT | 4e-05 | 4e-05 | 9e-05 | 3.5e-04 | 9e-05 | 9e-05 |
| BPTT+w | 5e-05 | 5e-05 | 1.1e-04 | 3.3e-04 | 6e-05 | 6e-05 |

Δ_credit = L_online − L_BPTT positive on all 5 seeds (precondition met).
Interaction I_i = (online−PC0) − (BPTT−BPTT+w): per-seed
[0.05599, 0.01930, 0.02590, −0.07696, 0.00446]; median **+0.0193**,
mean +0.0055, SD 0.0504. BPTT+w ≈ BPTT (±2e-05) ⇒ M_w does nothing on
exact gradients ⇒ the clipped-Adam interaction is **credit-regime
specific, not generic preconditioning**. Later B/M audits show that this
does not mean learned `w` reconstructs exact credit or has
optimizer-independent one-step utility.

## 3. Norm-matched PC0 (DEPLOYABLE diagnostic)

Same script/artifact as §2.

| arm | s0 | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|---|
| PC0_normmatched | 0.0167 | 0.0035 | 0.0027 | 0.1137 | 0.0073 | **0.0073** |

Identical to PC0 while ‖M_w g‖/‖g‖ pooled median per seed is
**[39.3, 63.9, 331.4, 563.6, 79.7]** — Adam absorbs the gain;
the improvement is **direction, not gain**.

## 4. D1 — exact missing-credit factorization (ORACLE diagnostic)

Script: `diagnostics/d1_exact_credit_factorization.py` (formerly
`prospective_offline2.py`) · artifact:
`results/prospective_offline2/summary.json` (git `b48aad1`).

exactD arm (eligibility × exact D⁻¹ at the correct upper-layer site vs
BPTT): **cos = 1.0 all 5 seeds, per-layer 1.0, relative error
2.4×10⁻¹⁵** (median). Placement/pole audit `placement_ok: true`.

## 5. D2 — modal-oracle representational ceiling (ORACLE diagnostic)

Scripts: `diagnostics/d2_modal_oracle.py` (formerly
`oracle_real_vs_complex.py`) + per-seed retrieval in
`controls/control_2x2_normmatch.py` · artifact:
`results/control_2x2_normmatch/summary.json` (fit batches 0–3, held-out 4–7).

| geometry | s0 | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|---|
| identity | — | — | — | — | — | 0.596 |
| per-mode real | 0.764 | 0.765 | 0.885 | 0.850 | 0.673 | **0.765** |
| per-mode complex | 0.901 | 0.893 | 0.921 | 0.977 | 0.813 | **0.901** |

No catastrophic seed (complex 0.813–0.977). Complex−real ≈ **+0.14** every
seed ⇒ phase is representationally valuable; the causal factorial tie is an
**identification gap**, not a representational one.

## 6. Teacher decomposition + deficit diagnostics (ORACLE)

| experiment | script | artifact | key numbers |
|---|---|---|---|
| routeA→PC gap decomposition | `diagnostics/teacher_decompose.py` | `results/teacher_decompose/summary.json` (`4f45945`) | medians A(same-batch exact) 0.0016, B(next-batch exact) 0.0014, C(next-batch causal=PC0) 0.0073 ⇒ **delay free (−0.9%), causal-teacher blindness +28.2%**; cos(r_exactNext, r_causal) 0.854 |
| κ-sweep (prospective objective) | `diagnostics/prospective_kappa.py` | `results/prospective_kappa/summary.json` (`04c3801`) | κ medians 0.0016/0.0029/**0.0014**/0.0013/0.0016/0.0025 for κ=0/.5/1/1.5/2/4 ⇒ **matched horizon κ\*≈1–1.5, plateau to 2, degrade at 4**; κ=0 bitwise == routeA, κ=1 bitwise == arm B |
| teacher-deficit persistence | `diagnostics/eps_perlayer.py` | same commit | ε = r_causal − r_exact: ρ_φ(1) pooled lower layers **+0.787** (per-seed 0.78–0.85), ρ_r(1) +0.760, ~0 at top layer; tangential energy ~0.60–0.71 |
| oracle lagged-deficit correction | `diagnostics/oracle_lagged_deficit.py` | `results/oracle_lagged_deficit/summary.json` (`5ef1c15`) | medians A1(=PC0, bitwise) 0.0073 / A2 radial-only 0.0157 / A3 tangential-only 0.0088 / **A4 full 0.0025**; A4 beats A1 4/5 paired, **closes 81% of the blindness gap, rescues seed 3 (0.0889→0.0053)**; correction must be JOINT radial+tangential |
| E1 radial vs tangential teacher | `diagnostics/e1_e2_identification.py` | `results/e1_e2_identification/summary.json` (`fead777`) | cos_r 0.596 vs cos_φ 0.488 (pooled), gap positive 4/5 seeds, tangential ≈71% of exact residual energy ⇒ causal deficit **disproportionately** (not exclusively) phase |
| E2 teacher × geometry 2×2 | same | same | E_C 0.0014 / E_R 0.0089 / C_C 0.0073 / C_R 0.0164; Δ_exact +0.0076, Δ_causal −0.0058, interaction +0.0134 (3/5) ⇒ complex geometry representationally valuable, **causal exploitation basin/seed-dependent** (frozen wording) |

## 7. Failed / negative arms (archived, do not reopen without new evidence)

| experiment | script | verdict |
|---|---|---|
| F1 self-bootstrap teacher | `archive/failed_self_bootstrap/bootstrap_teacher_f1.py` (`225d503`) | α=0 bitwise == PC0 ✓; α=0.5 NaN 2/5, α=1.0 NaN 3/5, 0/5 wins ⇒ self-teacher self-amplifies; **z_n observer stage NOT built (preregistered stop)** |
| TSS residual prospection | `diagnostics/route_pc_pro.py` (+ `_drift`) (`4f45945`) | κ=0 bitwise == PC0; stationary wins ≤3/5 incoherent; moving-delay ramp 0/5 every κ ⇒ **CORRECTION ONLY**; drift buried in batch noise |
| geometry factorial (causal) | `diagnostics/route_pc_factorial.py` | per-mode-complex 0.0073 best median but per-mode-real wins 3/5 paired ⇒ registered bar FAILS; rotation-vs-gain unresolved for the causal arm (cf. D2: representational, not identification) |
| Taylor/lead realization of D⁻¹ | `diagnostics/single_mode_control.py`, `archive/prospective_offline.py` | first-order pole-only prospective realization fails at realistic bandwidth (r=0.995); exact factorization (D1) unaffected |
| Wiener/LTI causal deployment | `archive/credit_filters/wiener_oracle.py`, `orient_wiener.py`, `matched_phase.py`(diag) | LTI deployment catastrophic (−6.97 of gap); orientation-alone no win at any K; staleness is the barrier |
| curvature-mass readings | `archive/recheck_curvature*.py` | w is not a curvature/Hessian object (corr to −0.90 anti-Newton) |
| holonomy | `archive/test_holonomy.py` | shallow phase not additive down the stack (0/3) |
| covariant-Adam gauge | `archive/covariant_adam.py` | defect is architectural (.real routing), not Adam's U(1) covariance |
| transfer | `diagnostics/transfer_phase.py`, `archive/credit_filters/transfer_m.py` | phase specific ✓ (random phases hurt), transferable ✗ at D=200 (task-bound) |

## 8. Generality lane (rot-RNN) and benchmark state

| item | script | status |
|---|---|---|
| rot-RNN generality (D6 v1/v2) | `diagnostics/rot_rnn_generality{,2}.py` | barrier phenomena generalize; P3 FAIL — the large frozen-orientation benefit is **S5-specific in magnitude** |
| CPU bench gates | `diagnostics/bench_copy.py`, `diagnostics/bench_smnist.py` | copy saturates (degenerate discriminator); sMNIST headroom 0.12 < 0.2 at CPU budget |
| **mechanism-first M1–M6 + null amendment** | `controls/m1_m6_action_mechanism.py`, `controls/m1_m4_specificity.py` | learned actual one-step utility null-to-harmful and not better than shuffled/random controls; static `w_C` strongly useful; reset Adam exposes a late benefit but clipped SGD/accumulated Adam do not; `hat w_F` action unstable 0/20; lag correlation fails shuffle/exact/top localization; **STOP mechanism search** |
| **S5 Stage 0** | `train_bench.py` + `scripts/stage0*.{sh,sbatch}` + `stage0_report.py` | paired BPTT/Online × clip 0/1.0, three-seed minimum, explicit clipping/audit manifests, **READY / UNLAUNCHED** |
| **large cluster benchmark** | `scripts/bench.sbatch/bench_grid.sh` + `bench_report.py` | explicit historical clip=0; RoutePCAdam/Phase included in reporting; **BLOCKED pending Stage 0 + correction-pilot report** |

## 9. Solver lane (separate positive, archived)

`archive/solver/`: PESM/Newton solver positive results (S5 spectrum solve
exact in 1 step; PLDS MC_Maze real neural data 2 NFEs/trial ~5 ms,
L-BFGS never converges). Distinct program line from RoutePC; preserved for
provenance. See `EXPERIMENTS.md` lane 1.

## 10. Addendum controls C1–C3 (claim-sharpening, branch `controls/c1-c3`)

Scripts: `controls/c1_phase_only_routepc.py` (+ `c1b_phase_only_15seeds.py`),
`controls/c2_real_w_diagnostics.py`, `controls/c3_matched_budget_bptt_w.py`.
Artifacts: `results/c1_phase_only_routepc/`, `results/c2_real_w_diagnostics/`,
`results/c3_matched_budget_bptt_w/`. Every replay gate vs stored finals
**bitwise PASS**; BPTT calls 0/0 in the causal arms.

### C1 — phase-only (unit-modulus) RoutePC

| arm | s0 | s1 | s2 | s3 | s4 | median |
|---|---|---|---|---|---|---|
| pcPhase | 0.0115 | 0.0085 | 0.0015 | 0.0580 | 0.0079 | **0.0085** |

(online/PC0 as in §1.) 5-seed verdict per registered rule
(median ≤ 1.5× PC0 AND beats online ≥ 4/5): **COMPETITIVE** — paired
ratios pcPhase/PC0 median 0.686; seed 3 fails for both arms but less
badly under unit modulus (0.0580 vs PC0 0.0889 vs online 0.0109).

15-seed extension (seeds 0–14, `summary_15seeds.json`):

| arm | median | paired-ratio median (arm/online) | failures (ratio > 1) |
|---|---|---|---|
| online | 0.0226 | 1.000 | 0/15 |
| PC0 | 0.0167 | 0.861 | **6/15** [3, 6, 7, 8, 12, 13] |
| pcPhase | **0.0137** | **0.718** | **4/15** [3, 6, 9, 13] |

pcPhase beats PC0 on 9/15 paired seeds (median ratio 0.686) and online on
11/15. **Unit modulus improves stability** (fewer catastrophic seeds) at
some cost where the gain channel was helping (s1, s11). Registered as a
**selectable arm** `--arm routePCphase` in `train_bench.py` (unit-modulus
verified in saved w); PC0 preserved unchanged. Honesty note: the frozen
5-seed headline (PC0 0.0073, 4/5) is a luckier-than-typical draw — over
15 seeds PC0 fails on 6/15 seeds and its median ratio is 0.861. Both arms
fail sometimes on this bistable task; per-seed reporting stands.

### C2 — why does the real geometry work?

- **Pr(w_j < 0) = 0.000** — every layer, every seed, trajectory and
  final. **Zero sign flips ever** (0 flips/mode, all layers).
- |w_j| at final: heavy-tailed, depth-increasing — medians L0 1.80,
  L1 11.2, L2 15.6, L3 29.1; p90 7.5–160; max 2724 (L1).
- temporal variation: relative |Δw|/|w| median < 5e-4 (quasi-static).

**Verdict: (b) relative modal gain structure** — a quasi-static positive
per-mode reweighting that changes the gradient DIRECTION before
clipping/Adam. Not sign flips (none), not time-varying gain (static).
Consistent with E1 (radial teacher alignment > tangential) and the
factorial tie being identification-limited.

### C3 — matched-headroom BPTT+w control

Budgets by registered rule (first K where median L_BPTT(K) ≤
{2×, 1×, 0.25×} online median final; L(K) = 25-step mean):
K = 129 / 184 / 280 (+1500 reference). Δ_credit positive **5/5 at every
K**. **BPTT+w is WORSE than BPTT at every budget on (nearly) every
seed** (e.g. K=129: 0.0445→0.0637; K=280: 0.0024→0.0114 medians) —
the geometry learned in the defective-online-credit regime actively miscalibrates
exact credit. Interaction I(K) medians: +0.0022 / +0.0037 / +0.0120 /
+0.0211 (final). **Verdict: the generic-preconditioning hypothesis is
rejected with headroom present, at every budget** — the final-step 2×2
floor was not the limitation. G3X/M1 later restrict this interaction to
clipped-Adam closed-loop behavior rather than static exact-credit repair.

## 11. Modal-geometry audit (G-program, branch `geometry/modal-audit`)

Full record: `FINAL_MODAL_GEOMETRY_AUDIT.md`. Scripts `controls/g*.py`;
artifacts `results/geometry_audit/`. All replay gates bitwise; causal
arms 0 BPTT.

- **G0**: Cartesian self-annealing FALSIFIED — |Δα|,|Δφ| grow with ρ
  (not ρ⁻²), ρ grows after (not before) bad-basin entry, successes reach
  the same final ρ (~13). Clip fire rate 1.000 everywhere.
- **G1/GA**: free/gauge polar NaN at original meta-LR; at η·1e-3/1e-4
  finite + beats online 5/5 but median 0.0137 (not competitive). Polar
  branch closed.
- **G3/G3X (clean no-clip)**: benefit vs online, clip → noclip: real
  +0.268→+0.156, PC0 +0.674→+0.294, pcPhase +0.620→+0.132; without
  normalization the C3 credit-specificity washes out (interaction ≈ 0
  every budget). The benefit is mediated by the clipped-Adam update
  geometry.
- **GC**: displacement decoupled from ρ (corr −0.91/−0.73 healthy);
  common radial scale is a near-exact gauge (post-clip direction cos
  ≥0.9990, Adam update cos ≥0.99996 at κ=100).
- **G4/G5 (oracle)**: full 2×2 0.922, rank-2 0.931 vs complex 0.901 —
  small, not material; no causal implementation.
- **GD/GE2 (action-Jacobian)**: E2 pins ρ≈1 exactly; 15-seed 0.0167,
  6/15 fails; sd(log|w|)≈0 (degenerates to phase-only). R_A: common
  radial action sensitivity 0.113 (null) vs per-layer relative 0.6–0.87.
  GE2B: radial residual is ~93% RELATIVE-subspace (common share 0.069).
- **GAA**: action-aware + Adam = 0.0191, 5/15 fails — no improvement.
- **G7/GE-SIG**: every failure = persistent meta-residual explosion
  (RESIDUAL_SPIKE); teacher-alignment collapse REFUTED as precursor.
- **G8**: only pc0_adam has a supported win over online (median 0.0120,
  ratio 0.597, 12/15, sign p 0.035, Wilcoxon 0.008; 3/15 failures with
  exact ratios 1.253, 1.767, 1.165 — not all marginal).

**Primary S5 candidate: pc0_adam (PC0 + Adam MetaOpt for w).**
Controls: PC0 full complex, routePCphase, E2 (mechanistic
action-Jacobian control), AA. Not carried: polar variants, causal
2×2/low-rank, block-metric MetaOpt (preconditions unmet).

## 12. Bridge audit (B1–B8, analysis-only)

15 bitwise-gated pc0_adam replays (training exact calls 0/0), K ∈
{500, 1000, 1500}, held-out probes. Scripts:
`controls/b1_b4_bridge_audit.py` and `controls/b6_b8_bridge_audit.py`;
artifacts: `results/geometry_audit/b1_b4_summary.json` and
`b6_b8_summary.json`.

- **B1**: learned-w exact-gradient alignment — ΔC ≈ 0 at every layer
  and checkpoint (K=1500 medians: L0 0.448→0.466, L1 0.326→0.322, L2
  0.503→0.540, L3 control 1.000→0.987), while the per-mode complex
  oracle at the same params is 0.82–0.99. **The winning arm's geometry
  is NOT a static exact-credit approximation at its own checkpoints.**
- **B2**: phase correspondence with the analytic credit statistic is
  strong — MRL(arg c_g^stat, arg w) = 0.80–0.96 per layer; relative
  log-gain correlation ≈ 0. B7 shows that this raw MRL is substantially
  distributional rather than mode-specific.
- **B3**: transplant at K=1500 — identity 0.361 / self 0.385 / off-diag
  0.366 (lower layers); B8 uncertainty does not support either small
  difference.
- **B4**: mode shuffle ≈ learned ≈ identity (static effect too small
  for assignment to matter).
- **B5**: exact pc0_adam failure ratios — s3 1.253, s9 1.767, s10
  1.165. They are not all marginal (PC0's seed-3 was 8.16).
- **B6**: learned phase × analytic gain does not rescue static alignment.
  At K=1500, `(C_id,C_learned,C_hybrid,C_oracle)` is
  `(0.448,0.466,0.421,0.958)` / `(0.326,0.322,0.291,0.823)` /
  `(0.503,0.540,0.467,0.897)` / `(1.000,0.987,0.897,1.000)` for
  L0–L3. Paired hybrid-minus-learned medians are
  `+0.002/-0.037/-0.068/-0.095`. The proposed “correct phase, destructive
  learned gain” explanation is **not supported**.
- **B7**: pooled median `|arg c_g^stat|` at K=1500 is
  `0.291/0.270/0.386/0.384` rad, so phase effects are nontrivial. But
  correct MRL `0.961/0.907/0.885/0.889` is close to the 256-shuffle null
  `0.916/0.887/0.864/0.837`; paired correct-minus-shuffle medians are only
  `+0.039/+0.012/-0.020/+0.056`. High raw MRL is weak evidence for
  mode-specific correspondence.
- **B8**: recipient-paired transplant differences cross zero:
  self−identity `+0.00375` (bootstrap 95% interval
  `[-0.00281,+0.03597]`, 9/15 positive); off-diagonal−identity `+0.00018`
  (`[-0.00198,+0.00485]`, 8/15). Do not infer shared defect structure.
  D2's identity 0.596 is global (L0–L3 + readout), RouteA params, five
  seeds; bridge 0.361 is lower-only (L0–L2), RoutePCAdam params, 15 seeds.
  At RoutePCAdam params/seeds 0–4, global aggregation raises 0.342→0.569
  (paired +0.183, interval [+0.113,+0.292]); the RouteA-vs-RoutePCAdam
  global difference is heterogeneous and the seed-set shift is small.

**Bridge reading:** learned phase has high raw MRL with the analytic
eligibility-credit statistic, but the shuffle audit shows that much of it
comes from common phase concentration. Neither learned nor hybrid geometry
improves static exact-gradient alignment. The causal benefit is separately
known to be mediated by clipped-Adam geometry; the mechanism connecting the
residual phase correspondence to that benefit remains open.

## 13. Mechanism-first action audit (M1–M6, analysis-only)

Script: `controls/m1_m6_action_mechanism.py`; artifact:
`results/geometry_audit/m1_m6_action_summary.json`. All 15 RoutePCAdam
replays are bitwise-gated and make zero exact-gradient/exact-lambda calls
during training. Offline probes use frozen clones only. The analytic
clip+Adam action gradient passes finite differences at `3.0e-6` relative
error.

- **M1 actual post-update utility:** learned-minus-identity median
  `F_{n+1}` differences at K=500/1000/1500 are
  `-4.26e-6/+1.22e-5/+2.40e-5` (improves 8/6/4 of 15). Learned phase only
  is small but consistently negative
  (`-1.01e-5/-1.79e-5/-1.05e-5`, 11/15 each). Static `w_C` is strongly
  useful (`-1.202e-3/-1.356e-3/-0.411e-3`, 15/15, 15/15, 13/15).
  **Learned `w` does not improve the actual one-step objective under its
  accumulated Adam state.**
- **M2 credit versus action oracle:** 60 bounded fits at 20 representative
  checkpoints all reached the iteration limit; 0/20 `hat w_F` estimates
  are stable across starts in either geometry or induced optimizer action.
  At K=500/1500, learned action cosine to the best fitted action is
  `0.699/0.651`, essentially identity's `0.696/0.650`; learned regret is
  `1.142e-3/0.985e-3`, again identity-level
  (`1.129e-3/0.960e-3`). `w_C` is closer (`0.856/0.797`) and has lower
  regret (`0.424e-3/0.396e-3`). Pairwise fit-action cosine itself is only
  `0.960/0.935` median (minimum `0.886/0.879`), so do not treat any raw
  `w` distance as primary or claim `w_C = w_F`.
- **M3 optimizer-state dependence:** accumulated Adam repeats M1. With
  `m=v=0` but optimizer time retained, medians are
  `-2.40e-4/+2.43e-4/-8.50e-4` (K=1500: 12/15 improve, sign `p=0.0176`);
  clipped SGD is near zero (`-2.57e-6/-5.14e-6/+0.53e-6`). The immediate
  globally normalized-gradient direction is not sufficient. Reset Adam's
  coordinatewise normalization exposes a late one-step benefit, while the
  actual accumulated state cancels it; the full-training win remains
  path-dependent.
- **M4 correlation identity, narrowly scoped:** at optimizer time `n`, the
  positive drive is `c_n=J_{n-1}^dagger g_n^on`; the descended
  meta-gradient is `-LR*c_n`. For corrected complex `B` blocks exactly,
  `c^B_{n,j}=sum_k conj(G^B_{n,jk})G^B_{n-1,jk}` (max relative error
  `3.33e-16`). The `(rho,theta)` recurrence block has additional derived
  reparameterization factors and is not assigned this raw identity; readout
  `c` is uncorrected. EMA/Adam-filtered optimizer-time correlation has raw
  phase MRL `0.894–0.966` / `0.838–0.916` with learned `w`, and EMA has
  `0.792–0.883` with sequence-time `c_g^stat`, but common phase
  concentration remains a confound.
- **M5:** B6/B7 remain the decisive phase-specificity controls. At K=1500,
  static-oracle median `|arg w_C|` is `1.013/0.933/0.736/~0` rad and median
  magnitude is `10.45/3.94/2.55/1.00` for L0–L3.
- **M6:** not feasible. No action-stable `hat w_F` exists at the 20 tested
  checkpoints, so target motion, tracking lag, and curvature were not
  compared.
- **M1 null amendment:** 128 deterministic within-layer learned-value
  shuffles and 128 marginal-matched random complex controls per checkpoint.
  Learned-minus-shuffle/random median `F_{n+1}` is
  `-1.23e-5/-1.24e-5` at K500, `+0.64e-6/+4.93e-6` at K1000, and
  `-0.76e-6/+3.03e-6` at K1500. Learned beats the two per-snapshot null
  medians on `10/11`, `7/7`, and `8/7` of 15 seeds. **No learned-controller
  specificity.**
- **M4 credit-specificity amendment:** for the exact complex-B drive
  `K_drive=conj(g_n^dagger g_{n+1})`, K1500 online MRL versus `c_g^stat` is
  `0.773/0.797/0.757/0.867`, but correct-minus-256-shuffle is only
  `+0.018/+0.039/-0.022/+0.044` for L0–L3. Exact-gradient controls give MRL
  `0.889/0.901/0.758/0.867` and specificity
  `+0.024/+0.030/+0.007/+0.039`. Although online/exact gradient relative
  error is `0.979/0.986/0.912` in L0–L2 and `2.0e-15` in exact L3, the
  correspondence is not stronger for online than exact gradients and does
  not localize below the top exact layer.

**Mechanism verdict:** static alignment, actual one-step utility of learned
`w`, and optimizer-aware one-step optimality all fail as explanations. The
exact B-block lag-one correlation and reset-Adam result identify real signal
and optimizer dependence, but learned `w` fails shuffled/random M1 nulls and
the correlation fails shuffle/exact/top-layer specificity controls.
`M1_specific=false`, `M4_specific=false`: **the pre-registered stopping rule
fires. Stop the mechanism search; do not automatically add a multi-step
counterfactual or algorithm variant.** The training benefit remains
clipped-Adam-mediated but its temporal-credit bridge is unsupported.

## 14. S5 clipping contract and Stage 0

- Historical compatibility restored: `train_bench.py --clip 0` is the
  default (unclipped Adam); cluster commands state clip explicitly.
- Every new metrics JSON retains `p_clip`, pre-clip norm distribution, and
  `chi=||g_pre||/C` distribution when C>0 (`chi.defined=false` when C=0),
  plus numeric structural exact-gradient/exact-lambda/BPTT counters.
- Every run has a manifest with git commit, arm, seed, clip threshold,
  optimizer/MetaOpt, sequence/model config, audit counters, clipping
  diagnostics, memory, and throughput.
- `scripts/stage0.sbatch` / `stage0_grid.sh` run only the matched
  `{BPTT,Online} x {clip=0,clip=1.0}` matrix. `stage0_report.py` requires
  at least three paired seeds, a positive clipped Online→BPTT gap on every
  seed, and median relative headroom ≥0.2 before allowing the small
  correction pilot. The large sweep remains unlaunched.
