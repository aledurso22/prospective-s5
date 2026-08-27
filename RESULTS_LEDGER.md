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
exact gradients ⇒ PC0 is **temporal-credit repair, not generic
preconditioning**.

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
| **cluster benchmark** | `train_bench.py` + `scripts/bench*.sbatch/bench_grid.sh` + `bench_report.py` | 8 arms × 3 tasks (+ opt-in `--arm routePCphase` from C1), paired streams, in-job headroom gate h≥0.2, CPU-smoke-clean, **UNLAUNCHED** |

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
the geometry learned for online-credit repair actively miscalibrates
exact credit. Interaction I(K) medians: +0.0022 / +0.0037 / +0.0120 /
+0.0211 (final). **Verdict: the generic-preconditioning hypothesis is
rejected with headroom present, at every budget** — the final-step 2×2
floor was not the limitation; PC0's benefit is credit-repair-specific.

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
  ratio 0.597, 12/15, sign p 0.035, Wilcoxon 0.008; 3/15 marginal-only
  failures).

**Primary S5 candidate: pc0_adam (PC0 + Adam MetaOpt for w).**
Controls: PC0 full complex, routePCphase, E2 (mechanistic
action-Jacobian control), AA. Not carried: polar variants, causal
2×2/low-rank, block-metric MetaOpt (preconditions unmet).
