# diagnostics/ — mechanism diagnostics

Experiments that explain **why** PC0/routeA work. All toy-rig. Numbers are
frozen in `../RESULTS_LEDGER.md`; one-line verdicts in `../EXPERIMENTS.md`.
Run from repo root as `python -m diagnostics.<name>`.

## Factorization and geometry ceiling
| script | what it establishes |
|---|---|
| `d1_exact_credit_factorization.py` (formerly `prospective_offline2.py`) | D1: eligibility × exact D⁻¹ at the correct upper-layer site == BPTT to 2.4e-15. Also Stage-A v2 fixed-operator arms (all ≤ cstat < base). |
| `d2_modal_oracle.py` (formerly `oracle_real_vs_complex.py`) | D2: held-out per-mode-complex oracle 0.901 vs real 0.765 vs identity 0.596 — phase is representationally valuable. |
| `gradient_cstat.py` | the gradient-level statistical projection cg^stat tracks the oracle exactly where isolated-mode structure holds (top layer 0.998). |
| `phase_probes.py` | defect localization: top recurrent layer online gradient is EXACT; the cross-layer instantaneous-error defect lives in shallow layers; arg w ≈ arg c* deep. |
| `spectrum_check.py` | the real/complex tie is NOT a low-frequency artifact (error signal broadband, ~300× outside Taylor condition). |
| `single_mode_control.py` | routing-free single-mode control: first-order Taylor/lead realization fails at realistic bandwidth. |
| `prospective_train.py` | Stage B: a poor pointwise reconstructor can still help closed-loop learning (D3-consistent). |

## Teacher and deficit analysis
| script | what it establishes |
|---|---|
| `teacher_decompose.py` | routeA→PC0 gap: batch-shift free (−0.9%), causal-teacher blindness +28.2%. |
| `e1_e2_identification.py` | E1: causal deficit disproportionately tangential/phase (cos_r 0.596 vs cos_φ 0.488); E2: complex advantage exact-teacher-exploitable, causal exploitation basin-dependent. |
| `eps_perlayer.py` | ε persistence per layer: ρ_φ(1) ≈ 0.79 lower layers, ~0 top. |
| `oracle_lagged_deficit.py` | lag-1 deficit prediction is ACTIONABLE (oracle): closes 81% of the blindness gap, rescues seed 3; correction must be joint radial+tangential. |
| `prospective_kappa.py` | κ-sweep: matched horizon (κ\*≈1–1.5), no stronger prospective tracking; κ=0 == routeA bitwise. |
| `pc_signal_audit.py` | the delayed-online teacher carries a real phase signal (cos 0.81 vs BPTT). |

## Learned-geometry trajectory / statics
| script | what it establishes |
|---|---|
| `phase_track.py` | learned phase trajectory is nearly static; momentum prediction not useful. |
| `optimum_track.py` | the credit-projection optimum MOVES (Var_train/Var_batch ≈ 196); learned w does not follow it — credit optimum ≠ learning-useful geometry. |
| `factorize_w.py` | frozen phase-only e^{i arg w} closes 113% of the online→full gap — phase is the mechanism (routeA regime). |
| `ablation_generic.py` | complex w 0.0016 vs real-only 0.0091 — phase load-bearing in routeA. |
| `transfer_phase.py` | phase is specific (random phases hurt) but task-bound (no transfer at D=200). |
| `matched_phase.py` | staleness (not variance) is the deployment barrier for derived phases; meta-learning buys the final 7×. |
| `pac_probe2.py` / `pac_analysis.py` | w identified as the optimal scalar credit projection (8/8 bars); meta-objective horizon = one step. |
| `d3_figure.py` / `d4_stability.py` / `d4_controls.py` | static credit fidelity ≠ deployment; credit-MSE does not determine the stability margin. |

## RoutePC variants (mechanism-complete, conclusions frozen)
| script | what it establishes |
|---|---|
| `route_pc_factorial.py` | geometry factorial: modal structure decisive; rotation-vs-gain unresolved for the CAUSAL arm (identification gap, cf. D2). |
| `route_pc_pro.py` / `route_pc_pro_drift.py` | TSS residual prospection: CORRECTION ONLY — no exploitable residual drift (third independent static-geometry measurement). |

## Generality + benchmark gates
| script | what it establishes |
|---|---|
| `rot_rnn.py` / `rot_rnn_generality.py` / `rot_rnn_generality2.py` | barrier phenomena generalize to a real 2D-rotation RNN; the large frozen-orientation benefit is S5-specific in magnitude (P3 FAIL at registered bar). |
| `bench_copy.py` / `bench_smnist.py` | CPU headroom gates: copy saturates; sMNIST h=0.12 < 0.2 at CPU budget → cluster phase required. |
