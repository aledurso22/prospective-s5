# EXPERIMENTS.md — complete ledger of everything tried and its result

**Program:** import prospective dynamics (NLA/GLE/VLE lineage) into
S5-style state-space models and find where — if anywhere — the
mechanism helps. **Branch:** `research/pesm-s5-spectrum` (all lanes
below; lanes 0–1 reach back to `main` and `research/prospective-credit-s5`).
**Discipline:** every experiment has a preregistered bar fixed before
running; paired seeds/init/data streams across arms; complex-conjugation
gates; post-hoc changes logged. Final-loss metric: mean of last 100 of
1500 steps, delayed copy D=50/T=128 (unless stated), median over seeds.

**Headline scoreboard:**

| lane | experiments | positive | negative / closed |
|---|---|---|---|
| 0 forward SSM prospection | 2 | 0 | 2 (memory cancellation; parasitic mode) |
| 1 solver/inference | 7 | 5 (solver suite + real data) | 2 (stiff-DEQ training; BISSM) |
| 2 credit filters | 5 | 1 (retracted) | 5 |
| 3 co-variational metric + audit | 9 | 2 (routeA; factorization) | 7 |
| 4 PAC (derive the phase) | 8 | 1 (probe, 8/8 bars) | 7 (deployment ≤40%) |
| 5 baselines & controls | 4 | — | 4 (all decisive) |

The two standing positives: **the PESM solver** (derived, real data)
and **routeA** (learned, 14×, mechanism identified). Everything else is
closed with a named mechanism.

---

## Lane 0 — prospective term in the SSM forward dynamics (main)

| script | question | result | verdict |
|---|---|---|---|
| `exact_failure.py` | does the exact prospective SSM (A=Λ, no Δ-scaling/clamps) work at real HiPPO init, N=64? | physical root fine at 1−ρ for every ρ; parasitic root μ₁μ₂=A drives overflow at step 13–14/784 regardless of ρ ∈ {0.5, 0.1, 1e-3} | **dead**: full prospection cancels the memory spectrum (τṡ=−s); the two-step Euler adds a parasitic numerical mode; collision threshold severe for slow modes |
| `ghost_demo.py` | stability boundary of the blended prospective step | measured boundary dt ≤ 2γτ (γ=1 stiffness-free) | closed; law used in solver line |

## Lane 1 — the solver metric (the derived positive)

| script | question | result | verdict |
|---|---|---|---|
| `pesm_s5_spectrum.py` | prospective Gauss-Newton on the real bilinear HiPPO spectrum vs GD/Anderson/Broyden | scan==dense 3e-16; one step == S5 rollout 1.5e-15; exact in 1 step on the quadratic chain, quadratic with tanh anchor; GD ~κ steps, Anderson stalls, Broyden diverges; κ up to 2.9e10 | **POSITIVE** |
| `s5_state_inference.py` | synthetic Poisson PLDS, stiff AR(1) latents | Newton 1–4 NFEs to 1e-8 flat across κ=4e2→4e8; L-BFGS false-converges at κ=4e8; Newton MAP == RTS posterior mean 2.8e-15 | **POSITIVE** |
| `plds_benchmark.py` | B1–B4 suite | Kalman gate pass; B4: dynamics gradient at loose solve 99% corrupted at κ=4e8 (solver quality is a training issue for implicit stiff models) | **POSITIVE** |
| `plds_mcmaze.py` | real neural data (NLB'21 MC_Maze, DANDI 000128, 40 trials) | Newton: **2 NFEs, ~5 ms/trial, residual ~1e-9 every trial**; L-BFGS median 2054 NFEs, never converges, false-converges 4/40; GD frozen | **POSITIVE (real data)** |
| `plds_mcmaze_fit.py` | alternating joint-MAP fit (λ, C, d) | Newton 51k vs L-BFGS 1.32M inner NFEs, same outer progress at ~26× fewer evals; held-out LL slightly favors loose arm (learning robust to loose solves) | **POSITIVE** with the loose-tolerance caveat |
| `registered_stiff_deq.py` | does the solver help DEQ *training* (Newton vs GD vs Broyden, G∈{64,256}, 3 seeds)? | P1/P2/P3 all NO WIN; rel_res@500 newton 0.53–0.63 vs gd 0.66–0.70 — not ≤0.1× | **KILL** — mechanisms/negative result |
| `registered_bissm.py` | (BISSM arm, exploratory) | superseded; user deprioritized | closed |

## Lane 2 — credit filters (branch `research/prospective-credit-s5`)

| script | question | result | verdict |
|---|---|---|---|
| `theory_checks.py` | is the prospective filter the matched filter of the credit operator? | phase identity arg H_pro = arg H_BPTT to machine precision (5/5) | **proven**: phase-exact, gain-inverted |
| `gradient_alignment.py` | six estimators × L×\|a\| sweep | prospective filter never beats online RTRL; gain error \|1−āe^{iω}\|²; ±ω mixing identified; four gated nulls pass | **dead as a filter** |
| `optimal_credit_filter.py` | closed-form optimal causal K-tap gains | apparent +0.3–0.45 cosine and transfer in the rig's dense-loss regime | exploratory positive, later retracted |
| `trained_credit_gains.py` v1/v2 | training races, oracle gains | v1 (unconstrained modes): gain arms explode (\|a\|>1 by step ~50); v2 (sigmoid-constrained): registered bar NO WIN (oracle 0.042 vs online 0.024, bptt 0.00003) | **NO WIN** |
| `registered_oracle_b.py` | confirm the v2 3-seed gap (5 seeds, bar fixed) | oracle_B flips to 2.67× WORSE than online on copy; adding task degenerate (~0.083 chance plateau for all arms incl. bptt) | **retracted** — credit lane closed except the co-variational cell |

## Lane 3 — the co-variational metric and its audit (this branch)

| script | question | result | verdict |
|---|---|---|---|
| `co_variational_metric.py` | free the action's metric: per-mode complex w on the descent field, learned online | routeA (meta-gradient): **0.0016 vs online 0.0224 (14×)**, all 5 seeds finite, no boundary-pushing; routeB (consistency residual): NaN all seeds. (Stale-flat bug found + fixed e538eb6; post-fix numbers only) | **POSITIVE (registered)** |
| `ablation_generic.py` | which parts of w are load-bearing? | complex w 0.0016 vs real-only w 0.0091 (phase load-bearing, 6×); unbounded-\|a\| 0.0016 (stability bound not load-bearing) | phase is the mechanism |
| `decompose_w_final.py` | is w the exact-credit correction α? | phase(w) ≈ phase(α) to ~0.1 rad in DEEP layers at trained params; \|w\|/\|α\| overshoot 3–16× growing with depth | partial alignment, deep only |
| `depth_law.py` | is the overshoot a law? | ratio gradient ~0.5 (input) → 50–200× (output) across L∈{1,2,4,8}, D∈{25,50,100} | output-proximal amplification; greedy one-step geometry |
| `transfer_m.py` | does the metric transfer? | meta-trained on delays {25,50,100}, frozen, deployed D=200/T=256: 0.0959 vs online 0.0932 | **NO WIN** — task-specific |
| `recheck_curvature.py` | is w secretly the curvature mass? | corr(\|w\|, 1/curv_a) = −0.03 | no |
| `recheck_curvature_matrix.py` | matrix-level: W_j vs (I+τH_j)⁻¹, τ swept | Hessian symmetric ⇒ zero rotational part, but layer 0 has 71% of W's energy in rotation; corr(\|w\|,‖M‖) to **−0.90 (anti-Newton)**; best-c_l residuals 0.48–0.995 | **curvature-mass story dead** (shape, scale, fit) |
| `factorize_w.py` | phase or gain? | frozen phase-only e^{i arg w} closes **113%** of online→full gap (median 0.0053 vs 0.0284), beats frozen-full every seed; mag-only ≈ full (0.0080) | **phase is the mechanism; gain redundant with Adam** |
| `transfer_phase.py` | is the phase specific? does it transfer? | random phases HURT (+14%: specific); frozen phase ties online at D=200 (0.0935 vs 0.0932: task-bound); capacity-vs-credit confound registered (BPTT headroom mandatory) | specific ✓, transferable ✗ |
| `derive_phase.py` | is the phase derivable from D(ω)? | analytic scalar phase **identically zero** (odd-phase cancellation / pole outside unit circle); α-at-init fails (11%); learned ≠ α in shallow layers | **NOT DERIVED** — scalar phase lives in signal statistics, end-of-training regime |

## Lane 4 — PAC: the phase as optimal scalar credit projection

| script | question | result | verdict |
|---|---|---|---|
| `pac_probe.py` / `pac_probe2.py` | is the learned phase the optimal scalar projection of exact onto causal credit (c* = Σ āᵏρ(k))? | **8/8 bars**: P5 error non-white (\|ρ(1)\| 0.15–0.50); REL ceilings (0.38–0.995); P1 R(c*,w) tracks ceilings (0.92–0.96 top); **CTRL: same at online-baseline params** (task+architecture, not trajectory); P3l AR(1) closure ≥; gates exact to 1e-15; P2 shallow attenuation = cross-layer term | **POSITIVE — what w IS, identified** |
| `pac_analysis.py` (prereg 50879e3) | A: does the resolvent combination beat its factors? B: is the horizon one step? | **A FAIL** — bare arg ρ(1) beats −arg(1−āρ(1)) at L2/L3; resolvent framing dropped. **B PASS** — monotone decline from H=1 (3/4 layers): the action's τ = the meta-objective's one-step horizon | A dead, B stands |
| `pac_deploy.py` | deploy full K = 1/(1−āβ) causally | g05 −0.18, g01 +0.31 of gap | NO WIN (magnitude channel fails again) |
| `pac_deploy2.py` | phase-primary + oracle-β control | phase-oracle **36%** (median 0.0188); EMA −0.06; full-oracle −0.18; align(late) 0.46–0.52 | **directionally right, not load-bearing** |
| `pac_deploy3.py` | raw e^{i arg ρ(1)} (exploratory) | 0.17–0.27, fracs −5 to −9 (~10× worse than online) | catastrophic — raw statistic unbounded/noisy; comb form is the variance stabilizer |
| `pac_deploy4.py` | horizon-1 form + estimation rate + frozen-periodic | c(1)-oracle 28%; slower EMA worse; **frozen-periodic 40% — best derived law**; STABILITY bar: barrier is **variance, not lag** | named barrier confirmed |

## Lane 5 — baselines and controls

| script | question | result | verdict |
|---|---|---|---|
| `tbptt_baseline.py` | does buffering W steps of exact credit beat the streaming rule? | medians: tbptt1 0.178, tbptt4 0.152, tbptt16 0.118 (all worse than online 0.0284); **tbptt64 0.0003 (works, beats routeA 0.0015 ~5×)**; bptt ~0.00003. (Complex-dtype bug caught by ComplexWarning, fixed, rerun clean) | buffered-64 wins on loss; streaming edge = O(1) memory, no backward pass; truncation below delay always loses to online |
| `test_holonomy.py` | is the shallow phase additive down the stack (holonomy)? | 0/3 seeds pass (increment concentrations 0.20–0.89 < 0.7 bar) | **NO HOLONOMY** — connection reading is decoration |
| `covariant_adam.py` | is the defect Adam's broken U(1) covariance? | seed 0: 0.0727→0.0028 (variance normalization rescue); healthy seeds ~1.5× worse; median frac **−0.15** | **BAR FAIL — gauge reading rejected; defect architectural (.real routing)** |
| `lr_control.py` | is any win just a learning rate? | best standard-Adam LR median 0.0136 — nothing near 0.0028 | rescue is structural, not rate. Note: task is bistable; float-order noise flips basins across processes — within-script pairing is the robust unit |

## What the ledger establishes, one paragraph

The prospective principle's derived gifts are exactly two: the
**curvature mass on the state side** (the PESM solver — the only
placement where the discretization is exact and memory survives) and
**adjoint orientation on the multiplier side** (the matched filter —
phase-exact by construction). Every other placement was asked for the
wrong commodity: memory (canceled), filter gain (inverted), rotation
from a symmetric mass (structurally unreachable), derivation of a
scalar phase from the operator alone (identically zero), causal
deployment of that phase (variance-limited, 40% best), optimizer
covariance (not the defect), holonomy (not additive). The learned
per-mode phase rotation (routeA) is the repair for the one scarce
resource — causal credit's orientation — found by meta-learning,
identified as the optimal scalar credit projection, bounded by physical
causality (MSR: advanced propagators unrealizable causally) and by
buffered exact credit (tbptt64) on loss. Physics unifications kept as
framing: Pontryagin/symplectic blocks, GENERIC split, Mori–Zwanzig/FDT
(ρ is the eliminated fluctuation's autocorrelation), MSR
(advanced = Hermitian conjugate of retarded = the phase theorem).

## Reproduce

```bash
python exact_failure.py && python ghost_demo.py        # lane 0
python pesm_s5_spectrum.py && python s5_state_inference.py
python plds_benchmark.py && python plds_mcmaze.py && python plds_mcmaze_fit.py
python registered_stiff_deq.py --grid --summarize      # lane 1
# lane 2 on research/prospective-credit-s5
python co_variational_metric.py && python ablation_generic.py
python decompose_w_final.py && python depth_law.py && python transfer_m.py
python recheck_curvature.py && python recheck_curvature_matrix.py
python factorize_w.py && python transfer_phase.py && python derive_phase.py
python pac_probe2.py && python pac_analysis.py
python pac_deploy.py && python pac_deploy2.py && python pac_deploy3.py && python pac_deploy4.py
python tbptt_baseline.py && python test_holonomy.py
python covariant_adam.py && python lr_control.py
```
