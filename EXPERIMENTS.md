# EXPERIMENTS.md — complete ledger of everything tried and its result

**Program:** import prospective dynamics (NLA/GLE/VLE lineage) into
S5-style state-space models and find where — if anywhere — the
mechanism helps. **Branch:** `research/pesm-s5-spectrum` (all lanes
below; lanes 0–1 reach back to `main` and `research/prospective-credit-s5`).
**Discipline:** every experiment has a preregistered bar fixed before
running; paired seeds/init/data streams across arms; complex-conjugation
gates; post-hoc changes logged. Final-loss metric: mean of last 100 of
1500 steps, delayed copy D=50/T=128 (unless stated), median over seeds.

> **Repository layout (2026-08 consolidation, commit `8cdc4d6`).** The
> scripts below moved into semantic directories; this ledger keeps the
> historical names in the tables. Mapping: the five shared modules
> (`trained_credit_gains`, `co_variational_metric`, `decompose_w_final`,
> `depth_law`, `route_pc`) are now `toyrig/{ssm_rig, route_a, probes,
> train_cell, routepc}.py`; `prospective_offline2.py` →
> `diagnostics/d1_exact_credit_factorization.py`;
> `oracle_real_vs_complex.py` → `diagnostics/d2_modal_oracle.py`;
> `check_route_pc/routeA_meta/online_s5` → `tests/test_pc0_regression /
> test_routepc_jax_meta / test_online_s5_jax.py`; all other experiment
> scripts keep their filenames under `diagnostics/`, `controls/`,
> `archive/` (see each directory's README). Everything runs as
> `python -m <dir>.<name>` from the repo root. Frozen numbers with full
> provenance: `RESULTS_LEDGER.md`.

**Headline scoreboard:**

| lane | experiments | positive | negative / closed |
|---|---|---|---|
| 0 forward SSM prospection | 2 | 0 | 2 (memory cancellation; parasitic mode) |
| 1 solver/inference | 7 | 5 (solver suite + real data) | 2 (stiff-DEQ training; BISSM) |
| 2 credit filters | 5 | 1 (retracted) | 5 |
| 3 co-variational metric + audit | 9 | 2 (routeA; factorization) | 7 |
| 4 PAC (derive the phase) | 8 | 1 (probe, 8/8 bars) | 7 (deployment ≤40%) |
| 5 baselines & controls | 5 | — | 5 (all decisive) |

The two standing positives: **the PESM solver** (derived, real data)
and **routeA** (learned, 14×, mechanism identified). Everything else is
closed with a named mechanism. The credit lane's final theory: causal
credit's missing piece is the anti-causal part of the adjoint operator,
whose H∞ (Hankel) distance is |a|/(1−|a|²) per mode — diverging with
mode slowness; linear causal reconstructions reach cosine 0.7+ with
horizon but destroy training out of regime (deploy K=64: −6.97); the
bounded per-mode phase is the stable remnant, and the learned phase is
the unique object both correctly oriented and stable along the
trajectory.

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

## Lane 7 — benchmark gates (CPU phase of the applied question)

| script | question | result | verdict |
|---|---|---|---|
| `bench_copy.py` | copy task (encode 20 / gap D / readout 20): headroom + routeA arms | D=50: h=1.00 at 800 steps but **saturates by 1500** (online 0.0001 ≈ bptt 0.0000 — all arms converge); D=100: h = −5.1 (noise) | **degenerate discriminator** — the toy copy regime is exhausted; gates must measure headroom at convergence, not mid-training |
| `bench_smnist.py` | sMNIST (T=196, subset 5k, N=32, L=2): online vs bptt credit gap | online 1.4212 / bptt 1.2492 train loss at 800 steps; **h = 0.12 < 0.2 bar**, but the absolute gap grows through the whole budget (0.017 → 0.30) | **no headroom at registered CPU budget**; trend positive — the real gate needs the cluster phase (bigger model/data/budget + faithful `online_full` reproduction) |

## Lane 6 — closed-loop temporal credit (the north star, CLOSED_LOOP.md)

| script | question | result | verdict |
|---|---|---|---|
| `orient_wiener.py` | does Wiener orientation (gain removed) survive deployment? | all arms worse than online; deployed cosine collapses to 0.05–0.23 | static orientation doesn't transfer |
| `matched_phase.py` | horizon vs granularity vs rate, in routeA's function class | frozen96 +0.41; refresh worse than frozen; **per-batch closed-form phase +0.65** | staleness is the barrier; meta-learning buys the final 7× |
| `d4_stability.py` + `d4_controls.py` | does credit-MSE control stability? | corr(MSE, η_max) = −0.287; ordering reverses at slow modes; Adam absorbs gain; at \|a\|=0.995 phase GD converges where online diverges | **gate A landed (defensible form)** |
| `d3_figure.py` | does static fidelity predict deployment? | best static objects at the bottom of deployment; best deployers statically modest | central figure data committed |
| `phase_track.py` | is the learned phase a predictable moving object? | hold error 0.0017 rad; momentum +6% < 20% bar | learned-trajectory prediction not useful |
| `optimum_track.py` | does the credit-projection optimum move? | Var_train/Var_batch ≈ 196; learned w doesn't follow it | **credit reconstruction optimum ≠ learning-useful geometry** |
| `rot_rnn.py` + `rot_rnn_generality.py` | do the phenomena replicate in a real 2D-rotational RNN? | rig FD-gated to 1e-9; P1/P2 replicate; P3 uninterpretable (no headroom at D=50) | barrier phenomena generalize |
| `rot_rnn_generality2.py` | same, headroom-gated (D=20, headroom 0.54) | routePhi < online all seeds but misses the 0.7× bar (0.0698 vs 0.057); frozenPhi ≈ online; scalarGain worse; perbatchOracle worst | **P3 FAIL — the win's magnitude is S5-specific; the ordering generalizes** |

| script | question | result | verdict |
|---|---|---|---|
| `wiener_oracle.py` | is causal credit a Wiener–Hopf problem — what is the LTI ceiling and does it deploy? | static cosine rises 0.32→0.73 (L3) with K=1→96, unsaturated; Hankel floor σ ≈ 11–14/mode (max ~62), corr 0.99 with \|a\|/(1−\|a\|²); Procrustes U1≈U2≈U4 (per-mode, not subspace); **deploy K=64: −6.97 of gap, one seed 26× blow-up** | **ceiling real but fixed-point; LTI deployment catastrophic — barrier confirmed at training level** |
| `orient_wiener.py` | does Wiener's long-history *orientation* (gain removed) survive deployment? | all arms worse than online (orient1 −0.9 → orient96 −6.2); deployed cosine collapses to 0.05–0.23 (vs static 0.32–0.73) | **NO WIN at every K** — static orientation doesn't transfer; per-timestep orientation is noise |
| `matched_phase.py` | matched function class: Wiener K reduced to constant per-mode rotation, frozen/refresh/per-batch | frozen K≤32 −0.9; frozen64 −0.7; frozen96 +0.41; refresh-200 worse than frozen; **per-batch closed-form phase 0.0111 (+0.65)** — best derived arm | horizon pays only at K≈96; staleness (not variance) is the deployment barrier; meta-learning buys the final 7× |
| `d4_stability.py` + `d4_controls.py` | does credit-MSE optimality control learning stability? | exact minimal model (single complex mode, quadratic): at |a|=0.95 Wiener has best MSE (0.54) and ρ>1 divergence (P=1.00) while online/phase converge; **corr(MSE, η_max) = −0.287** — margins REVERSE the MSE order (exact credit smallest margin; gain is what shrinks it); tuned rates all converge; Adam absorbs gain instability everywhere; at |a|=0.995 phase GD converges where online diverges | **gate A (defensible form): credit reconstruction error does not determine the stability margin**; slogan "better credit → worse learning" holds at shared step size, not at tuned rates |
| `tbptt_baseline.py` | does buffering W steps of exact credit beat the streaming rule? | medians: tbptt1 0.178, tbptt4 0.152, tbptt16 0.118 (all worse than online 0.0284); **tbptt64 0.0003 (works, beats routeA 0.0015 ~5×)**; bptt ~0.00003. (Complex-dtype bug caught by ComplexWarning, fixed, rerun clean) | buffered-64 wins on loss; streaming edge = O(1) memory, no backward pass; truncation below delay always loses to online |
| `test_holonomy.py` | is the shallow phase additive down the stack (holonomy)? | 0/3 seeds pass (increment concentrations 0.20–0.89 < 0.7 bar) | **NO HOLONOMY** — connection reading is decoration |
| `covariant_adam.py` | is the defect Adam's broken U(1) covariance? | seed 0: 0.0727→0.0028 (variance normalization rescue); healthy seeds ~1.5× worse; median frac **−0.15** | **BAR FAIL — gauge reading rejected; defect architectural (.real routing)** |
| `lr_control.py` | is any win just a learning rate? | best standard-Adam LR median 0.0136 — nothing near 0.0028 | rescue is structural, not rate. Note: task is bistable; float-order noise flips basins across processes — within-script pairing is the robust unit |

## Lane 8 — the applied benchmark harness + causal orientation learning (routePC)

| script | question | result | verdict |
|---|---|---|---|
| `train_bench.py` + `scripts/stage0*.{sh,sbatch}` + `stage0_report.py` | S5 Stage 0: paired {BPTT, Online} × {clip=0, clip=1.0}, matched seeds/model/data; validation, gap, p_clip/chi, memory, throughput, finiteness, numeric exact/BPTT audit counters | historical `--clip 0` default restored; every run writes explicit clip/audit manifest; Stage 0 report predeclares ≥3 seeds, positive clipped gap on every seed, median h≥0.2 | **READY, UNLAUNCHED** — run this gate before any correction pilot; no large sweep |
| `scripts/bench.sbatch` + `bench_grid.sh` + `bench_report.py` | full cluster benchmark and mechanism sweep | launcher states historical `--clip 0` explicitly; report includes routePCadam/routePCphase and excludes Stage 0 tags | **BLOCKED pending Stage 0 + small correction-pilot report** |
| `check_routeA_meta.py` | is the new autodiff machinery correct? | rotation vs numpy rig 7e-16; w=1 bitwise 0.0 (float32); no-meta fallback bitwise 0.0; teacher remap 4e-8; nested meta-gradient vs float64 FD 1.6e-6 (norm-referenced) | **ALL 4 GATES PASS** |
| `route_pc.py` | can the orientation be learned with ZERO BPTT — teacher replaced by the realized online gradient on the next batch (delayed correction; optional Simonetto prediction)? | medians: online 0.0224 / routeA 0.0016 / **PC0 0.0073** / PC1(β=.25) 0.0132 / PC1(β=.5) 0.0186; PC0 median R_gap **0.90**, beats online 4/5; BPTT_CALLS(PC) = 0 audited; caveat: \|w\| drifts to 30–1600 (Adam absorbs), and routeA itself basin-flips at seed 3 (0.0248) | **GATE PASS — causal orientation learning works on the toy**; prediction NOT load-bearing (correction-only retained; Simonetto stays dead) |
| `pc_signal_audit.py` | frozen audit: do h_same = g_online(θ′; B) and h_next = g_online(θ′; B′) carry the BPTT teacher's phase signal at all? | h_next (routePC's rule): cos vs BPTT 0.811, phase sign agree 0.625, phase-energy 0.433, cos vs realized Δw_RA 0.287; h_same: 0.876 / 0.703 / 0.378 / 0.350; BPTT ref phase-energy 0.373 | **SIGNAL PRESENT for both**, but B7/M1 show that signal presence is not a sufficient mechanism: phase correspondence is weakly mode-specific and learned `w` has no actual accumulated-Adam one-step utility |
| `route_pc_pro.py` | does TSS/Simonetto prospection of the META-RESIDUAL, r̂^pro = r̂ + κ(r̂ − r̂_{n−1}), beat correction-only? (NOT w-momentum — separate control arm) | synthetic fixed-w gate (corrected after review: the first version differenced across w and showed artifacts — κ-invariant drift lag, a fake κ stability boundary): e* = −v/α + κv exact, zero lag at κ=1/α, κ=50 overshoots but never diverges, sinusoid 32× at κ=1/α. SSM sweep: κ=0 **bitwise** reproduces PC0; stationary medians κ>0 ∈ [0.0069, 0.0145] vs PC0 0.0073 (wins ≤ 3/5, no coherence); moving-delay ramp: 0/5 wins for EVERY κ; wmom control 0.0132 (worse); zero BPTT calls audited | **CORRECTION ONLY** — the TSS term is correct-by-construction but there is no exploitable residual drift on this task class; third independent measurement of a static useful geometry (phase_track, optimum_track, now this) |
| `route_pc_pro_drift.py` | registered pre-measurement: systematic residual drift vs batch-noise floor, complex-vector form | t_vec(Δr̂) = 0.00–0.02 (no systematic increment drift; noise floor ~1), ac1(Δr̂) +0.21 stat / +0.09 mov, while the residual itself is persistent (ac1(r̂) 0.84/0.91) | **drift buried in noise → κ*≈0 confirmed as the predicted outcome**; increments carry no extrapolable signal |
| `check_route_pc.py` | freeze/regression of the deployable causal arm | stored PC0 bitwise on 5/5 seeds; paired-stream hashes identical; h_n bitwise w-invariant; 0 BPTT calls | **ALL PASS — PC0 frozen** |
| `route_pc_factorial.py` | geometry factorial: global-real / global-complex / per-mode-real / per-mode-complex | medians 0.0208 / 0.0324 / 0.0164 / **0.0073**; PC beats global-complex 4/5 (−77% median) but per-mode-real **wins 3/5 paired** (s0,3,4); bar FAILS | **modal structure decisive; rotation-vs-gain NOT resolved for the causal arm** — a live-learned per-mode real gain is at least as good on 3/5 seeds (factorize_w's frozen-phase story is a different object) |
| `teacher_decompose.py` | Route A→RoutePC gap: A same-batch exact / B next-batch exact / C next-batch causal, identical J_n and timing | medians A 0.0016, B 0.0014, C 0.0073; **batch-shift cost −0.9% of gap; causal-teacher blindness +28.2%**; r_exactNext vs r_causal: cos +0.854, norm ratio 1.04, Δφ 0.18 rad | **delay is free; the whole gap is teacher blindness** |
| `phase_probes.py` | per-layer falsification + arg w vs arg c* + frozen noise floor | top layer (l=3) gradient **exact**: rel 0.0, cos 1.0, every checkpoint/seed (RTRL identity: S-slot and J-slot contractions are the same double sum regrouped — no within-layer temporal defect exists in the gradient); error grows shallow (cos 0.2–0.9), phases largest where the error lives; arg w vs arg c* weighted MRL: deep 0.79–0.96, shallow 0.38–0.88; frozen batch SNR ≈ 8/mode; training-time residual motion ~450× the frozen floor but directionless | **the phase repairs the cross-layer instantaneous-error defect, concentrated in shallow layers**; prospection fails from directionless trajectory motion, not batch noise |
| `prospective_ops.py` + `prospective_offline.py` (Stage A v1) | fixed analytic prospective operator (demodulated c0*=1/(1−r), c1*=r/(1−r)² lead) at the credit pathway | base 0.440 > gain 0.307 > oppphase 0.269 > matched 0.249 > raw 0.174 (pooled gradient cos vs BPTT) | **INVALID placement**: filtered the already-routed q_l with pole a_l; pole/placement audit — the pathway operator is D_{l+1}⁻¹ at the UPPER site before routing (D_l⁻¹ comes free via Sa_l eligibility, RTRL identity); recorded only as "mis-placed pole-only approximation fails" |
| `prospective_offline2.py` (Stage A v2) | corrected site recursion + 4 diagnostics | **D1: exact-D⁻¹ factorization cos 1.000, rel 2.5e-15 all seeds** — factorization/placement validated; D2: per-mode-complex gradient oracle ceiling **0.901 held-out = in-window**; corrected arms: base 0.597 > cstat 0.448 > oppphase 0.340 > gain/raw/ema/matched ≤ 0.25 | **the exact prospective-credit factorization and the modal-gradient hypothesis stand**; instantaneous signal still beats every fixed analytic surrogate at gradient level |
| `gradient_cstat.py` | eligibility-weighted analytic scalar cg,j^stat = Σ āᵏR_j(k)/R_j(0), R_j(k)=Σ_t conj(s_tj)q_{t+k,j} | held-out cos: identity 0.596, cg^stat 0.475 (rel 13), z_oracle 0.901; MRL cg^stat≈w_oracle deep 0.87–0.90 / shallow ~0.5; cg^stat≈learned deep 0.89–0.94; **learned≈z_oracle: L3 0.998, L0 0.32** | the gradient-level projection (not signal-level) is the relevant object; it predicts the geometry exactly where isolated-mode structure holds; learned w IS the oracle at the top layer and does not reach it shallow |
| `oracle_real_vs_complex.py` | per-mode-real vs per-mode-complex held-out ceiling (fit 0–3, eval 4–7) | identity 0.596 / real 0.765 / **complex 0.901**; complex−real ≈ +0.14 every seed | **the factorial tie is an identification gap, not representational** — phase is worth real alignment at gradient level; the causal teacher identifies roughly the real-only part (cf. 28.2% blindness) |
| `e1_e2_identification.py` | E1: radial (gain) vs tangential (phase) teacher alignment on identical J_n; E2: crossed teacher×geometry 2×2 | E1: cos_r +0.596 vs cos_phi +0.488, per-seed gap +0.226 (4/5 seeds), tangential carries ~71% of exact residual energy; E2: E_C 0.0014 / E_R 0.0089 / C_C 0.0073 / C_R 0.0164 — Δ_exact +0.0076 (exact teacher: complex 6.4× better), Δ_causal −0.0058 (causal: none), interaction +0.0134 (3/5) | **BOTH registered predictions CONFIRMED — the causal teacher's blindness is disproportionately phase/tangential blindness** (not exclusively; radial deficit exists). Next algorithm should target phase identification specifically |
| `prospective_kappa.py` + `eps_perlayer.py` | registered prospective-objective sweep r^(κ) = (1−κ)r^(0) + κ r^(1), κ∈{0,.5,1,1.5,2,4}, exact teacher; ε = r_causal − r_exact directional persistence | κ medians 0.0016 / 0.0029 / 0.0014 / 0.0013 / 0.0016 / 0.0025 — **κ*≈1–1.5, plateau to 2** (κ=0 bitwise == routeA; κ=1 bitwise == arm B — the two timings are the same recursion relabeled); ρ_r(1) pooled lower +0.760/late +0.665, **ρ_φ(1) +0.787/late +0.711, per-seed 0.78–0.85, per-layer [0.77–0.82, ~0 at top]**, tangential energy 0.60 | **stronger prospective tracking NOT supported (matched horizon); the phase-teacher deficit IS temporally predictable** — but not via objective-time residual extrapolation (κ proved that); it must be harvested on the deficit's own persistent direction |
| `oracle_lagged_deficit.py` | oracle lagged-deficit test: r_causal corrected by −ρ·ε_{n−1} (fit window 250, held-out; per-mode LS lag-1 coefficients) | A1 == PC0 bitwise; medians: A1 0.0073 / A2 (radial-only) 0.0157 / A3 (tangential-only) 0.0088 / **A4 (full) 0.0025**; A4 beats A1 4/5 paired, **closes 81% of the A1→B blindness gap on median, rescues the seed-3 basin (0.0889→0.0053)**; single-component arms useless-to-harmful — correction must be joint (P2 verdict: GENERAL/mixed, NOT tangential-exclusive) | **the persistence is ACTIONABLE — lag-1 deficit prediction works**; but it's an oracle (ε needs BPTT for bookkeeping). The deployable question is now exactly: estimate ε without BPTT |
| `bootstrap_teacher_f1.py` | F1: causal bootstrap teacher g^teacher(α) = [(1−α)I + αM_w̄]g^on, w̄ = stop-grad EMA (β=0.99) of learned w; α ∈ {0, 0.5, 1} | α=0 bitwise == PC0 ✓, training-phase BPTT = 0 ✓, step-1 identical ✓; **α=0.5: NaN on 2/5 seeds, 0.093–0.137 elsewhere; α=1.0: NaN on 3/5, 0.098–0.240; 0/5 paired wins both; PRIMARY PREDICTION NOT CONFIRMED** | **the bootstrap teacher self-amplifies the geometry's own errors** — (I−M_w̄)g^on is NOT a sufficient causal measurement of the persistent deficit. Per the pre-registered rule: **STOP — no latent observer stage (z_n) built** |
| `package_core_routepc.py` + `results/core_routepc_reproduction/` | standalone core-algorithm artifact (read-only packaging; no retraining) | online 0.0224 vs PC0 0.0073 median, 4/5 paired wins, relative improvement 0.674 (per-seed values incl. seed-3 reversal reported plainly); zero BPTT both arms; per-seed RNG hashes all equal; stop-grad probe PASS | **the base algorithmic result stands alone, separated from all mechanism/v2 work** |
| `control_2x2_normmatch.py` | the real 2×2 control (online/PC0/BPTT/BPTT+w) + PC0_normmatched + D2 per-seed retrieval | BPTT+w ≈ BPTT on every seed (±2e-05) — M_w on exact gradients does nothing; interaction I_i median **+0.0193** = full PC0 gain on healthy seeds; Δ_credit positive on all 5 seeds; PC0_normmatched median **0.0073 == PC0** while ‖M_w g‖/‖g‖ pooled median is 79.7 (up to 564) — Adam absorbs the gain; D2 complex held-out per seed 0.813–0.977 (no catastrophic seed), real 0.673–0.885 | **credit-regime-specific interaction, not generic preconditioning**; later G3X/B/M audits restrict this to clipped-Adam closed-loop behavior, not static BPTT reconstruction or optimizer-independent repair |
| `prospective_train.py` (Stage B, v1 arms, unchanged) | can a poor pointwise credit reconstructor improve closed-loop learning? | matched median 0.0112 < online 0.0224 (3/5 paired wins), while Stage A showed the same arm DEGRADES pointwise alignment | D3-consistent data point only; not evidence about the corrected operator |
| `spectrum_check.py` (Stage C) | is the real/complex tie a low-effective-frequency artifact? | weighted mean \|arg c*\| = 0.49 rad (bar 0.2), 43% of weight < π/8; demodulated error rotates ~1.6 rad/sample off the pole (narrowband allowance ~0.005) | **tie NOT low-frequency**; the network's error signal is broadband/off-resonance ~300× outside the Taylor condition |
| `single_mode_control.py` (D4) | routing-free single-mode bandwidth control at r=0.995 | Taylor arms rel 8–32,000 in ALL regimes (even 0.03 rad/sample narrowband ≫ 0.005 asymptotic); cstat optimal scalar but any scalar ≈ identity at signal level (cos 0.16–0.27) | **first-order prospective/Taylor REALIZATION fails at realistic bandwidth** (not a claim about the mathematical limit); per-timestep signal fidelity is not the binding constraint for the gradient |

## Addendum controls C1–C3 (claim-sharpening; branch `controls/c1-c3`)

| script | question | result | verdict |
|---|---|---|---|
| `controls/c1_phase_only_routepc.py` + `c1b_phase_only_15seeds.py` | does unit-modulus (phase-only) RoutePC retain the benefit and improve stability? | 5 seeds: pcPhase 0.0085 vs PC0 0.0073 vs online 0.0224 — COMPETITIVE per registered rule (beats online 4/5; pcPhase/PC0 median ratio 0.686). 15 seeds: pcPhase median **0.0137** vs PC0 **0.0167** vs online 0.0226; failures (ratio>1) pcPhase **4/15** [3,6,9,13] vs PC0 **6/15** [3,6,7,8,12,13]; pcPhase beats PC0 9/15 paired, beats online 11/15. All replay gates bitwise, BPTT 0/0 | **unit modulus improves stability** (fewer catastrophic seeds) at some cost where the gain channel helped (s1, s11); registered as selectable arm `--arm routePCphase` in `train_bench.py` (w saved unit-modulus verified); PC0 preserved. Honesty: the 5-seed headline is a luckier draw — over 15 seeds PC0's median ratio is 0.861 with 6/15 failures on this bistable task |
| `controls/c2_real_w_diagnostics.py` | why is the causal per-mode-REAL geometry competitive (Adam normalizes gain)? | **Pr(w_j<0) = 0.000 everywhere — zero sign flips ever**; \|w\| heavy-tailed, depth-increasing (medians L0 1.8 → L3 29.1, max 2724); relative \|Δw\|/\|w\| median < 5e-4 (quasi-static) | the real arm's effect is **relative modal gain structure** (directional reweighting before clip/Adam) — NOT sign flips, NOT time-varying gain; consistent with E1 radial>tangential and the factorial tie being identification-limited |
| `controls/c3_matched_budget_bptt_w.py` | is BPTT+w≈BPTT a floor artifact? test at budgets where BPTT still has headroom | budgets K=129/184/280 (+1500), all four arms retrained with curves, bitwise gates 4/4; Δ_credit positive 5/5 at every K; **BPTT+w WORSE than BPTT at every budget on (nearly) every seed** (K=280 median 0.0024→0.0114); interaction I(K) medians +0.0022/+0.0037/+0.0120/+0.0211 | **generic-preconditioning hypothesis rejected WITH headroom present, at every budget** — the defective-online-credit geometry miscalibrates exact credit; later audits show the preferential benefit is clipped-Adam/path-dependent, not static exact-credit reconstruction |

## Modal-geometry audit (G-program; branch `geometry/modal-audit`; full record: `FINAL_MODAL_GEOMETRY_AUDIT.md`)

| script(s) | question | result | verdict |
|---|---|---|---|
| `controls/g0_cartesian_conditioning.py` | does large \|w\| self-anneal the Cartesian-SGD meta-learner into bad basins? | \|Δα\|,\|Δφ\| GROW with ρ (slopes +0.3..+0.6, not −2); ρ ~1.0 until AFTER basin entry on 5/6 failures; successes end at the same ρ (~13); clip fire rate 1.000 everywhere | **FALSIFIED by timing and by success controls** — radial drift is orthogonal to failure |
| `controls/g1_polar_arms.py` + `ga_polar_lr_sweep.py` | free vs gauge-fixed log-polar; Cartesian+Adam MetaOpt | polar NaN 5/5 at original LR (LR mismatch — at η·1e-3/1e-4 finite, beats online 5/5, median 0.0137, not competitive); pc0_adam sane | polar branch **closed** (no advantage); Adam MetaOpt controls the radius |
| `controls/g3_clipping_check.py` + `g3x_c3_noclip.py` | does the benefit survive removing the clip? | clip→noclip relative benefit: real +0.268→+0.156, PC0 +0.674→+0.294, pcPhase +0.620→+0.132; no-clip C3 interaction ≈ 0 every budget | **the benefit is mediated by the clipped-Adam update geometry**; without it the correction is nearly behaviorally inert |
| `controls/gc_displacement_gauge.py` | is common radial scale a true gauge? | corr(ρ,‖Δθ‖) negative (−0.91/−0.73); κ≤100 common scaling: post-clip direction cos ≥0.9990, Adam update cos ≥0.99996 | **near-exact gauge** (c blocks ~0.2% of norm) — gauge language quantitatively supported |
| `controls/gd_action_jacobian.py` + `ge2_15seeds.py` + `ge2b_radial_decomposition.py` | does respecting the clip+Adam Jacobian in the residual fix the geometry? | E1 0.0175 / E2 0.0121 (5 seeds, sane); E2→15: 0.0167, 6/15 fails; ρ≈1.0 exactly; sd(log\|w\|)≈0 (degenerates to phase-only); R_A: common radial action-null (0.113), per-layer relative substantial (0.6–0.87); radial residual ~93% relative-subspace (common share 0.069) | **E2 = mechanistic action-Jacobian control, not primary** — it pins the radius by discarding the load-bearing relative-gain channel |
| `controls/gb_adam_15seeds.py` + `gaa_action_adam.py` | residual × MetaOpt 2×2 completion | pc0_adam 0.0120, 3/15 fails (exact ratios 1.253/1.767/1.165; not all marginal), beats online 12/15, radius ~1.5–1.8 with sd(log\|w\|) 0.14–0.16 retained; AA 0.0191, 5/15 | **pc0_adam is the best cell; AA adds nothing** |
| `controls/g4_g5_oracle_geometry.py` | do full-2×2 or low-rank cross-mode exceed the 0.901 complex ceiling? | 2×2 0.922, rank1 0.908, rank2 0.931 held-out | gains +0.02/+0.03 — **not material; no causal implementation** |
| `controls/g7_failure_audit.py` + `ge_failure_signature.py` + `g8_statistics.py` | failure taxonomy + statistics | every failure = persistent meta-residual explosion (RESIDUAL_SPIKE); teacher-alignment collapse REFUTED (failures' cos if anything HIGHER); only pc0_adam has a supported win over online (sign p 0.035, Wilcoxon 0.008) | **primary S5 candidate: pc0_adam**; controls: PC0, pcPhase, E2, AA |

## Bridge audit (analysis-only; branch `geometry/modal-audit`)

| `controls/b1_b4_bridge_audit.py` | B1–B5: is learned pc0_adam a static exact-credit correction? | ΔC≈0 at every checkpoint/layer despite oracle 0.82–0.99; raw phase MRL 0.80–0.96; transplant pooled identity/self/offdiag 0.361/0.385/0.366; exact failure ratios 1.253/1.767/1.165 | **not static exact-credit reconstruction**; high raw phase MRL required shuffle-specificity audit |
| `controls/b6_b8_bridge_audit.py` | B6–B8: does analytic gain rescue learned phase; is phase MRL mode-specific; do transplant/D2 differences survive uncertainty? | B6 hybrid is null-to-worse (K1500 paired hybrid−learned +0.002/−0.037/−0.068/−0.095); B7 correct−shuffle MRL only +0.039/+0.012/−0.020/+0.056; B8 offdiag−identity +0.00018, bootstrap interval crosses zero. D2 0.596 vs bridge 0.361 is primarily global (L0–L3+readout) vs lower-only aggregation (+0.183 paired at the same RoutePCAdam params), not checkpoint mismatch | **learned-phase/analytic-gain explanation NOT supported; high raw MRL is weakly mode-specific; shared-defect transplant claim unsupported. Link from residual phase correspondence to clip-mediated training benefit remains OPEN** |
| `controls/m1_m6_action_mechanism.py` | M1–M6: does learned `w` help the actual next-batch objective; how does it compare with static-credit and clip+Adam action oracles; which optimizer state and residual blocks matter? | M1 learned−identity `-4.3e-6/+1.2e-5/+2.4e-5` at K500/1000/1500 (8/6/4 of 15 improve), while static `w_C` gives `-1.20e-3/-1.36e-3/-0.41e-3`; M3 reset-Adam K1500 `-0.85e-3` (12/15) but clipped SGD and accumulated Adam null-to-worse; M2 `hat w_F` unstable in action space 0/20 and learned action/regret is identity-like, while `w_C` is closer; M4 exact lag-one complex correlation verified only for corrected `B` blocks (≤3.33e-16), with EMA phase MRL 0.894–0.966 vs learned `w`; M6 infeasible | **no static or actual one-step explanation for learned `w`; raw `w-w_F` distance is non-diagnostic. The amended specificity controls below determine the stopping verdict** |
| `controls/m1_m4_specificity.py` | amended stopping controls: does learned M1 utility beat mode-shuffled and marginal-matched random `w`; does M4 correspondence beat `c_g^stat` shuffles, exact-gradient correlations, and the exact top-layer null? | K1500 learned−shuffle/random null median `-0.76e-6/+3.03e-6`, beating null medians only 8/15 and 7/15; online B-block correct−shuffle MRL `+0.018/+0.039/-0.022/+0.044` vs exact `+0.024/+0.030/+0.007/+0.039` at L0–L3. Despite online/exact error `0.979/0.986/0.912` in lower layers and `2.0e-15` at top, correspondence is not online-specific or lower-layer-localized | `M1_specific=false`, `M4_specific=false`: **PRE-REGISTERED STOP. Temporal-credit bridge unsupported; no multi-step diagnostic, geometry, or optimizer variant** |

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
buffered exact credit (tbptt64) on loss. New this lane: the orientation
need not come from a BPTT teacher — a fully causal delayed correction
off the realized online gradient (routePC, correction-only) closes ~90%
of the toy gap with zero BPTT calls, and the frozen audit shows the
delayed-online teacher carries the phase signal (cos 0.81 vs BPTT,
sign agreement 0.63); prediction remains useless there too. Physics
unifications kept as framing: Pontryagin/symplectic blocks, GENERIC
split, Mori–Zwanzig/FDT
(ρ is the eliminated fluctuation's autocorrelation), MSR
(advanced = Hermitian conjugate of retarded = the phase theorem).

## Reproduce

All commands from the repo root, module-style (`python -m ...`).
Directory READMEs map every script; frozen numbers: `RESULTS_LEDGER.md`.

```bash
python -m tests.test_pc0_regression    # THE freeze gate (bitwise PC0, 0 BPTT)
python -m tests.test_external_rig      # independent-rig identities
python -m tests.test_online_s5_jax && python -m tests.test_routepc_jax_meta
python -m tests.test_scan
python -m core.train_routepc           # canonical protocol (online/routeA/PC)
python -m core.train_online            # online baseline only
# controls:
python -m controls.control_2x2_normmatch && python -m controls.tbptt_baseline
python -m controls.lr_control
# key diagnostics:
python -m diagnostics.d1_exact_credit_factorization
python -m diagnostics.d2_modal_oracle
python -m diagnostics.teacher_decompose && python -m diagnostics.prospective_kappa
python -m diagnostics.e1_e2_identification
python -m diagnostics.oracle_lagged_deficit
python -m diagnostics.phase_probes && python -m diagnostics.gradient_cstat
# archived lanes (see archive/README.md):
python -m archive.solver.pesm_s5_spectrum      # solver positive
python -m archive.forward_prospection.exact_failure
# S5 Stage 0 (UNLAUNCHED; required before correction pilot):
#   bash scripts/stage0_grid.sh <partition> <account> smnist 0 1 2
#   then python stage0_report.py --task smnist
# large benchmark remains blocked pending Stage 0 + pilot report
```
