# Phase B31b — training with jointly-trained recurrent-Jacobian families, r=64, d=4

Branch `S5-CCM-scale-validation`. Follows B31a directly. Code:
`credit_memory/b31b_joint_recurrent_training.py` (`main()` reproduces
the numbers below); raw curves archived in
`credit_memory/b31b_training_result.json`.

**Important framing caveat, preserved per explicit instruction**: this
run's optimizer is **SGD plus a deterministic spectral-radius
projection on `R_V`** (`project_stable_R_V`, `rho_max=0.95`), applied
identically to all three (full/reduced/BPTT) parameter copies after
every update. Do not describe this as unconstrained SGD on the raw
parameterization — the projection is part of the training procedure
being tested, not an artifact removed before reporting. Gradient
equality is established *before* the optimizer step; the identical,
deterministic projection preserves exact matching of the three
post-update parameter trajectories.

## 1. Why the projection was needed

Raw SGD (no projection) on the jointly-trained `(R_V,K,B_V,C_V,C_U,
Phi)` family diverged to NaN within ~2–10 sequence-boundary steps at
several learning rates tried (0.03, 0.003 at T=128) — `R_V` is now
trainable with no structural spectral-radius guarantee, the same
recurring instability already diagnosed and fixed in B28's "ours"
architecture and recorded in memory (`b18-init-instability.md`: new
block-recurrent architectures need explicit stability projection). The
fix applied: rescale `R_V` toward `rho_max=0.95` whenever its spectral
radius exceeds it (no-op otherwise), applied to `R_V` only.

At `lr=0.001`, T=128, 100 steps: **stayed finite throughout** — no NaN
in any of the 100 train-loss or 20 validation-loss entries, confirmed
directly from the archived JSON.

## 2. Gradient / parameter equality (steps 0–4, before the optimizer step)

| step | ‖g_full−g_reduced‖ | ‖g_reduced−g_bptt‖ | ‖θ_full−θ_reduced‖ | ‖θ_reduced−θ_bptt‖ |
|---|---|---|---|---|
| 0 | 0.000e+00 | 2.677e-14 | 0.000e+00 | 1.008e-16 |
| 1 | 0.000e+00 | 1.040e-14 | 0.000e+00 | 1.150e-16 |
| 2 | 0.000e+00 | 7.828e-15 | 0.000e+00 | 1.419e-16 |
| 3 | 0.000e+00 | 7.056e-15 | 0.000e+00 | 1.582e-16 |
| 4 | 0.000e+00 | 3.127e-15 | 0.000e+00 | 1.618e-16 |

Full and reduced RTRL agree exactly (0.0) throughout; either agrees
with BPTT at float64 machine precision, with **no growth** across
steps at this learning rate (contrast the earlier divergent lr=0.003
attempts, where `‖g_reduced−g_bptt‖` grew from ~1 to ~1e105 within 2
steps as the trajectory went unstable — a floating-point-conditioning
effect of an ill-conditioned/near-unstable trajectory amplifying
differently under BPTT's single reverse-mode graph vs the per-step
RTRL loop, not a mathematical disagreement; full and reduced stayed
bit-identical to each other even then).

## 3. Training result (100 steps, T=128; 20 held-out validation sequences)

| | full | reduced | bptt |
|---|---|---|---|
| initial train loss (step 0) | 2.7349 | 2.7349 | 2.7349 |
| final train loss (step 99) | 3.1796e-02 | 3.1796e-02 | 3.1796e-02 |
| mean validation loss (20 sequences) | 2.6124e-02 | 2.6124e-02 | 2.6124e-02 |

All three identical at every logged step. Sample points: step 10:
9.553e-2; step 40: 5.714e-2; step 70: 2.015e-2; step 99: 3.180e-2
(noisy but clearly down from the ~2.73 start).

## 4. Recurrent-parameter movement

Family Frobenius norms, before → after (identical across all 3 paths):

| family | before | after | Δ |
|---|---|---|---|
| R_V | 2.4191 | 2.4183 | −0.0008 |
| K | 0.3050 | 0.3083 | +0.0033 |
| B_V | 1.0266 | 1.0171 | −0.0095 |
| C_V | 1.2080 | 1.2064 | −0.0016 |
| C_U | 1.4657 | 1.4633 | −0.0024 |

Distance to the TEACHER's params after training (identical across all
3 paths): R_V=3.2331, K=0.3986, B_V=1.3083, C_V=2.1420, C_U=2.0652 —
still far from the teacher despite the ~85x train-loss reduction; 100
steps at this conservative `lr` were not enough to recover the
teacher's exact recurrent parameters, only to fit the readout well.
Not oversold as full system identification.

**R_V spectral-radius projection activity**: NOT instrumented during
this run (`project_stable_R_V` does not log per-step activation or
`rho(R_V)`), and per instruction no further run was launched to obtain
it. What can be said without new computation: `R_V`'s Frobenius norm
barely moved (2.4191→2.4183, ≈−0.03%) between init and final, and
`R_V` was constructed with spectral radius exactly 0.80 by
`make_stable_dense` (well under the `rho_max=0.95` threshold) — this
is consistent with (not a direct measurement of) the projection rarely
or never actively clipping at this learning rate, unlike the earlier
diverging lr=0.003 attempts where the same projection was active but
evidently insufficient against larger raw gradient steps.

## 5. Recurrent-Jacobian change, init → final (identical across all 3 paths)

`E_{t,x}‖J_t^final−J_t^init‖_F` over 12 sampled states:
**mean_abs = 0.3480, mean_rel = 0.05572 (≈5.6%)** — confirms `J_t`
genuinely changed during learning, not frozen.

## 6. Structural diagnostics, before vs after (identical across all 3 paths)

| | before (init) | after (all 3 paths) |
|---|---|---|
| forward reachable rank | 64/64 | 64/64 |
| invariant residual `‖(I−PP⁺)J_tP‖` | **exactly 0** | **exactly 0** |
| commutator norm `‖[A_lin,Q]‖_F` | 1.041676 | 0.602750 |

The invariant module survives the entire training run exactly — 0
before and 0 after, at every one of the 12 sampled states — while the
commutator's magnitude changed (expected, since C_V/C_U/Phi moved) but
stayed clearly nonzero (still genuinely nonlinear/noncommutative both
before and after).

## 7. Persistent credit (unchanged by training, as expected — structural)

P_c=10,888. M_full=64×10,888=696,832 floats (5.575 MB).
M_reduced=4×10,888=43,552 floats (0.348 MB). **Ratio = 16.00x**,
identical to B31a — confirms the reduction survives actual joint
recurrent-parameter training, not just the gradient-correctness check.

## 8. Stop condition

Per explicit instruction: no further LR sweeps, longer runs, additional
seeds, T=256, or B31 variants were run after this one. B31 work stops
here; B32 begins next (moving-bundle generalization, r=2 d=1).

## 9. Commit hash

See the commit introducing this file.
