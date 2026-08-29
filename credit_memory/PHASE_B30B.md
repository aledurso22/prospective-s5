# Phase B30b — supervised learning equivalence, r=64, d=4 flag SSM

Branch `S5-CCM-scale-validation`. Follows B30a directly (same
architecture/structural matrices, reused unmodified via import). Code:
`credit_memory/b30b_flag_r64d4_training.py` (`main()` reproduces every
number below); raw curves archived in
`credit_memory/b30b_training_result.json`.

**Headline: the 16x-smaller exact reduced-RTRL credit state doesn't
just match gradients (B30a) — it trains the identical nonlinear
recurrent model. Full RTRL, reduced RTRL, and BPTT produce
bit-identical (machine-precision) gradients, parameter updates, and
loss curves across 100 sequence-boundary optimizer steps, with no
divergence between paths.**

## 1. Setup

Teacher/student system identification on the exact B30a flag
architecture (`T=V⊕U`, dim V=4, dim U=60, r=64, `P_c=10,088` trainable
Phi-MLP scalars; `R_U,D_U,R_V,K,B_V,C_V,C_U` fixed and IDENTICAL
between teacher and student, drawn once via B30a's `make_consts()`,
same seed as B30a). Teacher draws a frozen `Phi_theta*`
(`make_theta(seed=777)`); student initialized from a different draw
(`make_theta(seed=555)`). Fixed readout `W` (1×64, `W_u` small,
`W_v` order-1) reads a genuine mixture of U and V. Targets
`y_t*=W·s_t*` generated from the teacher along random input sequences
(`T=128` per sequence, `xs~N(0,0.5²)`, fresh `u0,v0` per sequence).
Loss: `L=(1/T)Σ_t(y_t-y_t*)²`.

**Noted structural fact, not hidden**: since `u_{t+1}` never reads
`v_t` or `theta`, and teacher/student share identical `R_U,D_U` and
input sequences, the U-component of the state is *exactly* identical
between teacher and student for any `theta` — so the readout's
U-component contributes exactly zero to the loss/gradient throughout
training. The mixture readout is retained per instruction as an extra
correctness safeguard (confirms the full-64-dim machinery doesn't
introduce spurious U-gradient contributions), not because U itself is
being identified.

Three independent training paths (A: full RTRL carrying `r×P_c`
sensitivity, B: reduced RTRL carrying `d×P_c` sensitivity, C: BPTT),
same plain-SGD optimizer, `lr=0.05`, same student init, same sequence
order (identical seeds), one gradient per full sequence (updates only
at sequence boundaries) — no per-step-online vs sequence-end mixing.

## 2. First-updates gradient/parameter agreement (steps 0–4)

| step | ‖g_full−g_reduced‖ | ‖g_reduced−g_bptt‖ | ‖g_full−g_bptt‖ | ‖θ_full−θ_reduced‖ | ‖θ_reduced−θ_bptt‖ |
|---|---|---|---|---|---|
| 0 | 0.000e+00 | 2.012e-15 | 2.012e-15 | 0.000e+00 | 2.519e-16 |
| 1 | 0.000e+00 | 7.863e-16 | 7.863e-16 | 0.000e+00 | 2.747e-16 |
| 2 | 0.000e+00 | 4.667e-16 | 4.667e-16 | 0.000e+00 | 2.893e-16 |
| 3 | 0.000e+00 | 3.008e-16 | 3.008e-16 | 0.000e+00 | 2.956e-16 |
| 4 | 0.000e+00 | 3.819e-16 | 3.819e-16 | 0.000e+00 | 3.026e-16 |

Full and reduced RTRL agree exactly (0.0, same underlying restricted-
Jacobian machinery as B30a); both agree with BPTT to float64 machine
precision (~1e-15/1e-16). **No falsification triggered**: the reduced
path's training trajectory never diverges from full/BPTT under
identical sequence-boundary updates.

## 3. Training result (100 steps, T=128; 20 held-out validation sequences)

| | full | reduced | bptt |
|---|---|---|---|
| initial train loss (step 0) | 0.34922 | 0.34922 | 0.34922 |
| final train loss (step 99) | 1.5110e-03 | 1.5110e-03 | 1.5110e-03 |
| mean validation loss (20 sequences) | 7.8518e-04 | 7.8518e-04 | 7.8518e-04 |

Loss trajectories are identical across all three paths at every logged
step (full curves in `b30b_training_result.json`), dropping from
~0.35 to the 1e-3–1e-4 range over 100 steps with the expected
step-to-step noise (each step uses a fresh random sequence, not a
repeated one). Sample logged points: step 10: 2.218e-3; step 40:
6.687e-4; step 70: 4.685e-4; step 99: 1.511e-3.

## 4. Persistent credit and cost (descriptive, not a speed claim)

| | value |
|---|---|
| P_c | 10,088 |
| full sensitivity | 64×10,088 = 645,632 float64 scalars = 5.165 MB |
| reduced sensitivity | 4×10,088 = 40,352 float64 scalars = 0.323 MB |
| reduction ratio | 16.00x |
| sequence length T | 128 |
| optimizer steps | 100 |
| wall-clock (all 3 paths, this run) | 372.1 s |

T=256 was not run — the T=128 result is already decisive (exact
agreement at every checkpoint, clean loss reduction) and B30a's own
precedent was to stop once a config is decisive rather than add
compute for its own sake.

## 5. Commit hash

See the commit introducing this file.
