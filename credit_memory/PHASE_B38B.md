# Phase B38b — minimal source-local selective SSM

Branch `S5-CCM-scale-validation`. B37a–c and B38a are frozen and imported
unmodified. The question is narrow: **can input-dependent selectivity be added
while retaining complete exact O(P) RTRL?**

Code: `b38b_selective.py`, `b38b_exactness.py`, `b38b_scaling.py`,
`b38b_train.py`, `b38b_sweep.py`. Logs `/tmp/b38b_*.log`,
results `results/b38b/*.json`. All timing **CPU-only** (no CUDA device on this
machine); everything ran locally, nothing needed the cluster.

## 1. Recurrence

    h_{t+1,j} = a_{t,j} h_{t,j} + b_{t,j} ξ_t,     a_{t,j} = exp(Δ_{t,j} A_j)

with `Δ_{t,j}, b_{t,j}` input-dependent, state split into `J` tiles of bounded
size `d_tile = 4`. Nothing else added: no Mamba-3 rotations, no trapezoidal
integration, no RoPE, no multi-head value states, no deep stacks, no language
modelling, no universal quotient `q`, no ProductLocal Jordan factors. Local
propagation is scalar-diagonal. `A = −softplus(Ã)` and `Δ = softplus(·)` give
`a ∈ (0,1)` by construction — no stability projection is added.

**The structural point.** `Δ` and `b` depend on `x_t` only, never on `h`, so

    J_t = ∂h_{t+1}/∂h_t = diag(a_t)   is EXACTLY diagonal in BOTH arms.

Hence each parameter's sensitivity column stays supported on exactly the
channels that parameter influences:

    e_{t+1}[j,φ] = a_{t,j} e_t[j,φ] + (∂a_{t,j}/∂φ) h_{t,j} + (∂b_{t,j}/∂φ) ξ_t
    ⟹  dim M_φ = |supp φ|,   Σ_φ dim M_φ = Σ_φ |supp φ|.

So a diagonal Jacobian does **not** by itself give O(P). What decides it is the
**fan-out of each trainable selector parameter**. That is precisely what the two
arms vary. The instantaneous `G_t` is a per-step, per-tile Jacobian of the
one-step map with `h` held fixed — no tape across time, i.e. ordinary RTRL.

## 2. Arms

**Arm S (shared selectors, negative control)** — Mamba-1-style: a shared input
projection `W, p` and a shared *scalar* `Δ_t` broadcast to all `N` channels.
Each of those parameters influences every channel, so its module has dimension
`N`. Per-channel `A_j, uB_j, cB_j` remain local.

**Arm L (source-local)** — every trainable selector parameter influences exactly
one tile (`c = 1`). Tiles read the input through **fixed, non-trainable** random
projections `R_τ` (structural constants needing no credit, as B37c's
multiplication tables are), so locality bounds trainable fan-out without
blinding a tile to the input. Same state size, same input/output interfaces.

Parameter counts are reported explicitly rather than forced equal (Arm L
replicates the projection per tile, Arm S shares one); the registered metric
`M_elig/P` normalizes for this.

## 3. Exact-credit verification (all selector families)

Reduced source-local RTRL vs full dense RTRL vs BPTT/autodiff, T=30, five
configurations, **separately for every recurrently influential family** —
verifying only the local recurrent `A` would miss the point of the phase:

| family | `A` (Ãtil) | `uD` | `cD` | `uB` | `cB` | `W` | `p` | `C` |
|---|---|---|---|---|---|---|---|---|
| worst rel err | 1.4e-17 | 1.3e-17 | 1.0e-17 | 6.5e-17 | 9.2e-17 | 3.5e-17 | 3.7e-17 | 6.5e-17 |

**Worst over all families and sizes: 9.226e-17 — PASS** (tolerance 1e-10).
`φ_Δ = (uD, cD)`, `φ_B = (uB, cB)`, input projection `= (W, p)`.

Arm S's dense RTRL was verified against BPTT to **1.136e-16**: the negative
control is exact, not approximated.

## 4. Shared-vs-local eligibility scaling

The module size was measured **empirically** as the number of structurally
nonzero entries of the dense RTRL sensitivity `S_t`, and the analytic formula
was validated against that measurement before being used at larger `N`:
**analytic == measured for 10/10 configurations, both arms.**

That check corrected a real error: my first Arm L count assumed every tile
parameter touched all `d` channels, giving 80 where the measurement said 52.
Only `W, p` act through the tile bottleneck `g` and touch all `d`; `Ã, uD, cD,
uB, cB` are per-channel and touch exactly one. The measurement was right.

`d_tile = 4` fixed, `m = J` (so the shared selector projection's parameter count
grows with model width, as in Mamba):

| N | P_rec L | M_elig L | **M/P L** | P_rec S | M_elig S | **M/P S** |
|---|---|---|---|---|---|---|
| 8 | 68 | 104 | **1.53** | 41 | 104 | 2.54 |
| 32 | 272 | 416 | **1.53** | 149 | 800 | 5.37 |
| 128 | 1088 | 1664 | **1.53** | 581 | 9344 | 16.08 |
| 512 | 4352 | 6656 | **1.53** | 2309 | 135680 | 58.76 |
| 2048 | 17408 | 26624 | **1.53** | 9221 | 2115584 | **229.43** |

**Arm L: `M_elig/P` constant at 1.53, bounded by `d_tile`, across a 256× range
of N ⟹ Σ_φ dim M_φ = O(P).** Registered prediction confirmed.

**Arm S: `M_elig/P` grows 2.54 → 229.43 (90×), `M_elig` is O(N²) = O(N·P_selector).**
Registered prediction confirmed. Both arms have an exactly diagonal `J_t`, so
this isolates fan-out as the cause.

Per §7, Arm S was not trained at scale; its purpose — demonstrating that a
small/diagonal `J_t` alone does not imply O(P) exact RTRL — is served by exact
small-N measurement plus the validated scaling analysis.

## 5. Actual training (Arm L)

Task: gated accumulator with input-driven reset,
`z_{t+1} = (1−g_t)·λ·z_t + v_t`, `y_t = z_{t+1}`, `λ=0.9`, gate rate 0.16, with
two noise channels. The decay is input-dependent, so no input-independent LTI
expresses it. J=4, d=4 (N=16), T=256, 200 epochs, Adam, LR grid {3e-3, 1e-2,
3e-2} selected on validation only.

**Selectivity control.** The same architecture with `Δ, b` made
input-independent is **4.1–6.1× worse** (0.220–0.249 vs 0.041–0.053) across
three seeds, confirming the task actually requires selectivity rather than
rewarding extra capacity.

**A (matched TBPTT) vs B (exact source-local RTRL)**, gradients verified at
every chunk during actual training with the RTRL gradient driving the update:

| seed | L | A NMSE | B NMSE | \|ΔNMSE\|/NMSE | worst grad err |
|---|---|---|---|---|---|
| 0 | 32 | 4.7384e-02 | 4.7384e-02 | 1.5e-16 | 3.8e-16 |
| 0 | 256 | 6.5622e-02 | 6.5622e-02 | 2.2e-12 | 5.6e-16 |
| 1 | 128 | 4.9904e-02 | 4.9904e-02 | 4.8e-14 | 4.5e-16 |
| 2 | 32 | 4.0936e-02 | 4.0936e-02 | 8.5e-16 | 3.7e-16 |

**Worst in-training gradient error across all 8 parameter families, 3 seeds and
3 chunk lengths: 5.906e-16.** Max final-NMSE disagreement: **2.159e-12**. Every
arm pair selected the same LR; no divergence anywhere.

## 6. Arm C — every-token online

`θ_t → θ_{t+1}` after every token, eligibility carried across parameter updates
and never reset; judged as an online algorithm, not a BPTT identity; optimizer
never differentiated through. 8192-token streams.

| seed | lr | online NMSE | held-out NMSE | diverged |
|---|---|---|---|---|
| 0 | 1e-2 | 2.36e-01 | 1.07e-01 | No |
| 1 | 3e-3 | 1.95e-01 | 1.43e-01 | No |
| 2 | 3e-3 | 1.85e-01 | 1.28e-01 | No |

Works, with **0/3 divergence**, but is **~2–3× worse than batch** (0.107–0.143
vs 0.041–0.094). Recorded as a partial result, not a win.

## 7. Timing — CPU-only, and a negative result

Median full-training wall-clock: **Arm A 0.91 s, Arm B 1.56 s** — the exact
source-local RTRL is **1.7× SLOWER** than matched TBPTT here. This is preserved
as a negative result. At N=16 with a per-tile `jacrev` inside the scan, the
instantaneous `G_t` Jacobian dominates and the model is far too small for the
O(P)-memory advantage to pay for itself in time. The guaranteed benefit
established by this phase is eligibility memory, not speed; no GPU was used and
no speed claim is made.

## 8. Verdict

- Exact gradient verification for **all** selector families: 9.23e-17 offline
  (three-way), 5.91e-16 during actual training.
- Shared-vs-local scaling: Arm L `M_elig/P` constant at 1.53 across N=8..2048;
  Arm S grows 90× with `M_elig = O(N²)`. Analytic counts validated against
  empirical support measurement, 10/10.
- Arm L RTRL reproduces matched TBPTT training to 2.2e-12 in final NMSE.
- Selectivity is genuinely required by the task (4.1–6.1× ablation gap).
- Every-token online works without divergence but is 2–3× worse than batch.
- CPU-only timing: RTRL is 1.7× slower at this scale.

# SOURCE LOCALITY RESTORES COMPLETE EXACT O(P) RTRL

Scope limits, stated plainly: this is one small selective recurrence at N ≤ 2048
for the memory analysis and N = 16 for training, on one synthetic task, with
source locality bought by giving each tile a fixed non-trainable input view.
Whether that restriction costs representational power at scale is **not**
tested here.

Not proceeding automatically to Mamba-3 or language modelling.

## Commit hash

See the commit introducing this file.
