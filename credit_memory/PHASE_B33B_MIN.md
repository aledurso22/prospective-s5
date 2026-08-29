# Phase B33b-min — actual trainable recurrent model, full-rank sensitivity, 2-scalar exact eligibility

Branch `S5-CCM-scale-validation`. Code:
`credit_memory/b33b_min_trainable.py` (`main()` reproduces every
number below; full log at `/tmp/b33b_min_full.log` this session).
Unlike B33a (explicitly abstract, `J_t`/`G_t` given directly), this
model's `J_t=D_hF_theta` and `G_t=D_thetaF_theta` are obtained via
genuine autodiff on a real recurrent parameterization — **this is
legitimate RTRL**, not a proof-of-principle.

**Headline: on an actual trainable r=P=64 recurrent model, rank(S_t)=64
at 100% of sampled timesteps across every setting (including T=1000),
while the exact dynamic eligibility state is just 2 scalars. Teacher/
student training with full RTRL, the 2-scalar lifted rule, and BPTT
produced IDENTICAL loss curves (train loss 0.0287→6.8e-7, ~42,000x
reduction) and gradient agreement at ~1e-17.**

## 1. Model

`r=P=64`. `theta∈R^64` persistent trainable parameter (same space as
`h`). `K=I-2vv^T` (Householder involution, `K^2=I`), applied ONLY
implicitly (`Kx=x-2v(v.x)`, O(r) cost/storage — never materialized
densely except for diagnostics).
`h_{t+1}=(alpha_t I+beta_t K)h_t+(gamma_t I+delta_t K)theta+c_t`.
`alpha_t,beta_t,gamma_t,delta_t,c_t` are causal/exogenous — generated
from an exogenous input `x_t` by a small FIXED, FROZEN (never trained)
MLP, with `lambda_{+,t}=0.95 tanh(u_{+,t})`, `lambda_{-,t}=0.95
tanh(u_{-,t})`, `alpha_t=(lambda_++lambda_-)/2`,
`beta_t=(lambda_+-lambda_-)/2` — both eigenspaces of K stay within
(−0.95, 0.95), guaranteeing stability regardless of theta (theta never
touches the transition operator, only the additive injection term, so
training theta carries no instability risk of the kind seen in B31b).

## 2. Part A — correctness (5 seeds × T∈{1,5,20,100,1000})

| quantity | value |
|---|---|
| worst full-vs-BPTT relative error | 2.237e-15 |
| worst lifted-vs-BPTT relative error | 2.354e-15 |
| worst reconstruction error (Ŝ_t vs S_t) | 4.996e-16 |
| worst query error (`S_t^T q` vs `a_t q+b_t Kq`, implicit K) | 7.772e-16 |
| `ALL < 1e-8` | **True** |
| mean fraction of sampled timesteps with rank(S_t)=64 | **1.0000** (every one of 25 settings, incl. T=1000) |

## 3. Storage accounting

| | value |
|---|---|
| Forward/model state | h_t=64, theta=64, Householder v=64 floats |
| Fixed frozen coefficient-generator params | 2,404 floats (never trained) |
| Additional persistent DYNAMIC credit, full RTRL | 64×64=4,096 floats |
| Additional persistent DYNAMIC credit, lifted | 2 floats |
| **Ratio** | **2,048x** |

This is the legitimate claim (4,096→2 in *additional persistent dynamic
credit storage* for this model). `K`/`v` is part of the forward model
needed by BOTH paths to run at all — not charged uniquely to the
reduced learner — and this is not a claim about total learner memory.

## 4. Part B — teacher/student training (T=128, 100 sequence-boundary SGD steps, lr=3.0)

| | full | lifted | bptt |
|---|---|---|---|
| initial train loss | 0.028666 | 0.028666 | 0.028666 |
| final train loss (step 99) | 6.8274e-07 | 6.8274e-07 | 6.8274e-07 |
| mean validation loss (20 held-out) | 6.1312e-07 | 6.1312e-07 | 6.1312e-07 |

Identical at every logged checkpoint (0,10,...,99) across all three
paths — a ~42,000x train-loss reduction. First-5-step gradient/param
agreement: `‖g_full−g_lifted‖`≈1.4–2.5e-17, `‖g_lifted−g_bptt‖`≈1.9–4.0e-17
(even tighter than B30b/B31b, since P_c=64 here vs ~10k there — less
floating-point accumulation). `rank(S_t)` sampled every 20 steps during
training: **[64,64,64,64,64]**, unchanged throughout. Distance to
teacher theta*: init=3.372 → 3.048 (all three paths identical) — moved
toward the teacher but not fully recovered in 100 steps; not oversold
as full parameter identification (the loss landscape need not
constrain every theta direction equally given `W`'s specific readout).

## 5. Part C — closure falsification (moderate eps only, per instruction — no B33a-style blowup)

`R_generic` confirmed 99.97% outside `span{I,K}`. `eps∈{0,1e-6,1e-4,
1e-3,1e-2}`, `T∈{5,20,100,500}` — max|S_t| stayed in [0.37,0.57]
throughout, no instability at any tested eps.

| eps | span_dim, T=500 | lifted_rel, T=500 | full_rel, T=500 |
|---|---|---|---|
| 0 | 2 | 1.376e-15 | 1.650e-15 |
| 1e-6 | 5 | 5.225e-06 | 1.130e-15 |
| 1e-4 | 9 | 5.227e-04 | 1.296e-15 |
| 1e-3 | 15 | 5.236e-03 | 1.485e-15 |
| 1e-2 | 36 | 5.325e-02 | 1.316e-15 |

At eps=0 the lifted span stays exactly 2 at every T (5,20,100,500).
For generic nonzero eps the span grows and the (deliberately blind)
lifted reconstruction/gradient becomes systematically wrong, growing
with eps; **full RTRL stays exact throughout, at every eps** (~1e-15 to
1e-16 relative error regardless of eps or T). No claim of monotonicity
in T is made.

## 6. Commit hash

See the commit introducing this file.
