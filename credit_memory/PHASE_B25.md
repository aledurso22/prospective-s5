# Phase B25 — nonlinear temporal-credit separation

Branch `S5-CCM-scale-validation`. Architecture:
`h_{t+1} = (I_n⊗R)h_t + (I_n⊗B)Φ_ψ(z_t,u_t)`, `z_t=(I_n⊗C)h_t`, with a
genuine tanh MLP `Φ_ψ` mixing all `n·k` entries of `z_t` nonlinearly.
`r=4`, `k∈{1,2}`, `n∈{2,4,8,16}` unless noted. Code:
`credit_memory/b25_nonlinear_credit.py` (new; `main()` reproduces every
number below). Uses JAX (float64) for exact autodiff Jacobians and an
independent BPTT reference — the factorized forward-RTRL recurrence
itself is the novel object under test, never the reference. No S5.

**Headline: STRONG CONFIRMATION with the expected qualification
(Decision C). An arbitrary differentiable Φ behind the bounded C/B
interface preserves an exact, n-independent temporal credit module —
verified via the Jacobian identity, algebra closure, three-way
naive/factorized/BPTT agreement, a genuine dimensional-reduction case,
a full gauge test, and depth to L=3, all at machine precision. Feature
coefficient storage grows with n·(parameter count) as predicted — not
a failure, exactly the theorem being tested. Nonlinear task quality
improves monotonically with width at fixed r,k, confirming the
architecture is worth the coefficient cost.**

## 1. Part 1 — the exact Jacobian identity

`z_t=(I⊗C)h_t`, `J_Φ,t=dΦ/dz` (JAX autodiff), uniquely expanded
`J_Φ,t=Σ_ab F_ab,t⊗E_ab` (`F_ab,t[p,q]` = the `(a,b)`-feature slice of
`J_Φ,t`, an `n×n` copy-mixing matrix). Predicted
`J_t = I⊗R + Σ_ab F_ab,t⊗Q_ab` (`Q_ab=BE_abC`, `r×r`, time-independent)
verified against a **direct** full autodiff Jacobian of `h_{t+1}` wrt
`h_t`:

| r | k | n | max\|J_pred − J_direct\| |
|---|---|---|---|
| 4 | 1 | 2 | 1.7e-16 |
| 4 | 2 | 4 | 2.2e-16 |
| 4 | 2 | 8 | 2.2e-16 |
| 3 | 1 | 16 | 1.1e-16 |

**Machine precision at every config.** This is the central structural
identity everything else depends on.

## 2. Part 2 — the temporal algebra

`Alg{R,Q_ab}` closed under multiplication (a genuine algebra-closure
iteration, orthogonalizing products until no new dimension appears).
`ρ=dim K(R,B)` (Krylov reachability of B under R), `ω=dim K(Rᵀ,Cᵀ)`
(observability), `deg(μ_R)` (minimal-polynomial degree via Krylov on
powers of R). Swept `n∈{2,4,8,16}` at fixed `r=4,k∈{1,2}`:

`d_T=16, ρ=4, ω=4, deg(μ_R)=4` — **identical at every n tested**, for
both k=1 and k=2 (as expected — these quantities depend only on
`R,B,C`, never on `n`). Bound `d_T ≤ min(r²,deg(μ_R)+ρω) = min(16,20)
= 16`: **holds, exactly tight** (`d_T=16=r²` at generic k=1) —
confirms the phase's own warning: "do not expect small k alone to
imply small d_T."

## 3. Parts 3/4 — naive vs. factorized forward RTRL vs. BPTT

Generic case (`r=3,k=1,n=2`, no reduction expected since `ρ=r`
generically): naive full RTRL (`S_t=(n,r)` per scalar param, updated
via `S_{t+1}=S_t@Rᵀ+Σ_ab F_ab,t@S_t@Q_abᵀ+Direct_t`) and factorized
RTRL (same recurrence restricted to the theory-predicted basis
`V_theta`, `X_{t+1}=X_t@Rmatᵀ+Σ_ab F_ab,t@X_t@Qmat_abᵀ+U_t`) both
compared against a genuine JAX-autodiff BPTT reference. Per-step
direct source terms `∂f/∂θ|_{h fixed}` computed via `jax.jacobian`
for every family, not hand-derived, to avoid introducing bugs
independent of the algorithm under test.

| family | d_naive | d_fact | \|naive−BPTT\| | \|fact−BPTT\| |
|---|---|---|---|---|
| R | 3 | 3 | 1.1e-16 | 1.1e-16 |
| B | 3 | 3 | 0.0 | 0.0 |
| C | 3 | 3 | 6.3e-17 | 5.2e-17 |
| ψ | 3 | 3 | 1.1e-16 | 2.2e-16 |

**Three-way machine-precision agreement.** No reduction visible here
(`ρ=r=3` generically) — see Part 5 for a case that shows genuine
savings.

## 4. Part 5 — parameter families, and a genuine reduction case

Predicted: `V_ψ=V_C=K(R,B)` (dimension `ρ≤r`, since ψ and C's direct
effect on `h_{t+1}` is *entirely* injected through `(I⊗B)`, and `Q_ab`
always maps into `im(B)⊆K(R,B)` regardless of input — making `K(R,B)`
a genuine `J_t`-invariant subspace for every `t`). `V_R=V_B=`full `r`
space (their direct seed is an arbitrary standard-basis `e_i`, since
`R[i,j]`'s seed is `h_t[p,j]·e_i` and `B[i,j]`'s is
`Φ_j(z_t,u_t)·e_i` — unrestricted, "larger source modules" as
predicted).

Generic `R` gives `ρ=r` (confirmed in Part 3/4, no reduction).
Built a genuine reduction case: `R` with a diagonalizable *repeated*
eigenvalue (3-dim eigenspace at `λ=0.6`, similarity-transformed dense)
— `deg(μ_R)=3<r=5`, giving `ρ=3<r=5`:

| family | d_naive | d_fact | \|fact−BPTT\| | \|naive−BPTT\| |
|---|---|---|---|---|
| C | 5 | **3** | 4.4e-16 | 1.4e-16 |
| ψ | 5 | **3** | 7.8e-16 | 2.2e-16 |

**Genuine dimensional reduction, still exact.** `d_T=10, ρ=3, ω=3,
deg(μ_R)=3` for this R (bound `min(25,3+9)=12`, holds — `10≤12`).

## 5. Part 6 — gauge test

Random well-conditioned `T` (SVD-clipped, condition number ≤10),
`h→(I⊗T)h`, `R→TRT⁻¹`, `B→TB`, `C→CT⁻¹`:

| check | result |
|---|---|
| z invariance | 6.4e-16 |
| h transform consistency (`h_new = (I⊗T)h_old`) | 2.6e-15 |
| temporal algebra dims (`d_T,ρ,ω`) unchanged | **exact match** (16,4,4 both sides) |
| BPTT gradient pullback (R,B,C via `TᵀgTᵀ⁻¹`-type rules; ψ identical, no pullback) | 2.8e-16 to 7.8e-16 |
| factorized-RTRL gradient pullback (same rules) | 3.3e-16 to 9.4e-16 |

**All machine precision.** The factorized RTRL is genuinely
gauge-covariant, not an artifact of a particular basis choice.

## 6. Part 7 — width sweep, temporal vs. feature reported separately

This sweep uses the same generic `r=4,k=2` config as Parts 1/2/3/4
(generic `R` gives `ρ=r`), so factorized and naive coefficient
storage happen to coincide here — the reduction demonstrated in Part 5
requires the degenerate-`R` case, not the generic one. Reported
honestly rather than picking a config that would flatter the
factorized method:

| n | TEMPORAL (d_T, ρ, ω) | naive floats | factorized floats |
|---|---|---|---|
| 2 | 16, 4, 4 | 992 | 992 |
| 4 | 16, 4, 4 | 3072 | 3072 |
| 8 | 16, 4, 4 | 10496 | 10496 |
| 16 | 16, 4, 4 | 38400 | 38400 |

**Temporal quantities (d_T, ρ, ω) are exactly n-independent at every
n, confirmed alongside a total FEATURE storage that scales with
`n·(param count)` as predicted — not a failure, exactly the theorem
under test.** The equal naive/factorized counts here are a property
of this *generic* R (ρ=r, so `V_R=V_B` already fill the full r-space
and `V_C=V_psi=K(R,B)` doesn't shrink either); Part 5's degenerate-R
config shows factorized genuinely undercutting naive (d=3 vs d=5) —
the saving is real but architecture/R-dependent, not automatic.

## 7. Part 8 — nonlinear capacity check

Teacher: a fixed, wider (`n=6`) instance of the same architecture at
`r=3,k=1`, genuinely nonlinear `Φ`. Approximators at the same `r,k`,
growing `n∈{1,2,4,8}`, trained via the **factorized forward-RTRL
gradients themselves** (a real training use, not a checked-but-unused
formula), 150 steps:

| n | final loss |
|---|---|
| 1 | 0.0151 |
| 2 | 0.0122 |
| 4 | 0.0106 |
| 8 | 0.0089 |

**Monotonically decreasing with width** — nonlinear task quality
genuinely improves with `n` at fixed `r,k`, while Part 2/6/7 confirm
the temporal credit module does not grow. This is the simultaneous
"capacity grows, temporal module doesn't" result the phase asked to
establish before any larger benchmark.

## 8. Part 9 (scoped) — depth L=2, L=3

**Explicit scope note**: implemented and verified the **naive**
forward-RTRL reference at depth (via JAX-computed per-step full state
and direct-parameter Jacobians of the whole multi-layer stack — no
new per-layer algebra derivation needed for this reference). Did
**not** implement the fully factorized deep-prefix extension (basis
`V_theta` per layer chained across layers, testing the predicted bound
`d_temp_{j←i} ≤ k_i·Σ_{l=i}^j r_l`) within this phase's time budget —
an honest scope limit, not a silent gap. Each layer is its own
instance of the Part 1 architecture; layer `l`'s `z` output feeds
layer `l+1`'s external input `u` (standard feedforward-through-time
stacking, matching B21/B23/B24's convention).

L=2 (`r=3,k=1,n=[2,2]`), cross-layer parameters (layer 0's own
parameters, propagated through to a loss defined on layer 1's state):

| layer.family | \|naive−BPTT\| |
|---|---|
| 0.R | 2.1e-17 |
| 1.R | 1.1e-16 |
| 0.B | 5.2e-18 |
| 1.C | 9.7e-17 |

L=3 (`r=3,k=1,n=[2,2,2]`), loss defined on the *final* (layer 2)
state, testing parameters at every layer including the earliest:

| layer.family | \|naive−BPTT\| |
|---|---|
| 0.R | 1.3e-18 |
| 1.B | 3.5e-18 |
| 2.C | 3.5e-18 |
| 0.C | 2.6e-18 |

**Machine precision at both depths**, confirming the architecture and
gradient machinery generalize correctly to depth at the naive level.
The factorized deep-prefix bound remains an open, well-motivated next
step (natural given B21/B23's linear precedent), not asserted here
without direct verification.

## 9. Verdict

**A — STRONG CONFIRMATION**, with the expected Decision-C
qualification made concrete and quantified rather than left abstract:

- Nonlinear width increases task/function capacity (Part 8: loss
  strictly decreases with n, 0.0151→0.0089 over n=1→8) **while** exact
  temporal credit dimensions (`d_T,ρ,ω`) stay exactly n-independent
  (Parts 2, 6, 7 — confirmed at every n and under a gauge
  transformation).
- Factorized forward RTRL matches full/naive RTRL and an independent
  BPTT reference to machine precision in every test run: the generic
  case (Part 3/4), a genuine dimensional-reduction case (Part 5), a
  gauge-transformed case (Part 6), and depth L=2,3 at the naive level
  (Part 9).
- The honestly-exposed cost: feature/coefficient storage
  (`n·d_theta·m` floats per family) grows with width and parameter
  count exactly as predicted (Part 7) — this is Decision C's
  qualification made quantitative, not a caveat discovered after the
  fact.

No new production online-credit training rule deployed. No S5 run.
Per the phase's own instruction, performance is not optimized here —
Part 8 is a capacity-existence check, not a benchmark.

## 10. What remains open, explicitly

The fully factorized **deep**-prefix construction (combining each
layer's own reduced `V_theta` basis with cross-layer propagation,
testing the predicted `d_temp_{j←i}≤k_i·Σr_l` bound) was not built —
flagged in Part 9, not silently dropped. This is the natural next
increment before any larger-scale benchmark, and the most direct
follow-up to this phase.

## 11. Commit hash

See the commit introducing this file.
