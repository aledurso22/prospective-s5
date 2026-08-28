# Phase B24.1 — theory reconciliation

Branch `S5-CCM-scale-validation`. Small, focused phase resolving two
load-bearing interpretation corrections to B24 before the nonlinear
successor. Code: `credit_memory/b24_1_reconciliation.py` (new; `main()`
reproduces every number below). No S5.

**Headline: both corrections confirmed exactly. Separable routes
(`F_l⊗Q_l`) collapse regardless of `k` or width — B24's k>1 result was
never about channel count, it was about non-separable/multi-generator
structure. Under the stricter fixed-scalar-I/O test, width is a genuine
but bounded TENSOR-RANK resource: n copies realize exactly the
rank-≤n coefficient matrices Γ (up to the ambient cap k), while the
external Hankel/McMillan order stays fixed and n-independent
throughout.**

## 1. Corrected statement (replaces "k>1 makes width useful")

**"Multiple independently feature-weighted temporal generators can
make width useful at fixed k, until the finite temporal-function
envelope saturates."** B24's k>1 architecture (shared `k_in×k_out`
core, per-copy `V[q]` mixing) is the non-separable/multi-generator
case: its numerator tensor `b[j,i,:]` is generically full-rank across
`(i,j)` (up to `min(k_in·k_out,r)`), giving `k_in·k_out` genuinely
distinct temporal generators `H_{ji}(z)` with independently adjustable
feature coefficients through `V[q]`. Plain channel count `k` was never
the active ingredient — separability is.

## 2. Part A — separable-route control

Built `make_separable_layer`: numerator tensor `b[j,i,s]=F[j,i]·q[s]`
— exactly rank-1 in `(i,j)` (one shared scalar temporal generator `q`
per layer, static gain matrix `F`), reusing B24's `miso_core_forward`/
`layer_forward`/`stack_forward` unchanged (a separable core is a
special case of the general one, not a new mechanism). Stacked `L`
such layers with scalar external input, arbitrary widths `n_list`,
fixed random scalar readout. Reference: the direct `L`-fold cascade of
the shared `(a_l,q_l)` SISO filters alone (n-independent by
construction).

| k | n_list | residual | relative | 
|---|---|---|---|
| 2 | [1,1] | 1.9e-17 | 2.7e-15 |
| 2 | [3,5] | 1.9e-17 | 1.2e-15 |
| 2 | [8,2] | 2.4e-17 | 2.0e-15 |
| 2 | [10,10] | 1.7e-16 | 8.6e-16 |
| 3 | [1,1] | 3.5e-17 | 9.1e-16 |
| 3 | [3,5] | 6.6e-17 | 3.1e-15 |
| 3 | [8,2] | 9.7e-17 | 6.7e-16 |
| 3 | [10,10] | 1.4e-16 | 9.5e-16 |

**Machine precision at every config** (worst relative residual
`3.1e-15`). Full transfer equals `gamma·H_temporal(z)` exactly,
independent of `n` and `k` — confirms separability, not `k=1`, is
what drove B23's original collapse result, and generalizes it to
arbitrary `k`.

## 3. Part B — explicit multi-generator k=2 counterexample

`H_n(z) = g(z)ᵀ Γ f(z)`, `f=(f_1,f_2)` and `g=(g_1,g_2)` two
INDEPENDENT (distinct-pole) order-`r` SISO generators each, each of
`n` copies contributing a rank-1 outer product `w_q⊗v_q` to an
effective `Γ=Σ_q w_q⊗v_q`. Target: a genuine full-rank (rank-2) `2×2`
`Γ`, built via SVD.

| test | result |
|---|---|
| target `rank(Γ)` | 2 |
| n=1 achieved `rank(Γ)` | 1 |
| n=1 matches best rank-1 reconstruction | `8.3e-17` |
| n=1 vs. the rank-2 target | **`0.244` — cannot match** |
| n=2 achieved `rank(Γ)` | 2 |
| n=2 `Γ` error vs. target | `2.2e-16` |
| n=2 output vs. target | `1.4e-16` — **exact** |
| n=3 (extra zero-weight copy) vs n=2 | `0.0` — exactly unchanged |

**Exactly as predicted.** `n=1` is provably incapable of the rank-2
target (a genuine `0.244` residual, not noise); `n=2` represents it to
machine precision; a further copy with zero weight changes nothing.
This is the cleanest possible demonstration that fixed-`k` hidden
linear width matters as a tensor-rank resource without ever growing
external output dimension.

## 4. Part C — tensor-rank / saturation check

Larger multi-generator stacks (`k=2,3,4`), `n=1..6` copies with
generic random `(w_q,v_q)`, accumulated `Γ=Σ_q w_q⊗v_q`. Measured
`rank(Γ)` and the Hankel rank of the resulting scalar impulse
response (`size=60`, checked for a clean singular-value gap before
trusting — e.g. r=3,k=2 showed a 12-order-of-magnitude gap,
`3.5e-6→4.9e-18`, confirming the reported rank is real, not a
tolerance artifact):

| r | k | gamma_rank (n=1..6) | hankel_rank (n=1..6) |
|---|---|---|---|
| 2 | 3 | 1,2,3,3,3,3 | 12,11,12,12,12,11 |
| 3 | 2 | 1,2,2,2,2,2 | 13,13,13,13,13,13 |
| 2 | 4 | 1,2,3,4,4,4 | 13,12,13,13,13,13 |

**Confirmed**: `gamma_rank` grows with `n` and saturates exactly at
`k` in every case. `hankel_rank` stays flat across the whole `n` sweep
(fluctuating by at most 1 across seeds/n — a numerical rank-detection
margin, not a trend; every value has a clean multi-order-of-magnitude
gap) — the external temporal order does **not** grow with `n`, even
as `Γ`'s rank (the internal tensor-rank resource) climbs from 1 to
`k`. No attempt is made here to derive the exact integer value of the
n-independent order from `(r,k)` in closed form — the qualitative
n-independence is what was asked for and is what's confirmed; a
precise formula is left as a candidate follow-up, not asserted.

## 5. Correction: `stack_backward` is an exact reference, not the online credit algorithm

`stack_backward` caches the full forward sequence, then traverses
backward — a genuine **reverse-mode adjoint**, and an excellent
independent exact reference (verified against FD at the FD precision
floor across L=2,3,4 in B24). It is **not** a forward-only online
algorithm, and its `O(n·k_in·r)` cost must not be reported as "the
scaling of our online credit state." What B24 actually established on
the learning side is narrower and still real: (A) exact local
`k`-input/`k`-output canonical eligibility formulas (`prefix_gradients_miso`,
verified against the adjoint to `2.2e-11` across the full grid), and
(B) a trustworthy full-stack analytic-adjoint reference against which
a future forward-only learner can be checked. No forward-only
prospective-credit recurrence for the multi-copy architecture has been
derived or implemented in B24 or here — that remains open, and is
exactly the "next major algorithmic problem" flagged for the nonlinear
successor.

## 6. Cap law: relabeled as architecture-specific

B24's `cap = dim_below·min(k_in·k_out, r)` is kept as an **empirical
exact-match law for the specific multi-copy architecture measured**
(shared `(a,b)` core, per-copy generic `V[q]`, output-trajectory-rank
diagnostic) — verified exactly at every tested configuration, not
asserted as a general theorem. A short derivation sketch was given in
`PHASE_B24.md` §5 (the shared-filter atom-pool argument), but it was
not re-derived here in a broader theory-level generality, and the
present phase's Part C shows a structurally different but related
quantity (the rank of `Γ`, capped at `k`, for the multi-generator
`f/g` construction) with a different — and here explicitly
undetermined — relationship to the external Hankel order. The two cap
phenomena (B24's `dim_below·min(k_in·k_out,r)` and B24.1's `rank(Γ)≤k`
saturation) are consistent in spirit (both are finite tensor-rank
saturation results) but have not been shown to be the same law under
a common general theorem — stated as open, per the phase's own
instruction not to overclaim generality.

## 7. Verdict

All four items from the phase's closing checklist hold:

- **separable k>1 width collapses exactly** — Part A, machine
  precision at every `(k,n_list)` tested.
- **multi-generator fixed-k scalar-I/O width grows by tensor rank** —
  Part B (explicit exact rank-1→rank-2 realization) and Part C (`Γ`
  rank grows with `n`, saturates at `k`, generic random copies).
- **growth saturates as predicted** — Part C, `gamma_rank` caps
  exactly at `k` in every `(r,k)` config tested.
- **temporal Hankel order stays n-independent** — Part C, confirmed
  with clean singular-value gaps (not a numerical artifact), flat
  across `n=1..6` in every config.

**The linear architecture theory is closed on the terms specified.**
Per the phase's own instruction: do not do more linear sweeps. Proceed
to the nonlinear successor (`z=(I_M⊗C)h`, `z̃=Φ(z)`, inject
`(I_M⊗B)z̃`), whose central open algorithmic problem is deriving and
implementing the forward-only exact credit recurrence for that
architecture — not attempted here, out of scope for this reconciliation
phase.

No new production online-credit training rule deployed. No S5 run.

## 8. Commit hash

See the commit introducing this file.
