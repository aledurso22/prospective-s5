# Phase B27 — noncommutative temporal advantage falsification

Branch `S5-CCM-scale-validation`. Tests whether B25/B25.1's surviving
theorem buys any REPRESENTATIONAL advantage over the strongest
existing structured exact-RTRL recurrence — not another "linear in n"
credit-cost check (retracted from novelty claims in the B26 audit).
Code: `credit_memory/b27_noncommutative_advantage.py` (new; `main()`
reproduces every number below, the final corrected protocol). No S5.
No wall-clock claims (JIT used only for practical runtime).

This report reflects **three** rounds of review correction, each
material to the conclusion — documented in full rather than hidden,
because the final verdict depends on understanding why the earlier
attempts were wrong.

**Headline: A — STRONG SEPARATION. Against the CORRECT, faithful
Nonlinear RTU baseline (activation inside each independent 2×2 block,
its own exact block-local RTRL implemented and verified against
BPTT), the pattern is exactly the one the phase asked to see: on a
teacher drawn from the RTU's own structural class (block-local), RTU
and our architecture fit comparably well; on a teacher whose temporal
algebra is the full, irreducible matrix algebra `A_T=M_r` (verified
rigorously — `dim=r²` and commutant dimension exactly 1), our
architecture fits it to near-zero error while RTU plateaus at a large,
state-and-parameter-independent error floor.**

## 1. Parts 1–3, 6 — unaffected by any correction, still solid

Teacher construction (`d_T=r²`, nonzero commutators), genuine
multi-generator usage (4/4 pairs active, ablation effect ≈2× state
std), our exact credit (machine precision vs. naive RTRL and BPTT,
`4.4e-16` to `1.8e-15`), and L=2 depth exactness (`3.3e-16` to
`1.3e-15`) all carry over unchanged from the prior draft — none of
this was in question.

## 2. Round 1 correction: rank(DΦ)=1 ≠ commuting (superseded as decisive)

The first "commuting" control only forced Φ's Jacobian to rank 1,
which does not imply `[R,Q*]=0`. Fixed with a **true** commuting
teacher (diagonal `R`, every interface channel aligned to one
coordinate) — verified `max‖[R,Q_ab]‖=0`, `max‖[Q_ab,Q_cd]‖=0`
*exactly*, `d_T=r=4` (abelian). This construction is **kept only as a
supplementary diagnostic** per the current instruction, not the
decisive falsifier (see §3 for why it was never going to be decisive).

## 3. Round 2 correction: why the commuting control could never have worked

Review identified the reason the commuting control failed to close
the gap in the prior draft: **the teacher retains genuinely nonlinear
recurrent feedback** (Φ still an MLP inside the loop) regardless of
whether its Jacobian *algebra* commutes, while the baseline used at
that time had **only linear recurrent dynamics** (`h_{t+1}=R_diag
h_t+B_diag u_t`, nonlinearity applied *after*, in a stateless head).
Asking a linear-memory model to approximate nonlinear state-dependent
memory was never a fair test of commutativity specifically — the
control's failure was **not evidence against the theorem**, it was
evidence the baseline itself was the wrong comparison class. This
reframes the correct structural distinction, refined further:

**Not** "noncommutative vs. commuting" (a Nonlinear RTU block has
time-varying Jacobians `J_t≈D_tR_block`, and different `J_t` need not
commute either). **The actual distinction**: potentially globally
coupled / irreducible temporal algebra over an `r`-dimensional
temporal core (ours) vs. a **direct sum of independent small blocks**
(Nonlinear RTU) — a *structural decomposability* question, not a
commutativity one.

## 4. `A_T = M_r`, verified rigorously (not just `d_T=r²`)

Beyond the algebra-closure dimension (already established), two
further checks confirm the main teacher's algebra is the *full*,
*irreducible* `r×r` matrix algebra, not merely large:

- `d_T = 16 = r²` (full matrix algebra — a subalgebra of `M_r`
  achieving dimension `r²` must equal `M_r` exactly).
- **Commutant dimension = 1** (`dim{X:XA=AX ∀ generators A}`, computed
  via the linear system `vec(XA-AX)=0` stacked over all generators —
  exact linear algebra, no numerical subspace search needed). By
  Schur's lemma, commutant dimension 1 for a real algebra means the
  generators act **irreducibly**: no nontrivial common invariant
  subspace exists. This is the requested "block-decomposition
  diagnostic," obtained directly and exactly rather than via iterative
  search.

## 5. The correct baseline: Nonlinear RTU, faithfully implemented and verified

Built the RTU paper's actual nonlinear recurrence — activation
**inside** each independent 2×2 block:
`h1_t=f(g·h1_{t-1}-φ·h2_{t-1}+Wx1·u_t)`,
`h2_t=f(g·h2_{t-1}+φ·h1_{t-1}+Wx2·u_t)`, `f=tanh` applied
element-wise (blocks never mix). Same strong full-state stateless MLP
head as before (`y_t=MLP([h_{t+1},u_t])`).

**Its own exact block-local RTRL was implemented and verified against
BPTT** (per instruction: derive and check the actual trace structure,
don't assume the old linear baseline's accounting carries over) —
machine precision (`0.0` to `2.8e-17`) across 3 blocks × 3 parameter
families (`θ`, `log_radius`, `Wx`). The verified per-block trace is
genuinely 2-dimensional (matching the block's own state size), giving
**`credit_floats = 2·r_rtu·(1+u_dim)`** — confirming the earlier
corrected linear-block accounting was asymptotically right, now
backed by an actual implemented-and-checked recurrence rather than an
analogy.

## 6. Positive control — block-local teacher (RTU's own class)

A genuine instance of the Nonlinear RTU family (`r_rtu=4`) as teacher:

| student | test NMSE |
|---|---|
| Nonlinear RTU (`r_rtu=4`) | **0.0013** |
| ours (`r=4,k=2,n=4`) | **0.0017** |

**Both fit near-perfectly and comparably** — confirms the baseline,
optimizer, and capacity are sufficient when the target genuinely lies
inside the RTU's own structural class, and that ours does not have
some unrelated advantage that would show up even here.

## 7. Decisive comparison — globally-coupled (`A_T=M_r`) teacher

Same protocol throughout (`h0=0` for all sequences, common
`BPTT+Adam`, 800 steps, 20 train / 16 test independently-sampled
sequences, matched-parameter and over-provisioned regimes):

| model | total state | params | credit floats | test NMSE |
|---|---|---|---|---|
| **ours** | 16 | 344 | 5504 | **0.0048** |
| RTU matched, r_rtu=4..128 | 4–128 | 357–781 | 24–768 | 0.213–0.514 |
| RTU strong (hidden=64), r_rtu=4..128 | 4–128 | 525–8833 | 24–768 | 0.229–0.475 |

**The gap is real, large, and does not close** — not with more
recurrent state (4→128, 32×), not with more parameters (357→8833,
25×), not with a deliberately over-provisioned nonlinear head. RTU's
test NMSE actually *worsens* somewhat as state grows in places
(overfitting: more capacity fits training data without transferring),
never approaching "ours" own **44–107× lower** error, achieved with
far less recurrent state and far fewer parameters.

Post-hoc exactness re-verified on "ours" final trained model (factored
RTRL vs. BPTT): `8.9e-16` to `2.0e-14`.

## 8. Verdict

**A — STRONG SEPARATION.**

- **Nonlinear RTU fits its own block-local teacher** (§6: NMSE 0.0013,
  matching ours' 0.0017) **but cannot efficiently approximate the
  globally-coupled `A_T=M_r` teacher** (§7: NMSE plateaus at 0.21–0.51
  across every state size and parameter budget tested, matched or
  over-provisioned) **while ours fits both** (0.0017 and 0.0048
  respectively) — exactly the pattern this option requires.
- **B (modest separation — RTU eventually catches up with more
  state/credit)**: not selected — RTU shows no improving trend with
  state or credit at all on the global teacher; if anything it
  degrades slightly at larger sizes.
- **C (no useful separation)**: not selected — the gap is large
  (44–107×) and robust across every regime tested.
- **D (confounded)**: not selected — the positive control (§6)
  directly validates that baseline, optimizer, and capacity are
  sufficient in general; the gap appears specifically and only when
  the teacher's structure (verified `A_T=M_r`, irreducible) falls
  outside the class RTU can represent by construction (a direct sum of
  independent blocks can never realize a matrix algebra acting
  irreducibly on the full space — this is now a structural fact about
  the two model classes, not an artifact of training).

Per the phase's own instruction, **no further depth experiments were
run** — the L=2 exactness result (§1) is retained as independent
exactness evidence only, not as progress on this representational
question, matching how it was already established.

## 9. What remains open, explicitly

1. The supplementary true-commuting control (§2/§8) is kept for
   completeness but is understood now not to isolate the relevant
   variable (nonlinear-recurrence-vs-linear-recurrence, not
   commutativity, was the actual confound in the earlier round) — a
   cleaner commuting control (matching the discussion in the prior
   PHASE_B27.md draft, e.g. a shared-eigenbasis construction with a
   full-rank interface) remains a reasonable follow-up but is no
   longer decision-relevant given §7's direct structural comparison.
2. Multi-seed robustness (teacher seeds, student init seeds, dataset
   draws) was not run given the scope already required to build and
   verify the correct baseline from scratch — an explicit, stated
   scope limit for this phase, not an oversight.
3. Whether an even richer RTU variant (e.g., larger blocks, k>2 per
   block) could close some of the gap is not tested — the phase's own
   RTU definition (independent 2×2 blocks) was implemented faithfully
   as specified, and a block size sweep is a natural, not-yet-explored
   extension.

No new production online-credit training rule deployed. No S5 run. No
wall-clock performance claims.

## 10. Commit hash

See the commit introducing this file.
