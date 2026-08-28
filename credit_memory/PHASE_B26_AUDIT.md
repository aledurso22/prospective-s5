# Final adversarial novelty audit (post-B26)

Branch `S5-CCM-scale-validation`. No code in this phase — a pure
literature/novelty audit of the B16–B26 research line, requested
before any further scaling or S5 work. No S5. No new claims are made
here that are not already established by prior committed phases; this
document only re-evaluates what those results are worth against prior
art.

**Headline verdict: B — NOVEL ONLY AS SYNTHESIS/DESIGN PRINCIPLE, with
one candidate (the B25/B25.1 deep temporal composition result) as the
sole piece with a real, but not yet literature-confirmed, chance of
surviving as a genuinely new theorem (bordering on A). Candidate 3, as
originally framed in B26, does NOT survive hostile scrutiny — it
compares against a strawman baseline, not against vanilla RTRL
correctly applied to the actual recurrent state, against which there
is zero saving. This is stated plainly and should be treated as
retracted from any novelty claim, not softened.**

## 0. The dangerous observation, confirmed by direct cost accounting

The user's own framing is exactly right, and checking it kills a
specific claim rather than just raising a concern. B26's cost table
states: parameters `O(nq)`, persistent exact credit `O(nq²)`. Since
`P` (total scalar parameters in `U,V,E,b`) is itself `O(nq)`, this is
`O(q·P)`. **Applying vanilla, textbook Williams–Zipser RTRL directly
to the q-dimensional state `s_t`** (tracking `ds_t/dθ`, a `q×P`
tensor, updated via `ds_{t+1}/dθ=(∂f/∂s)(ds_t/dθ)+(∂f/∂θ)` for an
*arbitrary* nonlinear transition `f`) gives cost `O(q·P)=O(q·nq)=
O(nq²)` — **identically**, with no factorization machinery required at
all. Nobody with any RTRL literacy would track `dx_t/dθ` (the
ephemeral n-dim pre-projection) as persistent state in the first
place, because `x_t` is not recurrent state — it is recomputed fresh
from `s_t` and `u_t` every step. B26's own Part 1b "genuine wide
realization" (`x_{t+1}=σ(UV^Tx_t+Eu_t+b)`) is a deliberately
constructed reformulation that makes `x_t` *formally* recurrent so
that a comparison against it is even possible — it is not a
formulation anyone would naturally reach for. **Against the correct,
natural baseline (vanilla RTRL on `s_t`), the "linear in n" result is
not an algorithmic saving; it is a restatement of what RTRL already
gives once applied to the right variable.**

This directly resolves candidate 3.

## 1. Candidate 1 — causal-realization view

*"Exact online credit complexity is controlled by the minimal
parameter-to-query causal tangent realization, not nominal hidden
width."*

For a single, flat, unstructured recurrent layer, this is
**definitional, not a finding**: the "nominal hidden width" (`n`)
never enters the persistent-state RTRL cost formula unless you
deliberately reformulate the ephemeral computation as if it were
state (exactly the strawman in §0). Any RTRL practitioner already
knows not to persist sensitivities of non-state variables.

The claim has genuine, non-trivial content **only** when re-read as
being about *structured/deep* architectures, where the "correct"
minimal dimension is *not* obvious in advance and naive tracking would
give a provably larger number than necessary — e.g., B20's original
super-additive over-realization (corrected in B20-correction), B22.1's
`r²→2r-1` gauge collapse, B21/B24's `O(L·r)` vs `r^L` prefix scaling,
B24's `dim_below·min(k_in·k_out,r)` cap. In that reading, candidate 1
collapses into candidate 4 — it is not an independent contribution.

## 2. Candidate 2 — parameter-invariant sufficiency

*"A state quotient sufficient at the current parameter value is not
sufficient for exact learning unless it remains valid under all
admissible parameter tangent directions."*

The mathematical content here is **classical, not new**: minimal
state-space realizations have been known since Kalman (1963) to be
unique only up to a similarity-transformation gauge group, and
structural/practical *identifiability* theory (Bellman & Åström 1970
and the substantial system-identification literature since) is
precisely the study of which parameter directions are and are not
distinguishable from input-output data — this is the same question,
in different notation. B22.1's residual-gauge accounting
(`r²-(r-1)²=2r-1`) and B26 Part 5's "perturbing `V` breaks `V^TD=0`"
finding are both direct, correct instances of this classical idea, not
extensions of it.

What *is* reasonably characterized as a useful synthesis: this
research line makes the check **operational** — an explicit,
repeatable falsification protocol (construct the "unobservable"
special case, verify the naive reduction fails generically and is
restored exactly in the special case, then check whether *every*
trainable parameter's own tangent direction respects the special
case) applied directly to the question "is this proposed online
learning-rule dimensionality reduction actually safe?" This is a
worthwhile methodological habit for this specific application domain,
not a new theorem. **Should be claimed as: "we adapt classical
identifiability/gauge reasoning into an explicit pre-registration
check for reduced-state online learning rules." Should not be
claimed as: a new sufficiency theorem.**

## 3. Candidate 3 — wide-to-small exact credit quotient

Resolved in §0: **this is standard state minimization applied to an
RNN, compared against a strawman.** The "learning/parameter-tangent"
angle does not rescue it — once `s_t` is correctly identified as the
state, there is no parameter-tangent subtlety left to resolve; Part
1b's three-way (wide/reduced/BPTT) exactness check is a correct but
unsurprising confirmation that two mathematically equivalent
formulations give the same gradient, not evidence of a nontrivial
reduction.

## 4. Candidate 4 — the deep temporal result (B25/B25.1)

This is the strongest candidate, and the one place a genuinely new
technical claim might survive: an *arbitrary* differentiable
nonlinear feature computation `Φ` behind a *bounded* linear `B/C`
interface preserves an exact, `n`-independent temporal generator
module (`d_T=dim Alg{R,Q_ab}`, `Q_ab=BE_abC`, computed via a genuine
algebra-closure/Krylov argument, bounded by
`min(r²,deg(μ_R)+ρω)`), and exact forward-only credit composes through
depth (B25.1's cross-layer `G_ab,t⊗P_ab` identity, verified to L=4).

**Honest, hostile comparison against the named literature, stated with
the confidence level actually warranted:**

- **Zucchet's online LRU** keeps the recurrence strictly *linear and
  diagonal*, pushing all nonlinearity into a *stateless* per-step
  readout applied *after* the recurrence. B25/B25.1's architecture has
  the nonlinearity *inside* the recurrence itself (`Φ` sits between
  the state and its own next value, behind the interface) — a
  genuinely different and structurally harder regime that Zucchet's
  construction sidesteps by design rather than solves.
- **Irie et al.'s exact element-wise RTRL** relies on *diagonal*
  recurrence structure specifically for its O(1)-per-parameter
  exactness; it does not, to my knowledge, address an arbitrary smooth
  nonlinearity behind a bounded interface with a formal algebra-bound
  on the resulting Jacobian-tensor rank, nor multi-layer cross-term
  factorization.
- **Deep e-prop / OSTL / Millidge's dynamic-programming formalization
  of e-prop** decompose credit into a *local temporal eligibility*
  term and a *spatial/instantaneous* cross-layer term — structurally
  very close in spirit to B25.1's own "local basis + cross-layer
  `P_ab`/`G_ab` injection" decomposition. **I do not have confident,
  verified knowledge of the exact generality of Millidge's specific
  result** (whether it already covers arbitrary smooth `Φ` behind a
  linear interface with an explicit algebra-closure dimension bound,
  or is scoped to spiking/specific nonlinearities) **or of the "2026
  streaming-POMDP" RTU work named in this request** (likely past or at
  the edge of my training knowledge). This is a real, not-dismissable
  risk to the novelty of candidate 4, and I am flagging it as an
  open verification item rather than either claiming clean novelty or
  conceding overlap I cannot substantiate.

**Recommendation, stated plainly**: before claiming candidate 4 as a
new result, do the literature check specifically against Millidge's
dynamic-programming e-prop paper and OSTL's stated generality
(nonlinearity class, interface-boundedness assumption, and whether
they prove an explicit n-independent module-dimension bound analogous
to `d_T≤min(r²,deg(μ_R)+ρω)`, or only an asymptotic/architectural
claim without the closed-form bound). If those works are scoped to
specific nonlinearities (spiking threshold functions, ReLU-like) or
lack the explicit algebra-closure bound, candidate 4 likely survives
as the narrow claim: *"exact algebra-closure characterization of the
temporal module dimension for arbitrary smooth `Φ`, with a verified
Krylov-computable bound and cross-layer composition law, is more
general/explicit than what e-prop-family methods establish."* If those
works already cover this generality, candidate 4 drops to the same
verdict as candidates 1, 2, 3, and the overall audit result moves from
B toward D.

## 5. Candidate 5 — combined (n,q,r,k) design

A reasonable, useful taxonomy, but not new on its own: separating
reservoir/mixing width from persistent state order is standard in
reservoir computing; separating temporal order from interface width is
standard classical linear-systems theory; deep state-space models
(S4/S5/Mamba-family architectures) already separate per-channel state
order from channel/head count informally. **The specific combination
of all four axes under one exactness-verified (naive/factorized/BPTT)
protocol** is not something I can point to a specific prior paper
doing in exactly this form, but the individual pieces are all
well-precedented enough that the combination does not, by itself, rise
above a synthesis/organizing-framework contribution.

## 6. Verdict

**B — NOVEL ONLY AS SYNTHESIS/DESIGN PRINCIPLE.**

What should be claimed:
- The explicit, operational parameter-invariant-sufficiency check
  (candidate 2) as a methodological contribution — a concrete
  falsification protocol for anyone proposing a reduced-state online
  learning rule, adapted from classical identifiability/gauge theory
  into this specific application.
- The verified `(n,q,r,k)` taxonomy (candidate 5) as a clear, fully
  cross-checked organizing framework for exact online learners — useful
  for design and communication, not claimed as a new complexity result.
- Candidate 4 (B25/B25.1's deep temporal composition), narrowly, as
  the strongest surviving technical claim — **pending** the literature
  check against Millidge/OSTL described in §4. This is the one place
  worth spending further verification effort before any publication
  framing.

What must **not** be claimed:
- That B26's "linear exact credit in n" result is itself a novel
  algorithmic saving. It is not, relative to vanilla RTRL correctly
  applied to the actual recurrent state (§0) — this should be
  explicitly retracted from any forward-facing summary of this
  research line, not left standing alongside the genuinely useful
  results.
- That the wide-to-small quotient (candidate 3) is a nontrivial
  reduction, in isolation.
- That candidate 1, stated for a flat/unstructured layer, is
  independent content beyond candidate 4.

## 7. The single strongest distinguishing experiment

The recommended experiment, aimed precisely at what this hostile audit
identifies as the *only* place a real advantage over RTU/diagonal-exact-RTRL
could live — not at reproducing a streaming result, but at forcing the
comparison onto ground where diagonal/linear-recurrence methods are
structurally disadvantaged:

**Construct a genuinely non-normal, densely-coupled temporal generator**
(a real generic dense `R` with a single noncommuting tangent
direction, verified via the B19/B24/B25 algebra-closure machinery to
give `d_T` close to the full `r²` — i.e., *provably* not reducible to
a diagonal or block-diagonal form by any fixed, parameter-independent
change of basis) **embedded behind a bounded k-dim interface, with a
Φ that genuinely mixes features nonlinearly** (not decomposable into
independent per-channel nonlinearities), **stacked to depth L≥2**.
Compare three things directly:

1. This framework's factorized forward-only credit: exact, at the
   provably-necessary `d_T` (not nominal width), verified against
   BPTT.
2. A diagonal/RTU-style architecture given the *same task*: either (a)
   quantify how much larger a diagonal state must be made to
   approximate the required non-normal dynamics to a fixed tolerance,
   or (b) fix the diagonal state size and quantify the resulting exact
   approximation error against true BPTT gradients — a number this
   framework's construction has no need to accept, by construction.
3. Report both as a genuine capability/cost frontier, not a single
   benchmark number: "at matched exactness, RTU-style diagonal
   recurrence needs X× the state; at matched state size, it accepts Y
   approximation error" — with X and Y measured directly, not asserted.

This is the sharpest test because it targets the actual structural
assumption (diagonalizability / linear-recurrence-only) that RTU,
Irie et al., and Zucchet's LRU all share and that this framework does
not require — rather than re-measuring a quantity (linear-in-n
persistent credit) that §0 shows is not actually a differentiator at
all.

## 8. What this licenses going forward

Per this audit's own recommendation: before any further scaling,
S5 run, or publication-facing claim, (a) retract/reframe the
"linear-in-n is novel" framing wherever it appears in prior phase
summaries for this line, (b) do the targeted Millidge/OSTL literature
check for candidate 4, and (c) if pursuing further empirical work,
prioritize the §7 experiment over any additional linear-architecture
scaling sweep, since the linear-architecture theory (per B24.1's own
closing verdict) is already closed and further sweeps there would not
address the actual open novelty question this audit identifies.

No new production online-credit training rule deployed. No S5 run. No
code changes in this phase.

## 9. Commit hash

See the commit introducing this file.
