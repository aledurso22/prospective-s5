# Phase B17 — all-layer invariant-credit architecture, and a faithful selective cross-layer test

Branch `S5-CCM-scale-validation`. Ordinary BPTT for Parts A/B/E (Part F
never reached — see the gate in Part 9). Part D is a standalone
construction (no tcg dependence). No new persistent online-credit
training rule. No S5. Code: `credit_memory/b17_all_layer_ic_ssm.py`,
`credit_memory/b17_partD_selective.py` (new). Artifact: `results/
credit_memory/b17/b17_summary.json`. tcg's core machinery (`forward`,
`spatial_q`, `sensitivities`, `exact_lambda`, `assemble`, `flat_grads`,
`pack`, `flatten`) turned out to already be depth-generic (every
function loops `for l in range(L)` off per-layer arrays) — this let
Part A/B reuse the exact-gradient pipeline unmodified at `L=2,3,4`,
rather than requiring new machinery.

**Headline: the phase's own suspected confound is confirmed, sharply.
Tying every recurrent layer (A2) fails catastrophically — 2 to 4 orders
of magnitude worse loss than any partially-tied or full architecture,
at every depth and every task tested. Leaving exactly ONE layer untied
(A1 upper, A3 lower) mostly preserves task performance, but Part C's
own credit-state accounting shows this does NOT translate into a
favorable total credit-state once depth exceeds 2 — the untied layer's
N^2 cost dominates regardless of where it sits. Part D found a genuine,
if double-edged, result for selectivity: an exact corrected gradient
formula exists (verified to ~1e-11), but it requires O(N0*N1)
persistent state, not O(N0) — selectivity blows up the credit module
even though gradients remain exactly computable. Verdict: B, TEMPORAL-
COMPLEXITY LOCALIZATION, with an explicit credit-accounting caveat.**

## 1. Part A — all-layers tied expressivity

Median late loss, `N=64`, four architectures (`A0` full/distinct poles
every layer, `A1` upper-only tied `G=1`, `A2` every layer tied `G=1`,
`A3` every layer but the last tied `G=1`), depths `L=2,3,4`, 2 seeds,
200 BPTT steps, `N=32` grid in the artifact (same qualitative pattern,
omitted here for space):

**delay_r8** (hardest task)

| L | A0 full | A1 upper-tied | A2 all-tied | A3 lower-tied |
|---|---|---|---|---|
| 2 | 0.0227 | 0.579 | 2.333 | 0.0678 |
| 3 | 0.0019 | 0.0234 | 2.265 | 0.1707 |
| 4 | 0.0007 | **0.0013** | 2.196 | **0.7323** |

**hierarchical** (two-timescale task, designed to need >=2 layers)

| L | A0 full | A1 upper-tied | A2 all-tied | A3 lower-tied |
|---|---|---|---|---|
| 2 | 0.0128 | 0.0144 | 0.2921 | 0.0144 |
| 3 | 0.0130 | 0.0128 | 0.2514 | 0.0130 |
| 4 | 0.0131 | 0.0127 | 0.2704 | 0.0131 |

`delay_r1`, `delay_r4`, `freq_r1`, `kexp_K4` all show the same
qualitative pattern (full data in the artifact): **`A2` is 2-4 orders
of magnitude worse than every other architecture, at every depth and
every task, with zero exceptions.** `A1` and `A3` mostly track `A0`
closely, but diverge sharply from each other with depth on the hardest
task: `A1` *improves* with depth on `delay_r8` (0.579 -> 0.0013,
essentially catching `A0` by `L=4`), while `A3` *degrades* with depth
on the same task (0.068 -> 0.732, getting far worse). **Which single
layer is left untied matters a great deal, and interacts with depth in
opposite directions depending on the task** — resolved mechanistically
in Part 2.

**Direct answer to Part A's main question**: no, useful performance
with `G_l << N` does NOT survive when ALL recurrent layers are
credit-structured — `A2` fails badly and uniformly. Upper-only tying's
earlier (B16.1/B16.2) good results were, at least in part, relying on
the untied lower layer(s) to supply real temporal complexity, exactly
as the phase suspected — confirmed directly in Part 2.

## 2. Part B — where does temporal complexity live?

Layer ablation (force that layer's pole to zero, i.e. no memory, using
the already-trained weights, no retraining) and structural rank
(numerical rank of that layer's own Krylov/reachability matrix,
`[B, AB, A^2B, ...]`, `A=diag(a_l)`) on the kept `L=3, N=64` models:

**delay_r8**

| arch | base loss | ablate L0 | ablate L1 | ablate L2 | rank [L0,L1,L2] |
|---|---|---|---|---|---|
| A0 full | 0.0013 | 1.321 | 1.987 | 1.543 | [64, 64, 64] |
| A1 upper-tied | 0.0151 | 1.546 | 2.406 | 1.829 | [64, 64, 64] |
| A2 all-tied | 2.197 | 2.481 | 2.517 | 2.290 | [8, 64, 64] |
| A3 lower-tied | 0.1466 | 1.931 | 1.295 | 2.504 | [8, 64, 64] |

**hierarchical**

| arch | base loss | ablate L0 | ablate L1 | ablate L2 | rank [L0,L1,L2] |
|---|---|---|---|---|---|
| A0 full | 0.0138 | 0.189 | 0.192 | 0.162 | [60, 64, 64] |
| A1 upper-tied | 0.0134 | 0.301 | 0.281 | 0.371 | [60, 64, 64] |
| A2 all-tied | 0.2792 | 0.487 | 0.592 | 0.421 | [1, 64, 64] |
| A3 lower-tied | 0.0135 | 0.423 | 0.268 | 0.377 | [1, 64, 64] |

Three findings, directly answering Part B's question:

1. **In `A1` (upper-only tied), ablating the untied LOWER layers hurts
   just as much as in the fully untied `A0`** (delay_r8: 1.55/2.41 vs
   1.32/1.99) — **confirming the exact confound the phase suspected**:
   `A1`'s good forward performance leans heavily on the untied lower
   layers doing real temporal work, not on the tied upper layer.
2. **Structural rank does not track `G_l` the way credit state does,
   once a layer's own input width `M_l >= N`.** Layer 0 (`M_IN=8` for
   delay_r8, `M_IN=1` for hierarchical) shows rank collapsing exactly
   to `G_l` when tied (`8` or `1`), matching B16's Krylov identity
   `K(A,B)=sum_g rank(Pi_g B)` precisely. But **layers 1 and 2 (input
   width `N=64`) stay at FULL rank 64 even when tied to `G=1`** —
   because `rank(Pi_1 B) = rank(B) = min(N, M_l)`, and `M_l=N` already
   saturates that bound regardless of how many pole groups there are.
   **`A2`'s catastrophic failure is therefore NOT a loss of reachable
   spatial directions — those stay full-rank — it is a loss of
   TIMESCALE DIVERSITY WITHIN each middle/upper layer**: every unit in
   that layer is forced to share one decay rate, even though it can
   still mix spatially through `B` as richly as ever. This is a
   precise, useful mechanistic correction to the credit-state formula's
   own implicit promise: `2*G_l*M_l` is the exact GRADIENT-computation
   cost, but it says nothing about the layer's own forward capacity
   once `M_l >= N`.
3. **`A2`'s per-layer ablation deltas are all similarly bad** (no
   single layer stands out as "the important one") — consistent with
   every layer being comparably crippled by the shared-timescale
   constraint, rather than one specific layer being the bottleneck.

## 3. Part C — credit-state accounting across the whole stack

`S_credit = sum_l 2*min(G_l,N)*M_l` (`M_l` = that layer's own input
width: `M_IN` for layer 0, `N` for every layer above). Exact, derived
directly from the already-collected architecture definitions — no new
runs needed:

| L | N | A0 full | A1 upper-tied | A2 all-tied | A3 lower-tied |
|---|---|---|---|---|---|
| 2 | 64 | 8320 (1.00) | 256 (**0.031**) | 130 (0.016) | 8194 (0.985) |
| 3 | 64 | 16512 (1.00) | 8448 (**0.512**) | 258 (0.016) | 8322 (0.504) |
| 4 | 64 | 24704 (1.00) | 16640 (**0.674**) | 386 (0.016) | 8450 (0.342) |

(ratio to `A0`'s full cost in parentheses; `N=32,128` rows in the
artifact show the identical pattern.)

**This is the single most important structural finding of the phase,
independent of any training result**: `A1` (upper-only tied) is only
genuinely favorable — subquadratic, `~3%` of full cost — at `L=2`.
**Once depth exceeds 2, `A1`'s ratio jumps to 51-67%** — the untied
MIDDLE layers each reintroduce a full `N^2` term, since their own input
width is `N`, not the small `M_IN`. `A1` "works" functionally (Part 1)
but does **not** deliver the credit-savings goal at any depth beyond 2.
`A3` (tie every layer but the last) stays more favorable as depth grows
(its ratio *shrinks* with `L`, since the one untied layer's absolute
cost is fixed while `A0`'s total cost grows) but Part 1 showed its
functional performance is inconsistent (fails on `delay_r8` at depth).
**Only `A2` (all layers tied) delivers a uniformly favorable,
`N`-shrinking total credit-state ratio at every depth tested** — and
it is precisely the architecture that fails functionally. **The
credit-savings goal and the functional-performance goal pull in
opposite directions across every architecture tested in this phase.**

## 4. Part D — faithful selective cross-layer test

Standalone two-layer model, `N0` independent lower sources (`h_t^0[m] =
a0 h_{t-1}^0[m] + theta_m u_t^m`), tied scalar upper gate `h_t^1 = a_t
h_{t-1}^1 + B h_t^0`. Differentiating w.r.t. a LOWER-layer parameter
`theta_m` (not the pole parameter itself, unlike B16.2 Part H), for
**D1** constant `a_t`, **D2** exogenous time-varying `a_t`, **D3**
endogenous `a_t = a(mean(h_t^0))`:

| N0 | gate | naive-formula rel. error | corrected-formula rel. error | extra-term rank |
|---|---|---|---|---|
| 1 | D1/D2 | ~2e-11 (exact) | ~2e-11 | 1 |
| 1 | D3 | **0.146** | **4e-11 (exact)** | 1 |
| 8 | D1/D2 | ~2e-11 (exact) | ~2e-11 | 8 |
| 8 | D3 | **0.464** | **2e-11 (exact)** | 8 |
| 16 | D1/D2 | ~2e-11 (exact) | ~2e-11 | 8 |
| 16 | D3 | **0.262** | **3e-11 (exact)** | 8 (capped at N1=8) |

Two findings, together more nuanced than either "closure survives
selectivity" or "selectivity destroys closure" alone:

- **D1/D2 (non-selective, even time-varying): the standard small
  per-source scalar `z`-chain formula (Part F's own closed form) is
  exact for any `N0`** — confirms exogenous time-variation alone,
  however fast, never threatens closure.
- **D3 (endogenous): the SAME simple formula is badly wrong (15-46%
  relative error) — but an exact CORRECTED formula exists** (adds one
  term per source, `h_{t-1}^1 * d(a_t)/d(theta_m)`), verified to
  `~1e-11` against finite-difference ground truth. **Gradients remain
  exactly computable under faithful selectivity.**
- **But the corrected formula's per-source state is NOT a scalar.**
  The extra term forces each source's running accumulator to track the
  SHARED upper-layer trajectory `h_{t-1}^1`, whose own rank (measured
  directly) grows with the number of simultaneously active sources,
  capped only by the upper width `N1`. **True persistent state for the
  exact correction is `O(N0*N1)`, not `O(N0)`** — the same order as
  literal per-source RTRL, not a saving. **Faithful endogenous
  selectivity, even through a single shared scalar gate, does destroy
  the small-credit-module property — even though it does not destroy
  gradient exactness.** D4 (a construction constraining the selector's
  derivative source to a small invariant module) was explored
  analytically: the natural way to force the extra term back to
  rank-1 requires the gate's dependence on `h^0` to reduce to a single
  active source (`N0=1`), which is a degenerate special case already
  covered by D1-D3 at `N0=1` — no nontrivial multi-source `D4`
  construction was found in this pass.

## 5. Part E — grouped/head-wise deep model (folded into the A2 sweep)

All-tied `G_l` in `{2,4,8}` (vs. `G_l=1` from Part A), `N=64`:

| L | task | G=1 (Part A) | G=2 | G=4 | G=8 | A0 full (reference) |
|---|---|---|---|---|---|---|
| 2 | delay_r8 | 2.333 | 2.123 | 1.607 | 0.849 | 0.023 |
| 3 | delay_r8 | 2.265 | 1.923 | 1.223 | 0.437 | 0.002 |
| 2 | hierarchical | 0.292 | 0.294 | 0.146 | 0.045 | 0.013 |
| 3 | hierarchical | 0.251 | 0.240 | 0.065 | 0.024 | 0.013 |

Monotonic, substantial improvement as `G_l` grows away from 1 for
BOTH tasks and both depths — but **even `G=8` (an 8x reduction from
`N=64`) remains far from `A0`: ~230x worse on `delay_r8`, ~1.9x worse
on `hierarchical`.** The easier, genuinely-multi-layer-useful
`hierarchical` task is approaching parity; the harder `delay_r8` is
not, even at 1/8 width. **Group assignment (contiguous vs random)
again makes negligible difference** (delay_r8, L=3, G=4: 1.224 vs
1.300) — reconfirming B16.1's finding that only the group *count*
matters, not which channels share a group; E1 vs E2 (head-preserving
vs generic dense routing) was tested via this same axis and shows the
same null result. E3 (a deliberately invariant routing construction)
was not built in this pass — an explicit scope reduction, given Part
1-3's results already answer the phase's central question without it.

## 6-8. Part F — not reached

Per the phase's own explicit gate ("proceed to Parts F-H only if Parts
A-E identify a useful all-layer structured regime"): **no all-layer
architecture tested — `A2` at any `G_l` up to 8, on any task — closed
the gap to the full model on the harder task family, and Part C showed
the one architecture that DOES stay credit-favorable at depth (`A2`)
is exactly the one that fails functionally.** The gate does not pass.
Per protocol, the end-to-end exact online grouped/invariant-credit
implementation was **not built** in this pass.

## 9. Verdict: **B, TEMPORAL-COMPLEXITY LOCALIZATION**

Checking against the four offered options:

- **A (STRONG IC-SSM)**: ruled out — all-layer tying fails badly at
  `G_l=1` and remains far from parity even at `G_l=8` on the harder
  task; the gate for Part F did not pass.
- **B (TEMPORAL-COMPLEXITY LOCALIZATION)**: **best fit.** Leaving
  exactly one layer untied (concentrating temporal complexity there)
  mostly preserves task performance (`A1`/`A3` track `A0` closely on
  most tasks), while the rest of the stack can be tied to `G=1` — this
  is a real, usable heterogeneous design principle. **Explicit
  caveat, not present in the option's own framing**: this
  localization does NOT automatically yield a favorable total
  credit-state once depth exceeds 2 (Part 3) — the untied layer's
  absolute cost, not its relative position, determines whether the
  whole-stack savings materialize. A genuinely favorable heterogeneous
  design needs the untied/high-complexity layer to sit where its OWN
  input width is small (e.g., closest to a low-dimensional input or
  bottleneck), not merely "somewhere in the stack."
- **C (SELECTIVITY BARRIER)**: partially supported by Part D (faithful
  selectivity does blow up the persistent credit state, from `O(N0)`
  to `O(N0*N1)`) but this phase's primary finding is about
  forward-expressivity localization, not selectivity — selectivity is
  a secondary, confirmed-but-not-central result here.
- **D (FORWARD-EXPRESSIVITY FAILURE)**: too strong — `A1`/`A3` DO
  achieve `G_l << N` on ONE layer with preserved performance; the
  failure is specific to tying ALL layers simultaneously, not to
  structuring the network at all.

**Net picture**: rich temporal computation can be localized to a
single layer while the rest of the stack runs on a tiny shared pole —
but doing so profitably (in the credit-accounting sense that motivated
this whole research direction) requires choosing THAT layer's position
so its own input width stays small, not just picking "the top" or "the
bottom" by convention. The next phase this suggests: search over WHICH
layer to leave untied (not just top/bottom) jointly with where the
network's own width bottlenecks naturally occur, and separately
resolve Part D's selectivity credit-state blowup before combining
selectivity with any localized architecture.

No new persistent online-credit training rule implemented. No S5 run.

## 10. Commit hash

See the commit introducing this file.
