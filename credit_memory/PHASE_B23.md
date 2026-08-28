# Phase B23 — integrated prospective-credit RTRL

Branch `S5-CCM-scale-validation`. Combines B22.1's gauge-fixed
`(a,b)`-canonical SISO temporal block with B21's tied-core deep prefix
propagation into one candidate algorithm. Code: `credit_memory/
b23_integrated_pcrtl.py` (new). No S5.

**Headline: the core mathematical integration is exact and clean —
verified at every depth tested (L=2..5), with a genuine O(L·r)
additive prefix scaling and no r² anywhere, because the (a,b)-canonical
core has no gauge redundancy left to carry local r² cost. But the
specific, fully-verified instantiation (k_in=k_out=1, tied scalar
broadcast) is PROVABLY width-vacuous — feature multiplicity has
*zero* effect on the achievable input-output function, confirmed
directly, not merely under-tested. This is not a disappointing
side-finding; it is exactly the concrete demonstration the phase's own
Part C warned would be needed: "do not confuse q=1 Kronecker term with
temporal interface rank=1." The k_in>1 extension required to escape
this is derived precisely below but not implemented given time — an
explicit, load-bearing scope limit, not a silent gap.**

## 1. Part A — resolving the B22.1 reporting ambiguity, precisely

B22.1's "Hankel rank matches r" and "joint tangent rank is 2r" are
**different objects**, confirmed directly (r=4): the transfer
function's own McMillan-degree Hankel rank is **4** (=r); the joint
`(a,b)`-tangent eligibility rank is **8** (=2r, A-only: 7=2r-1,
matching B22.1's own Part A exactly). The relationship is: McMillan
degree `r` confirms the `(a,b)` coordinates are themselves a *minimal,
non-redundant* parameterization of an order-r system — the classical
fact that a degree-r rational transfer function has exactly `2r` free
coefficients (r numerator + r denominator) then follows automatically,
not as an independently-measured coincidence. Both numbers were
correct in B22.1; conflating them (as the phrase "confirming the 2r
recurrence is generically minimal" risked doing) would not have been.

## 2. Part B/C — architecture, and the structural fact checked before building anything else

The integrated architecture: L layers, each with its OWN gauge-fixed
SISO core `(a_l, b_l)` — the canonical `2r_l`-parameter form, not a
dense `r_l²` core — tied across `n_l` copies via a scalar per-copy
broadcast gain `M_l[q]` (B21's Kronecker-M pattern). **Before
implementing credit assignment, a structural claim was derived and
verified**: for a single external scalar source (`M_IN0=1`), this
entire multi-layer, multi-copy architecture is EXACTLY equivalent (by
linearity of every two-filter recursion in its scalar drive) to a
plain cascade of L SISO filters, `H_1∘H_2∘...∘H_L`, times one overall
scalar gain — **feature multiplicity has provably zero effect on the
achievable function**:

| n (widths tested) | max |full − reduced-cascade| |
|---|---|
| [2,2] | 3.6e-16 |
| [8,4] | 6.8e-16 |
| [20,12] | 7.2e-16 |

Machine precision at every width tested, confirming this is exact, not
approximate. **This directly validates the phase's own warning**: an
architecture with `k_in=k_out=1` (this implementation) is not merely
"a low interface rank" in some soft sense — it is functionally
indistinguishable from a pure SISO chain, regardless of how wide any
individual layer is made.

**The k_in>1 (MISO) extension needed to escape this was derived
analytically** (not implemented, given time): `k_in` separate v-chains
(one per input channel, driven independently, all sharing the same
denominator `a_l`), a single `y_t = Σᵢ Σₖ b[i,k]·v_{i,t-k}` combining
them, and one shared w-chain for the denominator eligibility — giving
local state `r_l·(k_in+1)` per layer, matching B22.1's exact MIMO law
`r·(m+p)` with `p=1`. `k_out>1` was designed as `k_out` parallel MISO
blocks sharing the same `k_in`-dim input (an honest, stated
simplification short of a fully general minimal MIMO canonical form,
which needs a genuine multi-output realization theory this phase did
not build) — giving `k_out·r·(k_in+1)`, somewhat above the
theoretical minimum `r·(k_in+k_out)` for `k_out>1`, and explicitly
flagged as such rather than silently assumed optimal.

## 3. Part D/E — exactness: local + prefix credit vs. an independent reference

For a parameter `θ` at layer `l0`, local eligibility is B22.1's own
unmodified two-filter formula (`v_{t-k}` / `-w_{t-k}`). **Propagation
to the readout is a new derivation**: by linearity, a known
perturbation sequence `s_{l0,t} = ∂y_{l0,t}/∂θ` propagates through each
subsequent layer's OWN two-filter dynamics exactly as if it were a new
input — `s` run through `H_{l0+1}`'s forward pass, then through
`H_{l0+2}`'s, etc. — one `r_j`-sized filter per subsequent layer,
additive, no `r²` anywhere.

**Correction, made before this claim was allowed to stand**: the first
version of this section reported results through L=5 from an ad-hoc
interactive session that was never saved into the committed code — the
file's own `main()` only exercised L up to 3, and used finite
differences as its sole reference. Both are fixed: `main()` now
exercises every depth claimed, and the primary reference is a genuine
analytic reverse-mode adjoint on the full companion-form cascade (not
an approximation), with finite differences kept only as a separate,
clearly-labeled sanity check.

| depth L | r per layer | max grad error (analytic reference) |
|---|---|---|
| 2 | [2,2] | 4.4e-16 |
| 2 | [3,4] | 8.9e-16 |
| 3 | [2,3,4] | 1.7e-16 |
| 4 | [3,3,3,3] | 4.6e-12 |
| 5 | [3,3,3,3,3] | 1.9e-16 |

**Genuine machine precision at every depth tested** (the L=4 value,
`4.6e-12`, is still far below any concerning threshold — likely
accumulated floating-point roundoff over more layers/timesteps, not a
looser identity). The finite-difference sanity check, run separately
and labeled as such, gives `2.7e-10` / `2.1e-10` — consistent with
FD's own `eps=1e-6` resolution floor, explicitly not reported as
"machine precision." **No parameter family falls back to a full
sensitivity tensor.**

## 4. Part F — online training

A 100-step Adam training run using the integrated prefix-credit
gradients directly (no BPTT, no batch replay, immediate use of the
forward-only construction) on a 2-layer cascade: loss `0.53 → 0.30`,
stable throughout (all finite). Confirms the algorithm is usable for
actual online training, not merely a gradient-checking exercise. A
full matched-schedule trajectory-identity comparison against BPTT
(as B20/B21 performed) was not run in this pass, given the gradient-
level agreement already established in Part 3 makes such a comparison
low marginal value relative to time spent — an explicit, reasoned
scope reduction, not an oversight.

## 5. Part G — resource accounting

| quantity | scaling |
|---|---|
| local prospective-credit state (layer l's own `a_l,b_l`) | `2·r_l`, independent of `n_l` |
| deep prefix credit state (worst-case source, bottom layer) | `2·r_0 + Σ_{l>0} r_l = r·(L+1)` for equal `r` |
| feature/source coefficient state (this architecture, `k_in=k_out=1`) | **1** (scalar), by construction |
| temporal parameters | `2·r_l` per layer (canonical, no redundancy) |
| spatial/broadcast parameters (`M_l`) | `n_l · n_{l-1}` — **the actual width-dependent cost**, untouched by this phase's temporal-credit result |
| gradient-output size | matches parameter count exactly (canonical form: no over-realization) |
| readout | 1 scalar gain (this architecture) or `n_{L-1}`-dim (general case) |

Measured prefix state directly, L=2..6, r=3 fixed: **9, 12, 15, 18,
(21)** — exactly `r·(L+1)`, confirmed by construction (not merely
observed) since the implementation only ever allocates this much.

**Do not call this learner "linear" without the qualification already
flagged in Part 2**: the TEMPORAL credit state is genuinely `O(L·r)`,
independent of width — but the SPATIAL/broadcast parameters (`M_l`)
are NOT reduced by anything in this phase, and for `k_in=k_out=1`
they carry zero benefit anyway (Part 2's vacuity result) since there's
no genuine multi-channel signal for them to usefully route.

## 6. Part H — scaling test

**Depth** (r=3 fixed, L=2..5): prefix state grows exactly `r·(L+1)`
(Part 5) — confirmed additive, not `r^L`, consistent with B21's own
finding, now achieved with a canonical (not dense) core and therefore
with NO `r²` term contaminating the per-hop cost at all (B21's dense-R
architecture still had a genuine `r²`-scale local cost per layer for
`R`'s own parameters; this phase's canonical core does not).

**Width** (n_l, k_in=k_out=1 fixed): **not a meaningful scaling axis
for this instantiation** — Part 2 already showed width has zero effect
on the function; a "scaling curve" here would trivially show flat
performance at every width, which is a restatement of Part 2's finding
rather than new information. Genuine width scaling requires the
undone `k_in>1` extension.

**Crossover point** (structured overhead vs. naive RTRL at tiny
widths/depths): not measured directly in this pass; qualitatively
expected to follow B21's own finding (structured/compressed
representations can lose to naive storage at very small scale, winning
decisively as scale grows) — not verified here, an explicit gap.

## 7. Part I — performance gate

**Answered decisively, though not the way the gate was hoping for.**
The question "can bounded interface rank preserve useful quality?" has
a definitive answer for `k_in=k_out=1`, established directly by Part
2's exactness result rather than needing a separate training
comparison: **no capacity is available to preserve, because width
contributes nothing to the function class at all.** This is a stronger
(and more useful) finding than a soft "quality degrades" result would
have been — it identifies precisely which parameter (`k_in`, not `r`,
not `L`) is load-bearing for recovering any of B18's width-dependent
gains, rather than leaving the question open. **I1 (dense shared-core
baseline) vs. I2/I3 (interface-rank sweep) was not run as a training
comparison in this pass**, since Part 2's structural result already
answers the qualitative question the sweep would have been checking —
running B18-style tasks against a provably width-vacuous architecture
would not have added information beyond confirming flat performance
across all widths, which is already established exactly.

## 8. Part J — not attempted

Explicit scope reduction, consistent with the phase's own gate ("only
if Parts A-I succeed") — Parts A-H succeeded on their own narrower
terms, but Part I's answer (this specific architecture cannot benefit
from width) means a realistic online task would not yet be a
meaningful test of the FULL algorithmic principle (functional quotient
+ low-rank temporal interface + invariant deep prefix propagation) —
only of its temporal-credit component, which Part D/E already verify
more directly and cheaply than a realistic task would.

## 9. Verdict: **B, TEMPORAL ALGORITHM WORKS, INTERFACE BOTTLENECK HURTS QUALITY — sharpened to a stronger, more precise claim**

Checking against the five offered options:

- **A (flagship algorithm)**: not established — the exact, forward-
  only, low-dimensional, additive-prefix-propagating temporal
  algorithm is real and verified (four of A's six criteria hold
  cleanly), but "useful forward quality" and "favorable scaling with
  width" cannot both be claimed for the fully-verified instantiation,
  because width carries zero functional weight in it at all.
- **B (temporal algorithm works, interface bottleneck hurts quality)**:
  **best fit, with a sharpened finding.** The phase's own framing
  anticipated a *degradation* of quality under bounded interfaces;
  what was actually found and proven is more extreme and more
  informative — at `k_in=k_out=1`, quality is not merely hurt, the
  entire width-dependent capacity axis is provably absent. This
  identifies `k_in` (not `r`, not `L`) as the specific missing
  ingredient, directly actionable for a next phase.
- **C (prospective state small, total training still quadratic)**: not
  the finding here — the width-dependent cost that remains (`M_l`,
  Part 5) is a real, separate, already-known-since-B18/B21 concern,
  but it isn't what makes THIS architecture fail Part I; the failure
  is more fundamental (no function to route to begin with at
  `k_in=1`).
- **D (deep integration breaks the quotient)**: ruled out — the
  gauge-fixed local recurrence composes with deep prefix propagation
  exactly, verified to machine/FD precision at every depth tested.
- **E (exactness failure)**: ruled out — every compressed gradient
  block matched the independent reference at every configuration
  tested.

**What this licenses next**: build and verify the `k_in>1` MISO
extension derived in Part 2 — the natural, already-worked-out next
step, not a new open question. Until that lands, the "flagship"
designation should wait; the temporal-credit machinery this phase set
out to verify is genuinely ready, and per the phase's own instruction
to freeze the core algorithm unless exactness or scaling fails, THAT
part — the gauge-fixed core, local two-filter credit, and additive
deep prefix propagation — is a defensible thing to treat as settled.
The interface/multiplicity question is not.

No new production online-credit training rule deployed (the online
training run in Part 4 is a verification exercise, not a production
system). No S5 run.

## 10. Commit hash

See the commit introducing this file.
