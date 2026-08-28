# Phase B20 — end-to-end exact invariant-credit RTRL

Branch `S5-CCM-scale-validation`. **Scope note up front, stated
plainly**: this is the largest single ask of the entire B-phase series
(a genuine forward-only exact RTRL implementation, verified against
BPTT and full reference RTRL, plus structured routing, deep
selectivity, Pareto curves, and a realistic task). Given the time
already invested in this single phase, this pass delivers a real,
machine-precision-verified core (Parts A-D, `L=2`, dense core) and
explicitly does NOT reach Parts E-I or the sidearm. This is reported
honestly as an incomplete flagship, not dressed up as more than it is.

Code: `credit_memory/b20_ic_rtrl.py` (new). No BPTT through time is
used anywhere in the IC-RTRL/naive-RTRL code paths (verified by
construction: every recursion is forward-in-`t` only); BPTT itself is
retained purely as the reference to verify against. No S5.

**Headline: exactness is real and verified to machine precision,
including a genuine, working forward-only compression for one class of
cross-layer parameter (routing sources) — but the phase's own
distinction between compressible and non-compressible credit, adopted
from B19, is confirmed sharply: a lower layer's OWN dense-core R
parameters do NOT compress when propagated to upper layers (their
forcing depends on the network's own high-rank state, not a simple
source), and the current implementation is 30-220x SLOWER in wall-
clock than BPTT due to unvectorized per-parameter Python loops. Exact,
partially compressed, and currently impractical — reported as such.**

## 1. What was built and verified (Parts A/B)

Three algorithms, `L=2` (one "lower" recurrent layer, one "top"
recurrent layer), dense trainable core `R` at each layer (per B19's
own conclusion — the practically useful core, not a commuting one):

- `bptt_gradients` (reused from b18_temporal_core): the established
  reference throughout B16-B19.
- `naive_rtrl_gradients`: textbook forward-only RTRL. For every scalar
  trainable parameter, maintains an EXPLICIT chain of per-layer
  intermediate sensitivities (`dh_t^{l0}/dtheta`, `dh_t^{l0+1}/dtheta`,
  ...), each updated via its own layer's dynamics in the same causal
  order as the forward pass — never looks backward in time. Verified
  against BPTT to machine precision at `L=2` (`4.2e-17` max error) and
  `L=3` (`1.4e-17`) on small test cases, after fixing a real row/column
  index-order bug caught by the verification itself.
- `ic_rtrl_gradients_L2`: the compressed version. `R`'s own local
  gradient is O(r²) per layer (matching B19's "small-core-local and
  unavoidable" finding — not compressed further). Propagating a lower
  layer's routing parameters (`B_l[i,m]`) through the upper layer is
  compressed via a NESTED two-stage K-chain (derivation and a real
  index-order bug, both caught by direct verification — see below).

**Full gradient agreement, all five parameter blocks, machine
precision** (`R0`, `R1`, `B0`, `B1`, `c`; `1e-17`-`3e-17` range),
verified at `n1` (feature multiplicity of the top layer) from 4 to 40.

## 2. The compression, derived and corrected via verification

`B0[i,m]`'s local sensitivity at layer 0 is confined to copy
`p0=i//r0`'s own `r0`-dim block (layer 0's `A0` is block-diagonal and
never mixes copies) — a genuine `r0`-dim vector `v_t`. The FIRST
derivation attempt assumed this propagates through the top layer via a
single `r1×r1` K-matrix; **verification immediately caught this as
wrong** (`dB0` error `1.38`, not machine precision) — because `v_t` is
`r0`-dimensional, not scalar, so propagating it correctly requires `r0`
SEPARATE `r1×r1` K-matrices (one per component of `v_t`), each
following `K_t^{(k)} = R1·K_{t-1}^{(k)} + v_t[k]·I_{r1}`, combined as
`block_q = Σ_k K_t^{(k)} @ W_q[:,k]` (`W_q` = the top layer's own fixed
routing block for copy `q`). Fixed, then verified exact to machine
precision. **This is exactly the kind of error the project's
established verify-before-trust discipline exists to catch** — a
plausible-looking compression that undercounts a real degree of
freedom, caught immediately rather than silently shipped.

Storage per `B0` source: `r0·r1²`, INDEPENDENT of `n1` — measured
directly:

| n1 | N1 | naive storage/source | compressed storage/source | ratio |
|---|---|---|---|---|
| 4 | 8 | 8 | 12 | 0.7x (compression LOSES at small n1) |
| 20 | 40 | 40 | 12 | 3.3x |
| 40 | 80 | 80 | 12 | 6.7x |
| 100 | 200 | 200 | 12 | 16.7x |

**The compression genuinely pays off, but only past a crossover width**
(here around `n1≈12`, where `r0·r1²` first drops below `N1`) —
below that, the naive per-source vector is already smaller. This is a
real, useful, but not unconditional saving.

## 3. What does NOT compress (found directly, not assumed)

`R_l`'s own `r_l²` local parameters, when propagated to the top layer,
use the SAME naive (uncompressed, `O(N_upper)` per parameter)
propagation as `naive_rtrl_gradients` — deliberately, because the
forcing term `(I⊗E_jk)@h_prev_l` depends on `h_prev_l`, **the
network's own full recurrent state**, not a simple fixed-direction
source. B17's own structural-rank findings already established that
this state is generically full-rank once a layer's own input width is
large — so there is no hidden low-dimensional structure here to
exploit. **This directly confirms and sharpens B19's own framing**:
"accept the generic r² local gradient/output requirement" turns out to
mean not just the LOCAL computation is O(r²), but its FORWARD
PROPAGATION to the readout is genuinely O(r²·N_upper) per layer
boundary when done exactly online — a real, load-bearing cost that
B20's own optimistic "O(r) per relevant source" framing does not
automatically extend to a layer's own core parameters.

## 4. Exact training-trajectory reproduction (Part C)

Under a MATCHED update schedule (one Adam step per full trajectory,
identical init, identical data — the honest apples-to-apples
comparison the phase itself calls for, given "exact gradient equality
and identical training trajectory are different claims if update
schedules differ"): after fixing a second bug (the top layer's own
routing gradient, `dB1`, was silently left at zero in an early version
— caught immediately because gradient-block comparison showed a
`0.12` discrepancy where machine precision was expected, again the
verification discipline doing its job), **BPTT and IC-RTRL training
trajectories are bit-identical over 100 steps** (max loss difference
`2.2e-16`, exactly the floating-point floor — no drift at all).

**True per-timestep sequential updates** (applying an Adam step after
EVERY timestep, not once per trajectory — the genuinely "online"
schedule the phase's own Part C distinguishes from the matched-schedule
test) **were not implemented** — an explicit scope reduction. The
matched-schedule result establishes gradient/trajectory correctness;
it does not yet establish behavior under a truly sequential online
update regime, which changes the forward dynamics for the remainder of
each sequence mid-trajectory and was not attempted here.

## 5. Total cost accounting (Part D) — including the uncomfortable number

| quantity | measured/derived |
|---|---|
| forward recurrent state | `N_l` per layer (standard) |
| cross-layer temporal credit (`B_l`, l<top) | `r_{l}·r_{l+1}²` per source, **independent of n** |
| cross-layer temporal credit (`R_l`, l<top) | `N_{l+1}` per source (NOT compressed) |
| local dense-core gradient (`R_l`) | `r_l²·N_l` per layer |
| routing-gradient state (top layer's own `B`) | `N_L·M_L` (local, standard) |
| readout gradient | `N_L` |
| **wall-clock, this implementation, vs BPTT** | **33x-218x SLOWER**, growing with n1 |

**The wall-clock number is the load-bearing caveat of this phase.**
The per-source memory footprint for `B_l` genuinely shrinks relative
to `N_upper` exactly as predicted — but the CURRENT implementation
computes this via Python-level loops over individual scalar
parameters (`for i: for m: ...`), with none of the vectorization
`bptt_gradients` gets from batched `einsum` calls across all
parameters simultaneously. **This is an implementation-efficiency gap,
not a claim about the algorithm's fundamental complexity** — a
production version would vectorize the K-chain updates across all
`(i,m)` sources at once — but it was NOT built that way here, and the
measured wall-clock numbers must be read as "correctness demonstrated,
practical speed not yet demonstrated," not as evidence the approach is
currently fast.

## 6. Parts E-I and the sidearm — explicit scope reductions

**Not attempted in this pass**, given the time already invested in
Parts A-D's exact verification (which itself required catching and
fixing two real derivation bugs):

- **Part E (structured routing)**: not implemented in THIS phase's
  code. B18 Part E already found that structured (`M⊗I_r`,
  `ΣM_k⊗Q_k`) routing costs nothing relative to dense routing on the
  B18 task suite — a relevant, already-existing data point, but not a
  test of whether that finding survives being the trainable parameter
  set inside an actual online RTRL update rule.
- **Part F (symbolic scaling formula)**: partially covered by Part 5's
  table; not derived into a single closed-form expression across
  arbitrary `L`.
- **Part G (selectivity in the full deep learner)**: B18/B19 both
  verified selective-core gradients exactly in STANDALONE constructions
  (not integrated into this phase's actual IC-RTRL implementation).
  Whether the standalone formula survives being wired into the real
  online update loop is untested.
- **Part H (Pareto curves)**, **Part I (realistic task)**, and the
  **optional sidearm (intermediate core algebras)**: not run.
- **`L≥3` generalization**: `naive_rtrl_gradients` is verified correct
  at `L=3`; `ic_rtrl_gradients_L2`'s compression is implemented and
  verified ONLY for `L=2`. Composing the K-chain compression across
  two or more layer boundaries (per B19's own finding that bimodule
  composition can itself grow) was not attempted.

## 7. Verdict: **B, EXACT LEARNER WORKS, SCALING IS NOT YET ESTABLISHED — qualified**

Checking against the five offered options:

- **A (end-to-end exact scaling success)**: not established — only
  `L=2` verified, `R_l`'s own cross-layer propagation does not
  compress, structured routing was not tested in this phase's actual
  learner, and wall-clock is currently far worse than BPTT.
- **B (exact learner works, but routing kills scaling)**: **closest
  fit, with a correction to which piece is the bottleneck.** The
  phase's own framing anticipated dense routing PARAMETER COUNT as the
  scaling gate; what this phase actually found is a second, distinct
  gate — a lower layer's own dense-core parameters do not compress
  when propagated forward online, independent of the routing question
  entirely. Both are real; neither was resolved by structured routing
  here (untested), so "scalable exact temporal credit, not near-linear
  exact online training" is the accurate summary, extended to cover
  both gates.
- **C (selectivity kills full deep closure)**: not tested in the
  actual learner — no verdict possible from this phase's own work.
- **D (performance kill)**: not applicable — no forward-quality
  comparison was run in this phase (B18/B19 cover that ground
  separately).
- **E (exactness failure)**: explicitly ruled out — every gradient
  block verified to machine precision, and every bug found along the
  way was caught by the verification discipline itself and fixed
  before being reported, not left unresolved.

**What this phase actually establishes**: the theoretical machinery
from B19 is not just correct in isolation (as B19 itself showed) — it
composes into a genuine, exact, forward-only training algorithm, with
real (if bounded and width-dependent) memory savings for one class of
parameters. What remains before any "flagship" claim is warranted:
vectorized implementation (to make the wall-clock honest), the `R_l`
propagation cost resolved or accepted as a permanent limitation,
structured routing tested inside the actual learner (not just cited
from B18), and `L≥3` composition verified rather than assumed.

No new PRODUCTION persistent online-credit training rule was deployed
— per the phase's own instruction, this is a verification-grade
reference implementation only. No S5 run.

## 8. Commit hash

See the commit introducing this file.
