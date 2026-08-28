# Phase B24 — interface-rank frontier

Branch `S5-CCM-scale-validation`. Tightly focused on one architectural
axis: temporal interface rank `k`. Code: `credit_memory/b24_interface_frontier.py`
(new; `main()` reproduces every number below). No S5.

**Headline: the smallest genuinely useful interface is `k_in·k_out ≥ 2`
(e.g. `k=2`, since `k=1,k_out=2` or `k_in=2,k_out=1` already break the
k=1 collapse). Above k=1, functional capacity does NOT grow without
bound — it saturates at an exact, verified closed form (for this
specific architecture): `cap = dim_below · min(k_in·k_out, r)`. Exact
prospective credit for a full-stack REFERENCE ADJOINT stays linear in
width (`O(n·k_in·r)` per layer) throughout. A real trained-task check
confirms the practical stakes: a k=1 approximator plateaus 37× worse
than a k=2 approximator matched to a genuinely 2D-interface target.**

> **CORRECTION (Phase B24.1, see `PHASE_B24_1.md`)** — three wording
> issues in this document, load-bearing enough to flag inline:
> (1) the driver here is **non-separable, multi-generator structure**,
> not `k>1` per se — a `k>1` architecture built from a truly separable
> route (`F_l⊗Q_l`, one shared scalar temporal generator) collapses
> exactly regardless of `n` or `k`, verified to machine precision in
> B24.1 Part A. The correct statement is "multiple independently
> feature-weighted temporal generators can make width useful at fixed
> k, until the finite temporal-function envelope saturates," not
> simply "k>1 makes width useful." (2) `stack_backward`'s
> `O(n·k_in·r)` cost is the cost of an exact **reverse-mode adjoint
> reference**, not a forward-only online algorithm — it must not be
> read as "the scaling of our online credit state"; no forward-only
> credit recurrence for this multi-copy architecture has been derived.
> (3) the cap law below is an **empirical exact-match law for this
> specific architecture**, not a general theorem — see B24.1 §6 for
> what is and isn't established about its generality.

## 0. Reporting/verification cleanup (B23), done first as required

The committed B23 code (`b23_integrated_pcrtl.py`) was corrected before
this phase began: `bptt_gradients_cascade` (finite-difference only) was
kept but relabeled `verify_fd_sanity`, a genuine analytic reverse-mode
adjoint `analytic_gradients_cascade` was added as the primary
reference, and `main()` was rewritten to actually test `L=2..5` (it
previously stopped at `L=3`, while the report had claimed `L=5` from an
unsaved interactive session). `PHASE_B23.md`'s Part 3 table was updated
to the analytic numbers (machine precision, `1.9e-16` to `4.6e-12`,
across `L=2,2,3,4,5`), with the correction stated explicitly. This does
not change the k=1 structural theorem. See that file/commit for detail;
not reproduced here.

## 1. Part A — the k-input canonical core

Shared-denominator MISO core: `k_in` parallel v-chains driven by a
COMMON AR denominator `a` (order `r`), `k_out` independent numerator
readouts `b[j,i,:]`:

```
v_t^(i) = u_t^(i) - Σ_s a_s v_{t-s}^(i)         (i = 1..k_in, shared a)
y_t^(j) = Σ_i Σ_s b[j,i,s] v_{t-1-s}^(i)        (j = 1..k_out)
```

Local state: `r` per input channel → `k_in·r` total, matching the
predicted `r·(k_in+…)`-style scaling from B22.1's SISO case generalized
to MISO (`p=1`). `make_core_coeffs`/`miso_core_forward`.

## 2. Part B — exact gradient verification

Two independent methods: `analytic_gradients_miso` (reverse-mode
adjoint through the per-channel companion-form state) and
`prefix_gradients_miso` (closed-form local eligibility, `dy/db=V`,
`dy/da=-Σ_j w_j`, one w-chain per output). Swept `r∈{2,4,8}`,
`k_in,k_out∈{1,2,4}` (27 configs):

**Max deviation across the entire grid: `2.2e-11`** (worst case at
r=8, consistent with accumulation, not FD noise — both sides are
analytic). Full grid in the code's own printed output.

## 3. Part C — a real deep k-channel interface

Architecture: layer has `n` copies, each with its OWN `k_in`-dim input
via a copy-specific mixing matrix `V[q]` drawn from the FULL
`(n_below·k_out_below)`-dim lower output (not a scalar broadcast),
processed by a SHARED (tied) `(a,b)` core. `build_layer` / `layer_forward`
/ `stack_forward`.

**Explicit warning honored**: verified first, as a standalone check,
that k output channels which ARE scalar multiples of one shared signal
still collapse exactly — `Σ_j y_t^(j) = y_t^eff` with `b_eff=Σ_j g_j b_j`,
max error `1.4e-14`. This generalizes B23's k=1 collapse and confirms
the multi-copy architecture only avoids that trap because each `V[q]`
genuinely mixes an independently-informative lower-layer signal, not
because of the nominal channel count.

## 4. Part D — functional width test, before training

**Sanity check first** (must recover B23 exactly): a true stack, k=1
throughout (external scalar → k=1 layer → k=1 layer), swept over the
first layer's width `n0∈{1,2,4,8}` — **second layer's output rank stays
exactly 1 at every n0.** Confirms the k=1 collapse theorem still holds
in this new codebase, unconditionally on width below.

**Main sweep**: a single layer fed a genuinely `dim_below`-dimensional
external signal (representing what a richer lower stack can supply),
swept over `(r,k_in,k_out,n)`. Measured the rank of the actual
`n·k_out` stacked output trajectories (not a parameter-Jacobian — an
earlier, discarded version of this measurement tested local parameter
identifiability, a different question, and was fully consistent with
total functional collapse at k=1; corrected before use here).

| r | k_in | k_out | n | rank | n·k_out | predicted |
|---|---|---|---|---|---|---|
| 3 | 1 | 1 | 16 | 8 | 16 | 8 |
| 3 | 2 | 2 | 16 | 24 | 32 | 24 |
| 3 | 4 | 4 | 16 | 24 | 64 | 24 |
| 5 | 1 | 1 | 16 | 8 | 16 | 8 |
| 5 | 2 | 1 | 16 | 16 | 16 | 16 |
| 5 | 1 | 2 | 16 | 16 | 32 | 16 |
| 5 | 3 | 3 | 20 | 40 | 60 | 40 |
| 2 | 2 | 2 | 16 | 16 | 32 | 16 |

**All 8 exact.** (One config initially showed `39` vs predicted `40` at
`T=40` — not a real deviation but a time-horizon resolution artifact,
diagnosed by inspecting the singular-value spectrum directly: the 40th
value was exactly `0.0` given only 40 samples to resolve 40 dimensions,
not a numerical floor. `T=80` recovers `40/40` cleanly — the same
"insufficient time-horizon margin" lesson from B22/B22.1, re-applied.)

**Answer to Part D's central question**: functional rank does NOT grow
unboundedly with `n` at fixed `k>1` — it grows with `n` up to a hard
cap, then flatlines regardless of further width. The k=1 case is
correctly recovered as the special case `min(k_in·k_out,r)=1`.

## 5. Part E — the width-vs-k function class law

**`cap = dim_below · min(k_in·k_out, r)`** — exact at all 8 tested
configurations (Part D table). Derivation sketch: `a` is shared across
all `n` copies, so filtering ANY linear combination of the `dim_below`
lower-signal rows through the SAME order-`r` filter confines every
copy's internal state trajectory to the span of `dim_below` FIXED
`r`-vector-valued trajectories (one per lower-signal basis row) — an
atom pool of size `dim_below·r`. Reading out via `b[j,i,:]` (also
shared across copies) can only apply `min(k_in·k_out, r)` independent
linear projections of each atom's `r`-dim content (since there are only
`k_in·k_out` distinct `b`-vectors, each in `R^r`), never more than `r`
even if `k_in·k_out>r`. Copies (`V[q]`) only choose WHICH mixture of
lower-signal rows to feed in — they cannot exceed the projection budget
already fixed by `(k_in,k_out,r)`.

Checking against the phase's four candidate laws: this is closest to
**law B ("fixed k permits growth only until an O(k,r) saturation
point")**, sharpened to an exact formula that also depends on the
available input diversity `dim_below` — which, in a real deep stack,
is itself the previous layer's own `cap`, so the law composes
recursively down the stack (a `k=1` layer anywhere in the chain forces
`dim_below=1` into every layer above it, reproducing the flat-rank-1
Part D sanity check as the boundary case). Not law A (unbounded growth
— ruled out, cap is finite and reached quickly: e.g. r=3,k=2×2 needs
only `n=12` copies, not more, since `24/k_out=12`). Not law C exactly
(not literally "collapse to a bounded-order transfer family"
independent of `k` — the cap scales with `k_in·k_out` up to `r`, so
`k` matters up to that point). Not law D — no nonlinearity was needed;
this is a purely linear phenomenon.

## 6. Part F/G — deep exact credit, width scaling

`stack_backward`/`layer_backward`/`miso_backward`: a genuine analytic
reverse-mode adjoint through the WHOLE stack — companion-form per
input channel per copy, gradients w.r.t. shared `(a,b)` accumulated
across copies, gradients w.r.t. each `V[q]` and the propagated
`dL/d(lower_out)` computed via the chain rule through the mixing.
**Not finite differences** — FD used only as a secondary sanity check,
per the phase's own instruction not to call FD agreement "machine
precision":

| L | max |analytic adjoint − FD sanity check| (sampled `a` entries) |
|---|---|
| 2 | 1.2e-10 |
| 3 | 3.2e-11 |
| 4 | 3.1e-12 |

These sit at FD's own precision floor (`eps=1e-6`), correctly labeled
a sanity check, not machine precision. (First attempt at this
comparison gave large, non-uniform errors — `0.06` to `4.4` — traced
to an inconsistency in my OWN finite-difference test harness: it used
`np.mean` over both channels and time, while the code's `err`/`dL_dOut`
convention divides by `T` only [summing over channels] — not a bug in
`stack_backward` itself. Fixed the test harness, re-verified: floor-level
agreement above. Worth stating plainly, matching the phase's "do not
assume either side is correct" instruction.)

**Width scaling of credit state (Part G)**: `layer_backward` does
`O(k_in·r)` local adjoint work per copy, `O(n·k_in·r)` total per layer
— **linear in `n`**, independent of whether the functional cap has
already saturated. This is exactly the desired decoupling: adding
copies past the cap (Part D/E) wastes compute on capacity that doesn't
exist, but never makes the credit computation itself more expensive
than linear — no combinatorial blowup, no `r^n`, no `r²` per extra
copy.

## 7. Part H (light) — a real trained-task check

Target: a genuine 2-layer, `k=2`-interface stack (`k_in=1,k_out=2` then
`k_in=2,k_out=1`), fixed ground truth. Two approximators trained by
gradient descent using the exact `stack_backward` machinery
(1500 steps, clipped gradients):

- **k=1 (width-vacuous per B23/Part D, n=4 copies)**: final loss
  `0.197` — plateaus almost immediately after the initial large drop
  (loss at step 300 already `0.227`, step 1500 `0.197` — essentially
  flat), consistent with its layer-1 cap of `dim_below·min(1,3)=1`.
- **k=2 (interface-matched, n=1)**: final loss `5.3e-3`, still
  decreasing steadily at step 1500 (`0.014→0.011→0.008→0.007→0.005`).

**37× worse for the k=1 approximator** — the functional-rank ceiling
identified in Part D/E is not a linear-algebra curiosity; it directly
predicts which architectures can and cannot fit a task that genuinely
needs a multidimensional temporal interface. Full performance-frontier
sweep (multiple tasks, matched parameter budgets, wall-clock) is
explicitly out of scope here, gated correctly behind A–G per the
phase's own instruction — this is a single confirmatory data point, not
a benchmark.

## 8. Part I — verdict

Checking against the four offered kill-criteria:

- **B, "fixed k permits growth only until O(k,r) saturation" —
  confirmed, sharpened to an exact closed form.** `cap =
  dim_below·min(k_in·k_out, r)`, verified exactly at all 8 tested
  configurations (Part D/E), with the k=1 boundary case independently
  re-confirmed via a true stack sanity check (rank stays exactly 1
  regardless of lower-layer width). Exact prospective credit for the
  whole multi-copy, multi-layer stack is a genuine analytic adjoint
  (Part F), costing `O(n·k_in·r)` — linear in width, decoupled from
  whether the cap has already saturated (Part G). A real trained-task
  check (Part H) confirms the cap has practical bite: a k=1
  approximator is stuck 37× worse than a k=2 approximator on a target
  that needs the extra channel.
- **A (unbounded growth)**: ruled out — every swept config saturates,
  and saturation is reached at a small, predictable `n` (`≈cap/k_out`
  copies), not asymptotically.
- **C (bounded-order transfer family, k-independent)**: ruled out as
  stated — the cap DOES depend on `k_in·k_out` (up to `r`), so `k`
  matters up to the point where `k_in·k_out≥r`, not before.
- **D (needs nonlinearity)**: ruled out — every result here is exact
  linear algebra; no nonlinear readout or activation was used anywhere
  in Parts A–H.

## 9. What this licenses going forward (not attempted here)

The smallest genuinely useful interface is `k_in·k_out=2` (e.g.
`k_in=1,k_out=2` or `k_in=2,k_out=1`) — anything with `k_in·k_out=1`
provably collapses regardless of width, confirmed independently at two
different points in this phase (B23's original result and this
phase's own stack sanity check). Beyond that, useful width is bounded
and predictable via the exact cap formula, and exact credit for it is
cheap and already implemented (`stack_backward`). A genuine
performance-frontier study (real B18 diagnostic tasks, matched
parameter budgets, multiple architectures on the frontier, wall-clock)
is the natural next phase, now that the underlying capacity law and
the credit machinery are both established and verified — not
attempted here per the phase's own gating.

No new production online-credit training rule deployed. No S5 run.

## 10. Commit hash

See the commit introducing this file.
