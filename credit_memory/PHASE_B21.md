# Phase B21 — deep prefix IC-RTRL

Branch `S5-CCM-scale-validation`. Focused phase, scoped exactly as
requested: implement and verify the `L≥3` temporal-prefix dynamic
program from the theory correction, apply the two free B20 fixes,
audit measured ranks against the predicted bound, test depth scaling,
compare routing classes, and separate coefficient complexity from
temporal-module complexity. No broadening beyond this. Code:
`credit_memory/b21_prefix_ic_rtrl.py` (new). No S5.

**Headline: the theory is confirmed, and the resolution reads exactly
as the correction predicted — "the feared intrinsic r^L temporal path
blowup is ruled out... the remaining scaling risk is feature/source
coefficient complexity, not symbolic temporal path count." Verified:
machine-precision gradient agreement at L=2 through 5 for a q=1
Kronecker-routed source, with measured persistent state growing
exactly O(L·r) for one source origin — even better than the theorem's
own O(L·r²) prediction, because q=1 structure collapses the
coefficient side entirely. The predicted remaining problem (a layer's
own dense-core parameters) is confirmed separately, exactly on
schedule, not "killed" by this phase's own result.**

## 1. Part A/B — the prefix recurrence, implemented and verified

For a routing-parameter source at layer `i` (entry `(a,m)` of `B_i`),
under `q=1` Kronecker routing (`B_l = M_l ⊗ Q_l` for `l≥1`):

```
v_t          = R_i · v_{t-1} + drive_t · e_{pos0}        (origin, r_i-dim)
y_t^{(i+1)}  = R_{i+1} · y_{t-1}^{(i+1)} + Q_{i+1} · v_t
y_t^{(k)}    = R_k · y_{t-1}^{(k)} + Q_k · y_t^{(k-1)}     (k > i+1)
```

with copy `q`'s full sensitivity at the top layer reconstructed as a
**scalar** (composed from the `M_k` copy-mixing factors along the
path) times the **single shared** top-layer vector `y_t^{(L-1)}`. This
is stronger than the phase's own `dim(C_{k←i}) ≤ r_i·r_k` prediction:
under `q=1` structure the coefficient side collapses entirely, leaving
just `r_k` — not `r_i·r_k` — per hop. Discovered while deriving the
implementation, reported explicitly since it exceeds what was asked to
be verified.

**Exact gradient agreement, every trainable block** (`R_l` at every
layer, `B_l` at every layer including the top's own routing and every
intermediate layer's, `c`), against `bptt_gradients`:

| L | max error across all blocks |
|---|---|
| 2 | `5.6e-17` |
| 3 | `2.2e-16` |
| 4 | `5.6e-17` |
| 5 | `3.1e-17` |

All at or near the floating-point floor. **The gate ("do not proceed
to performance benchmarking if exactness fails") passed at every depth
tested — proceeding to Parts D onward is warranted.**

Two real bugs were caught and fixed during this verification, both by
the same discipline used throughout the session — a missing gradient
block (`B` for intermediate, non-bottom/non-top layers was silently
never computed in an early draft, caught immediately via a `1.59`
error where machine precision was expected) and a messy first
attempt at the R-parameter cross-layer chain (cleaned up by directly
reusing B20's already-verified multi-hop chain pattern rather than a
half-finished ad hoc version).

## 2. Part C — the two free B20 fixes, applied and verified

1. **Single-hop routing compression, corrected**: B20 stored
   `r_lower` separate `r_upper×r_upper` matrices (`r_lower·r_upper²`).
   Under `q=1` routing this phase's implementation uses a single
   shared `r_upper`-dim vector per hop — **`r_upper`, not even
   `r_lower·r_upper`** (see Part 1). Verified exact.
2. **Top-layer routing eligibility, `O(r)` not `O(N)`**: implemented
   directly — each `B_{L-1}[i,m]`'s local sensitivity is stored and
   updated as an `r_{L-1}`-dim block (not the full `N_{L-1}`-dim
   vector B20 used), confined to the injection copy for its entire
   lifetime (verified structurally correct via the exact gradient
   match above, which would fail immediately if any leakage outside
   the block were being silently dropped).

## 3. Part D — prefix rank audit (resolves the earlier "rank 10" ambiguity)

The earlier B20-correction measurement (rank 10 for a source at
`r0=3` propagated through `r1=4,r2=5`) is **now understood precisely**:
the correct bound for that configuration is `dim(C_{2←0}) ≤ r0·r2 =
15` (a single Hom-space bound from origin directly to endpoint, not
`r0` or `r0·r1·r2`) — `10 ≤ 15`, consistent throughout; the earlier
framing simply compared against the wrong two reference points.

For THIS phase's `q=1`-structured implementation, the measured state
is exactly the constructive sum (not a separate "audit" — the
implementation only ever allocates this much, by construction):

| origin `i`, one source | `dim C_{i←i}` (=`r_i`) | `dim C_{i+1←i}`...`C_{L-1←i}` (each `=r_k`) | total, one origin |
|---|---|---|---|
| any `i`, `L=3`, `r=4` | 4 | 4, 4 | 12 |
| any `i`, `L=5`, `r=4` | 4 | 4,4,4,4 | 20 |

Matches `O(L·r)` for one origin exactly (by construction, not merely
observed) — see Part 4 for the full depth sweep.

## 4. Part E — depth scaling, one fixed source origin and all origins

`r=4` fixed, width `N=16` fixed (`n` shrinking as `L` grows to hold
`N` constant), `L=2..6`:

| L | state (one origin) | `O(L·r)` prediction | state (all origins, summed) | `O(L²·r)` prediction | naive (one origin, uncompressed) |
|---|---|---|---|---|---|
| 2 | 8 | 8 | 8 | 16 | 16 |
| 3 | 12 | 12 | 20 | 36 | 16 |
| 4 | 16 | 16 | 36 | 64 | 16 |
| 5 | 20 | 20 | 56 | 100 | 16 |
| 6 | 24 | 24 | 80 | 144 | 16 |

**One-origin state matches `O(L·r)` exactly at every depth — not
`r^L`, confirming the theory's central claim decisively.** All-origins
state grows as a triangular sum (`r·ΣL_down_to_2`), genuinely `O(L²·r)`
but with roughly half the constant of the phase's own naive `L²·r²`-
style worst case, again because `q=1` drops the `r_i` factor.

**Honest additional finding, not asked for but relevant**: because
`N` was held fixed while `L` grew, naive (uncompressed) per-origin
cost stays flat at `16` while compressed cost grows from `8` to `24` —
**crossing over and becoming worse than naive by `L=5-6` at this fixed,
small width.** The compression's advantage is `N`-independent but
grows *linearly worse* with depth at fixed width — it wins decisively
as `N` grows (per B20's own measurement: 16.7x smaller at `n=100`)
and can lose at large `L`, small `N`. Both must be read together, not
one in isolation.

## 5. Part F — routing controls

**F1 (`q=1` Kronecker)**: the primary implementation above — fully
verified, exact, `O(r)`-per-hop.

**F3 (generic dense `B`)**: used only as a comparison baseline via
direct (uncompressed) propagation, matching B20's own naive-cascade
pattern. **A full multi-hop Hom-space reduction for dense routing
(the theorem's own `r_i·r_k` bound, not `q=1`'s further `r_k`
collapse) was not implemented in this pass** — B20's single-hop
correction achieves it for ONE boundary, but composing it correctly
across `L≥3` boundaries under generic (non-Kronecker) routing was not
re-derived here, an explicit scope limit consistent with the
instruction to stay focused. **F2 (bounded `q>1`) was not tested** —
same reason.

**What this means concretely**: this phase demonstrates the
temporal-module part of the theorem holds and is exploitable for
structured routing; it does NOT yet demonstrate the same for generic
dense routing at depth, though nothing found here contradicts the
theorem's own (weaker, `r_i·r_k`) prediction for that case either —
it is simply unverified past one hop.

## 6. Part G — coefficient complexity: the predicted remaining problem, confirmed on schedule

Exactly as the theory correction anticipated ("the remaining scaling
risk is feature/source coefficient complexity, not symbolic temporal
path count"), this phase's own implementation reproduces B20's
finding for a genuinely separate parameter family:

| parameter family | temporal dim `d_T` | coefficient dim `d_F` | total persistent state |
|---|---|---|---|
| lower routing param (`B_l`, `l<L-1`, `q=1`) | `r_upper` per hop | 1 (single fixed injection point) | `O(L·r)` — **small, verified** |
| top-layer routing param (`B_{L-1}`) | `r_{L-1}` | 1 (own copy, confined) | `O(r)` — **small, verified** |
| local `R_l` (own layer) | `r_l²` | 1 (local, no propagation) | `O(r_l²)` — **small, unavoidable (B19)** |
| **cross-layer propagation of `R_l`'s own params** | `r_upper` (per B19's `E=I`-style reduction, if isolated) | **`O(n_l)`** (inherits `h_prev_l`'s own multiplicity-rank, generically full) | **`O(n_l · r_l)` = `O(N_l)` per parameter — NOT small** |

**The temporal-module theorem (Parts A-E) does not, and was never
claimed to, resolve this last row.** It is a genuinely separate axis
(coefficient/multiplicity, not symbolic path count), confirmed exactly
where the correction said it would remain — R's own parameters inject
into every copy simultaneously via a forcing that depends on the
network's own generically-full-rank state, and no routing structure
tested here changes that (routing structure governs how SOURCES
propagate between layers; it does not touch what a layer's own core
parameters see LOCALLY, which is the network's own state regardless).

## 7. Part H — implementation cost (light, as instructed)

Not vectorized (explicitly out of scope for "mathematical exactness
first" per the phase's own Part H framing) — wall-clock is still
Python-scalar-loop-dominated:

| n1 | BPTT | prefix IC-RTRL | ratio |
|---|---|---|---|
| 4 | 0.7ms | 61ms | 89x |
| 12 | 0.7ms | 145ms | 196x |
| 32 | 1.4ms | 373ms | 270x |

Growing wall-clock with `n1` is **not** a failure of the state
compression (which stays fixed-size by construction, Part 4) — it
reflects the unavoidable fact that `B_l` genuinely has `O(n1·r)`
*parameters*, each needing at least one gradient output touched, and
this implementation touches each with its own Python-level loop
iteration rather than a vectorized batch op. Per-parameter state is
compressed; parameter *count* is not (nor should it be — that is
B19's own "unavoidable" axis, Part 5c of the earlier correction).

## 8. Verdict: **A, DEEP PREFIX THEORY CONFIRMED**

Checking against the four offered options:

- **A (deep prefix theory confirmed)**: **matches.** `L≥3` exact
  gradients agree with BPTT/naive RTRL to machine precision at every
  depth tested (2-5), and temporal state follows the predicted
  additive/polynomial prefix scaling (`O(L·r)` for one origin, even
  better than the theorem's own `O(L·r²)`) — not `r^L`. Proceed, per
  this option's own framing, to the real remaining problem: structured
  coefficient/routing scaling — which Part 6 shows is exactly where
  the difficulty now sits.
- **B (temporal passes, coefficients kill width scaling)**: partially
  true as a description of Part G's finding alone, but too strong as
  the PHASE verdict — coefficient cost is confirmed real for one
  specific parameter family (a layer's own core parameters), not
  established as killing scaling for the routing-parameter family this
  phase's own architecture is built around, which stays genuinely
  cheap and multiplicity-independent.
- **C (structured routing saves both)**: not established — this phase
  verified structured routing solves the TEMPORAL side; it did not
  test, and has no evidence, that structured routing also resolves
  Part G's coefficient problem for a layer's own dense-core parameters
  (a conceptually separate mechanism, as Part 6 argues).
- **D (prefix theorem fails)**: ruled out — every measurement matched
  the prediction or beat it.

No new production online-credit training rule deployed — verification-
grade only, matching this phase's own scope. No S5.

## 9. Commit hash

See the commit introducing this file.
