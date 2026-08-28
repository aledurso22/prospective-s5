# Phase B22.1 — reconciliation + constructive closure

Branch `S5-CCM-scale-validation`. Small, focused phase: resolve the
direct contradiction between B22's own "A-alone" measurement (`r²`)
and the theory's prediction (`2r-1`) before any further integration.
Code: `credit_memory/b22_1_reconciliation.py` (new). No S5.

**Headline: full reconciliation. B22's r² was correct for the object
it was actually measuring — hidden-state sensitivity `dx_t/dA`, an
exact match confirmed directly (both give rank 16 at r=4). The true
externally-observable output sensitivity `dy_t/dA` has rank exactly
`2r-1`, confirmed at r=2,4,8,16 with an unambiguous singular-value gap
(12+ orders of magnitude at every width). The gauge accounting closes
exactly: `r² − (r−1)² = 2r−1`. The constructive two-filter (V,W)
recurrence matches BPTT to machine precision at every depth tested,
after a real bug — an off-by-one indexing mismatch in my own
comparison harness, not the theorem — was caught and fixed.**

## 1. Part A — the true Markov-parameter Jacobian ranks

For `H_n = C·Aⁿ·B`, stacking `dH_n/dA[j,k]` (A-only, `r²` generators)
and `dH_n/dA, dH_n/dB, dH_n/dC` (joint, `r²+2r` generators) across
enough `n` to saturate rank:

| r | A-only rank | predicted `2r-1` | joint rank | predicted `2r` |
|---|---|---|---|---|
| 2 | 3 | 3 | 4 | 4 |
| 4 | 7 | 7 | 8 | 8 |
| 8 | 15 | 15 | 16 | 16 |
| 16 | 31 | 31 | 32 | 32 |

**Exact at every width.** At r=4, the singular-value spectrum near the
cutoff is `[0.161, 0.0091, 1.6e-15, 7.8e-16, 3.2e-16]` — the gap from
signal to numerical noise is 12+ orders of magnitude, completely
unambiguous. (r=16 initially showed a spurious `29/29` under the
default fast-decaying random matrix and tight time-horizon margin —
the same numerical-resolution issue diagnosed in B22 itself, where
signal decays smoothly into the double-precision floor with no clean
gap; resolved identically, by using a slower-decaying test matrix and
a more generous margin, confirming `31/32` cleanly — not a structural
issue, the same lesson re-applied.)

## 2. Part B — locating exactly what B22 measured

| object | rank at r=4 |
|---|---|
| old B22 "A-only" object (full `(a,b)`-indexed, independent) | **16** |
| B1: hidden-state sensitivity `dx_t/dA` | **16** |
| B2: external output sensitivity `dy_t/dA` | **7** |
| B4: nonzero-fixed-state sensitivity | **7** |

**Exact match, unambiguous**: B22's `d_credit_A_alone` computed
`H^(k)[a,b] = C·Aᵃ·E_k·Aᵇ·B`, treating `a` and `b` as independent
indices — this is precisely the hidden-state sensitivity `dx_t/dA`
(the sensitivity of an *internal* state reached `a` steps after a
perturbation that itself occurred `b` steps into a transient), not the
external output sensitivity. The true Markov-parameter Jacobian
`dH_n/dA` is the anti-diagonal SUM of that same object (`a+b=n-1`,
fixing the *total* elapsed time rather than allowing `a,b` to vary
independently) — a linear projection that collapses `r²` down to
`2r-1`. **The old result was correct for the object it was measuring;
it was measuring the wrong object for "what a future external teaching
signal can distinguish."** B4 landing at the same rank as B2 (7, not
16) additionally confirms that even holding an arbitrary *nonzero*
current state fixed (rather than the zero-state impulse response)
does not reintroduce the extra `r²-(2r-1)` directions — those are
gauge, not state-dependent.

## 3. Part C — gauge null-direction accounting closes exactly

- **Joint gauge exactness**: `δA=[X,A], δB=XB, δC=-CX` for random `X`
  gives max deviation of `δ(C·Aᵏ·B)` across `k=0..3r-1`: **`6.7e-16`**
  (machine precision).
- **Residual gauge dimension** (`X` with `XB=0, CX=0`, r=4): measured
  **9**, predicted `(r-1)²=9`. Exact.
- **Residual-gauge directions kill the A-only response too**:
  `δA=[X,A]` for a residual-gauge `X` gives Markov-Jacobian deviation
  **`2.8e-16`** (machine precision) — confirming these ARE genuine
  null directions for the A-only object specifically, not just for the
  full joint one.
- **The arithmetic closes**: `r² − (r−1)² = 2r−1` — `16 − 9 = 7` ✓,
  matching Part A's measured A-only rank exactly.

## 4. Part D/E — the constructive two-filter recurrence, and a bug found honestly

Implemented the `(V,W)` two-filter eligibility exactly as specified
(`v_t = u_t - Σaₖv_{t-k}`, `y_t = Σbₖv_{t-k}`, `w_t = y_t - Σaₖw_{t-k}`,
claimed `dy_t/db_k=v_{t-k}`, `dy_t/da_k=-w_{t-k}`), compared against a
companion-form state-space BPTT reference:

**First attempt failed** (errors `0.1`-`2.7`, nowhere near machine
precision) — investigated rather than accepted, per the phase's own
"do not assume either side is correct" instruction. Direct comparison
of raw output sequences (not yet gradients) showed the two
constructions' `y` sequences were **identical, staggered by exactly
one timestep** — a simple indexing mismatch in my own companion-form
reference (`y[t]=C·x[t]` vs. the two-filter convention's genuine
one-step delay, `y[t]` depending only on *past* `v`'s). Fixed by
delaying the state-space output by one step and re-deriving the
adjoint recursion for that shifted relationship (verified independently
against finite differences first: `2.5e-9`/`5.3e-10`, matching FD's
own precision floor, before trusting it as the reference). After the
fix:

| r | max err `grad_a` | max err `grad_b` |
|---|---|---|
| 2 | 1.1e-16 | 1.1e-16 |
| 3 | 6.9e-17 | 1.1e-16 |
| 4 | 2.8e-17 | 1.1e-16 |
| 6 | 7.6e-15 | 2.2e-15 |

**Machine precision at every depth tested.** The two-filter
construction is correct as specified; the earlier apparent failure was
entirely in the verification harness, not the theorem — worth stating
plainly since the phase explicitly asked not to assume which side was
wrong.

## 5. Part F — minimality (McMillan degree)

| r | Hankel rank | predicted |
|---|---|---|
| 2 | 2 | 2 |
| 4 | 4 | 4 |
| 8 | 8 | 8 |

Exact at every width — the `2r` recurrence is not merely sufficient,
it is generically minimal (the Hankel rank, the classical minimality
certificate, equals `r`, i.e. the transfer function itself has
McMillan degree `r`, and the `(a,b)` coefficient count `2r` matches
the joint tangent-realization rank from Part A exactly).

## 6. Verdict: **A, FULL RECONCILIATION**

Checking against the four offered options:

- **A (full reconciliation)**: **confirmed on every count.** External
  ranks are exactly `2r` (joint) and `2r-1` (A-only) at r=2,4,8,16. The
  two-filter recurrence matches exact gradients to machine precision
  at r=2,3,4,6. The old `r²` result is traced to a precisely identified
  object — hidden-state sensitivity `dx_t/dA` — confirmed by an exact
  numerical match (16=16), not merely a plausible guess.
- **B (theory rank fails)**: ruled out — every predicted rank matched
  exactly, with unambiguous singular-value gaps.
- **C (static rank passes but recurrence fails)**: **not the final
  outcome, but genuinely almost was** — the first implementation
  attempt looked exactly like this verdict (correct ranks, failing
  recurrence) until the discrepancy was traced to a test-harness bug
  rather than the construction. Worth flagging: an early, unverified
  report from this phase would have wrongly concluded C.
- **D (recurrence passes but only under zero-state queries)**: not
  observed — Part B's B4 check (nonzero fixed state) gave the same
  rank as the zero-state B2 check, and the two-filter construction
  itself operates causally on a genuine running trajectory (not a
  zero-state assumption), matching BPTT throughout.

## 7. What this licenses for B23 (not attempted here, per the phase's own gate)

Per Verdict A's own next-step framing: B23 should integrate gauge-fixed
functional temporal coordinates, genuinely low-dimensional temporal
interfaces, and B21's deep prefix propagation. This phase deliberately
stops short of that — no B21 modification, no low-rank deep interface,
no MIMO gauge quotient, no realistic-task benchmarking, no wall-clock
optimization, no routing redesign, exactly as instructed.

No new production online-credit training rule deployed. No S5 run.

## 8. Commit hash

See the commit introducing this file.
