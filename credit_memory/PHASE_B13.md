# Phase B13 — common temporal support audit: does relative orientation, not amplitude, explain r_tc?

Branch `S5-CCM-scale-validation` (current checkout). Theory/mechanism
audit only: **no new training algorithm, no S5, no new persistent
training arm.** Code: `credit_memory/b13_common_temporal_support.py`
(new). Artifact: `results/credit_memory/b13/b13_summary.json`. 8 seeds
for Parts A/B, 3 seeds for Parts C/F (cost-limited).

**Headline: all four of the task's own "most important falsifications"
were run, and none produced a decisive, systematic change in `r_tc`.**
Random Haar rotation of `V`'s temporal basis (destroying its alignment
with `U` while exactly preserving singular values), frequency-bin
permutation, band-stop removal of the dominant common frequency,
careful injection of a new shared spectral component, genuinely
distinct-frequency task complexity (not delays), and independent
teaching-dimension variation **all left the median effective rank at
~2**, with only seed-level noise comparable to what the unperturbed
data itself already shows. **Verdict: E, STILL UNEXPLAINED.**

## Setup: the double-whitened identity

Per B12, `U_white = L_U R_U^dagger`, `V_white = L_V R_V^dagger`, so
`V_white U_white = L_V (R_V^dagger L_U) R_U^dagger = L_V C_tc
R_U^dagger`, and since `L_V`/`R_U^dagger` are isometric,
`singular_values(V_white U_white) = singular_values(C_tc)` exactly —
`C_tc = R_V^dagger L_U` is the pure temporal-subspace-overlap matrix.
This phase asks whether that overlap comes from a genuine *common
temporal support* (particular shared directions/frequencies) as
opposed to something not yet identified.

## Part A — dominant modes of `C_tc`

Mapped `C_tc`'s own singular vectors back through `R_V^dagger`
(teaching-side) and `L_U` (eligibility-side) to physical time, FFT'd,
and compared: median dominant frequency of mode 0 is `0.10`
cycles/sample, mode 1 is `0.083` — **close but not identical**, and
only **1 of 8 seeds** shows the two dominant modes at the *exact* same
frequency (the signature of a real oscillatory mode's quadrature
pair). **The rank-2 structure looks more like two distinct, narrow
temporal atoms at nearby-but-different frequencies than a single real
oscillatory mode expressed as two quadratures** — though this
characterization varies somewhat across seeds and should not be
over-interpreted as a precise, stable "law."

## Part B — support-destroying nulls (median over 8 seeds)

| test | construction | median effective rank (90%) |
|---|---|---|
| true (unperturbed) | — | 2.0 |
| **B1: random Haar rotation of `V`'s temporal basis** | fresh, statistically independent orthonormal temporal directions assigned `V`'s own singular values (verified exact preservation, `2.2e-15`) | **2.0** |
| **B2: frequency-bin permutation** | `q1`'s FFT bins randomly permuted before inverse-transform (preserves total power and the bin-magnitude multiset) | **2.0** |
| **B3: band-stop** (`U`, `V`, or both) | dominant common frequency band (and its mirror) zeroed via FFT | **2.0** (all three variants) |
| **B4: inject new shared band** | a small-amplitude sinusoid at frequency `20/60` (previously weak support) added jointly to `Sa0` and `q1` | **2.0** |

**B1 is the single cleanest test in this phase** — it exactly preserves
`V`'s own singular value spectrum while replacing its temporal
directions with a *statistically independent, Haar-random* basis
unrelated to `U`'s. If common temporal support/orientation were the
cause, this should have driven the rank back up toward 6 (or at least
shown a large, consistent increase). **It did not** — median rank
stays at 2.0, with the same seed-to-seed spread (`1-4`) the
unperturbed data itself already shows. B2-B4 show slightly more
per-seed noise (e.g., B2 ranges `1-4` across seeds) but **no
consistent, systematic shift** in either direction. Amplitude
sensitivity was checked for B4 specifically: too-large an injection
(`scale=0.1-1.0`) makes the injected mode dominate everything
(artificially *lowering* apparent rank to 1, the opposite of the
naive prediction); only a carefully-tuned small amplitude
(`~0.01x` typical signal scale) gives a modest, inconsistent uptick
in a subset of seeds — not a robust, reproducible effect.

## Part C — genuine spectral-complexity task (not delays)

Per the task's own insight — a pure delay only multiplies by
`exp(-i omega tau)`, changing phase but not spectral *support* — this
constructs `r_spectral` independent oscillatory input channels at
genuinely disjoint frequency bins (`3, 11, 19, 27, 5, 13, 21, 29`
cycles per `T=60`), summed to a scalar target, with `M_IN` set to
`r_spectral`. Median over 3 seeds:

| `r_spectral` | median effective rank (90%) | median `K_epsilon` (5%) |
|---|---|---|
| 1 | 1.0 | 2.0 |
| 2 | 2.0 | 2.0 |
| 4 | 1.0 | 3.0 |
| 8 | 2.0 | 3.0 |

**No monotonic trend** — an 8-fold increase in genuinely disjoint
spectral task complexity does not move `r_tc` (fluctuates `1-2`,
consistent with seed noise), extending B12's delay-based null result
to a strictly stronger, spectral-support-based task-complexity test.

## Part F — teaching-dimension (`d_teach`) variation

`spatial_q` is exactly linear in the residual `r`
(`q[L-1]=conj(c)*r`, then a linear recursion), so `d_teach`
independent residual channels (same input `x`/eligibility, but
genuinely distinct target frequencies, `3, 11, 19, 27, ...`) combine
as `q1_combined = sum_k spatial_q(r_k)` — exactly reproducing what a
genuine `d_teach`-dimensional combined output/error signal would give,
without modifying the toy's core readout code. Median over 3 seeds:

| `d_teach` | median effective rank (90%) | median `K_epsilon` (5%) |
|---|---|---|
| 1 | 1.0 | 2.0 |
| 2 | 1.0 | 2.0 |
| 4 | 1.0 | 2.0 |
| 8 | 1.0 | 3.0 |

**Completely flat** — an 8-fold increase in independent teaching/error
temporal channels leaves the effective rank at exactly 1.0 throughout.
`K_epsilon` shows only a marginal uptick at `d_teach=8`. **The
teaching-bottleneck hypothesis is not supported** by this construction.

## Scope note: Parts D, E, G

Given the time already invested in Parts A-C/F (the tests the task
itself flagged as the four most important falsifications), **Parts D
(time-bandwidth/Slepian analysis), E (raw teaching-trajectory-matrix
factorization), and G (structural rank model `F_SV^dagger F_SU`) were
not built as separate dedicated pipelines in this pass** — an explicit
scope reduction, not a silently dropped requirement. Qualitatively:
Part A's frequency estimates (dominant modes near `0.08-0.10`
cycles/sample, i.e. periods of `10-13` samples against `T=60`) are
consistent with *some* structural time-bandwidth constraint being
present, but this was not formally connected to a `2WT`-style
prediction here.

## Part H — verdict: **E, STILL UNEXPLAINED**

All four of the task's own flagged "most important falsifications"
were run:

1. Random temporal rotation preserving singular values (B1): **no
   systematic change.**
2. Moving spectral support without changing marginal energy statistics
   (B2, B3): **no systematic change.**
3. Genuinely different frequency bands, not additional delays (C):
   **no effect**, extending B12's delay-based null.
4. Independently varying teaching/error dimension (F): **no effect**,
   completely flat at `er90=1.0` across an 8-fold range.

**None of A (common-support), B (teaching-bottleneck), or C
(time-bandwidth, not directly tested but qualitatively not implicated
by the flat spectral/teaching-dimension results) is cleanly
supported.** The rank-2(ish) collapse has now survived: task-specific
temporal-alignment nulls (B11: time-shift, cross-seed), amplitude/
whitening controls (B12), pole-architecture ablations (B12), delay-
based and now genuinely spectral-support-based task-complexity scaling
(B12, B13), and direct temporal-basis randomization plus targeted
spectral-support manipulation (B13). **This is now a remarkably
over-determined negative result** — the phenomenon is real, exactly
reproducible, and decision-relevant, but its causal mechanism has not
been identified across three full phases (B11, B12, B13) of
increasingly targeted, increasingly strong falsification attempts.

## Honest interpretation

Given the sheer breadth of interventions that failed to move `r_tc`,
the most defensible working hypothesis going forward is that the
effective rank observed here may be closer to a **generic,
near-unavoidable consequence of this toy's very small scale**
(`N=6`, `2N=12` candidates, `N_lower=6` modes) than a discoverable
mechanistic law — i.e., with so few degrees of freedom on both sides,
*most* reasonable signal pairs sharing broadly similar (low-pass,
narrow-timescale) smoothness properties may produce a similarly
concentrated cross-correlation structure, largely independent of the
specific manipulations tried. This is offered as a candidate
explanation for *why the search has been so unproductive*, not as a
verified finding — testing it directly (e.g. via B12's deferred
width-scaling experiment, D3, at meaningfully larger `N`) is the most
promising next step, ahead of further toy-scale mechanism hunting.

## Recommendation

Given three consecutive phases (B11, B12, B13) of extensive,
increasingly targeted falsification testing have not identified the
mechanism, **further toy-scale mechanistic search is not
recommended as the next step.** The practical result (periodic K-pool
CCM, B9.3) and the exact mathematics (B10/B10.1's identities,
independent of *why* the rank is low) remain fully valid and
unaffected by this open question. If the mechanism question is to be
pursued further, a width/scale-varying experiment (B12's deferred D3)
is the most likely candidate to distinguish "generic small-scale
artifact" from a genuine, scale-invariant law — this is a suggestion
for possible future work, not a commitment to a new phase.

No new persistent training arm added. No S5 run performed.
