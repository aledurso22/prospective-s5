# Phase B11 — shared/private communication-subspace mechanism audit

Branch `S5-CCM-scale-validation` (current checkout). Theory/mechanism
audit only: **no new training algorithm, no S5, no new persistent
training arm.** Code: `credit_memory/b11_shared_private_communication.py`
(new, builds on B10/B10.1/B10.2). Artifact: `results/credit_memory/
b11/b11_summary.json`. 8 seeds for Parts A-C, 4 seeds for Part D's
null-model tests (cost-limited), single-seed for D2's T-sweep.

**Headline: the central hypothesis is only partially supported, and
the null-model tests (Part D) — the most direct causal-style probe —
came back genuinely negative for its strongest form.** Real
low-dimensional, task-coupled structure exists (confirmed exactly,
again, via the Hankel-analogue spectral identity), but (1) the
"shared" component needs a *majority*, not a minority, of raw U/V
variance to reliably preserve decisions, contradicting the sharpest
form of the hypothesis at the most diagnostic (small-r) end; and (2)
**destroying cross-interface temporal alignment (time-shift and
cross-seed nulls) did not destroy the sharp low-rank interaction at
all** — suggesting the effect may be a structural property of the
shared, fixed pole-magnitude architecture rather than a task-learned
coherent communication channel.

## Construction note

Following B10.1's `K_tc = A Sigma_c B^dagger`, the shared temporal
bases are `Psi := L_U B_r` (U-side) and `Phi := R_V^dagger{}^H A_r`
(V-side) — **the Hermitian, not plain, transpose of `R_V^dagger`** is
required for `Phi` to be a genuine projection basis (verified: an
initial plain-transpose version passed its own orthonormality check
but failed to reproduce `K_tc`-truncated reconstruction, `0.94`
relative error; the corrected Hermitian version was verified
algebraically and is used throughout). `U_shared := Psi Psi^dagger U`,
`V_shared := V0 Phi Phi^dagger` — projections of the *raw* `U`/`V0`
onto the coupling's own canonical temporal directions, not each side's
independent top-r SVD (which B10.1 already showed is a worse
construction).

# PART A — shared/private latent test

Median over 8 seeds, P-branch:

| `r` | `U` var. shared | `V` var. shared | `R_shared` winner | `R_private` winner | `R_shared` regret | `R_private` regret |
|---|---|---|---|---|---|---|
| 1 | 34.7% | 31.2% | **0.33** | **0.67** | 0.80 | 0.07 |
| 2 | 62.9% | 56.6% | 0.67 | 0.50 | 0.32 | 0.58 |
| **3** | **85.4%** | **72.4%** | **0.83** | **0.33** | **0.0** | 9.13 |
| 4 | 90.1% | 88.9% | 0.75 | 0.25 | 0.0 | 21.8 |

**The interesting prediction — a small shared fraction dominating
decisions while a large private fraction is dispensable — is not
what's observed.** At `r=1`, where the hypothesis's strongest form
would need to hold (only 31-35% of variance retained), **`R_private`
actually preserves winners *better* than `R_shared`** (0.67 vs. 0.33)
— exactly backwards from the prediction. A clean separation (shared
clearly dominant, `R_shared` regret `=0`, `R_private` regret `>9`)
only appears at `r=3`, by which point the "shared" component already
holds **85%/72%** of the raw variance — a majority, not a minority.
**The cross-term reconstruction identity holds exactly** (`VsUs+VsUp+
VpUs+VpUp = VU` to `~1e-15`, a trivial but necessary sanity check),
and the individual cross-term norms (e.g. at `r=1`: `\|VsUs\|~9899`,
`\|VsUp\|~4461`, `\|VpUs\|~1238`, `\|VpUp\|~3724`) show substantial
mutual cancellation — the "energy fraction from shared-shared" exceeds
100% at every `r` tested (1.2-1.6x), meaning shared-shared and the
other three terms partially cancel rather than adding independently.

# PART B — balanced credit / Hankel analogue

**B1 (spectral identity):** `sigma_i(VU)^2 = lambda_i(W_t W_e)` (with
`W_e = U U^dagger`, `W_t = V0^dagger V0`, both `TB x TB`) verified
directly via full eigendecomposition: **max relative error `2.25e-13`
across all 8 seeds** — exact, as required. This is the most literal,
most expensive check in this phase (full `1920x1920` eigendecompositions)
and it holds cleanly.

**B3 (truncation-criterion comparison):** already established in
B10.1's Part D and not re-litigated here — balanced/`K_tc`-mode
truncation there clearly dominated independent `U`-only/`V`-only
energy truncation at matched rank (e.g. `r=4`: `K_tc` `0.0075`
Frobenius error / perfect winner preservation vs. `U`-only `0.061`
error / `48.0` regret). B11 confirms the identity underlying that
result (B1 above) rather than re-running the comparison.

**B4 (energy vs. credit importance):** median Spearman(`U`'s own
singular values, loading on `K_tc`'s dominant mode) `= 0.40`;
median Spearman(`V`'s own singular values, loading on dominant mode)
`= 0.51`. **Both are moderate, not strong, and the per-seed values are
highly variable** (`U`: range `-0.14` to `0.83`; `V`: range `-0.03` to
`0.94`, with no consistent side dominating — an initial single-seed
spot check suggested a sharp `U`-vs-`V` asymmetry that did **not**
hold up in the 8-seed aggregate). **"High state energy != high credit
importance" is weakly-to-moderately supported on both sides** — energy
carries real but incomplete information about credit relevance; it is
not a reliable proxy on its own, on either side.

# PART C — temporal / cross-spectral coherence

**C1/C2:** cross-spectrum built from raw `Sa0`/`q1` per (trajectory,
batch) realization (Welch-style averaging, 32 realizations per seed).
Median **7.5 of 60** frequency bins carry non-trivial cross-spectral
energy (`>5%` of peak); at those strong frequencies, the cross-spectral
matrix is **rank-1** (median); the single dominant frequency bin holds
**38%** of total cross-spectral energy (median). **Coherence is
concentrated in both frequency and rank simultaneously** — a small
number of frequency bands, each essentially rank-1.

**C3 (Parseval check):** time-domain and frequency-domain (summed,
`1/T`-normalized) cross-correlation match to **`1.2e-15`** — the
spectral construction is exact.

**C4 (pole-frequency matching):** `spectral_match[j] = sum_omega W(omega)
|H_j(e^{i omega})|^2`. Median Spearman(match, `|rho|`) `= 0.27` (weak-
to-moderate). **Pool members show substantially higher mean spectral
match than non-members** (`25.2` vs. `5.3`, averaged over all modes/
seeds) — a real, meaningful separation — but the correlation with the
exact `|rho|` ranking itself is modest, not tight.

**C5 (K-pole approximation of the coherent mode):** greedy K-term
fits of the dominant coherence weighting `W(omega)` using candidates'
own `|H_j(omega)|^2` profiles **do not converge well**: relative error
stays at **0.95 (K=1) -> 0.92 (K=2) -> 0.91 (K=4)** — essentially flat,
not a meaningful improvement. **This specific greedy/least-squares
operationalization does not explain why `r_tc < K`** — it is reported
as a negative result for this particular test, not as evidence against
the `r_tc -> K` bridge itself (which B10.1/B10.2 already established
directly via pool-regret curves, a more decision-relevant measure than
fitting a frequency profile).

# PART D — null models and finite-size scaling

**D1 (null models) — the most decisive test in this phase, and it
came back negative for the hypothesis's strongest form.**

| condition | median effective rank (90% energy) |
|---|---|
| true data | 2.0 |
| **independent circular time-shift null** (destroys precise cross-interface temporal alignment, preserves each side's own marginal spectrum) | **2.0** |
| **cross-seed null** (pairs seed `i`'s `U` with seed `j != i`'s `V`/`q1`/`B1`) | **2.0** |

**Neither null model destroys the sharp rank-2 structure.** Both
predicted-to-be-decoherence-inducing manipulations leave the effective
rank of `VU` completely unchanged. This directly contradicts the
central prediction that would distinguish "a genuine coherent
communication signal riding on high-dimensional private bulk" from a
simpler explanation: **the low-rank structure appears to be a
structural property of the shared architecture (identical pole-
magnitude grid `u0 = linspace(0.90, 0.995, N)` across every seed and
every trajectory), not a fragile, task-specific temporal alignment
between a given model's own eligibility and teaching dynamics.**

**D2 (T-scaling):** effective rank stays at **2 (90% energy)** across
`T = 30, 60, 120`, with no clean bulk-shrinks / coherent-signal-stays
separation visible in the raw singular-value magnitudes at these small
`T` — the dominant singular value of `(1/T)VU` grows with `T` rather
than the bulk visibly separating out; **no scaling law is claimed**,
per the task's own instruction not to force one that disagrees with
the data. Rank *stability* across `T` is itself a real, if narrower,
finding.

**D3 (width scaling):** not evaluated in this pass, given the time
budget already consumed by D1/D2 and Parts A-C — an explicit scope
limitation, not a null result.

# Verdicts

## G1 — Shared/private communication hypothesis: **B, PARTIAL**

Real shared low-dimensional structure exists and cleanly separates
decision quality at `r=3` (`R_shared` regret `0` vs. `R_private`
regret `9.1`) — but the shared component needs a **majority** (85%/72%
of `U`/`V` variance) to achieve this, and at the smallest, most
diagnostic rank (`r=1`), **private reconstruction actually beats
shared** — the reverse of the hypothesis's sharpest prediction. Not A
(variance economy claim fails at exactly the rank where it matters
most); not C (real, clean decision-relevant structure does exist at
`r=3`).

## G2 — Balanced-credit/Hankel hypothesis: **B, PARTIAL**

The mathematical analogy is **exact** (`2.25e-13` spectral identity,
the strongest possible confirmation of the algebra) and the framework
correctly explains B10.1's own finding that `K_tc`-truncation beats
one-sided energy truncation. But the natural *energy-only* diagnostic
this framework suggests (B4: does high singular-value energy on either
side predict dominant-mode loading) is only moderately predictive
(Spearman `0.40`/`0.51`, highly variable across seeds) — the balanced/
Hankel characterization is correct and useful, but doesn't add
predictive power *beyond* directly computing `K_tc`/`SVD(VU)`, which
B10.1 already established as the thing to compute.

## G3 — Temporal-coherence/pole-matching hypothesis: **B, PARTIAL**

Coherent temporal structure is real and sharply concentrated (rank-1
at strong frequencies, `38%` of energy in one bin), and K-pool members
do show meaningfully higher spectral match than non-members on average
(`25.2` vs. `5.3`) — but the match-vs-`|rho|` correlation is only
moderate (`0.27`), and the explicit test of whether a small number of
physical pole filters can *reconstruct* a dominant coherent mode's
frequency profile **failed** (error flat at `~0.91-0.95` from `K=1` to
`K=4`). Coherent bands exist and pool selection leans toward them on
average, but the relationship is not tight or mechanistically
demonstrated at the level of frequency-profile reconstruction.

## G4 — Bulk+coherent statistical-mechanics hypothesis: **C, FAIL**

This is the most direct causal-style test in the phase, and it came
back negative: **neither the time-shift null nor the cross-seed null
destroyed the sharp rank-2 interaction structure** — the central,
falsifiable prediction of the bulk-plus-coherent-signal picture. The
data are more consistent with the low-rank structure being a property
of the shared, fixed architecture (identical pole-magnitude spectrum
across seeds and trajectories) than of a fragile, task-specific
temporal coherence that null-shuffling should have disrupted. `T`-
scaling showed rank stability but no clean signal/bulk separation
either way. Width-scaling (D3) was not evaluated, so this verdict
rests on D1's null-model results specifically, which are unambiguous
within the tested scope.

## Bottom line

Task-relevant credit is genuinely low-dimensional and the mathematics
connecting `K_tc`, `VU`, and the balanced-Gramian analogue is exact —
but **why** it is low-dimensional is *not* well explained by a
task-specific shared/private communication-subspace story in its
strongest form. The most parsimonious reading consistent with all of
B10-B11's evidence is closer to: **the shared pole-magnitude
architecture (not the specific task or specific temporal alignment
between eligibility and teaching signals) is doing most of the work of
compressing the interaction** — a structural rather than a learned-
communication explanation. This does not undermine the *practical*
result (periodic K-pool CCM, B9.3) or the *exact* mathematics (B10/
B10.1's identities) — it changes the recommended *interpretation* of
why the compression exists, away from a neuroscience-style
communication-subspace story and toward an architecture/spectrum-driven
one.

No new persistent training arm added. No S5 run performed.
