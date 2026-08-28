# Phase B12 — structural-spectral theory audit: what actually causes r_tc?

Branch `S5-CCM-scale-validation` (current checkout). Theory/mechanism
audit only: **no new training algorithm, no S5, no new persistent
training arm.** Code: `credit_memory/b12_structural_spectral_theory.py`
(new). Artifact: `results/credit_memory/b12/b12_summary.json`. 8 seeds
for Parts B/C/D, 3 seeds for Part F (cost-limited), single-seed/
architecture-only for Part E.

**Headline: every mechanism tested in this phase failed to explain or
control r_tc.** Temporal whitening, PSD-matched independent
surrogates, pole radius/phase/spread ablations, and controlled task-
complexity scaling all left the effective rank at essentially the same
value (~1-3, median ~2) as the true, unperturbed system. Combined with
B11's null-model results (time-shift and cross-seed nulls also
preserved rank-2), **the rank-2 collapse in this toy is remarkably
robust to every specific mechanism tested — none of marginal spectral
concentration, the specific pole bank, or task temporal complexity (as
constructed here) is individually necessary for it.** The verdict is
**H-D, OTHER**: a real, replicated, decision-relevant phenomenon whose
specific cause this battery of tests did not identify.

## Scope note

Given the very large scope of this task and the time already invested
in Parts B-D-E-F (the most decisive, most directly falsifiable tests
per the task's own framing), **Part A's full marginal-PSD-product
formalism and Part G's spectral pole-utility score were not built as
separate dedicated analyses.** Part C's PSD-matched null (which is
built directly from each side's own marginal power spectrum) speaks
to Part A's central question directly; Part E's architecture-only
Gramian speaks to part of Part G's question. This is an explicit scope
reduction, not a silently dropped requirement.

## Part B — temporal whitening (the decisive test)

`U_whitened = L_U R_U^dagger` (flattens `U`'s own singular values to
1, keeps its temporal directions); `V_whitened` analogously for `V0`.
Median effective rank (90% energy) over 8 seeds:

| variant | median effective rank | median winner preserved | median pool regret |
|---|---|---|---|
| original | 2.0 | 1.0 | 0.0 |
| `U`-whitened | **2.0** | 1.0 | 0.0 |
| `V`-whitened | **2.0** | 0.33 | 0.80 |
| both whitened | **2.0** | 0.5 | 0.80 |

**The rank does not change under any whitening condition, including
full double-whitening.** Per the task's own stated falsification
criterion — "if rank ~2 survives double whitening, reject the simple
marginal-spectral explanation" — **the simple marginal-spectral
(energy-concentration) explanation is rejected.** A secondary, useful
finding: `U`-whitening alone costs *nothing* in decision quality
(winner preservation and regret both stay perfect), while `V`-
whitening *does* degrade decisions — `V`'s specific amplitude
weighting (not just its temporal directions) carries real,
non-redundant task information, even though removing it doesn't change
the *rank*.

## Part C — PSD-matched independent null

Fourier phase-randomization surrogate: the same random phase per
frequency is applied jointly across all channels of `Sa0` (or `q1`)
within a realization — preserving each side's own full multivariate
power spectrum exactly while destroying any true cross-relationship
between eligibility and teaching signals; 5 draws per seed.

| | median effective rank (90%) |
|---|---|
| true data | 2.0 |
| **PSD-matched independent null** | **1.5** |

Median top-2 energy fraction in the null: **96.8%** — matching the
true data's own sharp concentration almost exactly. **A synthetic,
statistically independent surrogate that only preserves each side's
marginal power spectrum reproduces the same sharp low-rank structure**
— marginal spectral shape (not true cross-coherence) is *sufficient*
to reproduce the phenomenon, consistent with, and further sharpening,
B11's cross-seed null finding.

## Part D — pole-architecture ablation

`U` is never filtered by poles (it is raw `Sa0`); only `V0`'s adjoint
filter depends on the pole set, so this ablation swaps in synthetic
poles for `V0`'s construction only, holding the real `Sa0`/`q1`/`B1`
fixed. Median effective rank (90%) over 8 seeds, all variants:

| variant | median effective rank | median Spearman vs. true `\|VU\|` |
|---|---|---|
| true poles | 2.0 | 1.00 |
| radius fixed, phase randomized | 2.0 | 0.65 |
| phase fixed, radius randomized | 2.0 | 1.00 |
| flat radius 0.90 / 0.95 / 0.99 | 2.0 / 2.0 / 1.5 | 0.99 / 1.00 / 0.99 |
| narrow / medium / broad magnitude spread (`[0.94,0.96]` / `[0.90,0.995]` / `[0.5,0.999]`) | 2.0 / 2.0 / 2.0 | 1.00 / 1.00 / 0.97 |

**None of phase randomization, radius randomization, flattening, or
spreading the pole timescales by an order of magnitude (`0.5` to
`0.999`) changes the effective rank at all.** Randomizing phase alone
*does* substantially change the actual entries (Spearman with the true
matrix drops to `0.65`) without changing the rank — confirming rank
and value-agreement are separate properties. **The specific pole bank
is not what controls r_tc.**

## Part E — finite-horizon analytic pole Gramian (architecture only)

`(P_T)_{ij} = d_i conj(d_j) [1-(lambda_i conj(lambda_j))^T] / [1 -
lambda_i conj(lambda_j)]` verified against direct summation: **relative
error `2.7e-15`** — exact, as required.

**Critically, this architecture-only controllability Gramian (no real
task signal at all, `d = 1`) is *not* especially low-rank**: effective
rank **5 (90%) / 6 (95%) / 9 (99%)** of a maximum 12 — nearly full
rank. **This directly contradicts "architecture creates the low-rank
envelope"** — the pole bank alone, absent any task-driven signal,
imposes almost no rank constraint. Combined with Part D (swapping
poles doesn't change the observed rank either), this rules out a
simple architectural explanation from two independent directions.

## Part F — task-complexity scaling

Controlled multi-delay task: `r_task` independent white-noise input
channels (`M_IN` set to `r_task`), each delayed by a distinct amount
(`5, 10, 15, ...`), summed to a scalar target — increasing the task's
own nominal temporal degrees of freedom without touching `N` or the
pole bank at all. Median over 3 seeds:

| `r_task` | median effective rank (90%) | median `K_epsilon` (5% regret) |
|---|---|---|
| 1 | 2.0 | 3.0 |
| 2 | 2.0 | 2.0 |
| 4 | 2.0 | 3.0 |
| 8 | 2.0 | 3.0 |

**Increasing task temporal complexity 8-fold (1 to 8 independent delay
channels) does not move `r_tc` at all**, and `K_epsilon` shows no
systematic trend (noise-level variation, `2.0-3.0`). In this specific
construction, **task temporal complexity (as measured by independent
delay-channel count) is also not what controls `r_tc`.** This should
not be over-read as ruling out *all* forms of task-complexity
dependence — a construction that varies task *frequency content*
rather than delay count, or that scales `N` jointly with task
complexity, was not tested — but the specific, natural construction
tried here shows no effect.

## Part G — spectral pole-utility formula

Not built as a separate dedicated test in this phase (see Scope
note); Part D's architecture-only Gramian (high rank, `5-9` of `12`)
already shows that a purely-architectural score `s_spec[j] approx
integral S_u(omega) |H_j(omega)|^2 d(omega)` built from marginal
spectra alone, without cross-phase information, would not be expected
to concentrate sharply — consistent with, but not a direct
replication of, this part's intended test.

## Answer to the four crucial falsification questions

1. **Does temporal whitening remove the rank collapse?** No (Part B).
2. **Do independent PSD-matched surrogates reproduce it?** Yes (Part C).
3. **Does controlled task complexity move `r_tc`?** No, in the tested
   construction (Part F).
4. **Do controlled pole-timescale changes move `r_tc`?** No (Part D),
   and the architecture-only Gramian is not itself low-rank (Part E).

## Part H — mechanistic verdict: **H-D, OTHER**

Not H-A (marginal-spectral): whitening should have flattened the
spectrum if energy concentration were the cause; it did not. The PSD-
matched null's success (Part C) is *consistent with* marginal
structure being sufficient in some sense, but Part B's whitening
result shows it can't be about *energy* concentration specifically —
the two results together are best read as "some property of the
marginal spectral *shape*, not its amplitude weighting, matters," which
is a real but incomplete piece of H-A, not confirmation of it in full.

Not H-B (architectural): pole radius/phase/spread ablations left the
rank unchanged, and the architecture-only Gramian is nearly full rank
on its own — the specific pole bank does not control `r_tc`.

Not (cleanly) H-C (task x architecture overlap) either, at least not
via the tested task-complexity axis: varying task temporal complexity
(delay-channel count) had no effect.

**H-D is the honest conclusion**: the rank-2 collapse survived every
specific mechanism tested (whitening, PSD nulls reproducing rather than
destroying it, pole ablations, task-complexity scaling), and B11's
earlier null models (time-shift, cross-seed) already showed the same
robustness. **This is a real, highly replicated, decision-relevant
phenomenon (confirmed at machine precision via the underlying algebra
throughout B10-B12) whose specific causal mechanism remains
unidentified after this battery of tests.** A plausible (not verified)
candidate for future work: the collapse may be closer to a structural/
dimensionality artifact of this toy's specific small scale (`N=6`) and
the general smoothness of both `Sa0` and `q1` as outputs of similar
low-pass SSM recurrences, rather than any of the specific mechanisms
tested here — this is speculation flagged as such, not a finding.

## Part I — algorithmic implications

**Given the H-D verdict, this phase does not provide support for
replacing full calibration with a cheap marginal-spectral or
architecture-only estimate.** None of the tested cheap proxies (pole-
bank-only Gramian, marginal PSD product, whitened/flattened spectra)
reliably tracked or controlled the actual low-rank structure — the
one thing that *did* reproduce it (the PSD-matched independent null)
is not itself cheaper to compute than the real calibration, since it
still requires the real marginal spectra of the real `Sa0`/`q1`
signals. **The existing empirical, `|rho|`-guided calibration approach
(periodic K-pool CCM, B9.3) remains the best-supported path** — no
"estimate a few structural modes, then select poles to cover them"
shortcut is justified by this phase's evidence. This conclusion is
reported honestly rather than forcing a positive spin toward the
future-algorithm idea the task described as "the dream result."

No new persistent training arm added. No S5 run performed.
