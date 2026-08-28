# Phase B15 — dense reduced-order credit realization: does it scale where K-pool does not?

Branch `S5-CCM-scale-validation` (current checkout). Theory/mechanism
audit only: **no new persistent training algorithm, no S5 benchmark
suite.** Code: `credit_memory/b15_dense_rom_theory.py` (new, builds on
B10/B14). Artifact: `results/credit_memory/b15/b15_summary.json`.
Widths `N_lower in {6,12,24,48}`, 2-3 seeds per width (cost-limited at
larger widths), `T=60, BATCH=8, N_CAL_TRAJ=4` matching the established
protocol; held-out test trajectories for Part F's rollout.

**Headline: D — NO SCALABLE LOW-DIMENSIONALITY.** The central
hypothesis (dense low-rank credit stays small and viable while sparse
pole-coordinate selection fails due to delocalization) is **not
supported** on either of its two load-bearing claims: (1) the
leverage/coherence-based explanation for K-pool's collapse does not
track `K_epsilon`'s actual behavior, and (2) the dense task-relevant
rank `r_grad` itself grows substantially with width — it does not stay
small. Dynamic (recurrent) realizability is additionally poor and
worsens with width. This branch should not be developed into a new
training algorithm.

## Part A/B — coordinate coherence, leverage, and `K_sub` vs width

Median over seeds, at matched rank `r`:

| `N` | `r` | coherence `mu` | `K_sub/M` (5%) | `r_grad(0.95)` | `K_epsilon/M` (5%) |
|---|---|---|---|---|---|
| 6 | 2 | 4.63 | 0.58 | 1-2 | 0.25 |
| 12 | 2 | 8.93 | 0.42 | 2-4 | 0.25-0.67 |
| 24 | 2 | 22.10 | 0.21 | 5 | **1.00** |
| 48 | 2 | 17.83 | 0.31 | 8-12 | 0.69-0.96 |

**The leverage-based coherence metric does not track `K_epsilon`'s
collapse.** If coordinate delocalization of the dominant (fixed-`r`)
subspace explained K-pool's failure, `K_sub/M` should climb toward 1
alongside `K_epsilon/M`. It does not — `K_sub/M` stays in a modest
`0.2-0.6` band across all widths, while `K_epsilon/M` climbs to `~1.0`.
**Coherence of a fixed small-`r` subspace is not the mechanism.**

## Part C/D — dense rank-`r` vs coordinate-`K`, and the minimum useful dense rank

At the toy's own `N=6` scale, dense rank-`r` truncation and coordinate-
`K` (QR/CUR) selection give **nearly identical** approximation quality
at matched degrees of freedom (e.g. `r=K=4`: Frobenius error `0.040`
dense vs `0.058` coordinate) — consistent with B10's own finding that
compression works comparably well by either route at this scale.

**But `r_grad` (minimum dense rank for `cos>=0.95` gradient fidelity)
itself grows substantially with width**: `1-2` (`N=6`) `-> 2-4` (`N=12`)
`-> 5` (`N=24`) `-> 8-12` (`N=48`). As a fraction of `M`, `r_grad/M`
stays in a roughly similar `~0.08-0.25` range across widths rather than
shrinking — **`r_grad` grows roughly in proportion to `M`, not
sub-linearly.** Per the task's own explicit instruction ("If not, stop
this branch"): **this is the first clear signal that the dense-ROM
hypothesis does not hold as stated** — the dense rank needed for
high-fidelity gradient reconstruction is not staying small as width
grows.

## Part E — `F`-invariance / Galerkin closure

Closure residual `\|\|(I-Phi_r Phi_r^dagger) F Phi_r\|\| / \|\|F Phi_r\|\|`
at the toy scale (`N=6`): **`0.13` at `r=1`, `0.47-0.48` at `r=2,4`** —
substantial, not negligible, and *worsening* with rank. Principal
angles between `span(F Phi_r)` and `span(Phi_r)` reach `40-64 deg` at
`r=2,4`. **The dominant task-credit subspace is not well `F`-invariant**
— naive Galerkin projection does not give a clean closed recurrence,
exactly the negative outcome Part E was designed to detect.

## Part F — actual reduced-order rollout (the critical gap)

Held-out test-trajectory rollout of `z_t = A_r z_{t-1} + b_r u_t`,
`x_hat_t = Phi_r z_t`, gradient-cosine vs. the exact full-bank value,
median over seeds:

| `N` | `r=1` | `r=2` | `r=4` |
|---|---|---|---|
| 6 | 0.658 | 0.851 | 0.962 |
| 12 | 0.818 | 0.715 | 0.752 |
| 24 | **-0.271** | **-0.019** | 0.265 |
| 48 | 0.072 | 0.271 | 0.180 |

At the toy's own scale (`N=6`), dynamic rollout is reasonable and
improves with rank, roughly tracking static quality. **At `N=24` and
`N=48`, dynamic rollout quality collapses — even going *negative*
(anti-correlated with the true gradient) at `N=24` for small `r`, and
staying poor (`<0.3`) even at `r=4`**, despite `r_grad(0.95)` at these
widths being `5-12` — i.e., even ranks that are still *below* the
rank needed for good *static* reconstruction give poor *dynamic*
rollout, and increasing `r` within the tested range does not reliably
fix it. **The gap between static projection quality and dynamic
rollout quality widens sharply with width** — precisely the failure
mode Part F was designed to surface.

## Part G — balanced/Petrov-Galerkin realization (scope note)

Given Part E's closure result and Part F's rollout collapse both point
toward poor dynamic realizability of the Galerkin reduction well before
reaching this part, **a full balanced-POD/Petrov-Galerkin pipeline
(separate trial/test bases with `Psi_r^dagger Phi_r = I`) was not
built as a separate implementation in this pass** — an explicit scope
reduction given the time already invested in the higher-priority
falsification tests (A/B, D, E, F) that the overall hypothesis
required. Qualitatively: a non-Galerkin (Petrov-Galerkin) test basis
constructed from the adjoint/teaching side could in principle reduce
closure error by allowing the projection and trial subspaces to
differ, but nothing in this phase's evidence suggests the *magnitude*
of `r_grad`'s growth (Part D, a rank problem, not a closure problem)
would be resolved by a better closure scheme — the rank itself is
already too large to be an obviously useful compression at bigger
widths, independent of how well any single fixed rank can be
dynamically realized.

## Part H — output/routing cost audit

Regardless of the state-propagation scheme, producing the deployed
gradient contraction `Ga[m] = 0.5 sum_j B1[j,m] sum_t conj(q1_t[j])
x_t[j] + c.c.` for a dense reduced state requires, for a basis `Phi_r`
(`M x r`): forming `Phi_r^dagger` contracted against the *per-mode*
routing/teaching structure (`B1[:,m]`, `q1`) — since `B1` and `q1`
vary by mode `m` and are **not** low-rank by construction in this toy
(no evidence from B10-B14 suggests `B1` or `q1`'s own structure is
compressible independent of the credit interaction itself), the
naive readout cost is **`O(r * M * N_lower)`** if `Phi_r^dagger`
must be re-contracted against `B1`/`q1` fresh per mode, or
**`O(r * N_lower)`** per step *after* an `O(r * M * N_lower)`
one-time precomputation of `Phi_r^dagger` applied to each mode's own
routing column — but that precomputation cost is the **same order**
as computing the full `M x N_lower` relevance matrix directly. **No
existing structure identified in this project (B1 has no established
low-rank property; `q1` is the naive online spatial error, also not
established as low-rank on its own) makes this readout cheaper than
what full calibration already costs.** This is reported plainly per
the task's own instruction not to hide it.

## Part I — end-to-end cost model

| method | state memory | state update | readout | total (this toy's regime) |
|---|---|---|---|---|
| full P/Q | `O(M N_lower)` | `O(M N_lower)` | `O(M N_lower)` | established baseline |
| periodic K-pool | `O(K N_lower)` | `O(K N_lower)` | `O(N_lower)` (K=selected) | wins only while `K/M` stays small — B14 showed it does not, past small widths |
| dense reduced (this phase) | `O(r N_lower)` | `O(r^2 N_lower)` or `O(r N_lower)` if `A_r` is diagonalizable | `>= O(r M N_lower)` one-time per calibration, per Part H | **never clearly cheaper** given `r_grad` itself grows with `M` and the readout cost floor matches full calibration |

**No width in the tested range (`N=6` to `48`) shows the dense reduced
realization winning** — at small `N` it offers no advantage over
already-cheap full calibration; at large `N`, `r_grad` has grown large
enough, and dynamic realizability degraded enough, that it offers no
advantage over (increasingly poor) K-pool either.

## Part J — matched Haar-random explanation of K-pool failure (revisited)

Given Part A/B's finding that the *fixed-rank* coherence metric does
not track `K_epsilon`, the more relevant framing (established already
in B14) is that the **amplitude-weighted rank itself** (not a fixed
small `r`'s coordinate coherence) grows with width and tracks matched
random-subspace geometry at every width tested. **This phase adds**:
even granting that growing rank, a coordinate-based (K-pool) selection
and a dense low-rank projection degrade at comparable rates once the
*effective* rank needed (`r_grad`) is accounted for — coordinate
sparsification is not uniquely disadvantaged relative to dense
reduction at matched degrees of freedom (Part C); both are limited by
the same underlying growth in required rank.

## Part K — verdict: **D, NO SCALABLE LOW-DIMENSIONALITY**

- The specific coordinate-delocalization mechanism proposed as the
  reason for K-pool's failure is **not supported** — `K_sub/M` (a
  direct measure of that mechanism) does not track `K_epsilon/M`'s
  collapse.
- The dense task-relevant rank `r_grad` **grows substantially with
  width**, roughly in proportion to `M` — it does not stay small. Per
  the task's own explicit stop condition, this alone is sufficient to
  not pursue a dense-ROM training algorithm from this evidence.
- Even where a rank is chosen that is adequate for *static*
  reconstruction, **dynamic (recurrent) rollout quality is poor and
  degrades further with width** (Part E/F) — a second, independent
  reason dense reduction does not currently offer a scalable path.
- The output/routing cost floor (Part H) means even a hypothetically
  perfect dense reduced state would not obviously reduce total cost
  below full calibration.

**This branch should not be developed into a new training algorithm.**
Combined with B14's finding that the K-pool compression benefit itself
erodes with width, **the cumulative evidence from B14-B15 is that
neither the sparse (K-pool) nor the dense (reduced-order) compression
strategy tested so far demonstrably scales beyond this toy's small
width regime.** The practical periodic K-pool CCM algorithm (B9.3)
remains valid and useful *at the scale it was validated on*; claims
about its scalability to substantially wider architectures are not
supported by B14 or B15's evidence and should not be assumed without
direct testing at the target scale.

No new persistent training arm added. No S5 benchmark suite run.
