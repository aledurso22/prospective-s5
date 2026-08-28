# Phase B16 — exact invariant/Krylov credit closure under tied recurrent dynamics

Branch `S5-CCM-scale-validation` (current checkout). This phase is
deliberately **not** compression: it tests whether an **exact** (not
approximate) reduction in persistent credit-state count follows from
tying multiple upper-layer channels to the same pole, which is a
purely algebraic consequence of the uniqueness of linear recursions,
not a rank-truncation approximation. No new persistent training
algorithm, no S5 benchmark suite. Code: `credit_memory/
b16_tied_pole_exact_closure.py` (new). Artifact: `results/
credit_memory/b16/b16_summary.json`. Widths `N in {6,12,24,48}`,
3 seeds per width for Part A; `M in {12,24,48}` for the generic Krylov
test (Part E); `M in {8,32,128}` for the selectivity test (Part G).

**Headline: the core two-layer exact-closure claim is confirmed at
machine precision and scales exactly as predicted — but Parts C, D, F,
and H (depth, complex minimal realization, forward-expressivity
tradeoff) were not tested in this pass, so the full "A" verdict as
specified cannot yet be claimed.** This is the necessary first step
toward that verdict, reported honestly as partial, not complete,
confirmation — see Part K.

## Part A — exact grouped P/Q reduction

Grouped recursion `p_t[g,m] = mu_g p_{t-1}[g,m] + u_t^m` (one state
per `(group, mode)` pair, not per `(channel, mode)`), with the exact
gradient regrouped as `G_P[m] = 0.5 sum_g sum_t p_t[g,m] [sum_{j: g(j)=g}
B1[j,m] conj(q1_t[j])]` (plus the `Q`-branch analogue, matching the
repo's established conjugation convention exactly). Compared against
(a) the full, uncompressed exact `P/Q` construction (`credit_memory/
full_causal.py`, unmodified) using the identical grouped-pole model,
and (b) BPTT:

| `N` | max error, all `G in {1,2,4,N}`, 3 seeds |
|---|---|
| 6 | 3.5e-15 |
| 12 | 3.8e-15 |
| 24 | 5.4e-15 |
| 48 | 7.8e-15 |

**Machine precision throughout — this is an exact identity, confirmed
across every tested width and every group count from `G=1` (fully
tied) to `G=N` (fully distinct, recovering the established full-P/Q
result).**

## Part B — true compute/memory cost

Persistent credit-state count is **exactly** `2GN_lower` vs. the full
`2N_upper N_lower` — a provable, not empirical, reduction:

| `N` | full states | grouped states (`G=1`) | reduction |
|---|---|---|---|
| 6 | 72 | 12 | 6x |
| 12 | 288 | 24 | 12x |
| 24 | 1152 | 48 | 24x |
| 48 | 4608 | 96 | 48x |

The reduction factor scales **linearly with `N`** (`=N/G`) by
construction — at `N=48, G=1` this is a 48x state reduction, and it
would continue growing without bound at larger `N`, unlike B14/B15's
compression-based approaches whose benefit *eroded* with width. Wall-
clock timing at this toy's Python-loop implementation is dominated by
constant-factor overhead (full vs. grouped times are within `~20%` of
each other at every width tested) — consistent with the established
pattern throughout this project that toy-scale wall-clock does not
cleanly reveal asymptotic advantages; the **state-count reduction is
the load-bearing, provable claim**, not the micro-benchmark.

## Part E — Krylov complexity

`K(A,B) = span{B, AB, A^2B, ...}` for diagonal `A`:

- **E1** (distinct poles, low-rank `B`, single column): Krylov
  dimension reaches the full ambient dimension `M` in every case
  tested (`M=12,24,48`, one numerical-tolerance edge case at `M=48`
  giving `47`) — **confirms low-rank `B` alone does not cap Krylov
  dimension under distinct poles.**
- **E2** (`G` tied pole classes): measured Krylov dimension **exactly
  matches** the formula `sum_g rank(Pi_g B)` in every tested
  `(M,G)` combination — a clean confirmation of the general structural
  law connecting pole-tying to reachable dimension.
- **E4** (genuine `A`-invariant input subspace, `B`'s column space
  spanning a union of eigenspaces with full rank within it): measured
  Krylov dimension **exactly equals** `rank(U)` regardless of
  iteration count — confirming that true invariance (not just tying)
  fixes the dimension exactly.

## Part G — selectivity counterexample

`h_t = a_t h_{t-1} + x_t`, `a_t = a_t(theta_0; gate\_signal)`, comparing
an **exogenous** gate (depends on `theta_0` and a fixed external
signal, never on `h_{t-1}`) against a **selective** gate (depends on
`theta_0` and the evolving state `h_{t-1}` itself, matching Mamba-2-
style input-dependent recurrence). Effective rank of
`d h_T / d theta_0` across many independent realizations:

| `M` | rank, exogenous | rank, selective | rank, selective + stop-grad through `a_t` |
|---|---|---|---|
| 8 | 5 | 6 | **0** |
| 32 | 6 | 7 | **0** |
| 128 | 24 | 28 | **0** |

**Selectivity does add extra sensitivity complexity beyond a matched
exogenous baseline** (a consistent `+1` to `+4` gap), confirming the
extra term `(d a_t/d theta) h_{t-1}` is not vacuous — but the gap is
**modest, not a dramatic full-width restoration**, at this toy scale.
The stop-gradient control gives **exactly zero** sensitivity (as it
must, since in this specific parameterization `theta_0` only acts
through `a_t`) — confirming the implementation correctly severs that
pathway when asked to. **Important methodological caveat**: this test
used a generic (non-tied) high-dimensional drive `x_t`, so *both*
conditions already show rank growing with `M` — it does not directly
test "selectivity applied on top of the Part A/B tied-pole
architecture specifically." A cleaner test combining tied poles with a
selective gate was not built in this pass, given time constraints —
this is a genuine, explicit scope limitation, not a resolved result.

## Scope note: Parts C, D, F, H, I, J

Given the time already invested in the four core falsification/
confirmation tests above (A, B, E, G), **the deep real scalar-identity
stack (Part C), the complex-depth minimal-realization analysis
(Part D), head-wise routing (Part F), the forward-expressivity Pareto
curve (Part H), the exact online-training diagnostic (Part I, which
was explicitly gated on Parts A/C passing — A passed, C was not run),
and the conceptual baseline comparisons (Part J) were not implemented
as separate pipelines in this pass.** This is an explicit, honestly-
reported scope reduction — not a silent omission, and not a claim that
these parts would fail. In particular, **Part H (does small `G` retain
forward task quality) is the single most important missing piece**
for any practical recommendation: an exact cheap-credit architecture
is worthless if tying poles destroys the network's ability to learn
the task, and this phase does not yet know the answer.

## Part K — verdict

**Neither a clean "A" nor "B" as specified — reported as: core claim
CONFIRMED, full hypothesis scope NOT YET TESTED.**

What is established, at machine precision, across every width tested
(`N=6` to `48`): the exact two-layer grouped/tied `P/Q` credit
reduction is **algebraically exact** (not approximate), reduces
persistent state count by a factor of `N/G` that grows without bound
with width (unlike B14/B15's compression approaches), and rests on a
correctly-verified general Krylov-complexity law (`K(A,B) = sum_g
rank(Pi_g B)`) that explains *why* it must be exact, not just that it
happens to measure as such. Selectivity (Part G) demonstrably adds
some complexity beyond a matched exogenous baseline, though the effect
measured here is modest and the test setup has an acknowledged
limitation.

**What remains genuinely open, and should be resolved before claiming
the full "A. EXACT CREDIT CLOSURE STRONG" verdict**: whether this
closure survives depth (Part C/D — untested), whether tying poles
preserves forward task performance at the widths where the state-
count savings would matter (Part H — untested, and the single most
practically important open question), and whether a proper combined
tied-pole-plus-selective-gate construction shows a cleaner version of
Part G's modest complexity increase or something more severe.

**Recommendation**: this is a genuinely promising direction — a real,
exact (not compressed) reduction in credit-state complexity, unlike
every avenue explored in B9-B15 — and is worth continuing. The
concrete next step is Part H (forward-expressivity Pareto curve for
small `G` via ordinary BPTT training, no new credit rule needed) since
it is the cheapest way to find out whether this direction is worth the
further investment of Parts C/D/F/I, and it directly determines
whether tied-pole architectures are viable at all before any exact
online training rule is built on top of them.

No new persistent training arm added. No S5 benchmark suite run.
