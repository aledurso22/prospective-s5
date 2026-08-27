# Phase B9.1 — selector/predictability diagnostic

Branch `S5-CCM-scale-validation` (current checkout). Diagnostic only,
per the task's explicit instructions: **no prediction-correction or
resurrection mechanism implemented, no training algorithm changed, no
S5 run.** Code: `credit_memory/streaming.py` (leak fix),
`credit_memory/b5_train.py` / `b6_prospective_tracking.py` (same fix
applied at their duplicated calibration call sites),
`credit_memory/phase_b4c_streaming_rank1.py` (sanity-check reference
fixed to match), `credit_memory/b9_1_selector_predictability.py` (new,
Parts 2-5). Artifact:
`results/credit_memory/b9_1_selector_predictability_summary.json`.
Same 8 seeds, `N=6, T=60, BATCH=8`, `N_CAL_TRAJ=4, N_TEST_TRAJ=4` static
protocol as B3/B4/B8/B9.

**Headline: the B9 low-correlation result was an artifact of B9's own
degenerate `S={top_j}` definition. Under the correct `S=empty`
(rank-1-from-scratch) formulation, `|rho_j|` finds the oracle-best
channel 73% of the time outright, 96% within its top-5, and the
resulting held-out gradient cosine loss from not always finding it is
negligible (0.9654 vs. an oracle ceiling of 0.9684). Poor rank
correlation does NOT cause meaningful gradient loss here. Separately,
none of five cheap (no-candidate-propagation) predictors come close to
matching `|rho_j|`'s power — the O(2N) propagation currently paid for
relevance scoring is doing real, currently-irreplaceable work.**

## Part 1 — StreamingRelevance leakage fix

`StreamingRelevance.reset_filter()` added (zeros `self.x`, leaves the
accumulated `self.rho` untouched — accumulating evidence across
independent calibration trajectories is the intended semantics).
Called at every trajectory boundary in the three places that build a
per-mode `StreamingRelevance` calibration loop:
`streaming.py::run_windowed_calibration`,
`b5_train.py::causal_calibration_selector`,
`b6_prospective_tracking.py::causal_prefix_selection`. (B3's own
original R1 derivation, `phase_b3c_relevance.py`, uses a different,
one-shot batch-pooled code path, not this class, and is out of scope
for this fix — noted for the record, not altered.)

**Quantified impact: none, in this toy config.** Rerunning B4C
(`phase_b4c_streaming_rank1.py`, the deployable streaming rank-1
result) after the fix gives **median cos = 0.9654**, bit-for-bit
identical per-seed to the pre-fix number, and the internal
streaming-vs-batch sanity check (also corrected to use a fair,
per-trajectory-reset reference) now passes at `1.75e-15` instead of
failing at `5.68e-1`. Rerunning B8's static benchmark gives **S0 median
cos = 0.9257** (previously reported 0.926 — unchanged within rounding).
For every one of the 8 seeds x 6 modes in this calibration draw, the
selected channel is identical with or without the leak — the toy's
`N_CAL_TRAJ=4, T=60` calibration window is evidently not (yet) long
enough relative to the slowest pole's decay time for the leakage to
flip a selection, even though Part 1 of B9 correctly identified the
leak as real and present. **The fix is correct and now in place; it
does not change any previously reported rank-1 number at this scale,**
but should be re-checked if `T` or the pole spectrum changes (e.g. at
S5 scale).

## Part 2 — extended oracle-utility metrics

B9's own script defined the active set as `S={top_j from rho}` and
asked for the marginal utility of *adding a second candidate* on top of
that — which makes the utility of `top_j` itself (already in `S`) a
self-referential, near-meaningless quantity, and buries the actually
interesting question (did the ORIGINAL choice of `top_j` make sense?)
inside a same-order noise floor. Here `S` is instead the **empty set**:
`U_j(S=∅) = |G|^2 - |G-gamma_j|^2` is the utility of picking candidate
`j` as the sole rank-1 channel from a cold start — exactly the decision
the selector actually makes. The identity `U_j = 2Re[conj(G)*gamma_j] -
|gamma_j|^2` re-verifies to `1.46e-11` under this definition. `gamma_j`
itself is also now computed leak-free (summed per-calibration-row, each
row's own filter state starting at zero, rather than B9's pooled/
non-reset construction).

| metric | value |
|---|---|
| median Spearman(`|rho_j|`, `U_j`) | **0.199** (was 0.052 under B9's `S`) |
| top-1 / top-3 / top-5 oracle hit rate of the `|rho|` ranking | **72.9% / 85.4% / 95.8%** |
| median selection regret, `U_oracle - U_rho` | **0.0000** |
| median selection regret, `U_oracle - U_random` | 42.16 |
| median `U` gap, oracle vs. median candidate | 73.38 (the oracle utility landscape is sharply peaked — most candidates are far worse than the best) |
| held-out gradient cos vs BPTT, using `j_rho` | **0.9654** |
| held-out gradient cos vs BPTT, using `j_oracle` instead | **0.9684** |
| held-out gradient cos vs BPTT, online baseline | 0.6278 |

**Interpretation.** Spearman is a weak/misleading summary here — the
oracle utility landscape is dominated by one or two clearly-best
candidates and a long tail of much worse ones (median gap 73 vs. best
`U` values in the hundreds), so a rank-correlation coefficient computed
over all 12 candidates is dominated by noise in the tail ordering,
which does not matter operationally. The metrics that matter for a
rank-1 selector — hit rate, regret, and actual downstream gradient
fidelity — all say the same thing: **`|rho_j|` already nearly always
finds the right channel, and on the rare occasions it doesn't, the
gradient-fidelity cost is small (0.003 median cos).** This directly
answers the motivating question: *poor rank correlation, as measured by
Spearman, does not translate into meaningful gradient loss.*

## Part 3 — cheap dormant-mode predictors

Five scores computed using **only** quantities that do not require
propagating any candidate's own `x_j`/`P_j`/`Q_j` filter state:

| score | formula | median Spearman | top-1/3/5 | median regret |
|---|---|---|---|---|
| A. random | uniform draw | +0.059 | 0.17 / 0.27 / 0.40 | 56.37 |
| B. `\|lambda_j\|` | pole magnitude | -0.064 | 0.02 / 0.21 / 0.42 | 60.47 |
| C. `1/(1-\|lambda_j\|^2)` | architecture controllability | -0.064 | 0.02 / 0.21 / 0.42 | 60.47 |
| D. `\|q_j\|^2 \|\|B_{j,:}\|\|^2` | error-weighted routing | +0.035 | 0.12 / 0.54 / 0.62 | 81.85 |
| E. D `x E_m / (1-\|lambda_j\|^2)` | D + eligibility energy + controllability | -0.042 | 0.15 / 0.42 / 0.62 | 75.74 |
| **`\|rho_j\|` (current, for comparison)** | full O(2N) propagation | **0.199** | **0.73 / 0.85 / 0.96** | **0.00** |

**None of the five cheap scores are competitive with `|rho_j|`** — all
sit at or below the random baseline on Spearman, and even the best
(D/E on top-3/5 recall, `0.54-0.62`) fall far short of `|rho_j|`'s
`0.85-0.96`, with regrets `70-80x` larger than `|rho_j|`'s essentially
zero median regret. B and C are exactly architecture-only
(`|lambda_j|` doesn't even vary across candidates sharing the same `j
mod N`, since P- and Q-branch poles have identical magnitude) and
perform at chance — confirming that pole magnitude alone carries no
information about which of the `2N` candidates will actually matter for
a *given* task/mode. D/E, which fold in the (per-upper-mode, not
per-candidate-propagated) naive error magnitude and routing weight, do
slightly better than chance on top-k recall but are still not close to
useful.

**Lag-corrected filter-energy score.** The logged lag-1 autocorrelation
`ac1` is a single scalar **shared identically across all `2N`
candidates for a fixed lower mode `m`** (it is a property of the raw
driving signal `Sa0[:,:,m]`, computed before any candidate's own
`lambda_j`-filter is applied). Multiplying every candidate's score by
the same shared scalar cannot change their relative order — this is an
algebraic fact, not an empirical one, so no numerical test was run.
**The existing shared lag-1 statistic is therefore insufficient to
build a genuinely discriminating lag-corrected score.** What would be
required is a statistic that varies *per candidate* — e.g., the
lag-`k` correlation of the driving signal evaluated at each candidate's
own characteristic phase/period, or equivalently a per-`j` projection
of the eligibility signal onto that candidate's own pole. Either is, by
construction, exactly as expensive as propagating that candidate's own
filter (the `O(2N)` cost this diagnostic is trying to avoid) — so no
such statistic was implemented, per the task's explicit "don't add
expensive per-`(j,m)` statistics yet."

## Part 4 — complexity audit

| score | asymptotic cost | measured (8 seeds x 6 modes, this toy config) |
|---|---|---|
| `\|rho_j\|` (current) | `O(2N)` per mode per calibration step -> `O(2N x T x N_CAL_TRAJ)` per mode, `O(2N^2 x T x N_CAL_TRAJ)` total | 0.040 s |
| B/C (architecture-only) | `O(2N)` **once**, reused for every mode (no `T`/`N_CAL_TRAJ`/`m` dependence at all) | (included below) |
| D/E (error/routing-weighted) | `O(N)` reads of already-computed `q1`/`B1` **once per seed**, no extra `T`-scan | (included below) |
| all cheap scores combined | | 0.026 s |

Measured ratio at this toy scale is only `1.5x` — unsurprising, since
Python-loop overhead dominates both at `T=60, N_CAL_TRAJ=4`. The real
asymptotic gap (`O(2N^2 T \cdot N_{cal})` vs. effectively `O(N)`,
independent of `T` and `N_{cal}` entirely) would widen sharply at
production scale (larger `T`, longer calibration windows, larger `N`)
— this toy measurement should not be read as "cheap scores aren't much
faster in practice," only as "at this scale, constant-factor overhead
swamps the asymptotic difference." Given Part 3's result, the cost
question is currently moot: none of the cheap scores are accurate
enough to be worth deploying regardless of how much cheaper they are.

## Part 5 — shared upper-pole active set across lower modes (report only)

**Mathematically compatible.** The exact decomposition indexes `j` over
the same `2N`-candidate set for every lower mode `m`; nothing requires
the selected `j` to vary with `m`. A single shared `j*` used for every
mode is a well-formed (if more restrictive) special case of the
existing per-mode selection — **not implemented**, but its effect was
measured using only already-computed quantities (aggregate score
`sum_m |rho_m[j]|^2`, picking the single best `j*` per seed, then
reusing the existing `deploy_selected_channel` unmodified with that one
channel for every mode).

**Cost benefit (if implemented):** deployment would need only *one*
persistent filter state per batch element instead of `N` (one per
lower mode) — an `N`-fold reduction in deployed state/compute,
`O(1)` instead of `O(N)` per training step.

**Accuracy cost, measured:** median held-out cos drops from **0.9654**
(per-mode-optimal, current approach) to **0.9111** (shared `j*`) — a
real, non-trivial `0.054` fidelity loss. Unlike Part 2/3's finding that
`|rho_j|`'s occasional sub-optimal per-mode pick barely costs anything,
forcing *one* channel to serve *every* mode costs meaningfully more,
because different lower modes route through genuinely different
upper-mode structure (`B1[:,m]` differs by column) and a single
channel cannot represent all of them well simultaneously. **Not
recommended as a drop-in replacement without further justification** —
the `N`-fold deployment-cost saving is real, but at `N=6` it is not
worth a `0.054` fidelity loss; whether this tradeoff changes at S5
scale (larger `N`, where the `O(N)` deployment cost is more likely to
matter, and where the routing structure may or may not concentrate more
sharply on one shared channel) was not investigated here.

## Bottom line / recommendation

**Cheap architecture/error-magnitude features do not contain enough
information to predict oracle marginal utility.** All five candidates
tested (random, `|lambda_j|`, controllability, error-weighted routing,
and their eligibility-energy-augmented combination) perform at or only
marginally above chance, `70-80x` worse in regret than the existing
`|rho_j|` statistic. The existing shared lag-1 statistic cannot help
because it is identical across candidates within a mode by
construction; a genuinely discriminating lag statistic would cost as
much as the `O(2N)` propagation it's meant to replace.

At the same time, **the existing `|rho_j|` selector is not the weak
link the B9 Spearman number suggested** — under a properly-posed
(empty-baseline) oracle comparison it finds the right channel most of
the time and costs almost nothing in gradient fidelity when it doesn't.
**No selector redesign is currently motivated by this diagnostic.** If
a cheaper score is wanted for cost reasons at S5 scale, this diagnostic
does not supply one — any future attempt should assume it needs
candidate-specific (not shared/architecture-only) information and
budget accordingly, rather than expecting a free lunch from `\lambda_j`,
routing weights, or the currently-logged shared statistics.
