# Phase B9 — state-lifecycle audit + oracle marginal-utility diagnostic

Branch `credit-memory-repair`. Requested **before** any new
predictor/resurrection mechanism and before any S5 run. Parts 1-2 are a
pure code-reading audit of the existing toy machinery (no new code
required, no training-algorithm changes). Part 3-4 add one new,
read-only diagnostic script, `credit_memory/b9_oracle_utility_audit.py`,
reusing B3/B4's exact static-benchmark protocol (8 seeds, `N=6, T=60,
BATCH=8`, `N_CAL_TRAJ=4`, imported from `phase_b2bc_hankel_truncation`).
Artifact: `results/credit_memory/b9_oracle_utility_summary.json`.

**No prediction-correction/resurrection mechanism is implemented here.
No training algorithm is modified. No S5 code is touched.**

## Part 1 — state lifecycle

Three logically distinct pieces of state exist in the current
implementation, and they have three different lifecycles. Conflating
them is what makes "dormant-state resurrection" look necessary; kept
separate, it isn't.

**(a) The deployed channel filter state** (`credit_memory/
b4_deploy.py::b4_layer0_gradient`, used by every training arm from B5
onward). `prevA`/`prevB` are allocated as fresh zeros **inside every
call** to `b4_layer0_gradient`, i.e. reset to zero at the start of
*every single optimizer step*, for *every* lower mode `m`, regardless of
which upper channel is currently selected or was selected on the
previous step. There is no persistent per-channel deployed state at
all — each step recomputes the selected channel's filtered readout from
scratch over that step's own batch. Consequence: **a newly-selected
channel is not "cold" relative to a channel that was already
selected** — both start every step from the identical zero state. There
is nothing to resurrect at the deployment layer, by construction.

**(b) The one-shot calibration estimator** (`credit_memory/
streaming.py::StreamingRelevance`, driven by `run_windowed_calibration`
in B4, and by `causal_prefix_selection` in B6). A *fresh* estimator
(`self.x`, `self.rho`, both zero) is created at the start of each
calibration event (the one-time bootstrap before training, and — for
arm T1 — every periodic recalibration call). **Within one calibration
event**, however, `self.x` is *not* reset between the `N_CAL_TRAJ`
independent trajectories fed into it — `run_windowed_calibration` /
`causal_prefix_selection` loop over trajectories and call `.step()`
trajectory after trajectory into the same instance without a reset in
between. This is a genuine, previously-unflagged inconsistency: each
calibration trajectory is drawn with the model's own recurrent state
starting at 0, but the *relevance estimator's* filter state carries the
tail of trajectory `k` into trajectory `k+1`. For the slowest calibrated
poles (`|a| up to 0.995`, e-folding time `~200` steps) against a
per-trajectory length `T=60`, this leakage is not negligible — a slow
channel's estimated `rho_j` is measurably contaminated by the previous
trajectory's tail. It does not affect the deployed gradient (which never
uses this estimator's persistent state directly, only its final
`top_channel()` decision), but it does affect the input to channel
*selection*, and should be fixed (reset `self.x` at each trajectory
boundary) before this calibration estimator is trusted at longer poles
or longer horizons than the current toy config.

**(c) The reactive tracking scalar** (`credit_memory/
b6_prospective_tracking.py`, arms T2/T3 only). `rho_cur[m]` /
`rho_prev[m]` are full `2N`-dimensional complex vectors that persist
continuously across *all* optimizer steps of a training run by design —
that is the entire point of "reactive" tracking. Critically, though, the
*per-step observation* feeding that EMA/predict-correct update
(`single_batch_observation`, wrapping `credit_memory/
lagcorr.py::per_coordinate_contribution` unmodified) allocates its own
filter state fresh, from zero, on every call — identical in spirit to
(a). So the *raw* per-step observation `r_obs` is stateless and
symmetric across all `2N` candidates; only the *running summary
statistic* `rho_cur[m]` (not a dynamical filter state, just a tracked
scalar) persists, and it is updated identically for every candidate,
selected or not, every step.

**Can the selected channel change inside a sequence?** No. Selection
(`hysteretic_select` in B6, or the one-shot `argmax` in B4/B5) is
computed once per optimizer step from that step's completed batch, and
is held fixed for the entirety of that step's gradient computation. It
can change at most once per optimizer step, never mid-sequence.

**What happens to a newly-selected channel's state on switch?** Nothing
special — per (a), *every* channel, whether just-selected or long-held,
starts each step's deployed computation from zero. Per (c), *every*
candidate's tracked `rho_cur[m][j]` has been updated every step all
along, selected or not — candidates are never actually dormant in the
tracking signal. **Conclusion: dormant-state resurrection is not
required by the current architecture.** The one real lifecycle issue
found is unrelated to resurrection — it is the cross-trajectory leakage
in the one-shot calibration estimator, (b) above.

## Part 2 — cost of adaptive relevance scoring

**Are all candidate P/Q states propagated to score relevance?** Yes,
unconditionally. `per_coordinate_contribution`'s internal filter array
has shape `(T, BATCH, 2N)` — literally every one of the `2N` candidates
per lower mode `m` is filtered every step, whether or not it is
currently selected. `StreamingRelevance.x` is `(BATCH, 2N)`, same story.
There is no per-candidate gating of the *propagation* — only the final
*readout/selection* is sparse.

**Cost, separated:**

| | persistent state (per lower mode `m`) | compute per step |
|---|---|---|
| **calibration / continuous adaptation** (T1/T2/T3 tracking) | `O(2N)` complex (all candidates' filter or tracked `rho`) | `O(2N)`: filter update for all `2N` candidates + building `c_t` from the full `q1` (`O(N)`) and `B1[:,m]` (`O(N)`) |
| **deployment** (the actual gradient used for the weight update, any arm) | `O(1)` complex (one selected channel) | `O(1)`: one channel's filter update, one scalar of `q1`, one entry of `B1` |

Summed over all `N` lower modes: **adaptation is `O(N·2N) = O(2N^2)`
state and compute per step; deployment is `O(N)`.** The ratio is a
factor of `N` — for the toy's `N=6` that's a 6x overhead; for a
production-scale S5 layer (`N` in the tens to low hundreds) the same
factor would make *continuous* tracking (T2/T3-style, paid every
optimizer step) far more expensive relative to the `O(1)`-per-mode
deployed gradient than it is in the toy regime.

**T1 vs T2/T3, amortized:** T1 only pays the `O(2N^2)` cost during its
periodic recalibration windows (`N_CAL_TRAJ · T = 240` steps' worth of
computation, incurred once every `t1_period=100` training steps in the
current B6 config) — an amortized overhead of `~2.4x` one training
step's cost, paid periodically. T2/T3 pay the (smaller, `O(2N)` per
mode, `O(2N^2)` total) tracking cost on *every* training step,
continuously, for the life of the run. Both are strictly on top of, and
separate from, the always-`O(N)` deployed-gradient cost that every arm
(including frozen T0) pays regardless.

## Part 3 — oracle marginal-utility diagnostic

New script `credit_memory/b9_oracle_utility_audit.py`. For each of the 8
B3/B4 seeds and each of the `N=6` lower modes, using the pooled
calibration data (same protocol as B3/B4/B8):

- `gamma_j[m]` = exact per-candidate contribution of upper channel `j`
  (`j=0..2N-1`) to the full causal gradient for mode `m`
  (`credit_memory/lagcorr.py::per_coordinate_contribution`, unmodified;
  `sum_j gamma_j[m] == G_causal[m]` exactly, as already established
  through B3/B4/B7).
- `S = {top_j}`, the existing rank-1 selection from the calibrated
  `|rho_j|` ranking (`StreamingRelevance`, unmodified).
- `U_j(S) = |G-G_S|^2 - |G-G_S-gamma_j|^2` computed directly, and
  independently via the identity `U_j(S) = 2 Re[conj(G-G_S)*gamma_j] -
  |gamma_j|^2`.

**Identity check: PASSES at machine precision** — max absolute
discrepancy across all 48 `(seed, mode)` rows and all `2N=12` candidates
is `2.9e-11`. The algebra (and its implementation here) is correct.

**Correlation with the existing `|rho_j|` ranking: weak.**

| comparison | median Spearman | top-1 agreement |
|---|---|---|
| `|rho_j|` vs `U_j(S)`, all `2N` candidates | **0.052** | **10.4%** |
| same, excluding the already-selected `top_j` (removes the self-referential degeneracy of scoring `top_j` against a residual that already excludes it) | **0.268** | — |

**Interpretation.** `|rho_j|` ranks candidates by the raw magnitude of
their *individual* contribution to the full sum. `U_j(S)` ranks them by
how much *replacing/complementing* the current selection would reduce
the squared error against the true total `G` — which depends on the
*phase alignment* of `gamma_j` with the residual `G-G_S`, not just its
magnitude. These are genuinely different quantities, and the data show
it: even restricted to non-selected candidates (removing the degenerate
"utility of a channel already in `S`" term), the ranking agreement is
weak-to-moderate (`0.268`), not strong. **The current `|rho_j|`
selector is a reasonable but demonstrably imperfect proxy for
marginal/oracle utility.** This does not contradict B3/B4/B7's
established result that `|rho_j|`-based rank-1 selection reproduces
BPTT training closely in practice (the oracle here is a *local, one-step
counterfactual*, not a measure of downstream training trajectory
similarity) — but it is a real, previously unmeasured gap between the
heuristic and the thing it is supposed to approximate, and it is the
natural quantity a future prediction-correction/resurrection mechanism
should be justified against.

## Part 4 — additional cheap diagnostics logged

All computed causally, without any dormant state, alongside Part 3, and
written into the same JSON per `(seed, mode)` row:

- `abs_lambda_per_candidate` — `|lambda_j|`, the pole magnitude of each
  of the `2N` candidates (architecture-only, no data needed).
- `abs_q_per_candidate` — RMS magnitude of the naive top-layer error
  `|q_j|` for each candidate's underlying upper mode over the
  calibration data.
- `B_row_norm_per_candidate` — `||B_{j,:}||`, the routing-weight row
  norm for each candidate's upper mode (how broadly that mode is used
  across all lower modes).
- `lower_eligibility_energy_E_m` — `sum_t |Sa0_t[m]|^2`, pooled over the
  calibration batch, per lower mode.
- `lag1_autocorr_Sa0` — lag-1 autocorrelation of the (real/imag-stacked)
  eligibility signal `Sa0[:,:,m]`, a cheap persistence diagnostic.

All are available at the same cost as the existing `O(N)`
deployment-time bookkeeping; none require materializing a full P/Q
teacher or storing a trajectory.

## Bottom line

1. **Dormant-state resurrection is not needed** — no per-candidate
   dynamical state persists across a channel switch in either the
   deployed gradient or the tracking mechanism; the only real
   lifecycle defect found is cross-trajectory leakage inside the
   one-shot calibration estimator (Part 1(b)), unrelated to
   resurrection.
2. **Adaptive scoring costs `O(2N^2)` vs. `O(N)` for deployment** — a
   factor-of-`N` overhead that is modest at the toy's `N=6` but would
   matter at S5 scale if paid continuously (T2/T3-style) rather than
   periodically (T1-style).
3. **The oracle marginal-utility diagnostic is implemented, verified to
   machine precision, and reveals that `|rho_j|` is only weakly
   correlated with true marginal utility** — a concrete, quantified
   target for any future selector redesign, without yet building one.
