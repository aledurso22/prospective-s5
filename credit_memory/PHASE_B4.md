# Phase B4 — deployable rank-1 credit identification, no exact teacher

Branch `credit-memory-repair`. Implementation + falsification only; no
new theory branch, prospective rule, or optimizer modification. L=2
only, per scope. Code: `credit_memory/streaming.py` (the deployable
estimator), `credit_memory/phase_b4{a,c,d}_*.py`. Artifacts:
`results/credit_memory/phase_b4{a,c,d}_*_summary.json`. Same N=6, T=60,
BATCH=8, 8 seeds, 4 calibration / 4 test trajectories as B2/B3 (imported
directly).

**Headline: Case A, with a Case B caveat for continuous adaptation, and
a Case-D-shaped depth caveat from bookkeeping.** A genuinely streaming,
no-teacher, no-BPTT estimator (`credit_memory/streaming.py`) reproduces
B3's R1 result essentially exactly — **median cos 0.9257** vs B3 R1's
0.926 — confirming a real, deployable L=2 rank-1 temporal-credit rule
exists. Continuously (EMA) adapting the same statistic, rather than
freezing after a fixed calibration window, is noticeably less reliable
on the already-hard seeds (occasional bistable channel-flipping). The
selection procedure itself (not the deployed rank-1 state) would scale
exponentially with depth if applied unchanged beyond L=2, per Phase A's
own channel-count derivation — flagged explicitly, not glossed over.

## B4A — exact identity check

Per natural pole channel `lambda_j` (`j=0..2N-1`), `rho_state[j] =
sum_t conj(c_t[j]) x_t[j]` (direct pole filter) vs `rho_lag[j] =
sum_k lambda_j^k (sum_t conj(c_t[j]) u_{t-k})` (independent lag-sum
re-derivation): **30 (seed,mode) checks, all `rel_err < 8.3e-14`** (min
`1.1e-15`). Convention note: implemented with `conj(c_t)`, matching the
Hermitian pairing verified throughout Phase A/B2/B3 (not literally
`c_t^T` as the handoff's schematic wrote it) — stated explicitly, not a
silent change.

Frequency-domain agreement (B3's R3 statistic) is markedly *worse* here
(median relative disagreement `0.90`) than in B3 (`0.41`), because this
check uses a single un-pooled trajectory (`T=60`) rather than B3's
4-trajectory-pooled calibration window — less data relative to this
toy's slow poles (`|a1[j]|` up to `0.995`) means more circular-DFT
leakage. Expected, reported, not a bug (the time-domain identity, which
is the actual convention check, is exact).

## B4B — the streaming estimator

`credit_memory/streaming.py`. Per lower mode `m`, per candidate channel
`j` (`j=0..2N-1`): two complex scalars — `x_j` (`x_j <- lambda_j x_j +
u_t`, the channel's own causal filter) and `rho_j` (the running
cross-statistic, `rho_j <- rho_j + conj(c_t[j]) x_j` windowed, or
`rho_j <- (1-gamma) rho_j + gamma conj(c_t[j]) x_j` for the continuous
EMA variant). **No `r_k` array, no stored trajectory, no P/Q
reconstruction** — this is literally the minimal sufficient statistic
per the task's own "direct pole-filter is sufficient" guidance.

**Bug caught before any B4C/D number was trusted**: an early draft
pooled the batch dimension into a single scalar drive *before* filtering
(`u_sum = sum(u_t)`, then `x <- lambda x + u_sum`). Filtering *is*
linear, so `sum_b filter(u_b) == filter(sum_b u_b)` holds exactly — but
the cross term `sum_b conj(c_t[b]) x_t[b]` is **bilinear** in
per-batch-element pairs that both vary independently, so pooling `u`
before the multiply silently paired every batch element's `c_t[b]` with
the *wrong* (summed) state. Fixed by keeping `x` per batch element,
`shape (BATCH, 2N)`, and only summing the already-paired cross term.

**Memory**: `O(2N)` complex per lower mode during calibration/selection
(state `x_j` for all `2N` candidates, since which one wins isn't known
yet), `O(1)` complex per stream once a single channel is selected and
frozen. **Compute**: `O(2N)` per step during calibration (one pole
multiply-add and one cross-multiply-accumulate per candidate channel);
`O(1)` per step once frozen (see B4E for what "frozen" actually needs to
read from `q1`).

## B4C — reproducing B3's rank-1 result, streaming, no teacher

Sanity check first: the streaming estimator's frozen `rho` (computed one
timestep at a time, no vectorized array reduction) matches B3's
batch-computed `g_p` exactly — `rel_err = 1.3e-15` (seed 0, mode 0; same
identity as B4A, different code path). Then: select `argmax|rho|` per
mode after the calibration window, freeze, evaluate on the same 4
held-out test trajectories as B3.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **median** |
|---|---|---|---|---|---|---|---|---|---|
| cos | 0.770 | 0.987 | 0.992 | 0.846 | 0.825 | 0.878 | 0.987 | 0.926 | **0.926** |

**Median cos 0.9257** — matches B3 R1's `0.926` to 3 decimal places
(the tiny residual difference is float-summation-order noise between
the two independent code paths, not a real discrepancy). C0 online
baseline on this same test set: `0.628`. Comparison table:

| method | median cos | uses exact teacher? | uses BPTT? |
|---|---|---|---|
| C0 online | 0.628 | no | no |
| **B4C streaming (this phase)** | **0.926** | **no** | **no** |
| B3 R1 (batch) | 0.926 | no | no |
| B3 R3 (frequency) | 0.940 | no | no |
| B3B exact-teacher rank-1 (optimized) | 0.992 | yes | no |

**Removing access to the exact teacher changes essentially nothing**
relative to R1/R3 (which also never used it) — as expected, since R1 was
already teacher-free. What B4C adds is proof that the *streaming
implementation itself* (the thing that would actually run online) loses
no accuracy relative to the batch computation. The real gap is between
*any* closed-form/single-statistic method (`~0.93`) and the
*optimized* teacher-aware diagnostic (B3B, `0.992`) — a `~0.07`
median-cosine gap that a single fixed statistic does not close.

**Primary gate `median cos >= 0.90`: PASS.**

## B4D — fully online (continuous EMA) adaptation

Same architectures, replayed as one long stream (40 trajectories, 2400
steps), `rho` updated continuously via EMA (no freeze), snapshotted
every 4 trajectories against the fixed B3/B4C test set. 4 "hard" seeds
(0, 3, 4, 5 — all `<0.90` median in B4C) plus 2 "easy" seeds (1, 6) for
contrast; `gamma in {0.005, 0.02, 0.08}`.

| | median final cos | range |
|---|---|---|
| hard seeds (0,3,4,5), all gammas | 0.826 | [0.533, 0.981] |
| easy seeds (1,6), all gammas | 0.979 | [0.974, 0.992] |

| gamma | median final cos | median tail variance (last 5 snapshots) |
|---|---|---|
| 0.005 (slow) | 0.879 | 0.00046 |
| 0.02 | 0.889 | 0.00155 |
| 0.08 (fast) | **0.969** | 0.00056 |

**Faster forgetting (larger gamma) does better on median final cos**,
somewhat counter-intuitively — a slow EMA (`gamma=0.005`, effective
memory `~200` steps) apparently hasn't shed the noisy early-transient
estimate by step 2400, while a fast one (`gamma=0.08`, effective memory
`~12` steps) locks onto the currently-dominant channel more decisively.

**Two qualitatively different "hard" failure modes, visible only in the
time series (not the medians)**:

- **Seed 0 — genuine bistability**: cosine oscillates between `~0.53`
  and `~0.96` repeatedly across the whole 2400-step replay, at every
  gamma tested (e.g. `gamma=0.005`: `0.51, 0.96, 0.96, 0.95, 0.76, 0.51,
  0.51, 0.96, 0.96, 0.53`). Two candidate channels are apparently very
  close in estimated relevance and the selector keeps flipping between
  them — this is a *noisy selection* failure, not a *ceiling* failure
  (13-14 channel switches recorded across the 10 snapshots for this
  seed alone).
- **Seeds 3 and 4 — stable but mediocre**: cosine stays in a tight band
  (seed 3: `0.78`-`0.85`; seed 4: `0.80`-`0.91`) across the entire
  replay and across all three gammas — the selector *converges* and
  *stays put*, it just converges to a channel whose ceiling is genuinely
  below `0.90`. This is a *ceiling* failure, not a *noise* failure.
- Seed 5 is intermediate (oscillates between `~0.67`-`0.98`, less
  violently than seed 0 but still not settled).
- Easy seeds (1, 6) are stable and high throughout, at every gamma.

**Per B4D's own framing**: the frozen-window estimator (B4C) already
handles seeds 3/4/5 about as well as the continuous version does at
convergence (comparable final medians), so continuous adaptation is not
obviously *better* than a one-shot calibration window here — its main
distinguishing behavior is the seed-0-type instability, which the frozen
protocol (a single fixed window, no further updates) does not exhibit
in the same way.

## B4E — architecture scaling accounting (bookkeeping only)

Notation: `N0` = lower (defective) layer mode count, `N1` = upper layer
mode count (`N0=N1=N` in this toy, but kept distinct below since real
architectures need not match). `d` = number of `Re(.)` boundaries in the
credit path (`d=1` for adjacent L=2 layers, per Phase A).

| method | deployed state | per-step compute | needs full `q1` at each step? |
|---|---|---|---|
| **exact L=2 P/Q teacher** | `2 * N0 * N1` complex (`P[j,m], Q[j,m]` all pairs) | `O(N0 * N1)` (one pole update per pair; building `c_t`/`q1`-weighted readout touches every `j`) | yes |
| **B2 `r`-truncated balanced reducer** | `r * N0` complex (`r` per mode) | `O(N0 * (r^2 + N1 * r))`, **dominated by `O(N0 * N1 * r)`** — the reduced *state* is small, but the readout is a *dense* projection `T_bal[:,:r]^dagger` of the *full* `2N1`-dim `c_t`, itself built from all `N1` components of `q1` | **yes, still** |
| **B4 rank-1 (coordinate-selection), deployed/frozen** | `1 * N0` complex (one selected channel per mode) | `O(N0)` — one pole update and one scalar multiply per mode, using only `Sa0[m]` (already computed by the existing online rule) and **one single component** `q1[j*]` | **no** — only the one selected component |
| **B4 rank-1, selection/calibration phase** | `O(N1)` per mode transiently (`2N1` candidate channels + accumulators) | `O(N1)` per mode — same order as what the existing online rule already computes every step (it already reads all of `q1` via `spatial_q`'s `B1^T` contraction), plus a small `O(N1)` accumulator on top | yes, but no *more* than the baseline already pays |

**The important, somewhat counter-intuitive finding**: the B2 balanced
reducer's smaller *state* (`r` vs `2N1`) does **not** translate into
smaller *compute*, because its readout is a dense rotation of the full
`c_t` vector — it still touches every one of `q1`'s `N1` components at
every step. B4's rank-1 construction is cheaper in a way balanced
truncation is not, *specifically because* R1 selects one of the
*existing, axis-aligned* natural channels rather than a rotated
combination — its deployed readout needs only the one `q1` component
tied to that channel. This is a property of *how* the reduction was
found (coordinate selection vs. basis rotation), not of the rank alone;
an `r`-channel version of B4's method (not tested here beyond `r=1`)
would cost `O(r)` deployed compute per mode, still far below balanced
truncation's `O(N1 r)`.

### Depth scaling (theory only, no new experiment)

Phase A (`credit_memory/PHASE_A.md`, Section A3) established that an
*exact*, fully causal (forward-only, no reverse-time pass) credit state
for an `L`-layer stack requires `O(2^{L-1})` channels per (bottom,top)
mode pair — one same-pole/conjugate-pole doubling per `Re(.)` boundary
crossed. R1's selection procedure, as implemented and tested here,
works by **enumerating every one of the natural candidate channels**
(`2N1` of them at `L=2`) and ranking them by empirical relevance. Applied
unchanged to a deeper stack, the *natural candidate pool to enumerate*
would be the full Phase-A channel set — `O(2^{L-1})` per mode pair — not
just `2N1`.

**This must be stated explicitly, per instruction**: the *deployed*
rank-1 (or rank-`r`) state remains small regardless of depth (that part
of B4's finding *does* generalize structurally — a single selected
channel is always `O(1)` per mode once chosen). But the **selection
statistic itself**, if computed the way R1 computes it here (exhaustive
enumeration and ranking of the full natural channel set), does **not**
remain cheap at depth — it inherits Phase A's exponential channel count.
Whether a *cheaper* (non-exhaustive, e.g. greedy/hierarchical) selection
procedure exists for `L>2` is an open question this phase does not
address and does not claim to have solved.

## B4F — decision

**Primary: Case A.** The frozen streaming, no-teacher, no-BPTT rank-1
estimator reaches median cos `0.9257 >= 0.90` on the same held-out
protocol as B3 (B4C), essentially reproducing B3 R1 exactly via a
genuinely online, O(1)-post-selection implementation. **A real, small,
deployable L=2 causal temporal-credit rule exists** for the architectures
and calibration protocol tested.

**Secondary: Case B applies specifically to continuous adaptation.**
B4D shows the always-on EMA variant is measurably less reliable than the
frozen protocol on the hardest seeds — not catastrophically (median
final cos on hard seeds, `0.826`, is comparable to B4C's hard-seed
median), but with genuine bistable oscillation on at least one seed
(seed 0) that the frozen, single-calibration-window protocol does not
exhibit. **Continuous adaptation is not yet a solved problem even though
the underlying statistic is sound** — this is consistent with, not
contradicting, the primary Case A verdict.

**Additional, non-exclusive caveat (bookkeeping, not Case D per se since
no `L>2` experiment was run)**: B4E's depth analysis shows the
*selection statistic*, as constructed and validated here, would need to
enumerate `O(2^{L-1})` candidate channels per mode pair for `L>2` if
applied unchanged — a genuine open scalability question for anything
beyond `L=2`, flagged now rather than left implicit.

## Artifacts

- `results/credit_memory/phase_b4a_identity_check_summary.json` — git
  hash, config, 30 per-channel identity rows, frequency-domain agreement.
- `results/credit_memory/phase_b4c_streaming_rank1_summary.json` — git
  hash, config, sanity check, per-(seed,test-trajectory) rows, median
  cos/frac-gap-recovered, gate pass/fail, selected channels per
  (seed,mode).
- `results/credit_memory/phase_b4d_online_adaptation_summary.json` —
  git hash, config, full cosine-over-time and channel-switch-count
  series per (seed,gamma).

## Not done in Phase B4 (by design, per scope)

No end-to-end task training, no RoutePC, no Meta-Adam, no prospective
coding, no least-action experiment, no S5 Stage 0, no `L=3+`
experiments (bookkeeping only), no new selection-statistic design for
deeper stacks.
