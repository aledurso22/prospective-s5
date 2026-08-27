# Phase B8 — resource-normalized rank-1 CCM selector

Branch `credit-memory-repair`. Not a test of causal credit (B7 settled
that decisively) — a test of whether the rank-1 **selection criterion**
is unnecessarily state-energy biased. Code:
`credit_memory/b8_normalized_selector.py` (the streaming `E_j`
extension + theory check), `credit_memory/b8_static_benchmark.py` (B8C).
Artifact: `results/credit_memory/b8c_static_benchmark_summary.json`.
Same 8 seeds, `N=6, T=60, BATCH=8`, `N_CAL_TRAJ=4, N_TEST_TRAJ=4`
(imported directly from `phase_b2bc_hankel_truncation` / reusing
`phase_b4c_streaming_rank1`'s exact deployment code), matching B3/B4's
own static benchmark protocol exactly.

**Headline: Case 4 — the normalized selector worsens the result,
cleanly and by a meaningful margin. Rejected. Stopped before B8D per
the task's own gate (only run B8D if B8C is positive; it is not).**

## B8A/B8E — hypothesis and theory check

Resource-normalized score:
```
R_j = |rho_j|^2 / (E_j + eps),   rho_j = sum_t conj(c_t[j]) x_{j,t}
                                   (the repo's established Hermitian
                                   pairing, unchanged),
                                  E_j = sum_t |x_{j,t}|^2
```

**Verified** (docstring derivation in `credit_memory/
b8_normalized_selector.py`): maximizing `|sum_t c_t^dagger (alpha
x_{j,t})|^2` over a free complex scalar `alpha`, subject to `sum_t
|alpha x_{j,t}|^2 <= 1`, gives objective `|alpha|^2 |rho_j|^2` under
constraint `|alpha|^2 E_j <= 1`; since the objective increases with
`|alpha|^2`, the constraint saturates at `|alpha|^2 = 1/E_j`, giving
objective `= |rho_j|^2 / E_j` — exactly `R_j`. The phase of `alpha` is
free (irrelevant to the objective), consistent with `alpha` representing
an unconstrained complex readout gain. This uses the same conjugation
convention verified throughout B1-B7 (`conj(c_t)`, not a plain-transpose
convention) — the algebra is correct and the criterion does follow from
the stated constrained-resource objective, exactly as claimed.

## B8B — streaming implementation

`StreamingRelevanceNormalized` subclasses `credit_memory.streaming.
StreamingRelevance` (unmodified in place), adding exactly one real
accumulator per candidate channel: `E_j <- E_j + sum_batch |x_j|^2`,
updated every step alongside the unchanged `x_j`/`rho_j` state. No BPTT,
no full P/Q teacher, no stored trajectory, no lag arrays, in the
algorithm. `eps=1e-9`, fixed, not tuned against BPTT.

## B8C — static benchmark result

**S1 is meaningfully *worse* than S0, not merely unhelpful:**

| selector | median held-out cos vs BPTT |
|---|---|
| online baseline | 0.628 |
| **S0 (unnormalized, current B4)** | **0.926** |
| **S1 (resource-normalized)** | **0.813** |
| (reference: exact-teacher-optimized rank-1, B3B) | 0.992 |

`S1 - S0 = -0.113` — a clear regression, not a rounding-level
difference. Per-seed detail (median cos over 4 test trajectories):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.770 | 0.987 | 0.992 | 0.846 | 0.825 | 0.878 | 0.987 | 0.926 |
| S1 | 0.663 | 0.717 | 0.932 | **0.889** | 0.350 | 0.818 | 0.988 | 0.629 |

**S1 beats S0 on only 2 of 8 seeds** (3 and 6, both by small margins);
on the other 6 it is comparable-to-much-worse, including a severe
failure on seed 4 (`0.825 -> 0.350`).

**Selection disagreement**: S0 and S1 pick different channels on
`28/48` mode-selections (`58.3%`) — the normalization materially changes
what gets selected, and that change is net harmful here.

**Why**: median `|lambda_j|` of the selected channel is *identical*
between S0 and S1 (`0.957` both) — normalization does not simply shift
selection toward a different pole magnitude. But median `E_j` of the
S1-selected channel is **less than half** that of the S0-selected
channel (ratio `0.42`). S1 systematically favors lower-*energy*
candidates among those with comparable pole magnitude — consistent with
the classic failure mode of ratio statistics with a small, noisy
denominator: a candidate whose `E_j` happens to be small (little
accumulated signal, only `N_CAL_TRAJ=4` trajectories of calibration)
can show an inflated `R_j` from denominator noise alone, without its
numerator `|rho_j|^2` actually reflecting better alignment quality. The
raw, unnormalized `|rho_j|^2` score turns out to be the more robust
statistic at this calibration sample size — exactly the opposite of the
motivating intuition in B8A.

**Gate (B8C, quoted)**: "If S1 does not improve the static rank-1
result meaningfully, stop B8 and recommend proceeding to S5 with the
current reactive A-CCM. Do not invent more selector heuristics." S1
does not improve — it regresses. **Gate fails. B8D was not run.**

## B8D — not run

Per the explicit instruction ("Only if B8C is positive"), and B8C is
decisively negative (a regression, not merely a non-improvement). No
end-to-end training comparison was performed for the normalized
selector.

## B8F — decision

**Case 4 — normalized selector worsens.**

> Reject it cleanly. Do not rescue it with parameter sweeps.

No `eps` retuning, no alternative normalization (e.g. a shrinkage/
regularized `E_j`, a minimum-sample-size floor before trusting `R_j`)
was attempted — the task explicitly forbids rescuing this heuristic
with further tuning, and this report follows that instruction rather
than exploring a fix.

**Recommendation**: proceed to S5 with the current (B6-established)
reactive unnormalized A-CCM selector (`S0`, `|rho_j|^2`). This closes
the toy-compression-criterion investigation started in B3 — B7 already
settled that the underlying causal mechanism is exact and complete;
this phase confirms the specific "normalize by state energy" refinement
of the *selection* heuristic is not the answer, at this calibration
sample size, on this toy.

## Artifacts

- `results/credit_memory/b8c_static_benchmark_summary.json` — git hash,
  config (including `eps`), full per-(seed, test-trajectory) rows,
  per-seed per-mode selection detail (`j_s0, j_s1, disagree,
  abs_lambda_s0/s1, raw_rho2_s0/s1, E_s0/s1, R_s0/s1`), aggregate medians,
  disagreement rate, gate result.

## Not done in Phase B8 (by design, per scope/gate)

No B8D end-to-end training (gate failed); no prospective prediction-
correction; no new EMA sweep; no BPTT-supervised selector; no dense
whitening/`MxM` deployed algorithm; no new model/task; no S5 launch; no
`eps` or normalization-scheme parameter search.
