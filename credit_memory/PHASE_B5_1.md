# Phase B5.1 — action-utility audit

Branch `credit-memory-repair`. Strictly diagnostic: the CCM (B4 rank-1)
algorithm was not changed, no new seeds were run, S5 was not launched.
Code: `credit_memory/b5_1_action_utility.py`. Artifact:
`results/credit_memory/b5_1/b5_1_action_utility_summary.json`. Reuses
exactly the 8 B5 seeds and their already-selected A2 channels (loaded
from the existing `results/credit_memory/b5/*.json`, never
re-calibrated).

**Headline: Case 2, with the mechanism behind "too small to accumulate"
identified.** BPTT has real, material one-step and full-training action
advantage over online (ruling out Case 1). CCM's action utility sits
between online and BPTT in 5 of 6 checkpoint x clip cells, and CCM's
*update vector* is consistently more aligned with BPTT's than online's
is, at the aggregate level (ruling out Case 4). But the block-level
audit (B5.1C) reveals *why* the effect is small: the frozen calibration
channel is strong near the calibration point (early-checkpoint `a0`
cosine `0.41 -> 0.88`, matching B3/B4's offline benchmark closely) and
decays substantially as training moves away from it (mid/late-checkpoint
`a0` cosine only `0.50 -> 0.51`; `b0` cosine actually **reverses**,
falling below online's own by step 600). This is a newly surfaced,
previously-untested failure mode — B3/B4's offline benchmarks never
probed post-calibration parameter drift.

## B5.1A — one-step action utility

Method exactly as specified: for each (seed, clip, checkpoint), clone
`(params, Adam m, Adam v)` from a bitwise-identical replay of the
online arm's own trajectory (same RNG draws as `credit_memory/
b5_train.py`, so this is re-deriving already-implied state, not a new
run); compute `g_on, g_CCM, g_BPTT` on the batch that would be drawn
next; apply one independent Adam step with each (same starting `m,v`,
no reset); evaluate all three, plus the pre-update baseline, on the
**same** fixed disjoint held-out batch; report `Delta L = L(theta') -
L(theta)`.

| clip | step | median dL_on | median dL_CCM | median dL_BPTT | ordering holds? |
|---|---|---|---|---|---|
| 0 | 100 | -0.00051 | -0.00055 | -0.00057 | yes |
| 0 | 300 | -0.00012 | -0.00015 | -0.00016 | yes |
| 0 | 600 | -0.00003 | -0.00006 | -0.00010 | yes |
| 1 | 100 | -0.00047 | -0.00055 | -0.00056 | yes |
| 1 | 300 | -0.00019 | -0.00017 | -0.00025 | **no** (CCM slightly worse than online here) |
| 1 | 600 | -0.00004 | -0.00011 | -0.00013 | yes |

**`Delta L_BPTT <= Delta L_CCM <= Delta L_on` holds in 5 of 6 cells** —
BPTT always gives the largest one-step improvement, CCM is consistently
intermediate except at `clip=1, step=300` where it is marginally *worse*
than online (a small, single-cell exception, not a pattern). This is
the literal signature of **Case 2**.

## B5.1B — optimizer-space (update-vector) comparison

| clip | step | median cos(dtheta_on, dtheta_BPTT) | median cos(dtheta_CCM, dtheta_BPTT) |
|---|---|---|---|
| 0 | 100 | 0.958 | **0.962** |
| 0 | 300 | 0.920 | **0.967** |
| 0 | 600 | 0.942 | **0.964** |
| 1 | 100 | 0.958 | **0.966** |
| 1 | 300 | 0.952 | **0.969** |
| 1 | 600 | 0.925 | **0.938** |

**CCM's post-Adam update vector is more aligned with BPTT's than
online's is, in all 6 cells** (median improvement `+0.02`-`+0.05`,
individually modest but directionally universal). This rules out **Case
4** ("gradient repair disappears in optimizer space") at the aggregate
level — the improvement in gradient cosine that B3/B4 established *does*
survive the Adam transformation into actual update space here, even
though (per B5.1C below) it does not survive uniformly across training
time or parameter blocks.

## B5.1C — parameter-block defect accounting (clip=0, step=300 shown; full table in the JSON)

| block | `D_b` (share of online-BPTT defect) | `R_b` (repair, absolute-residual) | `U_b` (share of BPTT's actual update norm) | `cos_on` | `cos_CCM` |
|---|---|---|---|---|---|
| **a0** (lower recurrent `a`) | **0.971** | -0.010 | 0.211 | 0.406 | 0.503 |
| **b0** (lower recurrent `b`) | 0.029 | 0.144 | 0.233 | 0.572 | 0.603 |
| upper (`a1`+`b1`) | 0.000 | 0.000 (trivial: nothing to repair) | **0.507** | 1.000 | 1.000 |
| readout `c` | 0.000 | 1.000 (trivial: already exact) | 0.052 | 1.000 | 1.000 |

**Two findings, and they are different things — reported separately so
neither is misread as contradicting the other:**

1. **`D_b`/`U_b` mismatch (explanation 3, confirmed)**: essentially
   100% of the online-vs-BPTT gradient *defect* lives in `a0` (97%) —
   exactly where B1-B4's mechanism targets. But `a0`+`b0` together
   account for only `21%+23%=44%` of BPTT's *actual update norm* — more
   than half (`51%`) of BPTT's real update comes from the upper-layer
   block, where online is **already exact** (Null-1, `D_b=0`, nothing
   for CCM to fix). Even a *perfect* fix of `a0`/`b0` is bounded by this
   44% ceiling on how much of BPTT's actual action it could ever recover.
2. **`R_b` (absolute residual) vs. `cos` (direction), a genuine and
   important distinction, not a contradiction**: `R_b` for `a0` is
   `-0.010` — CCM's absolute residual to BPTT is *no better* than
   online's for that block at this checkpoint — while `cos_ccm=0.503 >
   cos_on=0.406` shows CCM *is* directionally better. `R_b` is sensitive
   to magnitude/scale mismatch (the same `norm_ratio != 1` pattern
   documented throughout B1-B4: `norm_ratio_on=0.696`,
   `norm_ratio_ccm=0.634` here, neither close to 1). A correction can be
   right in direction and still show near-zero absolute-residual
   improvement if its scale is off. Both numbers are real; they answer
   different questions (B5.1C explicitly asked for both).

### The calibration-staleness finding (a0/b0 cosine across checkpoints, clip=0)

| step | `a0` cos_on | `a0` cos_CCM | `b0` cos_on | `b0` cos_CCM |
|---|---|---|---|---|
| 100 (early) | 0.414 | **0.879** | 0.215 | **0.812** |
| 300 (middle) | 0.406 | 0.503 | 0.572 | 0.603 |
| 600 (late) | 0.615 | 0.510 | 0.649 | **0.303** |

**This is the key mechanistic finding of this audit.** At the earliest
checkpoint (step 100, closest to the calibration point at initialization),
CCM's directional correction is strong and matches B3/B4's offline
benchmark closely (`a0`: `0.88`, `b0`: `0.81`, vs. B3/B4's established
`~0.90-0.99` range for a fresh calibration). By step 300 this has
decayed sharply (`a0`: `0.50`, `b0`: `0.60`), and by step 600 `b0` has
**reversed** — CCM's `b0` gradient is now *less* aligned with BPTT than
online's own naive `b0` gradient is (`0.303 < 0.649`). The calibration
(done once, before training, per B5D's primary protocol) selects a
channel that is well-matched to the network's parameters **at that
moment** — but layer 1's own `a1, B1` (which determine which channel is
"correct") continue to change throughout training, and the frozen
selection does not track that drift. B3/B4's offline experiments never
tested this: every offline evaluation there used calibration and
held-out data drawn from **the same fixed, never-updated architecture**
— this pilot is the first place a *moving* architecture (parameters
updated for hundreds of steps after calibration) was tested against a
frozen selection.

## B5.1D — BPTT training ceiling (existing B5 runs, no new experiments)

| clip | BPTT beats online (final loss) | median ratio(BPTT/online) |
|---|---|---|
| 0 | **8/8** | **0.862** |
| 1 | **7/8** | **0.904** |

**BPTT materially and consistently beats online on this toy** — final
loss lower by a median `14%` (clip=0) to `10%` (clip=1). This decisively
rules out **Case 1** ("the toy is not sensitive enough to exact temporal
credit"): the toy clearly *is* sensitive to exact credit; online is
leaving real, recoverable performance on the table, and BPTT recovers
it. Whatever limits CCM's task-loss benefit, it is not that BPTT itself
has nothing to offer here.

## B5.1E — decision

**Case 1 (BPTT has little action/task advantage): RULED OUT.** B5.1D
shows BPTT beats online on `8/8` (clip=0) and `7/8` (clip=1) seeds,
median ratio `0.86`-`0.90` — a real, material, consistent advantage.

**Case 4 (gradient repair disappears in optimizer space): RULED OUT at
the aggregate level.** B5.1B shows CCM's update-vector cosine to BPTT
exceeds online's in all 6 checkpoint x clip cells.

**Primary classification: Case 2** (`Delta L_BPTT <= Delta L_CCM <=
Delta L_on` holds in 5/6 cells) — **"CCM is doing the right thing, but
the toy effect is too small/noisy to accumulate strongly."** Per the
task's own rubric this recommends proceeding toward the prepared L=2 S5
GPU validation — **but this audit also identifies, precisely, why the
effect stays small on this toy**, which should inform how any S5
validation is designed:

1. **Calibration staleness** (new finding, this phase): the frozen
   channel's directional advantage is strong near the calibration point
   and decays — even reverses for `b0` — by 600 steps out. A validation
   that recalibrates more often, uses a shorter training horizon
   relative to calibration, or otherwise controls for this drift would
   be measuring the mechanism more fairly than this pilot's single
   frozen-at-init protocol did.
2. **Block-coverage ceiling** (B5.1C): even a hypothetically perfect
   `a0`/`b0` fix is bounded by the `~44%` of BPTT's actual update norm
   those blocks account for at L=2, N=6 — the other `>50%` of BPTT's
   real action improvement comes from upper-layer blocks online already
   gets exactly right, where there is nothing for a lower-layer-only
   correction to add.

Neither finding is a "Case 3" verdict (CCM's `Delta L` is not
`approx Delta L_on`, it is measurably better in 5/6 cells) nor a clean
"Case 4" verdict (the aggregate update-cosine advantage is real) — they
are the requested "identify which parameter blocks / optimizer
transformations explain the missing action benefit" analysis, and they
point at calibration staleness and block-coverage as the two concrete,
now-measured limiting factors, rather than a fundamental mechanism
failure.

## What this audit does and does not establish

- **Does establish**: BPTT has real action advantage here (Case 1
  closed); CCM's gradient advantage survives into optimizer-update space
  at the aggregate level (Case 4 closed); the one-step action-utility
  ordering (`BPTT <= CCM <= online`) holds in the strong majority of
  tested cells, consistent with a real but small, non-dominant effect
  (Case 2); and — the new contribution of this audit — *why* it stays
  small: calibration staleness under parameter drift, and a hard ceiling
  from which blocks actually dominate BPTT's real update.
- **Does not establish**: whether recalibrating periodically (continuous
  adaptation) would close this gap — B4D already showed continuous EMA
  adaptation introduces its own instability (bistable channel-flipping
  on at least one seed), so this is not a free fix, and per instruction
  no such experiment was run here. Also does not establish whether these
  same dynamics (staleness, block coverage) hold at S5 scale/architecture
  — this remains toy-L=2-specific evidence.

## Artifacts

- `results/credit_memory/b5_1/b5_1_action_utility_summary.json` — git
  hash, config, full per-(seed,clip,checkpoint) rows: `delta_L` for all
  three arms, update-vector cosines/relative-errors/norm-ratios, full
  per-block `D_b/R_b/U_b/cos_on/cos_ccm/norm_ratio_on/norm_ratio_ccm`
  (including `a0, b0, a1, b1, c`, and the aggregated `upper(a1+b1)`),
  and the B5.1D online-vs-BPTT summary.

## Not done in Phase B5.1 (by design, per scope)

No new seeds; no CCM algorithm changes; no prospective coding; no
continuous selector adaptation; no hyperparameter sweep; no S5 launch.
