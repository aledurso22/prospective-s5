# Phase B5 — first end-to-end training test of the B4 credit rule

Branch `credit-memory-repair`. First actual TRAINING use of the B4
rank-1 correction (B1-B4 were entirely offline/diagnostic). L=2 only.
No S5, no RoutePC, no Meta-Adam, no prospective coding, no exact/BPTT
information inside the A0/A1/A2 training update. Code:
`credit_memory/b4_deploy.py` (new: generalizes B4's verified "a"-block
correction to also cover "b", needed for a complete training gradient),
`credit_memory/b5_train.py`, `credit_memory/b5_report.py`,
`credit_memory/b5_selector_diagnostics.py`, `scripts/b5_pilot.sh`,
`scripts/b5_full_matrix.sh`. Artifacts:
`results/credit_memory/b5/*.json` (64 individual run files + 2
selector-diagnostic summaries).

**Headline: Partial pass, with an important statistical caveat.** A2
(B4 rank-1, causal calibration, frozen) maintains a real, directionally
consistent gradient-cosine advantage over the online baseline
throughout training, but task loss shows **no statistically
distinguishable difference** from online at this pilot's scale (n=8
paired seeds; sign test `p=1.0`, Wilcoxon `p=0.74`-`0.95` at both
clip settings). Per B5G: **credit repair is real; task-loss benefit is
not established at this scale — stop for review before S5, do not launch
the larger matrix without a decision on how to proceed.**

## B5A — arms, exactly as specified

- **A0 online**: `toyrig.ssm_rig`'s existing online rule, unchanged.
- **A1 b4_arch**: B4 rank-1, channel selected by a routing-weighted
  controllability score `|B1[j,m]|^2 / (1 - |a1[j]|^2)` computed from
  architecture alone (no data, no BPTT). This is **not** a zero-data
  version of R1's own statistic (see "A1's selector, explained" below)
  — R1's zero-lag cross-covariance is, by construction, an expectation
  of a product of two independent zero-mean quantities under any
  no-data prior, and is therefore identically zero/uninformative with
  no data at all. A1 instead reuses B2's "architecture + isotropic
  prior" logic (routing strength x controllability), the closest clean,
  well-defined zero-data analogue.
- **A2 b4_causal** (primary): B4 rank-1, channel selected via a short
  causal calibration prefix (the streaming estimator,
  `credit_memory/streaming.py`, windowed/frozen mode — B4D's continuous
  EMA variant is explicitly NOT used here, per instruction, to isolate
  the credit mechanism from the adaptation-instability problem it
  showed).
- **bptt** (reference only): exact BPTT teacher, for a performance
  ceiling; never informs A0/A1/A2.

Layers `>=1` and the readout `c` always use the unchanged online rule
in every arm — B3/B4 never found or claimed a defect there (Null-1: the
online rule is already exact for a layer with no further downstream
layers).

### A1's selector, explained

Per-mode `m`, P-type (pole `a1[j]`) and Q-type (pole `conj(a1[j])`)
channels have identical `|pole|` and identical `|B1[j,m]|`, so the score
above ties between them; A1 deterministically prefers the P-type index.
This is documented, not hidden.

## B5B — generalizing B4's correction from "a" to "b"

B3/B4 verified the rank-1 correction only for layer 0's **"a"**
(recurrence pole) gradient. A complete training update also needs the
**"b"** (input-weight) gradient. `credit_memory/b4_deploy.py` extends
the identical selected-channel construction to `Sb0` (Phase A's LEMMA 1
is stated for an arbitrary driving signal paired with an arbitrary local
sensitivity, so it applies to `Sb` exactly as it does to `Sa` — same
channel, same pole, same readout weight, only the paired local
sensitivity differs). **This is new, not previously numerically
verified** — checked here via checkpoint-0 diagnostics (see B5E):
`b4_causal`'s cosine at step 1 is `0.994`-`1.000` across all 8 seeds
(clip=0), matching the ~0.92-0.99 range B3/B4 established for the
"a"-only construction, confirming the "b"-block extension does not
break the mechanism at initialization.

## B5C/D — calibration protocol

**Primary (used for every A2 run reported here)**: (1) initialize
network at the run's seed; (2) run `N_CAL_TRAJ=4` trajectories forward
only (`forward`, `spatial_q`, `sensitivities` — **no parameter
updates**, matching B1-B4's calibration convention exactly); (3)
accumulate the streaming relevance statistic
(`credit_memory.streaming.StreamingRelevance`, windowed mode) causally,
one timestep at a time; (4) select `argmax|rho|` per lower mode; (5) no
recurrent hidden state carries across trajectories in this rig (every
`forward()` call starts from zero state — there is nothing to reset);
(6) begin ordinary training with the selected channel(s) frozen for all
`STEPS=600` steps. Calibration trajectories are drawn from a separate
RNG stream (seed offset `777+seed`), never reused as training batches.
**No BPTT, no exact P/Q teacher, no exact adjoint anywhere in
calibration or training** for A1/A2.

The "calibrate during the first K training steps" alternative (B5D's
secondary option) was **not run** in this pilot — the primary protocol
was designated first, per instruction, and the primary result was
informative enough to warrant stopping for review before spending more
runs on protocol variants.

## Fixed pilot config

`L=2, N=6, T=60, DELAY=20, BATCH=8`, delayed continuous-copy task
(matches `toyrig.ssm_rig`'s existing task family, DELAY scaled down from
the original 50/128 defaults to fit `T=60`), `STEPS=600`, Adam
`LR=1e-3`, checkpoints at steps `{1, 100, 300, 600}`. Pure numpy — no
GPU used or needed; **`~0.4`s wall time per run**, 64 runs total in
under 5 minutes.

## B5E — measurements

### Task loss (8 paired seeds, both clip settings)

| arm | clip | median final loss | median best loss | median late-training loss |
|---|---|---|---|---|
| online | 0 | 0.1368 | 0.0973 | 0.1331 |
| b4_arch | 0 | 0.1395 | 0.0943 | 0.1302 |
| **b4_causal** | 0 | **0.1336** | **0.0872** | **0.1253** |
| bptt (reference) | 0 | 0.1214 | 0.0766 | 0.1147 |
| online | 1 | 0.1350 | 0.0960 | 0.1335 |
| b4_arch | 1 | 0.1328 | 0.0902 | 0.1299 |
| **b4_causal** | 1 | **0.1307** | **0.0809** | **0.1205** |
| bptt (reference) | 1 | 0.1175 | 0.0717 | 0.1085 |

b4_causal's **medians** are consistently a bit better than online's at
both clip settings (and b4_arch sits in between, closer to online) —
but see the significance check below before reading anything into this.

**Paired A0-vs-A2 win count and significance** (final loss, `a2 - a0`
per seed):

| clip | A2 wins | sign test p | median log-ratio | Wilcoxon p |
|---|---|---|---|---|
| 0 | 4/8 | **1.000** | -0.0057 | 0.742 |
| 1 | 4/8 | **1.000** | +0.0066 | 0.945 |

**No significant difference by any paired test at n=8.** The favorable
medians above are consistent with (not proof of) a real small effect,
but this pilot cannot distinguish it from noise. Per-seed
`final_loss(A2) - final_loss(A0)`, clip=0: `[+0.022, +0.002, +0.032,
+0.032, -0.018, -0.004, -0.014, -0.010]` — a genuine mix of both signs,
comparable magnitude either way.

### Gradient-mechanism probes (checkpoint cosine vs BPTT, clip=0)

| step | online median | b4_causal median | b4_causal wins/8 | sign p |
|---|---|---|---|---|
| 1 | 0.994 | 0.996 | -- | -- |
| 100 | 0.874 | **0.936** | 5/8 | 0.727 |
| 300 | 0.829 | **0.894** | 6/8 | 0.289 |
| 600 | 0.848 | **0.897** | 4/8 | 1.000 |

Clip=1 (the gradient-mechanism advantage **erodes by the end of
training** here, unlike clip=0 where it stays roughly flat):

| step | online median | b4_causal median | b4_causal wins/8 |
|---|---|---|---|
| 100 | 0.865 | **0.947** | 6/8 |
| 300 | 0.748 | **0.917** | 5/8 |
| 600 | **0.833** | 0.745 | 3/8 |

**Even the cosine win-counts are not individually significant at n=8**
(best is `6/8`, `p=0.289`), though the *medians* consistently favor
b4_causal except at clip=1's final checkpoint. The much stronger,
statistically robust version of this same finding is B3/B4's own
dedicated offline experiments (32 held-out samples per cell, not 8) —
this pilot's within-training measurement is noisier, as expected for an
8-seed compound-over-600-steps setting, but points the same direction.

**bptt reference**: cosine is exactly `1.000` at every checkpoint for
every seed (self-consistency check on the diagnostic machinery: the
"bptt" arm's own recomputed gradient trivially equals the BPTT teacher
it's compared against) and reaches the best task loss of every arm at
both clip settings (median final loss `0.121`/`0.118`), as expected —
it is the unconstrained upper bound.

### Selector diagnostics (A2, clip=0; `credit_memory/b5_selector_diagnostics.py`)

Initial calibration relevance margin `(|rho|_top - |rho|_2nd) /
|rho|_top`, median over the 6 modes, per seed:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| median margin | 0.457 | 0.368 | 0.242 | 0.429 | **0.620** | 0.355 | **0.184** | **0.155** |
| final cos | 0.960 | 0.735 | 0.872 | 0.936 | 0.935 | 0.724 | 0.770 | 0.923 |
| final loss | 0.1857 | 0.1322 | 0.1700 | 0.1668 | 0.1312 | 0.1320 | 0.0776 | 0.1351 |

`corr(median_margin, final_cos) = 0.33`, `corr(median_margin,
final_loss) = 0.35` — both weak, and at `n=8` neither is a reliable
signal. **No clean relationship between calibration confidence and
downstream success is established** by this pilot; the "hard seeds"
identified in B3/B4 (architecture-level difficulty) do not obviously
map onto low-margin seeds here (e.g. seed 6 has the lowest margin,
`0.184`, but the *best* final loss of any b4_causal seed, `0.0776`).

A true offline "does the frozen channel stay top-ranked at later
checkpoints" re-audit (recomputing relevance from the network's
*trained* parameters at each checkpoint) was **not implemented** in this
pilot — flagged as a gap, not run, given the primary result was already
clear enough to warrant stopping for review.

## B5F — seed count

5-seed primary pilot (A0/A2, clip=0) run first, confirmed healthy and
finite (all 10 runs finite). Expanded to the full matrix (A0/A1/A2/bptt
x clip{0,1} x 8 seeds = 64 runs) since compute cost was negligible
(`~0.4s`/run). No cluster compute used or needed.

## B5G — classification

**Partial pass** (closest fit; stated with its full caveat):

> "Gradient cosine stays high but task loss is unchanged. → Credit
> repair is real, but optimizer/task geometry limits benefit. Stop for
> theory review before S5."

- Gradient cosine: directionally and consistently favors b4_causal at
  every checkpoint except clip=1's final one; medians are meaningfully
  higher (e.g. `0.894` vs `0.829` at step 300, clip=0); this is
  consistent with B3/B4's much more statistically robust offline
  findings.
- Task loss: **no significant difference** by paired sign test or
  Wilcoxon at either clip setting (`p >= 0.74` throughout); medians
  favor b4_causal modestly but the per-seed sign is genuinely mixed.

**Not "Strong pass"**: A2 does not clearly improve task loss on "most"
seeds — win count is an exact `4/8` tie at both clip settings, and the
paired tests confirm no detectable directional effect.

**Not "Mechanism failure"**: cosine does not collapse from initially-
high to low over the course of a run in the clip=0 condition (it stays
roughly flat, `0.936 -> 0.894 -> 0.897`); the clip=1 late-training
erosion (`0.917 -> 0.745`) is a partial, single-condition version of
this pattern and is flagged, not ignored, but does not dominate the
overall picture.

**Not "Failure"**: task loss does not clearly worsen — medians are
comparable-to-favorable for b4_causal, and the earlier appearance of a
clean online-wins pattern in the 5-seed pilot **did not survive**
expansion to 8 seeds (3 of the 3 added seeds all favored b4_causal on
task loss) — a direct, in-session demonstration of why the small-pilot-
then-expand protocol mattered.

## What this pilot does and does not establish

- **Does establish**: the B4 rank-1 mechanism, extended to a complete
  ("a"+"b") training gradient and deployed with a genuinely causal
  calibration protocol, trains stably (all 64 runs finite, no NaNs) and
  reproduces a real, directionally consistent gradient-alignment
  advantage inside an actual training loop, not just in offline
  diagnostics.
- **Does not establish**: that this gradient-alignment advantage
  translates into a statistically detectable task-loss benefit at this
  pilot's scale. The honest reading is "not yet resolved," not "resolved
  negative" — `n=8` paired seeds, each a single compound 600-step
  trajectory, is underpowered for the effect sizes observed (per-seed
  loss differences of order `0.01`-`0.03`, comparable to the run-to-run
  noise floor implied by the mixed signs).
- **Clip=0 vs clip=1**: no strong evidence either way that clipping
  mediates the effect the way it did for the historical RoutePC result
  — both conditions show similar (non-significant) task-loss patterns;
  the one clip-dependent difference found is in the *cosine trajectory
  shape* (stable under clip=0, eroding late under clip=1), not in task
  loss.

## B5H — lab-run entry point

**Branch**: `credit-memory-repair`. **Commit**: this phase's commit (see
`git log` after committing; recorded in every run's own JSON output
under `"git"`).

**Minimal required files**: `toyrig/ssm_rig.py` (unmodified),
`credit_memory/{hankel,teacher,streaming,b4_deploy,b5_train,b5_report,
b5_selector_diagnostics}.py`, `scripts/{b5_pilot,b5_full_matrix}.sh`.
**No S5 code required.** Python dependencies: `numpy`, `scipy`
(significance checks only) — no `jax`/`optax` needed for B5 itself
(those are only imported by B1/B2/B3's own scripts, not by B5).

**One-command pilot** (5 seeds, A0 vs A2, clip=0):
```bash
bash scripts/b5_pilot.sh
python -m credit_memory.b5_report --clip 0 --arms online,b4_causal
```

**One-command full matrix** (A0/A1/A2/bptt x clip{0,1} x N seeds,
default 8):
```bash
bash scripts/b5_full_matrix.sh          # 8 seeds
bash scripts/b5_full_matrix.sh 10       # 10 seeds
python -m credit_memory.b5_report --clip 0
python -m credit_memory.b5_report --clip 1
```

**Output directory convention**: `results/credit_memory/b5/
b5_{arm}_clip{clip}_s{seed}.json`, one file per (arm, clip, seed) cell.

**Resume behavior**: both launcher scripts are idempotent — they check
for the target output file and skip if it already exists; pass
`--force` (`bash scripts/b5_full_matrix.sh 8 --force`) to rerun
everything, or delete individual files under `results/credit_memory/b5/`
to force just those cells.

**Deterministic seed handling**: training RNG seeded `1000+seed`,
calibration RNG `777+seed` (always disjoint from training), diagnostic-
probe RNG `55555+seed` (disjoint from both) — set in
`credit_memory/b5_train.py`'s `train()`. Architecture itself
(`tcg.init_params(seed)`) uses `seed` directly, matching every other
phase in this project.

**GPU selection**: this pilot is pure numpy and has no GPU code path
(the toy rig in `toyrig/ssm_rig.py` never imports `jax`). The launcher
scripts honor `CUDA_VISIBLE_DEVICES` as a harmless no-op for consistency
with a future S5 launcher convention, but setting it changes nothing
here. Single-seed CPU debugging: `python -m credit_memory.b5_train --arm
b4_causal --seed 0 --clip 0`.

**Expected artifact filenames**: `results/credit_memory/b5/
b5_{online,b4_arch,b4_causal,bptt}_clip{0,1}_s{0..N-1}.json`, plus
`results/credit_memory/b5/b5_selector_diagnostics_clip{0,1}.json` if
`credit_memory/b5_selector_diagnostics.py` is run.

## Artifacts

- `results/credit_memory/b5/b5_{arm}_clip{clip}_s{seed}.json` x 64 —
  git hash, full loss curve, checkpoint diagnostics, selector info,
  `p_clip`, finite status, wall time.
- `results/credit_memory/b5/b5_selector_diagnostics_clip0.0.json` —
  per-seed calibration margins and correlations.

## Not done in Phase B5 (by design, per scope)

No S5, no RoutePC, no Meta-Adam, no prospective coding, no least-action
experiment; the "calibrate during first K training steps" protocol
variant was not run; the offline later-checkpoint channel-stability
re-audit was not implemented; no larger (>8-seed or >600-step) campaign
was launched pending review of this pilot.
