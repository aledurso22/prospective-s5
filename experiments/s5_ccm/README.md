# S5 CCM — Phase S1: L=2 scale validation

Branch `S5-CCM-scale-validation`. **Preparation only** — nothing in this
folder has been run at cluster scale or on GPU; every claim below is
either (a) verified on a tiny CPU config (`exactness_check.py`, run and
passing as of this branch's HEAD — see its own output/artifact) or (b) a
reasoned estimate, clearly marked as such.

## 0. The old prospective-S5 code (investigated, not touched)

There was an earlier "prospective SSM" hypothesis in this same repo
(`ssm/prospective/{layer,scan}.py`, wired into `ssm/model.py` as
`model_type="prospective"`). It is fully present on this branch (it was
never on a separate unmerged branch — all of its founding commits are
ancestors of `main`, and this branch descends from `main` via
`credit-memory-repair`). Its own docstring states the result plainly:
*"THIS LAYER DOES NOT TRAIN, AND THAT IS THE RESULT... diverges by
construction of the scheme"* — a second-order Euler discretization of
`tau ds/dt = -s + f(s,t) + tau df/dt` that provably cancels the SSM's
own learnable memory spectrum (see `docs/PROSPECTIVE_SSM_RESEARCH_
HANDOFF.md` for the full derivation). **Not used, not revived, not
touched by anything in this folder.**

What *is* reused, because it is ordinary shared infrastructure that has
nothing to do with the failed mechanism:

| piece | reused as |
|---|---|
| `ssm/model.py`, `ssm/shared/{block,params,hippo}.py` | architecture building blocks (`experiments/s5_ccm/ccm_core.py` extends the per-layer `model_type` idea, kept local rather than editing `ssm/model.py`) |
| `ssm/baseline_s5/{layer,scan}.py` | the BPTT reference arm and Sref |
| `ssm/online_s5/{layer,scan}.py` | the online arm (S0) and, unmodified, the machinery S1 builds on |
| `train_bench.py` | data loaders (`load_task`), `loss_fn`, `acc_fn`, `prep_x`, `provenance`, `peak_device_memory` — imported directly, not reimplemented |
| `scripts/train.sbatch`/`stage0.sbatch` conventions | SLURM script structure (`SLURM_SUBMIT_DIR`, `.venv` activation, GPU sanity print) |

`research/pesm-s5-spectrum` and the early commits of
`research/prospective-credit-s5` are also ancestors of `main` (same
code, not a separate lineage); that branch's *own* unique commits are a
standalone numpy analysis (`optimal_credit_filter.py`), unrelated to
`ssm/` and not used here.

## 1. Design note: how S1 is built (read this before trusting it)

**S1 needs no new custom-VJP code.** A 2-layer stack where the TOP layer
uses `ssm.baseline_s5` (ordinary autodiff — no approximation) and the
BOTTOM layer uses the existing `ssm.online_s5` (the Sa/Sb custom-VJP,
`credit_memory/PHASE_A.md`'s LEMMA 1: `Ga = sum_t conj(q_t) Sa_t` is
exact for *whatever* cotangent `q_t` is built from) gives, under plain
`jax.grad`:

- the top layer's own gradient exactly (ordinary autodiff, no custom VJP
  involved at all);
- the EXACT `dL/d(bottom-layer-SSM-output)` cotangent, backpropped
  through the (ordinary, exact) LayerNorm/GLU/Dropout/residual coupling;
- fed into the bottom layer's online custom-VJP, whose own formula is
  LEMMA-1-correct for *any* incoming cotangent — so it now reproduces
  the bottom layer's exact BPTT gradient too, using the SAME cheap
  forward Sa/Sb machinery the online rule already runs, no reverse-time
  pass for the bottom layer at all.

**Verified** (`experiments/s5_ccm/exactness_check.py`, tiny CPU config,
`d_model=8, state_size=6, T=20, batch=4`): every `params/SSM_0/*` leaf
(`Lambda`, `B`, `C`, `D`, `log_step` — the bottom layer's OWN recurrence
parameters, exactly what Phase A's construction claims to reconstruct)
matches BPTT to `cos ~ 1.0000000` (7-10 decimals), `rel_err` `6.5e-8` to
`7.2e-7` — float32 machine precision (this repo's own established bar,
matching `tests/test_online_s5_jax.py`'s `~1e-7`, not the pure-numpy
toy's `1e-15` float64 bar).

**Scope limit, found and documented, not hidden**: `S5Block_0`'s OWN
`LayerNorm_0` (which sits *before* the bottom SSM's input) does **not**
match BPTT (`rel_err ~ 0.08`-`0.11`, same order as the plain online
arm's own defect there) — because its gradient flows through the online
custom-VJP's `dx` output, which is *still* the defective instantaneous-
only input-cotangent (`dx = dy*(Re(C*Bb)+D)`, unchanged regardless of
what the layer above is). This is a genuinely new category of parameter
the toy never had (the toy's raw task input was not itself the output of
a learned pre-recurrence layer) — see "Transfer differences" #1 below.
S1's claim is scoped to the recurrence parameters, exactly as stated
above and as `exactness_check.py`'s gate enforces.

**A fully forward-only version of S1** (avoiding even the one ordinary
reverse-time pass BPTT does for the top layer — matching the toy's own
`P_t[j,m]=a1[j]P_{t-1}[j,m]+u_t[m]` construction literally) is
derivable: feed the time-varying, per-timestep local Jacobian of the
inter-block coupling (computable via `jax.vjp`/`jax.jvp` on the already-
available forward activations, no future information needed) *into* the
same pole-filtered forward accumulator, generalizing the toy's constant-
matrix `B1[j,m]` to a data-dependent per-step weight. This was **not**
implemented — the mixed-model-type construction above already satisfies
the load-bearing exactness requirement with much lower implementation
risk, and building the fully-forward version without GPU-based
validation capacity in this prepare-only phase was judged not worth the
risk of shipping subtly-incorrect custom-VJP code. Flagged as a possible
follow-up, not a blocker.

## 2. Implemented arms

| arm | CLI `--arm` | status | mechanism |
|---|---|---|---|
| S0 online | `online` | **ready** | both layers `ssm.online_s5`, unchanged |
| S1 full exact causal CCM | `s1_full_causal` | **ready, verified** | bottom `online`, top `baseline` (see above) |
| Sref BPTT | `bptt` | **ready** | both layers `ssm.baseline_s5`, ordinary autodiff |
| S2 rank-1 frozen | `rank1_frozen` | **not implemented** | see "S2/S3 implementation status" |
| S3 rank-1 reactive | `rank1_reactive` | **not implemented** | see "S2/S3 implementation status" |
| S4+ r=2/4/8 | `rank2`/`rank4`/`rank8` | **not implemented** | see "S4+ status" |

Selecting a not-implemented arm raises `NotImplementedError` immediately
(before any data loads or compute runs) with a pointer to this file —
it does not silently fall back to anything.

### S2/S3 implementation status

Full detail in `experiments/s5_ccm/ccm_rank1.py`'s module docstring and
inline "S2/S3 implementation status" section. Summary:

**Ready, independently verified pieces**:
- a plain-JAX reimplementation of the 2-layer forward pass
  (`ccm_rank1.two_layer_forward`), verified against the Flax baseline
  model's own forward output to `rel_err = 2.3e-7` (part of
  `exactness_check.py`'s output, "S2/S3 plain-JAX forward vs Flax
  model");
- `sa_forward`: the within-layer eligibility recursion, matching
  `ssm_online.py`'s own `Sa` exactly;
- `naive_top_layer_q`: extracts the exact *naive* (instantaneous, no-
  recursion-lookback) top-layer error `dy1_t` via an ordinary
  `jax.grad` on a "tail-only" closure (GLU+residual+pool+head, with the
  top layer's own SSM recurrence never touched by the closure) — no
  custom-VJP, low risk.

**Not implemented — the rank-1 selection/combination itself**, because
completing it exposed a genuine, S5-specific scaling question that was
not present in the toy: the toy selects one candidate channel *per lower
mode* from a pool of `2N` candidates; S5 adds a channel dimension the
toy never had, so the natural generalization is one selection *per lower
(channel, mode) pair* `(h0, n0)` from a pool of `H1 x N1 x 2` candidates.
At this repo's existing default S5 size (`H~96, N~64`) that is roughly
12,288 candidates per lower pair, `H0 x N0 ~ 6,144` pairs needing their
own selection — calibration would need to score on the order of 75
million `(candidate, target)` combinations, which is almost certainly
not the intended design. The natural restriction (candidates limited to
`h1 = h0` — same-channel-index only, reducing the pool to `N1 x 2` per
lower mode, `H0 x N0` total selections, matching how channels are
otherwise architecturally exchangeable in this codebase) is a genuine
design decision that needs an explicit answer before this is
implemented, not something to silently choose. Per instruction ("leave
arms scaffolded and report that rather than improvising" — licensed
explicitly for S4+, judged to apply with equal force once this scaling
question surfaced for S2/S3 too): left here, not shipped.

### S4+ status

Not started. B2's balanced-truncation machinery (`credit_memory/
hankel.py`) is built specifically for a *fixed, linear, time-invariant*
coupling matrix (the toy's constant `B1[j,m]`) via closed-form diagonal
Lyapunov equations. S5's actual inter-layer coupling (LayerNorm/GLU,
nonlinear and data-dependent — see "Transfer differences" #2) would need
this generalized to a genuinely time-varying system (time-varying
Riccati-type equations, not a closed-form diagonal solve) — substantial
new control theory, not an engineering port. Per instruction, left
scaffolded: no code beyond this note.

## 3. Transfer differences: toy vs S5 (found, all required care)

1. **A learned pre-recurrence layer exists in S5 and didn't in the
   toy.** `S5Block`'s own `LayerNorm` sits *before* each SSM. The toy's
   raw task input fed the recurrence directly (no learned parameters
   upstream of the lowest layer's own `a`/`b`). This is exactly the
   parameter category S1's exactness claim does not (and structurally
   cannot, via the mixed-model-type construction) cover — see Section 1.

2. **The inter-layer coupling is nonlinear and data/time-dependent, not
   a fixed matrix.** The toy's `x^{l+1} = Re(B1 * h^l)` is a *constant*
   linear map. The real S5 block computes `output = x + GLU(Dropout(
   SSM(LayerNorm(x))))` — `LayerNorm` and `GLU` (`Dense * sigmoid(
   Dense)`) are both nonlinear and depend on the actual runtime
   activations at each timestep. Both remain purely *instantaneous*
   (same-timestep only — no operation between stacked SSM layers mixes
   across time), so Phase A's core forward/backward duality (LEMMA 1)
   still applies exactly; what changes is that the toy's constant
   routing weight `B1[j,m]` generalizes to a time-varying, per-timestep
   local Jacobian, computable causally via `jax.vjp`/`jax.jvp` on
   already-available forward activations (see Section 1's "fully
   forward-only" note) rather than being a value fixed for the whole
   sequence.

3. **S5's poles are per-(channel, mode), the toy had only modes.** Each
   SSM layer discretizes `Lambda` (shared across channels) with a
   per-channel step size `Delta[h]` (`ssm/baseline_s5/layer.py:
   discretize_bilinear`), giving `Lambda_bar` shape `(H, N)` — a
   genuinely different pole per `(channel, mode)` pair, not shared
   across channels the way the toy's single-channel `N`-mode structure
   was. This is the direct cause of the S2/S3 candidate-pool blowup in
   Section 2.

4. **Batching is via `jax.lax.map` + `jax.checkpoint`, not `vmap`.**
   `ssm/baseline_s5/layer.py` and `ssm/online_s5/layer.py` both walk the
   batch dimension with `jax.lax.map(jax.checkpoint(run_batch), u,
   batch_size=16)` (a memory-safety valve, not a mathematical choice);
   channels are `vmap`-ed. `exactness_check.py`/`ccm_rank1.py` operate
   on one sample at a time (no batch dimension in the plain-JAX path);
   `train.py` relies on the existing Flax layers' own batching for S0/
   S1/Sref, so this only matters if/when S2/S3 are completed and need to
   decide their own batching strategy.

## 4. Exact-causal tensor/state shapes and expected complexity

Notation: `H0, N0` = bottom layer channels/modes; `H1, N1` = top layer
channels/modes; `T` = sequence length; `B` = batch.

| quantity | shape | memory | compute/step |
|---|---|---|---|
| S0 online, deployed | `Sa0, Sb0`: `(B, H0, N0)` (running, not stored per-`t`) | `O(B*H0*N0)` | `O(B*H0*N0)` |
| Sref BPTT, both layers | activation tape (checkpointed via `jax.lax.map`+`jax.checkpoint`, chunked `batch_size=16`) | per baseline_s5/layer.py's own comment: `~40MB` per full-size (`H=96,N=64,T=784`) sample's scan tensor, `x2` layers | `O(B*H*N)` per layer, parallel-scan `O(log T)` depth (`scan_impl=assoc`) |
| **S1 full causal CCM** | bottom: `Sa0,Sb0` (`O(B*H0*N0)`, unchanged online cost); top: ordinary BPTT tape for **one** layer only (`~40MB`/sample at full size, not `x2`) | strictly less than Sref (one layer's tape, not both) | bottom layer: same as online (cheap); top layer: same as one baseline-S5 layer |
| S2/S3 rank-1, deployed (if completed) | **one** selected `(h1,n1)` channel state per `(h0,n0)`: `O(B*H0*N0)` scalar states, same order as online | `O(B*H0*N0)` | `O(B*H0*N0)` -- matches the toy's own finding (B4E) that coordinate-selection, unlike balanced truncation, needs only the ONE selected upstream component, not the full upper-layer state |
| S2/S3 calibration (if completed) | transiently `O(H1*N1*2)` candidate channel states **per lower `(h0,n0)` pair** | see Section 2's scaling note -- this is the open question | matches whatever candidate-pool restriction is decided |

**BPTT vs full CCM vs compressed CCM, the actual trade** (only S1 is
measured; S2+ are the design target once completed):
- BPTT: needs the full activation tape (both layers) + a genuine
  reverse-time pass through both layers' own recurrence.
- S1 (full causal): needs the activation tape for **one** layer (the
  top) + a reverse-time pass through **that one layer only**; the
  bottom layer needs no tape and no reverse pass at all (Sa/Sb forward
  only). This is the concrete, measurable memory saving S1 offers over
  BPTT at `L=2` — it should grow if this pattern is later extended to
  deeper stacks (untested; `L>2` causal enumeration is explicitly out of
  scope for this phase, see Phase A's own `credit_memory/PHASE_A.md`
  `O(2^{L-1})` channel-count finding for why depth is not free even in
  principle).
- Compressed CCM (S2/S3, once completed): the toy's own finding (B4E) is
  that coordinate-selection rank-1 needs **no** persistent upper-layer
  state at all once deployed/frozen — only the one selected channel's
  own `O(1)` state, reading only the one component of the upper layer's
  signal it was matched to. This is the whole point of pursuing
  compression once the causal ceiling (S1) is established as real.

## 5. Cluster commands

```bash
git clone <repo-url> && cd prospective-s5
git checkout S5-CCM-scale-validation
bash setup.sh                       # builds .venv with CUDA jax, per the
                                     # existing repo convention
```

Environment variables (all optional; sensible repo-relative defaults if
unset — no hard-coded local paths anywhere in this folder):

| variable | default | purpose |
|---|---|---|
| `S5_CCM_RESULTS_ROOT` | `experiments/s5_ccm/results` | output JSON per run |
| `S5_CCM_DATA_ROOT` | `<repo>/data` | dataset root (passed through to `train_bench.load_task`'s existing MNIST loader) |
| `S5_CCM_CHECKPOINT_ROOT` | `<results root>/checkpoints` | reserved for future checkpoint writing (not yet used — `train.py` does not currently checkpoint mid-run, matching `train.py`'s own existing "no checkpoint/resume" note in `scripts/train.sbatch`; this run's own *output JSON* resume logic is separate and already active, see `--resume`) |

Direct invocation (any of these; `sbatch` wraps the same commands):

```bash
# exactness check (seconds, any device)
python -m experiments.s5_ccm.exactness_check

# one arm, one seed, tiny smoke config
python -m experiments.s5_ccm.train \
    --config experiments/s5_ccm/configs/l2_smoke.json --arm online

# primary L=2 config, one arm/seed (override task/seed on the CLI)
python -m experiments.s5_ccm.train \
    --config experiments/s5_ccm/configs/l2_primary.json \
    --task smnist --arm s1_full_causal --seed 0
```

Interactive (`srun`):
```bash
srun -p <partition> -A <account> --gres=gpu:1 --pty bash
source .venv/bin/activate
python -m experiments.s5_ccm.exactness_check
```

### Recommended first sbatch sequence

```bash
sbatch -p <partition> -A <account> experiments/s5_ccm/slurm/s5_ccm_exactness.sbatch
# wait, confirm "ALL params/SSM_0/* ... PASS: True" in the log
sbatch -p <partition> -A <account> experiments/s5_ccm/slurm/s5_ccm_smoke.sbatch
# wait, confirm all 3 arms (online, s1_full_causal, bptt) finish and write JSON
sbatch -p <partition> -A <account> experiments/s5_ccm/slurm/s5_ccm_rank1.sbatch smnist 3
# the primary L=2 experiment (named per the requested convention; currently
# runs online/s1_full_causal/bptt only -- see Section 2)
```

`train.py --resume 1` (default) skips any `(arm, task, clip, seed, tag)`
whose output JSON already exists and is finite — safe to resubmit the
same `sbatch` command after a preemption/timeout without redoing
completed cells.

## 6. Reproducibility

- **Commit**: this folder was committed on branch `S5-CCM-scale-
  validation`; every run's output JSON records its own `provenance`
  (git commit, via `train_bench.provenance()`, reused unmodified) and
  `config` (the full resolved argparse namespace).
- **Seeds**: `--seed` controls both `np.random.seed` (minibatch
  sampling) and the JAX `PRNGKey` (init + dropout) deterministically;
  `train_bench.load_task`'s own copy-task generator seeds from
  `args.seed + 777000`/`888000` (train/test), unchanged from the
  existing benchmark's convention.
- **Dependencies**: `requirements.txt`/`setup.sh` at repo root
  (`jax[cuda12]==0.11.0`, `flax==0.12.8`, `optax==0.2.8`) — unchanged,
  no new dependencies introduced by this folder.
- **Defaults** (primary config, `configs/l2_primary.json`): `L=2`,
  `d_model=96`, `state_size=64`, `dropout=0.1`, `scan=assoc`, `lr=1e-3`,
  `batch_size=64`, `epochs=3`, `clip=0.0` (this repo's own historical
  default, restored in `credit_memory`'s B-phase work — see
  `RESULTS_LEDGER.md` section 14). `clip=1.0` is not yet configured as a
  secondary sweep in this phase (per instruction: no large sweep; add
  `--clip 1.0` manually if wanted).

## 7. GPU memory/runtime considerations (estimated, not measured)

No GPU run has been performed in this preparation phase (per
instruction). Reasoned estimates only:

- **`s5_ccm_exactness.sbatch`**: tiny (`H=8,N=6,T=20,B=4`), CPU-fast
  (`<5s` measured on this machine's CPU); GPU launch overhead will
  likely dominate wall time. Negligible memory.
- **`s5_ccm_smoke.sbatch`**: `H=8,N=6`, `T~30`, `20` steps, 3 arms —
  seconds per arm on CPU; expect well under a minute total on GPU too.
  Negligible memory.
- **`s5_ccm_rank1.sbatch` (primary, `smnist`, `H=96,N=64,L=2,T=784`)**:
  by direct analogy to the existing `scripts/bench.sbatch`/`stage0.
  sbatch` full-sMNIST runs (same model size, `L=3` there vs `L=2` here —
  strictly less work), expect **peak GPU memory well under the existing
  benchmark's own `--mem=32G` job allocation** (this launcher requests
  the same). Sref (BPTT) is the most memory-hungry arm (full two-layer
  activation tape); S1 should measurably use less (one layer's tape
  instead of two, per Section 4); S0 (online) should be cheapest.
  **Read the actual `peak_device_memory_bytes` field in each run's own
  output JSON after the smoke run** for a real number before trusting
  this estimate on the primary run's larger config.
- **Wall time**: 3 epochs of full sMNIST at this model size, by analogy
  to the existing (unlaunched) `scripts/bench.sbatch` budget, is on the
  order of the `--time=04:00:00` that launcher itself requests per arm;
  this launcher requests `08:00:00` for the loop over 3 arms x N seeds
  as a conservative starting point — adjust after the smoke run reports
  `steps_per_sec`.
