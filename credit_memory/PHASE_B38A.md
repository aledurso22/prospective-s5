# Phase B38a — end-to-end training with exact causal ProductLocal credit

Branch `S5-CCM-scale-validation`. B37a–c are frozen and imported unmodified.
This phase is **not** about expressivity: ProductLocal is never compared
against the universal quotient here. The question is whether the practical
ProductLocal architecture can be **optimized using exact forward causal
credit** instead of BPTT.

Code: `b38a_engine.py`, `b38a_train.py`, `b38a_identity.py`,
`b38a_state_identity.py`, `b38a_sweep.py`, `b38a_bench.py`,
`b38a_stream_micro.py`, `b38a_online.py`, `b38a_analyze.py`.
Logs `/tmp/b38a_*.log`, results `results/b38a/*.json`.

## 0. Setup and honest protocol notes

Architecture is B37c's, unchanged: `A_π = ∏_j K_j[ε_j]/(ε_j^{d_j})`,
`z_{t+1} = u z_t + Σ_c b_c x_{c,t}`, native local multiplication only, no
global `q`, companion matrix, CRT transform, or parameter-dependent basis.
Gradients use a **native adjoint multiplication** — `M_e^T` is never
materialized — so the whole gradient path is O(P) work and memory with no tape
(verified against explicit `M_e^T` to 9.3e-17).

Three things are stated up front rather than buried:

1. **Index convention.** B37c's `y_t = C_out z_{t+1}` is kept (the B37b/c
   teachers are generated as `y_t = C_* h_{t+1}`); the spec's `y_t = C_out z_t`
   is the same equation under a relabelling of `z`. Trace recursions unaffected.
2. **Target normalization.** Each teacher's `C_*` is rescaled so the target has
   unit variance, applied **identically to arms A, B and C**. At T=256 the
   high-transient families reach a large steady-state output amplitude
   (`exact_jordan` mean y²=2.8e6, `nonnormal` 2.6e7), which would confound a
   training-ALGORITHM comparison with a pure target-scale effect. Scaling `C_*`
   is an exact output relabelling; B37b's `make_teacher` is imported unmodified.
3. **LR grid.** `{3e-4, 1e-3, 3e-3, 1e-2}`, selected on validation only. This is
   a **common grid settled during pilot/debugging, not preregistered** — an
   initial narrower grid missed `exact_jordan`'s working LR, so it was widened
   before the sweep and held fixed thereafter for every arm.

Tasks: the nine B37c LTI families at r=8, T=256, 3 seeds, known compatible π
(so architecture mismatch cannot contaminate the training-algorithm test).

## 1. Q1 — gradient engine: RTRL == matched TBPTT

Arm A is ordinary autodiff/TBPTT; Arm B computes the same chunk gradient from
forward eligibility states only. At every chunk boundary the incoming hidden
state is stop-gradiented, the eligibility is reset, the same chunk losses are
accumulated, and one update is made. Parameters are fixed across each
differentiation interval.

**Verified during actual training, not on an isolated derivative test** — both
gradients computed at every chunk, with the RTRL gradient driving the update.
Over **19,800 training chunks** (9 families × L ∈ {32,128,256} × 200 epochs):

| block | worst relative error |
|---|---|
| `u` | **1.215e-15** |
| `b_c` | **5.257e-16** |
| `C_out` | **3.356e-15** |

Step-locked from identical initialization, comparing full parameter vectors and
both Adam moments after every block of epochs:

| quantity | worst deviation |
|---|---|
| parameters | **1.217e-15** |
| Adam `m` | 1.356e-13 |
| Adam `v` | 1.360e-14 |

Across the 81 matched (family, seed, L) cases, 80 selected the same LR in both
arms; there the validation-trajectory deviation is ≤ **3.8e-13** and the final
held-out NMSE agrees to **6.9e-11**. Held-out NMSE and Markov error are
identical to printed precision at every family and every L (Tables 1–2 in
`/tmp/b38a_analysis.log`), with zero divergence in either arm.

**The one exception, reported rather than smoothed over.** `stiff` seed 1
L=32 selected different LRs per arm. Direct step-locked comparison at that
point: at lr=3e-3 the trajectories stay at 3.4e-15 for all 200 epochs; at
lr=1e-2 they begin at 3.4e-15, reach 3.6e-09 by epoch 50, and end at 2.2e-01.
This is exponential amplification of floating-point differences in a chaotic
optimization regime — the stiffest family (ρ=0.99) at the most aggressive LR —
**not** a gradient discrepancy: the per-chunk gradients at those same steps
still agree to 1e-15. Gradient equivalence is exact; *trajectory* equivalence
holds only where the optimization is not chaotic.

## 2. Derivative-memory scaling

The registered claim is specifically:

> **persistent RTRL eligibility memory is independent of sequence length T.**

No claim is made that *total* compiled memory is T-independent: in the batched
benchmarks the whole `B × T` input tensor is resident, and it dominates.

**True streaming microbenchmark** (`b38a_stream_micro.py`) — tokens arrive in
chunks of 256, the full sequence is never materialized:

| tokens seen | eligibility bytes | state | optimizer | chunk buffer | process RSS |
|---|---|---|---|---|---|
| 256 | 1024 | 512 | 384 | 32768 | 2011.4 MB |
| 65,536 | 1024 | 512 | 384 | 32768 | 2011.4 MB |
| **1,048,576** | **1024** | 512 | 384 | 32768 | **2011.4 MB** |

Eligibility is constant at 1024 bytes = `(m+1)·r·B·8` exactly, over a
**1,048,576-token** stream, with **+0.0 MB** RSS drift.

**XLA compiled scratch, time-major inputs** (B=8, r=8, full window). Time-major
input *did* remove the residual O(T) compiler scratch, so it is reported rather
than engineered around further:

| T | BPTT temp | RTRL temp |
|---|---|---|
| 128 | 0.280 MB | 0.004 MB |
| 512 | 1.116 MB | 0.004 MB |
| 2048 | 4.458 MB | 0.004 MB |
| 8192 | 17.827 MB | 0.004 MB |
| 32768 | **71.305 MB** | **0.004 MB** |

BPTT scratch grows exactly linearly in T (255× over a 256× range); RTRL is
flat. State-dimension scaling at T=512 is equally clean: as r goes 8→512,
BPTT temp 1.116→67.265 MB while RTRL temp 0.100→0.291 MB.

**The three regimes, distinguished:**

| method | derivative memory | temporal credit |
|---|---|---|
| full BPTT | activation/tape grows with the differentiation horizon | untruncated |
| fixed-L TBPTT | bounded | **truncated at L** |
| **ProductLocal RTRL** | **O(P) persistent, `P_dyn=(m+1)r`** | **untruncated as stream length grows** |

RTRL is the only one of the three that gets both. No OOM point was reached
locally for either method (CPU with ample RAM), so a "maximum sequence length
before memory is limiting" is reported as the measured scaling trend rather
than an observed failure point.

## 3. Wall-clock — CPU-ONLY, no GPU claim

**This machine exposes a CPU-only JAX backend (no CUDA device).** Peak *GPU*
memory was not measured and **no GPU speed claim is made**. All timings are CPU,
with `jax.block_until_ready` before and after every timed region, compilation
excluded via warmup, median of ≥5 reps.

| L | arm A (s) | arm B (s) | B/A |
|---|---|---|---|
| 32 | 0.28 | 0.22 | 0.78 |
| 128 | 0.25 | 0.21 | 0.84 |
| 256 | 0.22 | 0.15 | 0.68 |

Throughput at T=32768, B=8, r=8: BPTT 1.59e6 tok/s, RTRL 2.63e6 tok/s.

**These ratios must not be read as architectural speedups.** At r=8 and B=8 the
model is tiny and the measurement is dominated by scan/dispatch overhead on a
CPU backend; the guaranteed advantage established here is
sequence-length-independent derivative memory, and wall-clock superiority
remains an open empirical question. The scaling and memory results are
preserved in `results/b38a/bench.json` so a single matched GPU benchmark can be
run on the cluster later without re-deriving anything.

## 4. Q2 — Arm C, every-token online updates

`θ_t → θ_{t+1}` after every observed loss, eligibility carried across parameter
updates and never reset. Judged as an **online learning algorithm**, not as a
numerical reproduction of BPTT: once θ changes every step there is no single
fixed parameter vector that generated the history, so the carried trace is the
exact sensitivity under the fixed/path-shift interpretation but is not the
frozen-current replay gradient. The optimizer is never differentiated through.
Architecture and optimizer unchanged from A/B; same LR grid; 8192-token streams.

| family | online NMSE | held-out NMSE | Markov | tokens to 90% drop | div |
|---|---|---|---|---|---|
| random_stable_diag | 4.78e-02 | 2.76e-05 | 3.41e-03 | 512 | 0.00 |
| distinct_real | 2.16e-02 | 1.08e-03 | 2.03e-02 | 512 | 0.00 |
| complex_conjugate | 2.86e-02 | 2.14e-05 | 4.20e-03 | 512 | 0.00 |
| repeated_poles | 1.29e-02 | 4.42e-06 | 2.23e-03 | 256 | 0.00 |
| multi_jordan_shared | 7.56e-02 | 4.14e-03 | 2.55e-02 | 768 | 0.00 |
| nearly_defective | 1.83e-02 | 3.26e-06 | 1.55e-03 | 256 | 0.00 |
| nonnormal | 4.66e-02 | 4.12e-03 | 3.60e-02 | 768 | 0.00 |
| **exact_jordan** | 1.00e+00 | **1.00e+00** | 3.96e-01 | not reached | 0.00 |
| **stiff** | 1.17e+00 | **3.10e-01** | 1.44e-01 | 1920 | 0.00 |

**22 of 27 runs** reach held-out NMSE < 0.1; **0 of 27 diverge**. Adaptation is
fast where it works — 256–768 tokens to a 90% drop in online loss. Single-stream
throughput ≈ 1.4e5 tokens/s with a parameter update after every token (CPU).

**Failures, preserved by family.** `exact_jordan` fails in all 3 seeds — but
**matched BPTT on the identical model fails identically** (arms A and B both
0.980–0.997 at every L), so per the phase's own rule this is *not* an RTRL
failure; it is a shared optimization failure of the single degree-8 jet factor
on this task. `stiff` is the one family where Arm C is genuinely worse than
batch: 3.10e-01 online vs 3.24e-04 for A/B — and it is also the family whose
batch trajectories are chaotic at high LR (§1), which is consistent.

**One observation flagged, not claimed.** `nonnormal` trains here in all three
arms (A/B 1.2e-03, C 4.1e-03), whereas B37c's generic-init arm did not
(0.85–0.97). The protocols differ in several ways at once — T=256 vs 64,
chunked updates with hidden-state carry vs full-sequence updates, 1600 vs
400/4000 updates, and unit-variance targets. This is **not** presented as
overturning B37c, which is frozen and used a different protocol; it is recorded
as a difference worth understanding later.

## 5. Verdict

**Q1 — can forward ProductLocal RTRL replace BPTT as an exact gradient engine?**
Yes. Per-chunk agreement of 1e-15 over 19,800 chunks of real training,
parameter/optimizer trajectories identical to floating point, identical final
NMSE and Markov error, zero divergence, with the single chaotic-regime
exception characterized above.

# EXACT CAUSAL PRODUCTLOCAL CREDIT MATCHES BPTT

**Q2 — does the same machinery support useful continuously-updating online
learning?** Mostly. 22/27 runs succeed with zero divergence and fast
adaptation, but two families fail or degrade — one of which (`exact_jordan`)
fails identically under matched BPTT and so is not attributable to online
updating.

# EVERY-TOKEN ONLINE TRAINING: PARTIAL

Not proceeding to Mamba-3. Per the phase plan the next step, if taken, would be
a source-local Mamba-3-like model (complex/selective recurrence + bounded
recurrent tiles + source-local selectors); that is a separate decision.

## Commit hash

See the commit introducing this file.
