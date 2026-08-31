# Phase B38c — source-local selective language model (local stage)

Branch `S5-CCM-scale-validation`. B37a–c, B38a and B38b are frozen and untouched.
This commit delivers the **local correctness + cluster-ready implementation**.
The GPU phase runs after cloning; its commands are in `credit_memory/b38c/README.md`.

Self-contained package: `credit_memory/b38c/` imports nothing from the parent
repo and can be cloned and run on its own (JAX + NumPy only).

## Model

Single-layer, byte-level (V=256), state split into `J` tiles of bounded size:

    g_{t,τ} = tanh(E_τ[byte_t])                  tile-LOCAL trainable embedding
    Δ_{t,τ} = softplus(uD_τ g + cD_τ),  A_τ = −softplus(Ãtil_τ)
    a_{t,τ} = exp(Δ_{t,τ} A_τ) ∈ (0,1)           by construction, no projection
    b_{t,τ} = uB_τ g + cB_τ
    h_{t+1,τ} = a_{t,τ} ⊙ h_{t,τ} + b_{t,τ},   logits_t = W_out h_{t+1} + c

Every recurrently influential parameter (`E, Ãtil, uD, cD, uB, cB`) affects
exactly one tile; there is no trainable dense input projection shared across
tiles. The dense readout is memoryless and does not feed back. No attention, no
second recurrent layer, no Mamba-3 rotations/trapezoidal state, no universal
quotient coordinates.

**Arm C control** (`init_shared`): one shared embedding feeding every tile
through trainable per-tile mixing, so each embedding parameter affects all
tiles. Embedding width `q_s = J·q` keeps the embedding parameter count matched.
Trained with BPTT only — no exact-RTRL claim is made for Arm C.

## Analytic instantaneous source derivatives

B38b showed per-tile `jacrev` dominated CPU runtime, so `G_t` is written in
closed form. With `α_k := h_k A_k a_k σ(s_k)` (pre-update `h`):

    ∂/∂cD_k = α_k        ∂/∂uD_{ki} = α_k g_i      ∂/∂Ãtil_k = −h_k Δ_k a_k σ(Ãtil_k)
    ∂/∂cB_k = 1          ∂/∂uB_{ki} = g_i
    ∂/∂E_{v,i} = [α_k uD_{ki} + uB_{ki}](1 − g_i²) · 1[byte_t = v]

No `jacrev` in the hot path. The RTRL kernel is `batch × tiles × local state`
with no Python loop over tiles or parameters; the only sequential dependency is
`lax.scan` over time.

## Local correctness gate — all six checks pass

| check | result |
|---|---|
| 1. analytic `G_t` vs jacrev/autodiff (4 configs, all 6 families) | **1.041e-16** |
| 2. reduced RTRL = dense RTRL = BPTT (4 configs, all 8 families) | **1.150e-16** |
| 3. 50 matched Adam steps from identical init | **3.331e-16** |
| 4. eligibility vs T = 128…8192 | 1.063 MB, **identical at every T** |
| 5. eligibility vs N = 32…2048 | `elig/P` **constant at 30.76** |
| 6. XLA scratch − resident input buffers | **constant 0.1480 MB** at every T |

Check 2 covers every recurrent family separately including the tile embeddings
`E`, plus the readout. Check 6 is the "no accidental O(T) scratch" requirement:
total scratch does grow with T, but subtracting the resident input buffers
leaves exactly 0.1480 MB at T = 64, 256, 1024 and 4096 — the T-dependence is
input buffers, which are **not** eligibility memory.

**Eligibility composition** (a real systems finding): the tile-local embedding
dominates at `B·J·d·V·q`, while all selector eligibility together is only
`B·J·d·(3+2q)`. The ratio `M_elig/P` is nonetheless constant in N — O(P) holds,
with a large constant set by `B·d`. A lazy-decay optimization for embedding rows
is possible but was not implemented; correctness was the goal here.

## Local smoke training (CPU)

Tiny local corpus, J=16, d=4, q=2, B=8, T=128, 300 steps, float32:

| arm | val CE (nats) | bits/byte | tok/s | wall |
|---|---|---|---|---|
| A — local + BPTT | 2.206637 | **3.1835** | 87,077 | 3.5 s |
| B — local + exact RTRL | 2.206637 | **3.1835** | 49,682 | 6.2 s |

`|Δ bpb| = 8.6e-08`, i.e. float32 roundoff; the same in-training check in
float64 gives **2.453e-16**. Learning is real (4.33 → 3.18 bpb against an 8.0
bpb uniform-byte baseline) but this is a *smoke test on a tiny corpus*, not a
language-quality result — that is the cluster run's job.

**Negative result preserved:** CPU wall-clock RTRL is **1.75× slower** than
matched BPTT, consistent with B38b's 1.7×. No speed advantage is preregistered
or claimed; whether analytic tile kernels change this at GPU parallel scale is
precisely what the GPU benchmark is for. **Do not infer GPU behaviour from
these CPU timings.**

## Dataset honesty

`data.py` targets **enwik8** and downloads it into `$B38C_DATA_DIR`. The tiny
local corpus is a *separate, explicitly named* `smoke` dataset: with
`require_enwik8: true` (the default in every GPU config) a missing enwik8 is a
hard error, so the final dataset cannot be silently substituted. Every result
JSON records the dataset name, path, byte count and content hash.

## Cluster commands

    git clone <repo> && cd credit_memory/b38c
    pip install --upgrade "jax[cuda12]" numpy
    export B38C_DATA_DIR=/path/to/data ; export B38C_OUT_DIR=/path/to/out
    ./scripts/gpu_correctness.sh    # same code exact on GPU, float64
    ./scripts/gpu_train.sh          # A local+BPTT, B local+RTRL, C shared+BPTT
    ./scripts/gpu_bench.sh          # T∈{128..32768}, N∈{128,512,2048}, fixed batch

The benchmark reports tokens/sec, wall/update, eligibility bytes, model bytes,
optimizer bytes, input-buffer bytes, XLA scratch and device peak separately, in
two regimes (batched-context with identical resident B×T inputs; streaming RTRL
with state + eligibility carried and the sequence never materialized). If BPTT
OOMs, the first OOM horizon is recorded rather than quietly lowering its batch.

## Status

- **Credit correctness: established locally.** exact RTRL = dense RTRL = BPTT to
  1.15e-16 in float64 for every recurrent family including tile embeddings.
- **Language viability: not yet answered.** Smoke training learns, but the real
  enwik8 run is a cluster job.
- **Systems result: not yet answered.** CPU shows a 1.75× RTRL slowdown; the GPU
  measurement is pending and is not predicted here.
- **Representational cost of locality (Arm C): not yet answered.** The control is
  implemented and configured but must be run on the cluster.

Not starting Mamba-3, Transformer hybrids, or large-scale hyperparameter search.

## Commit hash

See the commit introducing this file.
