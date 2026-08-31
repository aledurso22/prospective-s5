# B38c — source-local selective byte-level LM with exact O(P) RTRL

Self-contained. Nothing here imports from the parent research repo; it can be
cloned and run on its own. Python + JAX + NumPy only.

## What this is

A single-layer, byte-level (V=256) selective recurrent language model in which
**every recurrently influential trainable parameter affects exactly one tile**:

    g_{t,τ} = tanh(E_τ[byte_t])                  tile-LOCAL embedding
    Δ_{t,τ} = softplus(uD_τ g + cD_τ),  A_τ = −softplus(Ãtil_τ)
    a_{t,τ} = exp(Δ_{t,τ} A_τ)                   ∈ (0,1) by construction
    b_{t,τ} = uB_τ g + cB_τ
    h_{t+1,τ} = a_{t,τ} ⊙ h_{t,τ} + b_{t,τ}
    logits_t  = W_out h_{t+1} + c                memoryless, does not feed back

There is no trainable dense input projection shared across tiles, no attention,
no second recurrent layer, no Mamba-3 rotation/trapezoidal state, and no
universal-quotient coordinates.

Because Δ and b depend on the byte only (never on h), `J_t = diag(a_t)` is
exactly diagonal, so each parameter's eligibility stays supported on the tile it
influences and the total eligibility is O(P). The instantaneous source
derivatives are written **analytically** (see `model.source_grads`) — there is
no `jacrev` in the hot path.

`init_shared` provides the **Arm C control**: one shared embedding feeding every
tile through trainable per-tile mixing, so each embedding parameter affects all
tiles. Arm C is trained with BPTT only; no exact-RTRL claim is made for it.

## Files

| file | role |
|---|---|
| `model.py` | architecture, analytic `G_t`, reduced source-local RTRL, eligibility accounting |
| `verify.py` | the 6-part correctness gate (float64) |
| `data.py` | enwik8 loader (+ explicit `smoke` fallback that cannot be silently substituted) |
| `train.py` | matched BPTT vs exact RTRL training, in-training gradient checking |
| `bench.py` | memory / throughput, batched-context and streaming regimes |
| `configs/`, `scripts/` | reproducible configs and runners; no machine-specific paths |

Outputs go to `$B38C_OUT_DIR` (default `./out`); data to `$B38C_DATA_DIR`
(default `./data`).

## Local (CPU) — already run, results below

    pip install -r requirements.txt
    ./scripts/local_smoke.sh

## Cluster GPU

    git clone <this-repo> && cd b38c
    pip install --upgrade "jax[cuda12]" numpy
    export B38C_DATA_DIR=/path/to/data     # enwik8 is downloaded here if absent
    export B38C_OUT_DIR=/path/to/out

    ./scripts/gpu_correctness.sh   # 1. same code is exact on GPU (float64)
    ./scripts/gpu_train.sh         # 2. A=local+BPTT, B=local+RTRL, C=shared+BPTT
    ./scripts/gpu_bench.sh         # 3. memory / throughput sweep

`gpu_bench.sh` sweeps T ∈ {128, 512, 2048, 8192, 16384, 32768} and
N ∈ {128, 512, 2048} at a **fixed batch for both algorithms**. If BPTT OOMs, the
runner records the first OOM horizon (`oom: true` plus the error) rather than
lowering BPTT's batch while leaving RTRL unchanged. Adjust the sweep with
`--T`, `--N`, `--batch`.

The benchmark reports, separately and without conflation: tokens/sec,
wall-clock per update, eligibility-state bytes, model bytes, optimizer bytes,
resident input-buffer bytes, XLA scratch, and device peak bytes. **Input-buffer
memory is never reported as eligibility memory.** Both a batched-context regime
(identical resident B×T inputs for both algorithms) and a streaming RTRL regime
(state + eligibility carried across chunks, sequence never materialized) are run.

## Local CPU results (correctness gate — all passed)

| check | result |
|---|---|
| analytic `G_t` vs jacrev/autodiff | **1.04e-16** |
| reduced RTRL = dense RTRL = BPTT, all 8 families | **1.15e-16** |
| 50 matched Adam steps from identical init | **3.33e-16** |
| eligibility vs T (128→8192) | 1.063 MB, **identical at every T** |
| eligibility vs N (32→2048) | `elig/P` **constant at 30.76** |
| XLA scratch − input buffers | **constant 0.1480 MB** at every T |

Smoke training (tiny local corpus, 300 steps, float32): BPTT and RTRL reach
**identical 3.1835 bits/byte** (|Δ| = 8.6e-08, float32 roundoff; the same check
in float64 gives 2.45e-16). CPU wall-clock: RTRL **1.75× slower** than BPTT —
consistent with B38b's 1.7×, and preserved as a negative result. No speed
advantage is claimed; whether analytic tile kernels change this at GPU scale is
exactly what `gpu_bench.sh` is for.
