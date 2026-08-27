# scripts/ + cluster runbook

SLURM launchers for the S5 benchmark. Machine-specific details
(partition/account) stay OUT of the repo — pass them as arguments.

## Files

| file | what |
|---|---|
| `stage0_grid.sh <partition> <account> <task> [seeds...]` | submits only the paired Stage 0 optimizer/credit matrix; defaults to three seeds |
| `stage0.sbatch` | four matched cells per `(task, seed)`: BPTT/Online x clip 0/1.0; no correction arms |
| `bench_grid.sh <partition> <account> [seeds...]` | submits one job per (task, seed) |
| `bench.sbatch` | the per-(task,seed) job: gates first (baseline, online), inline headroom check (h ≥ 0.2, exit 42 = gate-skip), then mechanism arms + tbptt |
| `train.sbatch` | plain `train.py` wrapper (baseline/prospective models) |

## Cluster setup (once)

```bash
git clone --depth 1 --single-branch --branch s5-routepc \
    https://github.com/aledurso22/prospective-s5.git
cd prospective-s5
python -m venv .venv && source .venv/bin/activate
# GPU build — do NOT rely on nvidia-smi detection from a login node:
pip install "jax[cuda12]==0.11.0" numpy flax==0.12.8 optax==0.2.8
# MNIST must be fetched on the LOGIN node (compute nodes may be offline;
# data/ is gitignored):
python -c "from train import load_mnist; load_mnist(downsample=1)"
```

## Smoke tests (run BEFORE any grid, on one GPU node)

```bash
# correctness gates (CPU-capable, ~15 min total):
python -m tests.test_scan && python -m tests.test_online_s5_jax && \
    python -m tests.test_routepc_jax_meta && \
    python -m tests.test_modal_geometry_convention

# training smokes (tiny, minutes each):
python train_bench.py --task smnist --arm baseline --subset 2000 \
    --epochs 1 --batch-size 32 --d-model 32 --state-size 16 --n-layers 2 \
    --clip 0
python train_bench.py --task smnist --arm online   --subset 2000 \
    --epochs 1 --batch-size 32 --d-model 32 --state-size 16 --n-layers 2 \
    --clip 0
python train_bench.py --task smnist --arm routePC  --subset 2000 \
    --epochs 1 --batch-size 32 --d-model 32 --state-size 16 --n-layers 2 \
    --clip 1.0
python train_bench.py --task smnist --arm baseline --state-prospective \
    --gamma 0 --rho-init 1e-3 --subset 2000 --epochs 1 --batch-size 32 \
    --d-model 32 --state-size 16 --n-layers 2 --clip 0
```

(For state-prospective, use `--rho-init 1e-3` at T~784: the parity
collision scale is ~1/rho_init tokens — the default 0.1 NaNs at full
sMNIST length. This is the known ghost-lane regime, not a wiring bug.)

Check each `results/bench/metrics_*.json` for the `provenance`,
`instrumentation`, and `audit` blocks.

## Stage 0 (required before any correction pilot)

Clipping is an explicit factor. The trainer's historically compatible
default is `--clip 0`; Stage 0 passes both thresholds explicitly.

```bash
bash scripts/stage0_grid.sh <partition> <account> smnist 0 1 2
# after all three paired jobs finish:
python stage0_report.py --task smnist
```

The report requires at least three paired seeds, a positive Online→BPTT
loss gap on every seed, and median relative headroom `h >= 0.2` in the
clipped regime. Passing authorizes only the small correction pilot. It never
authorizes or launches the large sweep.

## Grid launch (expensive sweep — blocked pending Stage 0 + pilot report)

```bash
bash scripts/bench_grid.sh <partition> <account> 0 1 2
# after jobs finish:
python bench_report.py --gate     # per-(task,seed) headroom gate h >= 0.2
python bench_report.py            # bars A/B/C + per-seed tables
```

Per-job resources (bench.sbatch): 1 GPU, 8 CPUs, 48 GB, 48 h. Model:
H=96 N=64 L=3, 3 epochs; copy uses `--seq-len 220`; tbptt window 64.
