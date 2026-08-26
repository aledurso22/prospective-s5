#!/usr/bin/env bash
# SLURM batch script — benchmark pipeline, one (task, seed) per job.
#
# Runs sequentially inside the job:
#   1. GATE arms: baseline (exact BPTT) + online
#   2. headroom gate: h = (L_online - L_baseline)/L_online >= 0.2,
#      else the job stops (mechanism arms are meaningless without
#      headroom — the D6 lesson)
#   3. mechanism arms: routeA, scalarLive, routePC (exploratory),
#      frozenPhase, frozenMag (w from THIS job's routeA run), tbptt
#
# Submit via scripts/bench_grid.sh, or directly:
#   sbatch -p <partition> -A <account> scripts/bench.sbatch smnist 0
#   sbatch -p <partition> -A <account> scripts/bench.sbatch copy 0
#
#SBATCH --job-name=pssm-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-bench-%x-%j.out
set -euo pipefail

TASK="${1:?usage: sbatch scripts/bench.sbatch {smnist|psmnist|copy} SEED}"
SEED="${2:?usage: sbatch scripts/bench.sbatch TASK SEED}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  task: $TASK  seed: $SEED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

# task configs ------------------------------------------------------------
if [ "$TASK" = "copy" ]; then
    TASKFLAGS="--seq-len 220 --copy-k 10 --copy-alpha 8 --subset 20000 --test-subset 2000 --eval-subset 2000"
else
    # full sequential MNIST: T=784, 60k train, final eval on all 10k
    TASKFLAGS="--subset 0 --test-subset 10000 --eval-subset 2000"
fi
MODELFLAGS="--d-model 96 --state-size 64 --n-layers 3 --dropout 0.1 --batch-size 64 --lr 1e-3 --scan assoc --epochs 3"

run() {  # run ARM [extra flags]
    local ARM="$1"; shift
    echo "=== $TASK s$SEED $ARM ==="
    srun python train_bench.py --task "$TASK" --arm "$ARM" --seed "$SEED" \
        $MODELFLAGS $TASKFLAGS "$@"
}

# 1. gates -----------------------------------------------------------------
run baseline
run online

GATE_RC=0
python - "$TASK" "$SEED" <<'EOF' || GATE_RC=$?
import json, os, sys
task, seed = sys.argv[1], int(sys.argv[2])
base = f"results/bench/metrics_{task}_%s_s{seed}.json"
lo = json.load(open(base % "online"))["final_test_loss"]
lb = json.load(open(base % "baseline"))["final_test_loss"]
h = (lo - lb) / lo
print(f"headroom gate: online {lo:.4f} baseline {lb:.4f} h {h:.3f} "
      f"(bar 0.2)")
if h < 0.2:
    print("GATE FAIL — skipping mechanism arms for this (task, seed)")
    sys.exit(42)   # distinct code: gate-skip, not a crash
EOF
if [ "$GATE_RC" -eq 42 ]; then
    echo "gate not passed; job done (gates only)."
    exit 0
elif [ "$GATE_RC" -ne 0 ]; then
    echo "gate check itself failed (rc=$GATE_RC)"; exit "$GATE_RC"
fi

# 2. mechanism arms ---------------------------------------------------------
run routeA
run scalarLive
run routePC
run routePCreal
run routePCphase   # opt-in exploratory (C1: unit-modulus variant)
WFILE="results/bench/w_${TASK}_routeA_s${SEED}.npz"
run frozenPhase --w-file "$WFILE"
run frozenMag   --w-file "$WFILE"
run tbptt --tbptt-window 64

echo "done. metrics under results/bench/; aggregate with bench_report.py"
