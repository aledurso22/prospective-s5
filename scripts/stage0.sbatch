#!/usr/bin/env bash
# S5 Stage 0: paired optimizer/credit regime matrix for one (task, seed).
# Exactly four cells, with clipping stated explicitly:
#   baseline/BPTT x clip {0, 1.0}; online x clip {0, 1.0}.
# This job does not launch RoutePC or any large sequence-length sweep.
#
# Submit through scripts/stage0_grid.sh, or directly:
#   sbatch -p <partition> -A <account> scripts/stage0.sbatch smnist 0
#
#SBATCH --job-name=s5-stage0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-stage0-%x-%j.out
set -euo pipefail

TASK="${1:?usage: sbatch scripts/stage0.sbatch {smnist|psmnist|copy} SEED}"
SEED="${2:?usage: sbatch scripts/stage0.sbatch TASK SEED}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  task: $TASK  seed: $SEED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

if [ "$TASK" = "copy" ]; then
    TASKFLAGS="--seq-len 220 --copy-k 10 --copy-alpha 8 --subset 20000 --test-subset 2000 --eval-subset 2000"
else
    TASKFLAGS="--subset 0 --test-subset 10000 --eval-subset 2000"
fi
MODELFLAGS="--d-model 96 --state-size 64 --n-layers 3 --dropout 0.1 --batch-size 64 --lr 1e-3 --scan assoc --epochs 3"

run() {  # run ARM CLIP TAG
    local ARM="$1"
    local CLIP="$2"
    local TAG="$3"
    echo "=== Stage 0: $TASK s$SEED $ARM clip=$CLIP ==="
    srun python train_bench.py --task "$TASK" --arm "$ARM" --seed "$SEED" \
        --clip "$CLIP" --tag "$TAG" $MODELFLAGS $TASKFLAGS
}

run baseline 0   _stage0_clip0
run online   0   _stage0_clip0
run baseline 1.0 _stage0_clip1
run online   1.0 _stage0_clip1

echo "Stage 0 cells complete for $TASK seed $SEED."
echo "After all paired seeds finish: python stage0_report.py --task $TASK"
