#!/usr/bin/env bash
# S5 CCM Phase S1 -- STEP 3, the primary L=2 experiment. Named
# "s5_ccm_rank1.sbatch" per the requested cluster convention, but
# CURRENTLY RUNS ONLY THE READY ARMS (online, s1_full_causal, bptt) --
# S2/S3 (rank-1 frozen/reactive) are not implemented in this preparation
# phase (see README.md "S4+ status" and ccm_rank1.py's "S2/S3
# implementation status"). Re-run this same script once S2/S3 land; the
# ARMS list below is the only line that needs to change.
#
# Usage:
#   sbatch -p <partition> -A <account> \
#       experiments/s5_ccm/slurm/s5_ccm_rank1.sbatch <task> <n_seeds>
#   e.g.
#   sbatch -p gpu -A myaccount experiments/s5_ccm/slurm/s5_ccm_rank1.sbatch smnist 3
#
# Runs the exactness gate first (cheap, fails loudly if S1 is somehow
# broken on this environment before spending the real training budget).
#
#SBATCH --job-name=s5ccm-l2-primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

TASK="${1:-smnist}"
N_SEEDS="${2:-3}"
ARMS=(online s1_full_causal bptt)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  task: $TASK  seeds: $N_SEEDS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

echo "=== exactness gate ==="
srun python -m experiments.s5_ccm.exactness_check

CONFIG="experiments/s5_ccm/configs/l2_primary.json"
for ARM in "${ARMS[@]}"; do
    for SEED in $(seq 0 $((N_SEEDS - 1))); do
        echo "=== $ARM task=$TASK seed=$SEED ==="
        srun python -m experiments.s5_ccm.train \
            --config "$CONFIG" --task "$TASK" --arm "$ARM" --seed "$SEED"
    done
done

echo "done. metrics under \${S5_CCM_RESULTS_ROOT:-experiments/s5_ccm/results}/"
