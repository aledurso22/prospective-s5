#!/usr/bin/env bash
# S5 CCM Phase S1 -- STEP 2 of the recommended sbatch sequence. Tiny
# smoke run (all 3 ready arms, 20 optimizer steps each, the "copy" task)
# to confirm the training loop itself runs end-to-end on the actual GPU
# node before committing to the primary L=2 experiment's wall-clock
# budget.
#
# Usage:
#   sbatch -p <partition> -A <account> experiments/s5_ccm/slurm/s5_ccm_smoke.sbatch
#
#SBATCH --job-name=s5ccm-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

for ARM in online s1_full_causal bptt; do
    echo "=== smoke: $ARM ==="
    srun python -m experiments.s5_ccm.train \
        --config experiments/s5_ccm/configs/l2_smoke.json --arm "$ARM"
done

echo "Smoke run done. Next: sbatch experiments/s5_ccm/slurm/s5_ccm_rank1.sbatch"
echo "(that launcher currently runs 'online' and 's1_full_causal' -- see"
echo " experiments/s5_ccm/README.md 'S4+ status' / ccm_rank1.py for why"
echo " rank1/rank2/rank4/rank8 are not yet available)."
