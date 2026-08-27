#!/usr/bin/env bash
# S5 CCM Phase S1 -- STEP 1 of the recommended sbatch sequence. Verifies
# g_full_causal == g_BPTT on a tiny CPU-fast config before anything else
# runs. Cheap (seconds); still requested via --gres=gpu:1 so it exercises
# the SAME environment/device the real runs will use, catching env
# problems early.
#
# Usage:
#   sbatch -p <partition> -A <account> experiments/s5_ccm/slurm/s5_ccm_exactness.sbatch
#
#SBATCH --job-name=s5ccm-exactness
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

srun python -m experiments.s5_ccm.exactness_check

echo "Exactness gate passed. Next: sbatch experiments/s5_ccm/slurm/s5_ccm_smoke.sbatch"
