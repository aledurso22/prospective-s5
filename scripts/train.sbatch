#!/usr/bin/env bash
# SLURM batch script — sequential MNIST, one arm per job.
#
# Setup once on the login node (builds .venv with the CUDA build of JAX):
#     bash setup.sh
#
# Then submit one job per arm:
#     sbatch -p <partition> -A <account> scripts/train.sbatch baseline
#     sbatch -p <partition> -A <account> scripts/train.sbatch prospective "--rho-init 1e-3 --tag _rho1e-3"
#
# NOTE: the prospective arm runs the derivation verbatim — there is no
# stabilizer — so its loss goes NaN within the first steps (the forward
# pass overflows float32 ~14 steps into the 784-step recurrence; see
# exact_failure.py). That divergence IS the experiment, not a failed run.
#
# The 2nd argument is appended verbatim to train.py, so it can override any
# flag set below (later flags win in argparse).
#
#SBATCH --job-name=pssm-smnist
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

MODEL="${1:?usage: sbatch scripts/train.sbatch {baseline|prospective} [extra flags]}"
EXTRA="${2:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}   partition: ${SLURM_JOB_PARTITION:-?}"
echo "model     : $MODEL   extra: ${EXTRA:-(none)}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

# NOTE: train.py has no checkpoint/resume. If the job hits --time the run is
# lost, so raise the limit rather than restarting.

# Full sequential MNIST at the repo's default budget: T=784 (no downsampling),
# 60k train, H=96 N=64 L=3, 3 epochs, associative scan (the GPU-appropriate
# choice — "lax" is only faster on few CPU cores).
srun python train.py --model "$MODEL" \
    --epochs 3 --subset 0 --test-subset 10000 --eval-subset 2000 \
    --batch-size 64 --d-model 96 --state-size 64 --n-layers 3 \
    --lr 1e-3 --dropout 0.1 --seed 0 --scan assoc $EXTRA

echo "done. metrics written under results/"
