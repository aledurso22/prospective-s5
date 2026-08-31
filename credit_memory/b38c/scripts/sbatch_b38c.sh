#!/usr/bin/env bash
# B38c GPU phase as a batch job.
#
# Site values below are real for the PGI cluster (partition/nodelist confirmed
# from `sinfo -s`). The node is PINNED because /Local is node-local storage:
# the repo, venv and enwik8 live on pgi15-gpu3 and exist nowhere else.
#
# Submit with (paths passed in, nothing machine-specific hard-coded):
#   export B38C_DATA_DIR=/path/data B38C_OUT_DIR=/path/out
#   sbatch -o /path/out/slurm_%j.out scripts/sbatch_b38c.sh
# (the venv is taken from $VIRTUAL_ENV if you submit with it activated)
#
#SBATCH --job-name=b38c
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_b38c_%j.out
set -uo pipefail
cd "$(dirname "$0")/.."
# Default the venv to whatever was active at submit time ($VIRTUAL_ENV is
# exported, and sbatch forwards the submitting environment by default), so the
# submit line stays short enough not to be broken by terminal line-wrapping.
B38C_VENV="${B38C_VENV:-${VIRTUAL_ENV:-}}"
: "${B38C_VENV:?set B38C_VENV, or submit from an activated virtualenv}"
: "${B38C_DATA_DIR:?set B38C_DATA_DIR}"
: "${B38C_OUT_DIR:?set B38C_OUT_DIR}"
source "$B38C_VENV/bin/activate"
export B38C_DATA_DIR B38C_OUT_DIR
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec ./scripts/run_all.sh
