#!/usr/bin/env bash
# B38c GPU phase as a batch job.
#
# Site values below are real for the PGI cluster (partition/nodelist confirmed
# from `sinfo -s`). The node is PINNED because /Local is node-local storage:
# the repo, venv and enwik8 live on pgi15-gpu3 and exist nowhere else.
#
# Submit with (paths passed in, nothing machine-specific hard-coded):
#   sbatch --export=ALL,B38C_VENV=/Local/durso/prospective_ssm_project/.venv,\
#B38C_DATA_DIR=/Local/durso/data,B38C_OUT_DIR=/Local/durso/b38c_out \
#     --output=/Local/durso/b38c_out/slurm_%j.out scripts/sbatch_b38c.sh
#
#SBATCH --job-name=b38c
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_b38c_%j.out
set -uo pipefail
cd "$(dirname "$0")/.."
: "${B38C_VENV:?set B38C_VENV (path to the virtualenv)}"
: "${B38C_DATA_DIR:?set B38C_DATA_DIR}"
: "${B38C_OUT_DIR:?set B38C_OUT_DIR}"
source "$B38C_VENV/bin/activate"
export B38C_DATA_DIR B38C_OUT_DIR
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec ./scripts/run_all.sh
