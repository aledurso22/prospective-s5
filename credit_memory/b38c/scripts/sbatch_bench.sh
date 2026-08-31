#!/usr/bin/env bash
# Bench-only batch job (training already done; see slurm log of the prior run).
#
#   cd <b38c dir>
#   export B38C_VENV=/path/.venv B38C_OUT_DIR=/path/out
#   sbatch -o ~/b38c_bench_%j.out scripts/sbatch_bench.sh
#
# NOTE: do NOT ssh/srun into the compute node while this runs. With
# pam_slurm_adopt your login session is adopted into the job, and ending it
# kills the job. Read ~/b38c_bench_<jobid>.out from the head node instead.
#
#SBATCH --job-name=b38c_bench
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_b38c_bench_%j.out
set -uo pipefail
ROOT="${B38C_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$ROOT" || { echo "ERROR: cannot cd to ROOT=$ROOT"; exit 2; }
[ -f ./bench.py ] || { echo "ERROR: bench.py not under ROOT=$ROOT"; ls -la .; exit 2; }
B38C_VENV="${B38C_VENV:-${VIRTUAL_ENV:-}}"
: "${B38C_VENV:?set B38C_VENV, or submit from an activated virtualenv}"
: "${B38C_OUT_DIR:?set B38C_OUT_DIR}"
source "$B38C_VENV/bin/activate"
mkdir -p "$B38C_OUT_DIR"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec python -u bench.py --batch 8 --T 128,512,2048,8192,16384,32768 \
                        --N 128,512,2048 --out "$B38C_OUT_DIR/gpu_bench.json"
