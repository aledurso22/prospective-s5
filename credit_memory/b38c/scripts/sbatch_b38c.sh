#!/usr/bin/env bash
# B38c GPU phase as a batch job.
#
# Site values below are real for the PGI cluster (partition/nodelist confirmed
# from `sinfo -s`). The node is PINNED because /Local is node-local storage:
# the repo, venv and enwik8 live on pgi15-gpu3 and exist nowhere else.
#
# Submit with (paths passed in, nothing machine-specific hard-coded):
#   cd <this b38c directory>
#   export B38C_DATA_DIR=/path/data B38C_OUT_DIR=/path/out
#   sbatch -o ~/b38c_slurm_%j.out scripts/sbatch_b38c.sh
# Send Slurm's own output somewhere SHARED (e.g. $HOME) if /Local is node-local,
# otherwise you cannot read it from the head node when the job fails.
# (the venv is taken from $VIRTUAL_ENV if you submit with it activated)
#
#SBATCH --job-name=b38c
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodelist=pgi15-gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_b38c_%j.out
set -uo pipefail
# NOTE: under sbatch, $0 is Slurm's spooled copy of this script, NOT its path in
# the repo -- so `dirname $0` must not be used to locate the package. Prefer the
# submit directory; fall back to $0 only for direct (non-sbatch) invocation.
ROOT="${B38C_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$ROOT" || { echo "ERROR: cannot cd to ROOT=$ROOT"; exit 2; }
if [ ! -x ./scripts/run_all.sh ]; then
  echo "ERROR: ./scripts/run_all.sh not found under ROOT=$ROOT"
  echo "       (submit from the b38c directory, or set B38C_ROOT to it)"
  ls -la . ; exit 2
fi
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
echo "ROOT=$ROOT  VENV=$B38C_VENV"
echo "DATA=$B38C_DATA_DIR  OUT=$B38C_OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec ./scripts/run_all.sh
