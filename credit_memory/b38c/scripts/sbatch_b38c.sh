#!/usr/bin/env bash
# Slurm template. EDIT the marked lines for your cluster (partition/account/time),
# then:  sbatch scripts/sbatch_b38c.sh
#SBATCH --job-name=b38c
#SBATCH --gres=gpu:1                 # <-- EDIT if your site uses a different GPU flag
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00              # <-- EDIT
#SBATCH --output=b38c_%j.out
##SBATCH --partition=<your-partition> # <-- UNCOMMENT AND EDIT
##SBATCH --account=<your-account>     # <-- UNCOMMENT AND EDIT
set -uo pipefail
cd "$(dirname "$0")/.."
source "${B38C_VENV:?set B38C_VENV to your venv path}/bin/activate"
export B38C_DATA_DIR="${B38C_DATA_DIR:?set B38C_DATA_DIR}"
export B38C_OUT_DIR="${B38C_OUT_DIR:?set B38C_OUT_DIR}"
./scripts/run_all.sh
