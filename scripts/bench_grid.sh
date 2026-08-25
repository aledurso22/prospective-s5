#!/usr/bin/env bash
# Submit the benchmark grid: 3 tasks x 3 seeds, one job each
# (each job self-sequences gates -> headroom gate -> mechanism arms;
# see scripts/bench.sbatch).
#
# Usage:  bash scripts/bench_grid.sh <partition> <account> [seeds...]
# Example: bash scripts/bench_grid.sh gpu alessandro 0 1 2
#
# Afterwards, on the login node:
#   python bench_report.py --gate     # headroom summary -> gate.json
#   python bench_report.py            # full table + registered bars A/B/C
set -euo pipefail

PARTITION="${1:?usage: bench_grid.sh <partition> <account> [seeds...]}"
ACCOUNT="${2:?usage: bench_grid.sh <partition> <account> [seeds...]}"
SEEDS=("${@:3}")
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(0 1 2); fi

for TASK in smnist psmnist copy; do
    for SEED in "${SEEDS[@]}"; do
        sbatch -p "$PARTITION" -A "$ACCOUNT" \
            --job-name="pssm-bench-${TASK}-s${SEED}" \
            scripts/bench.sbatch "$TASK" "$SEED"
    done
done
echo "submitted 3 x ${#SEEDS[@]} jobs."
