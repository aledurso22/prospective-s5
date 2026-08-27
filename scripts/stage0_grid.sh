#!/usr/bin/env bash
# Submit only the S5 Stage 0 four-cell matrix for one task.
# Usage: bash scripts/stage0_grid.sh <partition> <account> <task> [seeds...]
# Example: bash scripts/stage0_grid.sh gpu myaccount smnist 0 1 2
set -euo pipefail

PARTITION="${1:?usage: stage0_grid.sh <partition> <account> <task> [seeds...]}"
ACCOUNT="${2:?usage: stage0_grid.sh <partition> <account> <task> [seeds...]}"
TASK="${3:?usage: stage0_grid.sh <partition> <account> <task> [seeds...]}"
SEEDS=("${@:4}")
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(0 1 2); fi

case "$TASK" in
    smnist|psmnist|copy) ;;
    *) echo "task must be smnist, psmnist, or copy"; exit 2 ;;
esac

for SEED in "${SEEDS[@]}"; do
    sbatch -p "$PARTITION" -A "$ACCOUNT" \
        --job-name="s5-stage0-${TASK}-s${SEED}" \
        scripts/stage0.sbatch "$TASK" "$SEED"
done
echo "submitted Stage 0 only: $TASK x ${#SEEDS[@]} paired seeds"
echo "No RoutePC arms or large sweep were submitted."
