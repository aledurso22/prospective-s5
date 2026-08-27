#!/usr/bin/env bash
# B5 full small matrix: A0/A1/A2 x clip{0,1} x N_SEEDS paired seeds
# (default 8). Pure-numpy toy rig, no GPU needed. Run scripts/b5_pilot.sh
# FIRST and confirm it looks healthy before running this.
#
# Usage:
#   bash scripts/b5_full_matrix.sh            # 8 seeds (default)
#   bash scripts/b5_full_matrix.sh 10          # 10 seeds
#   bash scripts/b5_full_matrix.sh 8 --force   # rerun everything
set -euo pipefail
cd "$(dirname "$0")/.."

N_SEEDS="${1:-8}"
FORCE=0
if [ "${2:-}" = "--force" ]; then FORCE=1; fi

OUTDIR="results/credit_memory/b5"
mkdir -p "$OUTDIR"

SEEDS=$(seq 0 $((N_SEEDS - 1)))

for ARM in online b4_arch b4_causal bptt; do
    for CLIP in 0 1; do
        for SEED in $SEEDS; do
            OUT="$OUTDIR/b5_${ARM}_clip${CLIP}_s${SEED}.json"
            if [ -f "$OUT" ] && [ "$FORCE" -eq 0 ]; then
                echo "skip (exists): $OUT"
                continue
            fi
            .venv/bin/python -m credit_memory.b5_train \
                --arm "$ARM" --seed "$SEED" --clip "$CLIP" --out "$OUT"
        done
    done
done

echo "Full matrix done ($N_SEEDS seeds). Summarize with:"
echo "  .venv/bin/python -m credit_memory.b5_report --clip 0 --arms online,b4_arch,b4_causal,bptt"
echo "  .venv/bin/python -m credit_memory.b5_report --clip 1 --arms online,b4_arch,b4_causal,bptt"
