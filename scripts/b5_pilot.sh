#!/usr/bin/env bash
# B5 primary pilot: 5 paired seeds, A0 (online) vs A2 (b4_causal), clip=0
# only. Pure-numpy toy rig (toyrig/ssm_rig.py via credit_memory/b5_train.py)
# -- no GPU needed, runs in well under a minute total on a laptop CPU.
#
# Usage:
#   bash scripts/b5_pilot.sh                 # CPU, all 5 seeds
#   CUDA_VISIBLE_DEVICES=0 bash scripts/b5_pilot.sh   # harmless no-op here
#                                                       (this pilot has no
#                                                       GPU code path; the
#                                                       env var is honored
#                                                       for consistency with
#                                                       future S5 launchers,
#                                                       see PHASE_B5.md)
#
# Resume: idempotent -- re-running skips any (arm, seed) whose output JSON
# already exists. Delete the specific file under results/credit_memory/b5/
# to force a rerun of just that cell, or pass --force to rerun everything.
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; fi

OUTDIR="results/credit_memory/b5"
mkdir -p "$OUTDIR"

SEEDS="0 1 2 3 4"
CLIP=0

for ARM in online b4_causal; do
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

echo "Pilot done. Summarize with:"
echo "  .venv/bin/python -m credit_memory.b5_report --clip 0 --arms online,b4_causal"
