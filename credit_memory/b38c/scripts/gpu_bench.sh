#!/usr/bin/env bash
# Step 3: memory / throughput. Identical architecture under BPTT vs exact RTRL.
# BPTT OOM horizons are RECORDED, not worked around by lowering its batch.
set -euo pipefail
# Resolve OUT to an absolute path BEFORE cd, so a relative $B38C_OUT_DIR
# is interpreted relative to where YOU invoke the script.
OUT="${B38C_OUT_DIR:-$PWD/out}"; case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
python bench.py --batch 8 --T 128,512,2048,8192,16384,32768 --N 128,512,2048 \
                --out "$OUT/gpu_bench.json"
