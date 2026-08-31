#!/usr/bin/env bash
# Step 3: memory / throughput. Identical architecture under BPTT vs exact RTRL.
# BPTT OOM horizons are RECORDED, not worked around by lowering its batch.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${B38C_OUT_DIR:-./out}"; mkdir -p "$OUT"
python bench.py --batch 8 --T 128,512,2048,8192,16384,32768 --N 128,512,2048 \
                --out "$OUT/gpu_bench.json"
