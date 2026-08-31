#!/usr/bin/env bash
# Step 2: matched language training. A = BPTT, B = exact RTRL, C = shared-selector BPTT.
set -euo pipefail
# Resolve OUT to an absolute path BEFORE cd, so a relative $B38C_OUT_DIR
# is interpreted relative to where YOU invoke the script.
OUT="${B38C_OUT_DIR:-$PWD/out}"; case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
export B38C_DATA_DIR="${B38C_DATA_DIR:-./data}"
python train.py --config configs/gpu_train_local_bptt.json  --out "$OUT/gpu_A_local_bptt.json"
python train.py --config configs/gpu_train_local_rtrl.json  --out "$OUT/gpu_B_local_rtrl.json"
python train.py --config configs/gpu_train_shared_bptt.json --out "$OUT/gpu_C_shared_bptt.json"
