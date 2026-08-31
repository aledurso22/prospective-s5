#!/usr/bin/env bash
# Step 1 on the cluster: confirm the SAME committed code is exact on GPU.
set -euo pipefail
# Resolve OUT to an absolute path BEFORE cd, so a relative $B38C_OUT_DIR
# is interpreted relative to where YOU invoke the script.
OUT="${B38C_OUT_DIR:-$PWD/out}"; case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
export B38C_DATA_DIR="${B38C_DATA_DIR:-./data}"
python -c "import jax; print('backend', jax.default_backend(), jax.devices())"
python verify.py                                   2>&1 | tee "$OUT/gpu_verify.log"
python train.py --config configs/gpu_correctness.json --out "$OUT/gpu_correctness.json"
