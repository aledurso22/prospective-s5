#!/usr/bin/env bash
# Step 1 on the cluster: confirm the SAME committed code is exact on GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${B38C_OUT_DIR:-./out}"; mkdir -p "$OUT"
export B38C_DATA_DIR="${B38C_DATA_DIR:-./data}"
python -c "import jax; print('backend', jax.default_backend(), jax.devices())"
python verify.py                                   2>&1 | tee "$OUT/gpu_verify.log"
python train.py --config configs/gpu_correctness.json --out "$OUT/gpu_correctness.json"
