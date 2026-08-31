#!/usr/bin/env bash
# Local CPU correctness gate + tiny smoke training. No GPU, no cluster.
set -euo pipefail
# Resolve OUT to an absolute path BEFORE cd, so a relative $B38C_OUT_DIR
# is interpreted relative to where YOU invoke the script.
OUT="${B38C_OUT_DIR:-$PWD/out}"; case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
python verify.py 2>&1 | tee "$OUT/verify.log"
python train.py --config configs/smoke_local_f64.json  --out "$OUT/smoke_f64.json"
python train.py --config configs/smoke_local.json      --out "$OUT/smoke_rtrl.json"
python train.py --config configs/smoke_local_bptt.json --out "$OUT/smoke_bptt.json"
