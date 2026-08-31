#!/usr/bin/env bash
# Detached-friendly runner: all B38c GPU work, sequentially, each step logged
# separately so one failure does not lose the others. Safe under nohup/sbatch.
#
#   B38C_DATA_DIR=/path/data B38C_OUT_DIR=/path/out nohup ./scripts/run_all.sh &
#
# Deliberately NOT `set -e`: a failing arm must not abort the remaining arms.
set -uo pipefail
OUT="${B38C_OUT_DIR:-$PWD/out}"; case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
export B38C_DATA_DIR="${B38C_DATA_DIR:-./data}"

step () {   # step <name> <config-or-cmd...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name ===" | tee -a "$OUT/run_all.log"
  ( "$@" ) 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  echo "=== [$(date +%H:%M:%S)] END   $name (exit $rc) ===" | tee -a "$OUT/run_all.log"
}

echo "backend check:" | tee -a "$OUT/run_all.log"
python -c "import jax; print(jax.default_backend(), jax.devices())" 2>&1 | tee -a "$OUT/run_all.log"

step A_local_bptt  python -u train.py --config configs/gpu_train_local_bptt.json  --out "$OUT/gpu_A_local_bptt.json"
step B_local_rtrl  python -u train.py --config configs/gpu_train_local_rtrl.json  --out "$OUT/gpu_B_local_rtrl.json"
step C_shared_bptt python -u train.py --config configs/gpu_train_shared_bptt.json --out "$OUT/gpu_C_shared_bptt.json"
# bench last: a deliberate OOM at large T must not endanger the training results
step bench         python -u bench.py --batch 8 --T 128,512,2048,8192,16384,32768 \
                                      --N 128,512,2048 --out "$OUT/gpu_bench.json"

echo "ALL DONE $(date)" | tee -a "$OUT/run_all.log"
