#!/usr/bin/env bash
set -euo pipefail

# Reproduce the “gold standard” residual benchmark (32 768 points / d=3 / 1 024 queries / k=50)
# that yielded the 24.20 s build / 0.046 s query result on 2025‑11‑17.
#
# Usage: ./benchmarks/run_residual_gold_standard.sh [output_log]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${1:-$ROOT_DIR/bench_residual.log}"
GRID_WHITEN_SCALE="${GRID_WHITEN_SCALE:-}"

# Ensure we run with the default dense traversal / no chunking knobs, since
# the reference result was captured before the sparse/Numba paths were enabled.
unset COVERTREEX_ENABLE_SPARSE_TRAVERSAL
export COVERTREEX_SCOPE_CHUNK_TARGET=0
export COVERTREEX_ENABLE_NUMBA=1
export COVERTREEX_BATCH_ORDER=natural
export COVERTREEX_PREFIX_SCHEDULE=doubling
export COVERTREEX_ENABLE_DIAGNOSTICS=0

if [[ -n "$GRID_WHITEN_SCALE" ]]; then
  export COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE="$GRID_WHITEN_SCALE"
  echo "[run_residual_gold_standard] using COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE=$GRID_WHITEN_SCALE"
fi

python "$ROOT_DIR/benchmarks/queries.py" \
  --dimension 3 \
  --tree-points 32768 \
  --batch-size 512 \
  --queries 1024 \
  --k 50 \
  --metric residual \
  --baseline gpboost \
  --seed 42 | tee "$LOG_PATH"

echo "\nGold-standard residual benchmark log written to $LOG_PATH"
