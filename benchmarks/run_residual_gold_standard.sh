#!/usr/bin/env bash
set -euo pipefail

# Reproduce the “gold standard” residual benchmark (32 768 points / d=3 / 1 024 queries / k=50)
# that yielded the 24.20 s build / 0.046 s query result on 2025‑11‑17.
#
# This script now *always* runs the Python/Numba traversal for the primary gold-standard run,
# regardless of environment or compiled Rust extensions. We explicitly disable the Rust path
# to keep the reference numbers stable and comparable over time.
#
# A configurable comparison run can be executed afterwards via COMP_ENGINE (default: rust-hilbert).
# Set COMP_ENGINE=none to skip the comparison, or override to any supported engine
# (python-numba | rust-natural | rust-hybrid | rust-hilbert | rust-fast).
#
# Usage:
#   ./benchmarks/run_residual_gold_standard.sh [output_log]
#   COMP_ENGINE=rust-fast ./benchmarks/run_residual_gold_standard.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${1:-$ROOT_DIR/bench_residual.log}"
GRID_WHITEN_SCALE="${GRID_WHITEN_SCALE:-}"
ENGINE="python-numba"  # hard lock to reference path
# Optional comparison engine (runs after gold standard). Use COMP_ENGINE=none to skip.
COMP_ENGINE="${COMP_ENGINE:-rust-hilbert}"
RESIDUAL_CHUNK_SIZE="${RESIDUAL_CHUNK_SIZE:-512}"

# Ensure we run with the default dense traversal / no chunking knobs, since
# the reference result was captured before the sparse/Numba paths were enabled.
# Explicitly disable Rust to force Python/Numba traversal even if the extension is present.
unset COVERTREEX_ENABLE_SPARSE_TRAVERSAL
export COVERTREEX_SCOPE_CHUNK_TARGET=0
export COVERTREEX_ENABLE_NUMBA=1
export COVERTREEX_ENABLE_RUST=0
export COVERTREEX_BATCH_ORDER=natural
export COVERTREEX_PREFIX_SCHEDULE=doubling
export COVERTREEX_ENABLE_DIAGNOSTICS=0

if [[ -n "$GRID_WHITEN_SCALE" ]]; then
  export COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE="$GRID_WHITEN_SCALE"
  echo "[run_residual_gold_standard] using COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE=$GRID_WHITEN_SCALE"
fi

ENGINE_ARGS=(--engine "$ENGINE" --residual-chunk-size "$RESIDUAL_CHUNK_SIZE")

python -m cli.pcct query \
  --dimension 3 \
  --tree-points 32768 \
  --batch-size 512 \
  --queries 1024 \
  --k 50 \
  --metric residual \
  --baseline gpboost \
  --seed 42 \
  "${ENGINE_ARGS[@]}" | tee "$LOG_PATH"

echo "\nGold-standard residual benchmark log written to $LOG_PATH"

# Optional comparison run (typically Rust) to keep a fresh side-by-side reference.
if [[ -n "$COMP_ENGINE" && "$COMP_ENGINE" != "none" ]]; then
  echo "\n[run_residual_gold_standard] running comparison baseline COMP_ENGINE=$COMP_ENGINE"
  COMP_LOG_PATH="${LOG_PATH%.log}_${COMP_ENGINE}.log"
  # Allow Rust for comparison; keep other knobs aligned.
  export COVERTREEX_ENABLE_RUST=1
  COMP_ENGINE_ARGS=(--engine "$COMP_ENGINE" --residual-chunk-size "$RESIDUAL_CHUNK_SIZE")
  python -m cli.pcct query \
    --dimension 3 \
    --tree-points 32768 \
    --batch-size 512 \
    --queries 1024 \
    --k 50 \
    --metric residual \
    --baseline gpboost \
    --seed 42 \
    "${COMP_ENGINE_ARGS[@]}" | tee "$COMP_LOG_PATH"
  echo "\nComparison baseline log written to $COMP_LOG_PATH"
fi
