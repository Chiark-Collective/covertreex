#!/usr/bin/env bash
set -euo pipefail

# Gold standard dense benchmark
export COVERTREEX_ENABLE_NUMBA=1
export COVERTREEX_BATCH_ORDER=natural
export COVERTREEX_PREFIX_SCHEDULE=doubling
export COVERTREEX_ENABLE_DIAGNOSTICS=0
export COVERTREEX_SCOPE_CHUNK_TARGET=0

echo "Running Dense Baseline Benchmark..."

python -m cli.pcct benchmark \
  --dimension 3 \
  --tree-points 32768 \
  --batch-size 512 \
  --queries 1024 \
  --k 50 \
  --metric residual \
  --baseline none \
  --seed 42 \
  --repeat 1 \
  --log-file artifacts/benchmarks/residual_dense_telemetry.jsonl

echo "Benchmark complete."
