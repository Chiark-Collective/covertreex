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
RUNS="${RUNS:-1}"
GRID_WHITEN_SCALE="${GRID_WHITEN_SCALE:-}"
ENGINE="python-numba"  # hard lock to reference path
# Optional comparison engine (runs after gold standard). Use COMP_ENGINE=none to skip.
COMP_ENGINE="${COMP_ENGINE:-rust-hilbert}"
RESIDUAL_CHUNK_SIZE="${RESIDUAL_CHUNK_SIZE:-512}"
# Predecessor mode for Vecchia GP queries (neighbor j < query i)
PREDECESSOR_MODE="${PREDECESSOR_MODE:-}"

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
# Explicitly pin the residual radius floor so the Numba pruning path
# (si_cache-based triangle bound) matches the validated configuration.
export COVERTREEX_RESIDUAL_RADIUS_FLOOR=1e-3

calc_stats() {
  python - "$@" <<'PY'
import sys, math
vals = [float(x) for x in sys.argv[1:]]
if not vals:
    print("0.0 0.0")
    sys.exit(0)
mean = sum(vals) / len(vals)
var = sum((v - mean) ** 2 for v in vals) / len(vals)
std = math.sqrt(var)
print(f"{mean:.4f} {std:.4f}")
PY
}

parse_metrics() {
  python - "$1" <<'PY'
import re, sys, json, pathlib
text = pathlib.Path(sys.argv[1]).read_text().splitlines()
line = next((l for l in reversed(text) if l.startswith("pcct |")), None)
if not line:
    sys.exit(1)
m_build = re.search(r"build=([0-9.]+)s", line)
m_qps = re.search(r"throughput=([0-9.,]+)", line)
if not (m_build and m_qps):
    sys.exit(1)
build = float(m_build.group(1))
qps = float(m_qps.group(1).replace(",", ""))
print(f"{build} {qps}")
PY
}

run_suite() {
  local engine_label="$1"
  local enable_rust="$2"
  local summary_path="$3"
  local log_prefix="$4"

  export COVERTREEX_ENABLE_RUST="$enable_rust"

      # Optional residual kernel type for comparison engine.
      RESIDUAL_KERNEL_TYPE="${RESIDUAL_KERNEL_TYPE:-}"
  
      builds=()
      qps_list=()
  
      : > "$summary_path"  # truncate summary for this suite
  
      for i in $(seq 1 "$RUNS"); do
        run_log="${log_prefix}_run${i}.log"
        echo "[run_residual_gold_standard] run $i/$RUNS engine=$engine_label logging to $run_log"
        CMD=(
          python -m cli.pcct query
          --dimension 3
          --tree-points 32768
          --batch-size 512
          --queries 1024
          --k 50
          --metric residual
          --baseline gpboost
          --seed 42
          --engine "$engine_label"
          --residual-chunk-size "$RESIDUAL_CHUNK_SIZE"
        )
        if [[ -n "$RESIDUAL_KERNEL_TYPE" ]]; then
          CMD+=(--residual-kernel-type "$RESIDUAL_KERNEL_TYPE")
        fi
        if [[ -n "$PREDECESSOR_MODE" && "$PREDECESSOR_MODE" == "1" ]]; then
          CMD+=(--predecessor-mode)
        fi
        "${CMD[@]}" | tee "$run_log" | tee -a "$summary_path"
  
    read -r build_val qps_val <<<"$(parse_metrics "$run_log")"
    builds+=("$build_val")
    qps_list+=("$qps_val")
  done

  read -r build_mean build_std <<<"$(calc_stats "${builds[@]}")"
  read -r qps_mean qps_std <<<"$(calc_stats "${qps_list[@]}")"

  best_qps=$(printf '%s\n' "${qps_list[@]}" | sort -nr | head -n1)
  best_idx=1
  for idx in "${!qps_list[@]}"; do
    if [[ "${qps_list[$idx]}" == "$best_qps" ]]; then
      best_idx=$((idx + 1))
      break
    fi
  done

  {
    echo ""
    echo "[run_residual_gold_standard] summary engine=$engine_label runs=$RUNS"
    echo "  build_mean=${build_mean}s build_std=${build_std}s"
    echo "  qps_mean=${qps_mean} q/s qps_std=${qps_std} q/s"
    echo "  best_run=${best_idx} best_qps=${best_qps} q/s log=${log_prefix}_run${best_idx}.log"
  } | tee -a "$summary_path"
}

if [[ -n "$GRID_WHITEN_SCALE" ]]; then
  export COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE="$GRID_WHITEN_SCALE"
  echo "[run_residual_gold_standard] using COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE=$GRID_WHITEN_SCALE"
fi

GOLD_SUMMARY="$LOG_PATH"
run_suite "$ENGINE" 0 "$GOLD_SUMMARY" "${LOG_PATH%.log}"
echo "\nGold-standard residual benchmark logs written to ${LOG_PATH%.log}_run*.log and summary to $GOLD_SUMMARY"

# Optional comparison run (typically Rust) to keep a fresh side-by-side reference.
if [[ -n "$COMP_ENGINE" && "$COMP_ENGINE" != "none" ]]; then
  echo "\n[run_residual_gold_standard] running comparison baseline COMP_ENGINE=$COMP_ENGINE"
  COMP_SUMMARY="${LOG_PATH%.log}_${COMP_ENGINE}.log"
  run_suite "$COMP_ENGINE" 1 "$COMP_SUMMARY" "${LOG_PATH%.log}_${COMP_ENGINE}"
  echo "\nComparison baseline logs written to ${LOG_PATH%.log}_${COMP_ENGINE}_run*.log and summary to $COMP_SUMMARY"
fi
