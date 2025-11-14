# Residual Dense Baseline Regression (2025-11-12)

The full investigative log now lives under [`docs/journal/2025-11-12_residual_dense_regression.md`](journal/2025-11-12_residual_dense_regression.md) so this document can stay focused on the current status and entry points back into the history.

## Current Snapshot — 2025-11-17

- Dense residual builds hit **≈29 s total traversal** on the maskopt_v2 preset at HEAD (`pcct-20251114-082601-d2e6df`), using the defaults below.
- Dynamic query blocks are **enabled by default**; disable via `--residual-dynamic-query-block 0` if you need to reproduce the 4 k guardrail numbers.
- Latest scaling checkpoints:
  - 32 k: `artifacts/benchmarks/residual_dense_32768_maskopt_v2_dynamic_fix.jsonl` (median `traversal_semisort_ms≈257 ms`).
  - 48 k: `artifacts/benchmarks/residual_dense_49152_maskopt_v2_default.jsonl` (median `≈244 ms`, traversal ≈53 s).
  - 64 k: `artifacts/benchmarks/residual_dense_65536_maskopt_v2_default.jsonl` (median `≈232 ms`, traversal ≈76 s).

### Reproducible Baseline Command (Hilbert / maskopt_v2)

```bash
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline none \
  --log-file artifacts/benchmarks/residual_dense_32768_maskopt_v2_dynamic_fix.jsonl
```

- Prefix schedule defaults to `doubling` for residual metrics, and `residual_dynamic_query_block` / survivor ladder are on by default.
- Adjust `--tree-points` (e.g., 49152, 65536) to reproduce the scaling results listed above.
- For 4 k guardrail runs, append `--residual-dynamic-query-block 0` to avoid the small-scale regression.

### Scaling sweep helper

Use `tools/residual_scaling_sweep.py` to benchmark multiple tree sizes in sequence (defaults: 4 k, 8 k, 16 k, 32 k, 48 k, 64 k) and capture medians directly from the JSONL logs:

```bash
python tools/residual_scaling_sweep.py \
  --tree-sizes 4096,8192,16384,32768,49152,65536 \
  --log-prefix residual_scaling_maskopt_v2
```

Logs land under `artifacts/benchmarks/scaling/`; rerun with different flags to compare features (e.g., toggling the residual bitset) without hand-editing commands.

## How to Use This File

1. Read the archived journal entry for the detailed telemetry, commands, and mitigation attempts.
2. Update [`BACKLOG.md`](../BACKLOG.md#dense-residual-regression-bisect) when the bisect lands or when the dense baseline is healthy again.

### Quick Links

- [Historical journal entry](journal/2025-11-12_residual_dense_regression.md)
- [November 2025 journal index](journal/2025-11.md)
