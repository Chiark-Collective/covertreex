# Residual Dense Baseline Regression (2025-11-12)

The full investigative log now lives under [`docs/journal/2025-11-12_residual_dense_regression.md`](journal/2025-11-12_residual_dense_regression.md) so this document can stay focused on the current status and entry points back into the history.

-## Current Snapshot — 2025-11-17

- Dense residual builds now land at **≈20.6 s wall / 16.9 s dominated traversal** when the bitset scope path is enabled by default (`artifacts/benchmarks/artifacts/benchmarks/residual_dense_32768_dense_streamer_bitset.jsonl`, `run_id=pcct-20251114-141549-e500d8`). Dominated batches sit at `traversal_semisort_ms≈47 ms` (p90 ≈66 ms).
- The 4 k guardrail run (`artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl`, `run_id=pcct-20251114-105559-b7f965`) stays well under the `<1 s` threshold with `traversal_semisort_ms≈42 ms`.
- Dynamic query blocks **and** the dense scope streamer are enabled by default; disable via `--residual-dynamic-query-block 0` or `--no-residual-dense-scope-streamer` if you need to reproduce the historical multi-pass telemetry.
- Bitset dedupe is now part of the default preset (`--residual-scope-bitset` / `COVERTREEX_RESIDUAL_SCOPE_BITSET=1`); the new Numba helper keeps both cache-prefetch hits and scope inserts inside compiled code. Disable the flag for regressions that need the old Python set semantics.
- Latest scaling checkpoints:
  - 32 k: `artifacts/benchmarks/artifacts/benchmarks/residual_dense_32768_dense_streamer_gold.jsonl` (median `traversal_semisort_ms≈62 ms`, total traversal ≈18.8 s, build summary `pcct | build=21.1754 s`).
  - 48 k: `artifacts/benchmarks/residual_dense_49152_maskopt_v2_default.jsonl` (median `≈244 ms`, traversal ≈53 s) — rerun pending with the dense streamer.
  - 64 k: `artifacts/benchmarks/residual_dense_65536_maskopt_v2_default.jsonl` (median `≈232 ms`, traversal ≈76 s) — rerun pending with the dense streamer.

### Reproducible Baseline Command (Hilbert / dense streamer)

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
      --residual-scope-bitset \
      --log-file artifacts/benchmarks/residual_dense_32768_dense_streamer_bitset.jsonl
```

- Prefix schedule defaults to `doubling` for residual metrics, and both `residual_dynamic_query_block` plus the dense streamer are on by default.
- Adjust `--tree-points` (e.g., 49152, 65536) to reproduce the scaling results listed above; rerun the scaling sweep once the dense streamer data lands.
- For 4 k guardrail runs, the dense streamer already keeps dominated batches well below 50 ms (log `artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl`). Use `--no-residual-dense-scope-streamer` only if you need the legacy maskopt_v2 numbers.

### Scaling sweep helper

Use `tools/residual_scaling_sweep.py` to benchmark multiple tree sizes in sequence (defaults: 4 k, 8 k, 16 k, 32 k, 48 k, 64 k) and capture medians directly from the JSONL logs:

```bash
python tools/residual_scaling_sweep.py \
  --tree-sizes 4096,8192,16384,32768,49152,65536 \
  --log-prefix residual_scaling_dense_streamer
```

Logs land under `artifacts/benchmarks/scaling/`; rerun with different flags to compare features (e.g., toggling the residual bitset) without hand-editing commands.

### Masked scope append experiments (2025-11-14)

- Command (dense streamer preset, diagnostics on):

  ```bash
  COVERTREEX_BACKEND=numpy \
  COVERTREEX_ENABLE_NUMBA=1 \
  COVERTREEX_SCOPE_CHUNK_TARGET=0 \
  COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
  python -m cli.queries \
    --metric residual --dimension 8 --tree-points 32768 \
    --batch-size 512 --queries 1024 --k 8 --seed 42 \
    --baseline none --log-file artifacts/benchmarks/residual_dense_32768_dense_streamer_maskappend_on_run1.jsonl
  ```

- With masked append **on** (default) the run `pcct-20251114-134307-62e1f4` reports `build=20.0–21.0 s`, dominated `traversal_semisort_ms≈50–56 ms`, and traversal medians ≈262 ms. The alternative seed (`pcct-20251114-134406-a7ebba`, log `…_on_run2.jsonl`) shows similar medians with slightly longer traversal (≈300 ms) but the same per-batch semisort cost.
- Forcing the legacy Python path via `--no-residual-masked-scope-append` (`pcct-20251114-134434-cf31f8` and `pcct-20251114-134507-6efbf5`) keeps semisort medians within ±2 ms but adds ≈4–5 s wall time on the worst Hilbert seeds because every cache-prefetch hit now round-trips through `np.nonzero`. Keep both logs around for regression diffs.

## How to Use This File

1. Read the archived journal entry for the detailed telemetry, commands, and mitigation attempts.
2. Update [`BACKLOG.md`](../BACKLOG.md#dense-residual-regression-bisect) when the bisect lands or when the dense baseline is healthy again.

### Quick Links

- [Historical journal entry](journal/2025-11-12_residual_dense_regression.md)
- [November 2025 journal index](journal/2025-11.md)
