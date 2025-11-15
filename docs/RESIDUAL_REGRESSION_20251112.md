# Residual Dense Baseline Regression (2025-11-12)

The full investigative log now lives under [`docs/journal/2025-11-12_residual_dense_regression.md`](journal/2025-11-12_residual_dense_regression.md) so this document can stay focused on the current status and entry points back into the history.

## Current Snapshot — 2025-11-17

- Dense residual builds now land at **≈17.8 s wall / ≈16 s dominated traversal** with pair-count shard merging + buffer reuse enabled (`artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`, `run_id=pcct-20251114-214845-7df1be`). The log captures `pcct | build=17.7892 s`, `traversal_semisort_ms≈36.1 ms` (p90 ≈58.7 ms), and ~35 k `conflict_scope_chunk_pair_merges` with `conflict_arena_bytes≈2.7 MB`.
- The prior dense baseline (pair merge disabled) now serves as a regression control: `pcct-20251114-215010-fa37f4` (`artifacts/benchmarks/residual_dense_32768_gold_best5_run4.jsonl`) posted **18.6479 s** wall with `traversal_semisort_ms≈37.0 ms` (p90 ≈57.3 ms). Keep both logs for ablations.
- The refreshed 4 k guardrail log (`artifacts/benchmarks/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl`, `run_id=pcct-20251114-105559-b7f965`) reports `traversal_semisort_ms≈41.7 ms` with total dominated traversal ≈0.42 s; coverage + gate prunes remain at zero until we rerun with `COVERTREEX_RESIDUAL_FORCE_WHITENED=1 / --residual-gate`.
- Dense streamer, masked append, dynamic query blocks, bitset scopes, level-cache batching, **pair-count shard merging**, and the Numba buffer arena are all default-on; disable via `--no-residual-*`, `--no-scope-chunk-pair-merge`, or `--no-scope-conflict-buffer-reuse` only when bisecting.
- **Validation run (2025‑11‑14 17:59 UTC).** `pcct-20251114-175936-d3cd5d` (`artifacts/benchmarks/residual_dense_32768_degree_cap_smoke_rerun.jsonl`) replays the 32 k preset with the conflict-degree telemetry wired in. Totals: **≈22.4 s dominated traversal**, `median traversal_ms≈333 ms`, `p90≈621 ms`, `median traversal_semisort_ms≈40 ms` (p90 ≈59.5 ms), and `conflict_degree_cap=0` with `conflict_arena_bytes=0`. This run is purely an instrumentation smoke test—the gold reference is the pairmerge log above.
- Latest scaling checkpoints:
  - 32 k: `artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl` (median `traversal_semisort_ms≈36 ms`, `pcct | build=17.7892 s`).
  - 48 k: `artifacts/benchmarks/residual_dense_49152_maskopt_v2_default.jsonl` (median `≈244 ms`, traversal ≈53 s) — rerun pending with batching enabled.
  - 64 k: `artifacts/benchmarks/residual_dense_65536_maskopt_v2_default.jsonl` (median `≈232 ms`, traversal ≈76 s) — rerun pending with batching enabled.

### Reproducible Baseline Command (Hilbert / dense streamer + batching)

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
      --log-file artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl
```

- Prefix schedule defaults to `doubling` for residual metrics, and all residual hot-path toggles remain on by default.
- Adjust `--tree-points` (e.g., 49152, 65536) to reproduce the scaling results listed above.
- For 4 k guardrail runs, reuse the command with `--tree-points 4096`; the latest log lives at `artifacts/benchmarks/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl`.
- Toggle pair-merge or arena reuse only for regressions by appending `--no-scope-chunk-pair-merge` / `--no-scope-conflict-buffer-reuse`.

**Best-of-five (diagnostics on):**

| Config | Env deltas | Builds (s) | Best run |
| --- | --- | --- | --- |
| Pair-merge + buffer reuse | _default_ (`COVERTREEX_SCOPE_CHUNK_PAIR_MERGE=1`, `COVERTREEX_SCOPE_CONFLICT_BUFFER_REUSE=1`) | 18.47, 18.96, 20.98, 28.51, **17.79** | `pcct-20251114-214845-7df1be` (`residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`)
| Legacy dense (pair merge off) | `COVERTREEX_SCOPE_CHUNK_PAIR_MERGE=0`, `COVERTREEX_SCOPE_CONFLICT_BUFFER_REUSE=0` | 18.96, 21.43, 21.01, **18.65**, 21.86 | `pcct-20251114-215010-fa37f4` (`residual_dense_32768_gold_best5_run4.jsonl`)

Diagnostics-off sweeps still hit <20 s, but the on-by-default configuration above is the new audit anchor.

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
