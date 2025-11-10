# Residual PCCT Build Optimisation — Living Plan

_Last updated: 2025-11-09_

This file tracks ongoing work on the residual-correlation PCCT build pipeline. It captures the current state, recent experiments, and the next concrete steps so anyone can pick up the thread quickly.

## Current State (2025-11-09)

- **Baseline performance (dense traversal, grid conflict graph).** `python -m cli.queries --metric residual --tree-points 32768 --queries 1024 --k 8 --baseline gpboost --seed 42` with diagnostics on measures **≈63 s build / 0.035 s query (29.5 k q/s)**. Grid telemetry shows ~1.5 k cells and ~510 leaders per batch; `traversal_gate1_* = 0` because the gate never runs on the dense path.
- **Sparse traversal ergonomics.** `_collect_residual_scopes_streaming()` now feeds `(query, member)` pairs through the bucketed CSR builder (shared with the conflict graph). This removes the Python list/tuple churn and guarantees CSR ordering matches the dense path, but the residual distance scans still dominate (minutes per batch) when sparse traversal is enabled.
- **Gate‑1 default lookup.** Runtime now points at `docs/data/residual_gate_profile_32768_caps.json`, aligning the lookup with the 32 k corpus captured on 2025‑11‑08. Dense runs are unaffected; sparse runs still trip the audit because the lookup thresholds remain too aggressive for the dominated batches we see at 32 k.

## Recent Experiments

1. **Grid scale sweep (0.75 vs 1.50).** Changing `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` nudges grid telemetry but doesn’t touch the traversal bottleneck (still 0.7–1.4 s per dominated batch). Logs: `residual_grid_scale0p75_20251109180103.jsonl`, `residual_grid_scale1p50_20251109180226.jsonl`.
2. **Gate‑1 audit runs (sparse traversal).** Lookup margins 0.02 and 0.5 both failed the audit immediately. Disabling the audit let the run finish, but `traversal_semisort_ms` ballooned to 30 s+ per batch (minutes per run) and Gate‑1 only pruned ~1.4 % of candidates. Logs: `residual_gate_lookup_sparse_scale1p00_*.jsonl`.
3. **CSR streaming refactor.** `_collect_residual_scopes_streaming()` now produces CSR directly from the streamed `(query, member)` pairs (Numba). Sparse vs dense outputs now match exactly (tests added), and the tuple/list mismatch is gone.

## Next Steps

1. **Numba scope streamer.** Move the residual scope collection (distance scans, dedupe, cache updates) into a dedicated Numba kernel so sparse traversal no longer spends minutes per batch. This should include:
   - Processing batches in fixed chunks with early exits.
   - Emitting the `(owner, member)` arrays directly to the CSR builder.
   - Recording telemetry (scans, points, dedupe hits) without Python loops.
2. **Lookup refresh.** Regenerate the Gate‑1 lookup using the 32 k corpus (including per-level radius metadata) so the audit can pass with realistic thresholds. The existing `tools/build_residual_gate_profile.py` can be extended to ingest recorded artifacts (e.g., `benchmark_residual_*.jsonl`) instead of synthetic Gaussian data.
3. **Gate benchmarking.** Once (1) and (2) land, rerun the 32 k benchmark with sparse traversal enabled and Gate‑1 auditing on. Record build/telemetry deltas for:
   - Gate off (dense vs sparse).
   - Gate on (sparse streamer).
   - Sensitivity to `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` under the new traversal.

## Handy Commands

```bash
# Dense baseline (diagnostics on)
python -m cli.queries \
  --dimension 8 --tree-points 32768 --batch-size 512 \
  --queries 1024 --k 8 --metric residual \
  --baseline gpboost --seed 42 \
  --log-file artifacts/benchmarks/residual_grid_default_<ts>.jsonl

# Sparse traversal smoke test (still slow until the streamer lands)
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 \
python -m cli.queries --metric residual ...

# Regenerate synthetic gate profile (placeholder until the 32 k ingestion tool exists)
python tools/build_residual_gate_profile.py docs/data/residual_gate_profile_diag0.json \
  --tree-points 2048 --dimension 8 --seed 42 --bins 512 --pair-chunk 64
```

## 2025-11-10 — Sparse Streamer Benchmarks & CPU Utilisation Notes

### Runs captured

- **Dense control (residual_dense_20251110103412.jsonl).** `python -m cli.queries --metric residual --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --baseline gpboost --seed 42 --log-file artifacts/benchmarks/artifacts/benchmarks/residual_dense_20251110103412.jsonl`
  - Build wall-clock: 264.3 s (sum of per-batch timings ≈241.9 s).
  - Median per-batch timings: `traversal_ms≈1.11 s`, `traversal_semisort_ms≈0.34 s`, `conflict_graph_ms≈2.42 s`.
  - Scope chunk telemetry stays at zero because dense traversal ignores `scope_chunk_target`; Gate‑1 counters remain zero.
- **Sparse streamer + capped traversal (residual_sparse_streamer_20251110104928.jsonl).** `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 COVERTREEX_ENABLE_NUMBA=1 COVERTREEX_SCOPE_CHUNK_TARGET=8192 python -m cli.queries --metric residual ... --log-file artifacts/benchmarks/residual_sparse_streamer_20251110104928.jsonl`.
  - Build wall-clock: 597.4 s (sum of per-batch timings ≈581.4 s).
  - Median per-batch timings: `traversal_ms≈10.37 s`, `traversal_semisort_ms≈9.88 s`, `conflict_graph_ms≈0.036 s`.
  - Every query hits the scan cap: `traversal_scope_chunk_points=8192`, `scope_chunk_scans=16`, `scope_chunk_saturated=512`, `scope_chunk_dedupe=0`, `scope_cache_* = 0` per batch. Observed residual radii top out around 1.0, so the ladder still feeds Gate‑1 nothing (all gate counters zero).

### CPU oscillation diagnosis

- The node spends ~35 ms with all cores busy (JAX+NumPy phases inside conflict graph/MIS), then ~10 s with a single core maxed out. That single-core phase is `_collect_residual_scopes_streaming` walking 512 queries sequentially: for each query we synchronously call `kernel_provider`, run `compute_residual_distances_with_radius`, update dedupe flags, and append to CSR buffers. Only the inner `_distance_chunk` loop is vectorised (prange over 512 candidates); the outer loop is Python/NumPy and holds the GIL.
- Because `kernel_provider` is an arbitrary Python callable, today we can’t invoke it from inside a `@njit(parallel=True)` loop, so the streamer can’t fan queries across threads. Hence the “all cores → one core” pattern every few seconds during the sparse run.

### Parallelisation options

1. **Threaded shard (near-term).** Split the per-query loop across a `ThreadPoolExecutor` (e.g. 8 workers × 64 queries). Give each worker its own `collected_flags`, scope buffer, telemetry accumulators, and optional level cache; after the pool finishes, concatenate the chunks and feed `build_scope_csr_from_pairs`. The heavy sections (`kernel_provider`, `_distance_chunk`, `residual_scope_append`) all release the GIL, so the workers will run concurrently and keep CPU utilisation flat. Risks: slightly higher memory (one cache per worker) and more complicated telemetry reduction.
2. **Fully numba-fied streamer (mid-term).** Port `_collect_residual_scopes_streaming` to a single Numba kernel (similar to the Euclidean `_collect_scopes_csr`) so we can parallelise over queries with `prange`. To do that we need Numba-native kernel tiles, either by:
   - Reconstructing kernel entries inside Numba from the Vecchia caches (`v_matrix`, `p_diag`, `kernel_diag`), or
   - Prefetching Python-side slabs (multiple queries × chunk) and passing them to the kernel.
   This keeps dedupe, cache reuse, and CSR assembly inside one parallel kernel and eliminates host-side GIL contention entirely.
3. **Gate integration.** Once the streamer is parallel, hoist Gate‑1 into the same kernel so we don’t reintroduce serial work when the lookup is re-enabled.

### Action items

1. Prototype the threaded shard to validate that multi-core utilisation and wall-clock improve before investing in the full Numba rewrite. Target: keep the scan cap at 8 192 but drop per-batch traversal below the current ~10 s.
2. Extend `ResidualCorrHostData` with a Numba-friendly kernel tile helper (or inline the RBF reconstruction) so we can remove the Python `kernel_provider` bottleneck.
3. After the streamer is parallel, regenerate the Gate‑1 lookup from the new telemetry (per-level radii + observed ladder) and rerun the sparse suite with audit on.

All logs mentioned above live under `artifacts/benchmarks/…` for future comparisons.

### 2025-11-10 (afternoon) — Numba Chunk Streamer rollout

- **Implementation.** `_collect_residual_scopes_streaming_parallel()` now batches queries, streams kernel tiles once per block, and calls a new batched residual-distance kernel that lives alongside the existing Vecchia helpers. Gate‑1 is still respected: the parallel path only activates when the gate is off, and the previous serial function remains in place for the audit/prefilter path. Per-query telemetry (scan counts, dedupe hits, cache stats) is emitted directly from the new kernel, so downstream consumers and tests were untouched.

- **Dense control (pcct-20251110-105043-101bb9, `artifacts/benchmarks/artifacts/benchmarks/residual_dense_20251110115043.jsonl`).** No material change: build ≈257 s, `median(traversal_ms)=1.23 s`, `median(conflict_graph_ms)=2.51 s`, scope chunk telemetry remains zero.

- **Sparse streamer (pcct-20251110-105526-68dddf, `artifacts/benchmarks/artifacts/benchmarks/residual_sparse_streamer_20251110115526.jsonl`).** Build sum dropped to ≈493 s (previously ≈581 s). Median per-batch timings improved to `traversal_ms≈8.65 s` (−15 %), `traversal_semisort_ms≈8.11 s` (reflecting the multi-core chunk streamer), while `conflict_graph_ms≈0.038 s` stayed tiny. Every query still hits the 8 192-point scan cap (`scope_chunk_scans=8 192`, `scope_chunk_points=4.19 M`, `scope_chunk_saturated=512`), so the next wins have to come from actually pruning those scans (Gate‑1 + better cache reuse) rather than host overhead.
- **Profile capture (pcct-20251110-112936-0d2a25, `artifacts/benchmarks/residual_sparse_streamer_profile_20251110122936.jsonl`).** Same sparse harness but with `COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH=$PWD/docs/data/residual_gate_profile_32768_caps.json`, which refreshed the lookup with real traversal telemetry (≈2.10 M samples recorded). The file at `docs/data/residual_gate_profile_32768_caps.json` now carries a 2025‑11‑10 metadata block, so downstream runs automatically pick up the refreshed thresholds.
- **Gate‑1 validation (pcct-20251110-112456-69dfdd, `artifacts/benchmarks/artifacts/benchmarks/residual_sparse_gate_20251110124256.jsonl`).** With the new lookup, audit stayed green across all 64 batches (`traversal_gate1_pruned=0`, `traversal_gate1_candidates≈2.33×10^8`), confirming the profile is safe but still too conservative to prune anything yet. Traversal timings matched the profile run because every dominated batch still saturated the scan cap.

- **Observations.**
  1. Cache prefetch/hit telemetry is still zero across dominated batches, so the per-level cache needs smarter heuristics (or to be disabled) when every query saturates the cap.
  2. Because the streamer now finalises scopes immediately after each block, the per-level cache can be fed back into the very next block; this should matter once Gate‑1 or scope caps start producing smaller scopes again.
  3. The dense path is unchanged, so regression comparisons use the new dense log above while we focus on shrinking the sparse traversal phase.

### Next steps (post-streamer)

1. Rebuild the 32 k Gate‑1 lookup from the new telemetry and re-enable the gate (with audit) so the scan cap isn’t hit on every query.
2. Teach the per-level cache to emit block-prefetch hints so cache hits show up in telemetry again.
3. Re-run the full suite (dense, sparse, sparse+gate) after (1)-(2) to quantify the drop in dominated batches (we have the verification run logged above; next we need pruning >0 before we can call it done).

Append new findings or TODOs as the work progresses.
