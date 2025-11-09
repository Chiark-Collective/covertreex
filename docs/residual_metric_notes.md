# Residual-Correlation Metric Integration (2025-11)

This note summarizes the current host-side implementation for the Vecchia residual-correlation metric inside `covertreex`.

## Host Caches & Configuration

We introduced `ResidualCorrHostData` in `covertreex/metrics/residual.py`. It packages the host-resident artefacts supplied by the VIF pipeline:

- `v_matrix` — the low-rank factors \( V = L_{mm}^{-1} K(X, U) \) for all training points.
- `p_diag` — per-point residual diagonals \( p_i = \max(K_{x_ix_i} - \|V_i\|^2, 10^{-9}) \).
- `kernel_diag` — raw kernel diagonals \( K_{x_ix_i} \) (provides a fallback for bound computations).
- `kernel_provider(rows, cols)` — a callable that returns the raw kernel slice \( K(X_{rows}, X_{cols}) \) over integer dataset indices.
- `point_decoder` — optional decoder that maps tree payloads to dataset indices (defaults to treating them as integer ids).
- `chunk_size` — preferred batch size for host streaming (defaults to 512).

`configure_residual_correlation(...)` installs the residual metric hooks. We intentionally keep Euclidean metrics untouched: the residual path is only active when `COVERTREEX_METRIC=residual_correlation` and custom caches are registered.

## Traversal Path

### Early-Exit Parent Search

- `_residual_find_parents` (in `covertreex/algo/traverse.py`) streams the tree indices in `chunk_size` tiles.
- For each chunk, we request the raw kernel block and feed it to the chunk kernel (`compute_distance_chunk` from `metrics/_residual_numba.py`).
- The kernel accumulates `V_i · V_j` incrementally, uses cached \( \|V_i\|^2 \) and \( p_i \) to bound the residual correlation, and aborts if the best possible distance still exceeds the caller’s current best. This replicates the residual bound from `_ResidualCorrBackend.lower_bound_train`.
- We track the minimum distance per query, yielding the same parent as the dense path.

### Streaming Scope Assembly

- `_collect_residual_scopes_streaming` reuses the chunk kernel to gather per-query conflict scopes.
- For each parent, we stream candidate tree nodes, apply the residual bound (lower bound via kernel diagonal) and exact distance checks only for survivors, and accumulate dataset indices into the scope. Parent chains (`Next`) are appended afterwards, ensuring deterministic ordering (descending level then index).
- Radii are derived the same way as the Euclidean path: \( \max(2^{\ell_i+1}, S_i) \).
- While streaming we now record the **observed maximum residual distance** per query into `TraversalResult.residual_cache.scope_radii`. These measurements form the “residual radius ladder” used during conflict-graph construction: when the metric is residual-correlation, `build_conflict_graph` swaps the Euclidean fallback \( \max(2^{\ell_i+1}, S_i) \) for the observed value (bounded below by `COVERTREEX_RESIDUAL_RADIUS_FLOOR`, default `1e-3`). This keeps newly inserted nodes from inheriting `radius≈2` regardless of depth and feeds Gate‑1 with radii that reflect the actual residual distances seen during streaming.
- `_collect_residual_scopes_streaming` now tracks chunk-level telemetry (chunks scanned, points touched, dedupe hits, saturation flags). These counters flow into `TraversalTimings` (`traversal_scope_chunk_{scans,points,dedupe,saturated}`) and the JSONL writer so profiling the 16 384-cap hit is straightforward.
- Per-level scope caches remember up to `scope_limit` tree positions from the most recent query at each level. Later queries with the same parent level prefetch those nodes first (`traversal_scope_cache_{prefetch,hits}` capture how many were re-checked and re-used) before the chunk streamer emits new kernel tiles.
- When `COVERTREEX_SCOPE_CHUNK_TARGET>0`, the same value now caps **both** the scope size and the number of tree points we are willing to scan per query. Once the scan budget is exhausted we mark the scope as saturated, stop requesting new kernel tiles, and rely on the telemetry above to highlight the truncation.
- Practical impact: the 32 768-point Hilbert run with `COVERTREEX_SCOPE_CHUNK_TARGET=8192` (`benchmark_residual_scopecap_20251108.jsonl`) now finishes in **687.29 s build / 0.027 s query (37.6 k q/s)** with the scan budget tripping on 48 of 64 dominated batches (`traversal_scope_chunk_points` per batch ≈4.19 M = 512 queries × 8 192 cap, `traversal_scope_chunk_saturated=48`). The earlier cap-only replay (no scan guard) took 900.2 s, so this reuse alone saves ~24 % wall time without touching the conflict graph.
- Residual batches emit `traversal_scope_radius_{obs,initial,limit,cap_values}_*` summaries plus `traversal_scope_radius_cap_hits` and `*_delta_{mean,max}` in every JSONL record. Those stats let you inspect how aggressive the ladder currently is (observed maxima) versus the limits you feed into traversal.
- Optional per-scope caps can rein in traversal radii before the gate runs. Provide a JSON table via `COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH=/path/to/caps.json` (schema `{"schema": 1, "default": <optional>, "levels": {"0": 0.5, "1": 0.75}}`) and/or a global fallback with `COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT`. The caps clamp the runtime radius but still honour `COVERTREEX_RESIDUAL_RADIUS_FLOOR`; cap hits show up in the JSON logs so you can confirm how often each level is being throttled.
- `docs/data/residual_scope_caps_scope8192.json` captured the legacy flat 1.25 cap, while `docs/data/residual_scope_caps_32768.json` now records the per-level medians (+0.05 margin) from the November 8 32 768-point replay. Derive fresh tables without hand-editing JSON by adding `--residual-scope-cap-output /path/to/out.json` (plus optional `--residual-scope-cap-percentile` / `--residual-scope-cap-margin`) to `cli.queries`; the new `ResidualScopeCapRecorder` streams per-level ladder samples directly from each batch insert, annotates the active `run_id`, and emits summaries at shutdown.
- Residual gate profiles recorded via `tools/build_residual_gate_profile.py` now ship with embedded metadata (`run_id`, seeds, dataset size, kernel hyperparameters, etc.). The builder writes to `artifacts/profiles/residual_profile_<run_id>.json` by default; override with an explicit `output` path or pass `--run-id` to control the identifier if you need deterministic filenames.

## Conflict Graph Pipeline

### Pairwise Matrix

- Inside `build_conflict_graph`, when residual mode is active we decode dataset ids for the batch and materialise the full \( n \times n \) matrix once per batch. That matrix now feeds every downstream stage (dense, segmented, and filter) when available; we fall back to on-demand kernel streaming only when the cache is absent.

### Adjacency Filter

- The dense adjacency builder (`_build_dense_adjacency`) now accepts an optional `residual_pairwise` matrix. When provided, the Numba helper receives the residual distances directly.
- When the cached matrix is available we also reuse it during the post-build radius filter, so pruning simply indexes into that \(n\times n\) buffer. Only when the cache is missing (e.g., legacy dense traversal) do we stream kernel tiles through `compute_residual_distances_from_kernel`.
- The segmented builder piggybacks on the same residual matrix (used when `COVERTREEX_CONFLICT_GRAPH_IMPL=segmented`).
- Chunk telemetry now records pair-wise saturation as well: JSONL batches expose `conflict_scope_chunk_pair_cap`, `conflict_scope_chunk_pairs_before`, and `conflict_scope_chunk_pairs_after`, making it obvious when `_chunk_ranges_from_indptr` fuses tails because the estimated pair budget is tiny even if membership volume looks large.

## Chunk Kernel (Numba)

`covertreex/metrics/_residual_numba.py` houses the Numba implementation. Given:

- query factor `v_query`, chunk factors `v_chunk`
- cached radii/diagonals `p_i`, `p_chunk`
- raw kernel entries

It emits both distances and a mask indicating which entries fall below a caller-specified radius. We expose `compute_residual_distances_with_radius` in `metrics/residual.py` so traversal and conflict graph code paths can reuse the same accelerated helper with CPU fallback.

### Gate‑1 (whitened Euclidean bound)

- `ResidualCorrHostData` now materialises a float32-whitened copy of `v_matrix` (`gate_v32` + `gate_norm32`) along with telemetry counters and user-tunable parameters (`gate1_alpha`, `gate1_margin`, `gate1_eps`, `gate1_audit`).
- `compute_residual_distances_with_radius` consults `RuntimeConfig.residual_gate1_*`, runs the Numba `gate1_whitened_mask` before calling `_distance_chunk`, and scatters survivor distances back into the full chunk order so existing callers keep the same contract.
- Per-batch telemetry (`traversal_gate1_{candidates,kept,pruned}`, `traversal_gate1_ms`) is emitted via the traversal timings and JSONL sink; audit mode optionally replays `_distance_chunk` on pruned rows and fails fast if a true neighbour slips through the gate.
- The gate is feature-flagged via `COVERTREEX_RESIDUAL_GATE1=1` plus the associated `..._ALPHA`, `..._MARGIN`, `..._EPS`, and `..._AUDIT` knobs so we can tune the α/ε mapping on benchmark data before enabling it by default.
- Latest diag0 sweeps (`benchmark_residual_gate_p2k_alpha{0_75,1_0,1_25,1_5,2_0,2_25,2_5,3_0,4_0,10_0}_diag0.jsonl`) still fail the audit as soon as dominated scopes appear—even with the clamped `si_cache` and the residual radius ladder—so Gate‑1 remains *off* in production unless you opt in explicitly. The only run that completes is the intentionally loose `benchmark_residual_gate_p2k_alpha100_diag0{,_run2}.jsonl`, and those logs show zero pruned candidates (the gate keeps everything), confirming we previously lacked a safe but selective threshold.

### Calibration & Lookup Table

- `tools/build_residual_gate_profile.py` builds an empirical profile from a synthetic residual workload. It reproduces the diag0 harness (2048 points, dimension 8, seed 42) and samples every pair, recording (residual distance, whitened distance) into evenly spaced radius bins.
- The default artefact, `docs/data/residual_gate_profile_diag0.json`, contains 2,096,128 samples across 512 bins with zero audit escapes. Check it into source control so CI and local runs share the same lookup without recomputing the NxN matrix.
- To regenerate or explore alternative datasets, run for example:

  ```bash
  python tools/build_residual_gate_profile.py docs/data/residual_gate_profile_diag0.json \
      --tree-points 2048 --dimension 8 --seed 42 --bins 512 --pair-chunk 64
  ```

- At runtime you can consume the lookup by setting:

  ```bash
  export COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1
  export COVERTREEX_RESIDUAL_GATE1=1
  export COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH=docs/data/residual_gate_profile_diag0.json
  export COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN=0.02   # optional safety buffer
  export COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP=10.0      # optional cap for huge radii
  ```

  (You can omit the cap to let the lookup see larger radii; we cap it when we want deterministic thresholds even if the residual ladder spikes.) The lookup supplies per-radius thresholds directly, so `residual_gate1_alpha` becomes a fallback rather than the primary tuning knob.

- The CLI now takes care of those flags for you: `python -m cli.queries --metric residual --residual-gate lookup ...` wires sparse traversal, enables the gate, and points at `docs/data/residual_gate_profile_diag0.json`. Override the path (`--residual-gate-lookup-path`), margin (`--residual-gate-margin`), or radius cap (`--residual-gate-cap`) as needed. Use `--residual-gate off` to explicitly keep the gate disabled during experiments.

- Gate‑1 still defaults to off globally—we only flip it on in telemetry or experimental runs until we finish the sparse traversal rollout and confirm the lookup holds for larger corpora. When you opt in, make sure `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` is set; otherwise the dense traversal path bypasses the streaming helper and the gate never engages.
- Regardless of whether Gate‑1 is enabled, residual batches now materialise the whitened cache so the grid builder can operate in the same space as the Euclidean runs. If the default cell width feels too loose/tight, set `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` (>1 tightens the cells, <1 loosens them; default `1.0`) to dial in `grid_*` telemetry before capturing regression artefacts.

- `docs/data/residual_gate_profile_scope8192.json` captures a larger synthetic workload (8 192 points, 33 550 336 samples, same 512 bins) for comparison. Relative to the diag0 artefact the median threshold is +9.66, the 90th percentile delta is +19.93, the maximum absolute delta is ≈61.9, and 387/512 bins have higher maxima. `docs/data/residual_gate_profile_32768_caps.json` replaces those synthetic probes with the full 32 768-point capture (caps enabled, audit on) from 2025‑11‑08 so gate experiments can finally reference a workload that matches the production corpus.

- Fresh 32 k runs with the lookup-enabled gate:
  - `benchmark_residual_gate_lookup_32768_default_cap10.jsonl` (clamped/default build) reports build ≈269.8 s. `traversal_gate1_*` counters stay at zero because the per-level radii continue to exceed the lookup cap.
  - `benchmark_residual_gate_lookup_32768_chunked_cap10.jsonl` (chunk target 16 384) lands at ≈270.8 s with the same gate stats (0 candidates/kept/pruned).
  - `benchmark_residual_lookup_32768_final_run2.jsonl` (Hilbert batches + new per-level caps + chunked traversal) builds in 900.2 s and sustains 37.7 k q/s (1 024 queries, k=8) with audit enabled. Gate‑1 still rejects nothing (`traversal_gate1_pruned=0`) because the medians now sit near 1.05; the lookup remains a no-op until we either raise the caps again or derive a tighter radius floor.
- Setting `COVERTREEX_RESIDUAL_PREFILTER=1` enables the lookup-driven gate as a chunk-level SIMD prefilter (defaults: diag0 lookup, margin 0.02, radius cap 10). Override the lookup via `COVERTREEX_RESIDUAL_PREFILTER_LOOKUP_PATH`—for example, point it at `docs/data/residual_gate_profile_32768_caps.json` to reuse the November 8 corpus. Combining the prefilter with the scan cap and per-level cache (`benchmark_residual_cache_prefilter_20251108.jsonl`) yields **700.71 s build / 0.027 s query (37.6 k q/s)** while keeping `traversal_scope_chunk_points≈4.19 M` and logging `traversal_scope_cache_prefetch≈2.1 M` on saturated batches.

## Tests

- `tests/test_metrics.py` exercises the chunk kernel (distance + radius masks) and validates that residual distances computed via kernel reuse match the dense path.
- `tests/test_traverse.py` now includes a residual sparse traversal regression (dense vs. streamed scopes) to ensure parents/levels/scopes remain consistent.
- `tests/test_conflict_graph.py` gained a residual parity check to confirm dense Euclidean and residual-aware models produce identical CSR structures.

## Residual Grid Scale Sweep (2025‑11‑09)

**Objective.** Measure whether the new `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` knob changes the dominated residual workload enough to impact build time, and capture telemetry deltas that auditors can compare against future runs.

**Setup.** `python -m cli.queries --dimension 8 --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --metric residual --baseline gpboost --seed 42 --log-file <artifact>` with

- `COVERTREEX_BACKEND=numpy`, `COVERTREEX_ENABLE_NUMBA=1`
- `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`, `COVERTREEX_PREFIX_SCHEDULE=doubling`
- diagnostics on, dense traversal (`COVERTREEX_SCOPE_CHUNK_TARGET=0` from default gold-standard harness)
- swept `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` ∈ {0.75, 1.50}

Artifacts live under `artifacts/benchmarks/artifacts/benchmarks/` and share the run IDs below.

| Scale | Log file | Run ID | PCCT Build (s) | Throughput (q/s) | Grid leaders (mean) | Grid local edges (mean) | Traversal ms (p50) | Conflict ms (p50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | `residual_grid_scale0p75_20251109180103.jsonl` | `pcct-20251109-170103-ce1129` | 56.10 | 28 802 | 510.17 | 130 115 | 737 ms | 69.6 ms |
| 1.50 | `residual_grid_scale1p50_20251109180226.jsonl` | `pcct-20251109-170226-df7a7f` | 55.73 | 28 698 | 511.70 | 130 713 | 711 ms | 69.9 ms |

**Observations.**

- The grid stays fully saturated (domination ratio ≈0.984) regardless of scale: every batch emits ~1.5 k cells, ~510 leaders, and zero conflict edges, so MIS time remains negligible.
- Tightening the scale increases local grid edges by <0.5 % and shaves ~25 ms off the traversal median, but the overall build time is bounded by traversal (0.7–1.4 s per dominated batch) rather than grid work.
- RSS peaks around 1.53–1.60 GB in both runs; memory pressure doesn’t change with the scale adjustments.

**Traversal-focused follow-ups.** Grid tuning alone cannot deliver material build wins while traversal dominates:

1. **Gate‑1 rollout.** The sweep still shows `traversal_gate1_* = 0`; enabling the lookup-backed gate (with `COVERTREEX_RESIDUAL_GATE1=1` plus the diag0 profile) should prune candidates before `_distance_chunk` and directly attack the 0.7–1.4 s dominated batches.
2. **Scope trimming / cache hits.** Saturated batches hit zero cache hits at the start and only ~344/512 by the tail. Investing in per-level cache prefetch heuristics (or raising `residual_scope_cap_default`) should reduce the number of fully scanned scopes.
3. **Sparse traversal gating.** Re-enabling the sparse traversal path once the bucketed CSR builder lands (see AUDIT.B) will remove the O(n log n) semisort tail (`traversal_semisort_ms` still >20 ms median) and keep dominated scopes bounded without relying on scope chunk caps.

Until those traversal improvements ship, keep the residual grid in place for determinism (leader telemetry + MIS seeds) but don’t expect the whiten scale alone to replicate the Euclidean build-time gains.

### Gate‑1 Experiments (2025‑11‑09 evening)

**Goal.** Re-enable the float32 Gate‑1 (lookup-backed) and confirm whether it can prune enough dominated candidates to lower traversal time without violating correctness. Runs used the same 32 768‑point benchmark as above.

| Config | Key env knobs | Outcome |
| --- | --- | --- |
| `residual_gate_lookup_sparse_scale1p00_20251109191729.jsonl` | `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1`, `COVERTREEX_RESIDUAL_GATE1=1`, lookup margin 0.02, audit on | Gate tripped audit in the second batch (`RuntimeError: Residual gate pruned a candidate that lies within the requested radius`). Only the first batch logged, `traversal_gate1_*` counters stayed at 0 because the failure happened during the audit replay. |
| `..._margin0p5_20251109191744.jsonl` | Same but lookup margin 0.5 | Audit still failed immediately. Loosening the margin alone does not prevent false negatives with the current lookup profile. |
| `residual_gate_lookup_sparse_noaudit_scale1p00_20251109191757.jsonl` | Same as the first run, but `COVERTREEX_RESIDUAL_GATE1_AUDIT=0` (unsafe) | Run completed but took **>34 minutes** (our harness killed the CLI after ~2 050 s). Telemetry: `traversal_semisort_ms` averaged **31.2 s** per batch (sparse traversal is still sorting scope members), Gate‑1 processed ~5.5×10^8 candidates but only pruned ~7.6×10^6 (~1.4 %), and the gate itself cost ~2.85 s per batch. Whatever pruning we gained was dwarfed by the sparse semisort overhead. |
| `residual_gate_dense_scale1p00_20251109195236.jsonl` | Gate enabled, sparse traversal **disabled** | Gate counters remained zero for all 64 batches; as expected, the dense traversal path never calls the gate and behavior matches the baseline dense run. |

**Lessons.**

1. **Audit fails immediately.** Even the “loose” lookup margin (0.5) still prunes legitimate neighbors on the first dominated batch. We need either a safer lookup profile (probably drawn from the 32 k corpus) or an adaptive bound that accounts for the per-batch radius ladder before we can ship Gate‑1.
2. **Sparse traversal must be fixed first.** With audit off, Gate‑1 prunes millions of candidates, but the sparse traversal’s semisort dominates runtime (dozens of seconds per batch). The bucketed CSR builder from AUDIT.B is a prerequisite; otherwise enabling sparse traversal regresses build time by 40–60×.
3. **Gate on dense traversal is a no-op.** Without the streaming helper the gate never runs, so toggling `COVERTREEX_RESIDUAL_GATE1` alone does nothing in production today.

**Traversal notes.** To make Gate‑1 viable we need the bucketed CSR path (so sparse scopes don’t explode), a lookup derived from the latest 32 k telemetry (to keep audit happy), and probably a cheaper gate kernel (current gate spent ~2.85 s/batch processing 5.5×10^8 candidates). Only after those land will Gate‑1 have a chance of chipping away at the 0.7–1.4 s dominated batches we see in the dense runs.

## Operational Notes

- Residual mode requires `backend.name == "numpy"` (NumPy backend) until the GPU/JAX kernels are ported.
- The decoder must map tree payloads to dataset indices; if trees store transformed buffers, supply a custom `point_decoder` when creating `ResidualCorrHostData`.
- The chunk kernel honours `chunk_size`; tune it to balance host-side streaming vs. cache reuse.
- Setting `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` now engages residual streaming automatically when the metric is residual-correlation; otherwise, traversal falls back to the dense mask path.
