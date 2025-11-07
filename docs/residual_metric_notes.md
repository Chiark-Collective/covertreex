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

## Conflict Graph Pipeline

### Pairwise Matrix

- Inside `build_conflict_graph`, when residual mode is active we decode dataset ids for the batch and materialise the full \( n \times n \) matrix by streaming kernel tiles through `compute_residual_distances_from_kernel`. This preserves compatibility with both dense and segmented builders.

### Adjacency Filter

- The dense adjacency builder (`_build_dense_adjacency`) now accepts an optional `residual_pairwise` matrix. When provided, the Numba helper receives the residual distances directly.
- The post-build radius filter no longer touches Euclidean norms. Instead, it groups outgoing edges by source, streams the kernel rows, and calls `compute_residual_distances_from_kernel` to prune edges one chunk at a time.
- The segmented builder piggybacks on the same residual matrix (used when `COVERTREEX_CONFLICT_GRAPH_IMPL=segmented`).

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
  export COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP=2.0       # cap radii for the lookup, optional
  ```

  (You can omit the cap to let the lookup see larger radii; we cap it when we want deterministic thresholds even if the residual ladder spikes.) The lookup supplies per-radius thresholds directly, so `residual_gate1_alpha` becomes a fallback rather than the primary tuning knob.

- Gate‑1 still defaults to off globally—we only flip it on in telemetry or experimental runs until we finish the sparse traversal rollout and confirm the lookup holds for larger corpora. When you opt in, make sure `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` is set; otherwise the dense traversal path bypasses the streaming helper and the gate never engages.

## Tests

- `tests/test_metrics.py` exercises the chunk kernel (distance + radius masks) and validates that residual distances computed via kernel reuse match the dense path.
- `tests/test_traverse.py` now includes a residual sparse traversal regression (dense vs. streamed scopes) to ensure parents/levels/scopes remain consistent.
- `tests/test_conflict_graph.py` gained a residual parity check to confirm dense Euclidean and residual-aware models produce identical CSR structures.

## Operational Notes

- Residual mode requires `backend.name == "numpy"` (NumPy backend) until the GPU/JAX kernels are ported.
- The decoder must map tree payloads to dataset indices; if trees store transformed buffers, supply a custom `point_decoder` when creating `ResidualCorrHostData`.
- The chunk kernel honours `chunk_size`; tune it to balance host-side streaming vs. cache reuse.
- Setting `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` now engages residual streaming automatically when the metric is residual-correlation; otherwise, traversal falls back to the dense mask path.
