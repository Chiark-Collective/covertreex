# Refactor Plan — 2025-11-06

This document captures the near-term refactor opportunities identified during
the repository audit. The focus is to retire accumulated technical debt and
stabilise the parallel compressed cover tree (PCCT) code paths before the next
optimisation sprint.

## 1. Harden Backend & Configuration Plumbing ✅ (shipped 2025-11-06)

- Replace the global `DEFAULT_BACKEND` singleton with an injectable runtime
  context that can be passed to trees, algorithms, and diagnostics.
- Delay backend and device resolution until it is explicitly requested, so
  unit tests and long-lived services can switch configurations without
  re-importing modules.
- Trim the eager environment mutation inside `covertreex.config` in favour of
  a pure `RuntimeConfig` builder that returns a structured object; only apply
  side effects (e.g. logging setup, JAX flags) when the context is “activated”.
- Expected outcome: fewer hidden global dependencies, easier backend swapping,
  and cleaner test parametrisation across NumPy/JAX/Numba combinations.

Status: Completed via `RuntimeContext` + `get_runtime_backend` refactor (2025-11-06).

## 2. Modularise Traversal Strategies

- Extract Euclidean dense, Euclidean sparse, and residual traversal flows into
  dedicated strategy classes or functions instead of branching inside
  `traverse_collect_scopes`.
- Centralise shared helpers (radius computation, chunk telemetry,
  `TraversalTimings`) so each strategy maintains its own data conversions and
  kernels.
- Ensure the dispatcher respects the runtime context (metric, sparsity flag,
  backend) and exposes identical outputs, keeping the public API untouched.
- Expected outcome: reduced duplication, easier integration of the sparse
  traversal milestone, and clearer performance profiling per strategy.

Status: Strategy-based traversal dispatcher landed 2025-11-06.

## 3. Rework Persistence Cloning

- Introduce a journal-based update pipeline that batches tree mutations
  (parents, levels, child chains, caches) during batch insert/delete planning.
- Add a backend-aware “apply journal” helper (NumPy now, Numba path later) that
  performs a single copy-on-write sweep per batch instead of multiple
  round-trips through NumPy.
- Update persistence tests to cover the new path for both NumPy and JAX
  backends to preserve immutability guarantees.
- Expected outcome: lower allocation pressure during rebuilds and a clear
  abstraction boundary between planning and persistence application.

Status: Completed via `PersistenceJournal`, pooled `JournalScratchPool`, and the
backend-aware `apply_persistence_journal` helper now invoked by `batch_insert`.
NumPy/Numba journal sweeps and the copy-on-write fallback both ship with new
unit coverage (2025-11-06), and the refreshed 32 768-point benchmarks report
PCCT builds at 42.13 s (Euclidean) / 56.09 s (residual) with steady-state query
latencies below 0.23 s (see docs/CORE_IMPLEMENTATIONS.md).

## 4. Extract Conflict Graph Builders

- Split dense vs segmented vs residual conflict-graph construction into
  self-contained modules that return a shared `ConflictGraphArtifacts`
  structure.
- Keep the top-level `build_conflict_graph` as orchestration only (scope
  preparation, metric selection, telemetry wiring).
- Align the implementation with the pending “scope chunk limiter” work so each
  builder can be benchmarked and optimised independently.
- Expected outcome: narrower files, more targeted profiling, and simpler
  extension points for new batching strategies.

Status: Completed by introducing `conflict_graph_builders.py` (dense / segmented /
residual helpers + `AdjacencyBuild` telemetry), rewiring `build_conflict_graph`
into orchestration-only mode, and keeping the residual post-filtering hook that
now reuses builder CSR outputs (2025-11-06).

## 5. Relax Hard Dependency on JAX ✅ (shipped 2025-11-06)

- Move `jax`/`jaxlib` requirements into an optional extra (e.g. `pcct[jax]`) so
  CPU-only deployments can install the library without those wheels.
- Guard JAX-specific code paths with clearer error messages and fall back to
  the injected runtime backend automatically.
- Adjust tests that currently `importorskip("jax")` to tolerate environments
  without JAX by parametrising over available backends.
- Expected outcome: smoother installation in CPU environments, better support
  for lightweight CI jobs, and more accurate signalling about which features
  require JAX.

Status: JAX moved to an optional extra; JAX-only tests now skip when the backend is not available (2025-11-06).

## Recent updates — 2025-11-07

- **32 768-point residual sweep (scope clamp verified).** Replayed `benchmarks/queries.py --metric residual --baseline gpboost --dimension 8 --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --seed 42` twice with the new JSONL sink. The unclamped adjacency run (`COVERTREEX_SCOPE_CHUNK_TARGET=0`, log: `benchmark_residual_default_20251107_run2.{log,jsonl}`) now measures **1058.58 s build / 0.208 s query (4 919 q/s)** for PCCT versus **2.79 s / 11.09 s (92.3 q/s)** for GPBoost. Every dominated batch hit the 16 384-member traversal cap (63/64 batches trimmed) yet `conflict_scope_chunk_segments` stayed at 1 with `conflict_scope_chunk_max_members≈8.4 M`, and steady-state `conflict_adj_scatter_ms` averaged **54 ms** with >70 ms spikes. Enabling chunking (`COVERTREEX_SCOPE_CHUNK_TARGET=8192`, log: `benchmark_residual_scope8192_20251107.{log,jsonl}`) reduced build time to **625.19 s** and kept both traversal and conflict chunk maxima at the target (8 192 members, ≈470 segments emitted per dominated batch) while sustaining **0.252 s query / 4 063 q/s**. The chunked pass still shows `conflict_adj_scatter_ms≈33 ms` steady-state but the first dominated batch spends **364 ms** scattering across 1.9 k shards, so there is headroom to merge chunk tails once candidate pairs drop.

- **Scope chunk limiter wiring.** `RuntimeConfig.scope_chunk_target` now flows through the Euclidean sparse traversal path and the conflict-graph builders, and the new `COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS` guard keeps the Numba builder from exploding into thousands of shards. Chunk segmentation stats (`scope_chunk_segments/emitted/max_members`) are captured in `TraversalResult` timings and forwarded to the batch-insert logger so dominated batches expose chunk-hit telemetry without extra instrumentation.
- **Residual pipeline tightening.** Residual traversal streams now carry a cached pairwise matrix (`ResidualTraversalCache`) that the conflict-graph builder reuses instead of re-encoding kernels. `_collect_residual_scopes_streaming` enforces a configurable scope cap (default 16 384, overridable via `COVERTREEX_SCOPE_CHUNK_TARGET`) to keep residual scopes from exploding, and the integration tests cover both the capped traversal path and the cached pairwise reuse.
- **Benchmark logging sink.** `benchmarks/queries.py` gained `--log-file` plus the `BenchmarkLogWriter` helper so every batch insert emits JSONL telemetry (candidate counts, timing split, chunk stats, RSS deltas). The new artefacts above are the first long-form runs produced via this sink, giving us structured before/after traces instead of console scrapes.

## Next Steps

1. **Keep residual scopes clamped through adjacency.** The traversal-side cap is working (max 16 384 members with 63/64 scopes trimmed) but adjacency still sees 8 M-member groups when `scope_chunk_target=0`. Push the new `scope_chunk_max_segments` guard into the dense builder so it automatically shards these scopes (target: `conflict_scope_chunk_segments≈256`, `conflict_adj_scatter_ms≤50 ms`) even when chunking is nominally disabled.
2. **Retune chunked adjacency.** Chunked runs now cap memberships at 8 192 but still emit hundreds of shards, and the first dominated batch pays 364 ms scattering across 1.9 k segments. Teach `_chunk_ranges_from_indptr` to merge tail shards when candidate counts collapse so steady-state `conflict_scope_chunk_segments` drops into the “low hundreds” called out by the plan. The `--log-file` sink (done) gives us the per-batch telemetry to validate this.
3. **Refresh published metrics.** After (1) and (2) land, regenerate the 32 768-point Euclidean + residual rows in `docs/CORE_IMPLEMENTATIONS.md` and ship the new JSONL artefacts / chunk telemetry CSV so auditors can see the recovered 4.0–5.0 k q/s steady state next to the GPBoost baseline numbers.
