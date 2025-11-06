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

## Next Steps

1. Wire the scope chunk limiter (runtime `scope_chunk_target`, segmented builder) through the new builder split and surface chunk-hit telemetry in the batch logs.
2. Optimise the residual path by reusing the cached pairwise matrix inside the adjacency filter and tightening `_collect_residual_scopes_streaming` so residual scopes stop ballooning past 16 k members.
3. Promote the Tier-C async refresh + GPU smoke tests so the journal pipeline and builder split are exercised under concurrency/backends beyond NumPy.
