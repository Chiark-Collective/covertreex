# Residual Audit Implementation Plan

_Last updated: 2025-11-12_

This document tracks the concrete engineering work required to execute the November 2025 residual audit directions. Each section cites the current code, describes the implementation gaps, and outlines the changes we intend to land.

## Status snapshot (2025-11-12)

**What’s done.**
- Phase 0 telemetry is feature-complete: every residual traversal path (parent search, scope streaming, dense fallbacks) emits `traversal_{whitened_block,kernel_provider}_{pairs,ms,calls}`, `cli/queries` prints a coverage summary at the end of each run, and `docs/residual_metric_notes.md` now explains how to interpret the new CLI block for auditors.
- Residual traversal selection is deterministic: `_ResidualTraversal` triggers for all NumPy residual builds, the Euclidean dense fallback only applies when `metric="euclidean"`, and `cli/queries` refuses to run the residual metric without Numba + NumPy so we can depend on cached pairwise blocks everywhere.
- Diagnostics guardrails are in place: `tools/export_benchmark_diagnostics.py` now surfaces the metric column and raises if any residual batch reports `conflict_pairwise_reused=0`; regression tests cover both the aggregator and the reuse behaviour across dense/sparse chunk targets.

**What’s next.**
- Phase 5 validation: rerun the residual Hilbert presets starting from a 4 k-point shakeout (to catch regressions quickly) before replaying the 32 k workload with the builder-only path so `traversal_semisort_ms` collapses; dump JSONL/CSV telemetry for every run under `artifacts/benchmarks/` and compare against the November 11 baselines.
- Track the `/dev/shm` semaphore warning (Numba parallel init) noted in the Phase 0 validation snapshot and decide whether to fix it in the runtime guardrails (Phase 7) or via environment setup.

## Phase 0 — Hotspot Telemetry Injection

**Current code.** Residual traversal now tracks `whitened_block_pairs/ms` and `kernel_provider_pairs/ms` inside `compute_residual_distances_with_radius`, threads those counters through both streaming helpers, and exposes them via `TraversalTimings`, the benchmark JSONL (new `traversal_whitened_block_{pairs,ms}` / `traversal_kernel_provider_{pairs,ms}` fields), and `tools/export_benchmark_diagnostics.py`. We also measure a per-plan `build_wall_seconds` so telemetry can consistently attribute “total traversal” time instead of relying on the surrounding CLI logs. Dense traversal still lacks the counters, and we haven’t plumbed the numbers into any dashboards beyond the CSV exporter, so the acceptance gate hasn’t been evaluated yet.

**Plan.**
- ✅ Mirror the counters in the dense Euclidean fallback and conflict-graph reuse path so every residual build (even when the gate is disabled or traversal falls back to dense chunks) reports the same metrics. (`_residual_find_parents`, `compute_residual_pairwise_matrix`, and both streaming helpers now share a single `ResidualDistanceTelemetry`/`ResidualWorkspace`.)
- Teach `cli/queries.py` / downstream notebooks to surface the new fields (e.g., histogram them per batch) and document how to interpret “pairs” vs. “milliseconds”. `tools/export_benchmark_diagnostics.py` now emits per-call/per-pair summaries, but the CLI still needs a first-class view.
- ✅ Re-run the Hilbert 32 k suite with the new telemetry enabled, capture the ≥80 % coverage target, and stash the resulting numbers here for later phases.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/traverse/base.py](../covertreex/algo/traverse/base.py), [covertreex/algo/batch/insert.py](../covertreex/algo/batch/insert.py), [covertreex/telemetry/logs.py](../covertreex/telemetry/logs.py), [tools/export_benchmark_diagnostics.py](../tools/export_benchmark_diagnostics.py).

**Progress:** Instrumentation merged 2025‑11‑11: dense parent search, pairwise cache builds, and both streaming helpers now report `traversal_{whitened_block,kernel_provider}_{pairs,ms,calls}`, and `docs/residual_metric_notes.md` explains how to read the “pairs” vs. “milliseconds” columns. The Hilbert gate replay (below) now hits the ≥80 % coverage bar, and as of 2025‑11‑12 the CLI emits the same telemetry summary after every run so we no longer need CSV exports for spot checks. Remaining docs/backlog work: add a short How-To in `docs/RESIDUAL_PCCT_PLAN.md` once we have the gate-on vs gate-off comparison plots (tracked separately).
- 2025‑11‑11 — Gate preset replay completed for the Hilbert 32 k workload: `python -m cli.queries --metric residual --dimension 8 --tree-points 32768 --batch-size 512 --queries 256 --k 8 --seed 0 --batch-order hilbert --residual-gate lookup --residual-gate-margin 0.2 --log-file residual_phase01_hilbert_gate.jsonl --residual-scope-cap-output residual_scope_caps_gate.json`. Aggregates at `artifacts/benchmarks/residual_phase01_hilbert_gate.csv` show `whitened_block_pairs_sum=7.61×10^8`, `kernel_provider_pairs_sum=7.78×10^8`, and `whitened_to_kernel_pair_ratio=0.9785` (>0.8). The raw telemetry lives at `artifacts/benchmarks/residual_phase01_hilbert_gate.jsonl`, scope-cap derivation at `artifacts/benchmarks/residual_scope_caps_gate.json`.
- 2025‑11‑12 — `python -m cli.queries --metric residual ...` now streams the residual counters directly to stdout after each run. The new `ResidualTraversalTelemetry` helper hooks into every batch insert, aggregates `whitened_block_{pairs,ms,calls}` vs `kernel_provider_*`, prints per-batch medians/p90s, the aggregate coverage ratio, and an explicit “pair/time mix” sentence so we can quote ≥80 % coverage without exporting CSVs. `docs/residual_metric_notes.md` documents the summary format and clarifies how to interpret the “pairs” vs. “milliseconds” lines for auditors.

**Validation snapshot (2025‑11‑11).**
- Command: `python -m cli.queries --metric residual --dimension 8 --tree-points 4096 --batch-size 512 --queries 256 --k 8 --seed 0 --log-file artifacts/residual_phase01.jsonl --residual-scope-cap-output artifacts/residual_scope_caps.json`
- Run ID: `pcct-20251111-140854-e97e63`
- Telemetry confirms `traversal_engine="residual_parallel"` with the new `traversal_whitened_block_{pairs,ms,calls}` / `traversal_kernel_provider_{pairs,ms,calls}` fields populated.
- Artifacts: JSONL at `artifacts/benchmarks/artifacts/residual_phase01.jsonl`, scope caps at `artifacts/benchmarks/artifacts/residual_scope_caps.json`.
- Warnings observed: Numba emitted `/dev/shm` semaphore warning (permissions), and CLI prints the new engine/thread banner (`engine=residual_parallel gate=off blas_threads=32 numba_threads=32`).
- Follow-ups: fix `/dev/shm` permissions so Numba parallel init is quiet; consider `--blas-threads` / `--numba-threads` overrides in `cli/queries` (currently default to `os.cpu_count()` when unset).

## Phase 1 — Whitened-V Block GEMM (Primary Lever)

**Current code.** `covertreex/metrics/residual/core.py` now ships `ResidualWorkspace` plus `compute_whitened_block(...)`, and the serial streamer calls `compute_residual_distances_with_radius(..., kernel_row=None, workspace=...)` so Gate‑1 can prune candidates before we ever request kernel tiles. Survivors trigger lazy kernel fetches (either the cached row or a subset) which preserves the audit/profile hooks. When Gate‑1 is disabled we stay on the parallel streamer, but it still instantiates fresh scratch buffers and skips the Phase 0 counters, so we cannot yet attribute time spent in the GEMM vs. kernel fallback.

**Plan.**
- Finish threading the workspace through every chunk consumer: the shared `ResidualWorkspace` now backs both serial and parallel streamers, so the next steps are (1) keeping the parallel fallback in pure SGEMM even when the gate is disabled and (2) validating that wider `scope_chunk_target` settings (8 k–16 k) don’t fragment the allocator.
- With Phase 0 counters available end-to-end, log `whitened_block_pairs/ms` vs. `kernel_provider_pairs/ms` for the Hilbert sweep to prove the SGEMM path dominates the traversal budget.
- Apply the gate / radius checks directly on the SGEMM output for dense and sparse conflict-graph callers (the CSR builder still shells out to `_distance_chunk` in a few fallback cases).
- Acceptance criteria (unchanged): median dominated-batch `traversal_ms` ≤150 ms and 32 k residual builds finish in “tens of seconds” before later phases.

**Key files:** [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [covertreex/metrics/_residual_numba.py](../covertreex/metrics/_residual_numba.py), [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/_residual_scope_numba.py](../covertreex/algo/_residual_scope_numba.py).

**Progress:**
- 2025‑11‑11 — `_collect_residual_scopes_streaming_parallel` now reuses the shared `ResidualWorkspace` and calls `compute_residual_distances_with_radius(..., force_whitened=True)` for every chunk, so the “parallel” path emits the same `whitened_block_{pairs,ms}` telemetry even when Gate‑1 is disabled. Cached level probes reuse the same helper (and telemetry), meaning dense runs consistently report SGEMM coverage while still sharing the block kernel that produces the raw distances. `docs/residual_metric_notes.md` documents the new `force_whitened` hook plus the CLI command we’ll use to snapshot Hilbert 32 k coverage (target ≥80 % whitened pairs / batch). Conflict-graph call-sites continue to use the cached pairwise matrices; the remaining manual `_distance_chunk` fallbacks are tracked under Phase 3.
- 2025‑11‑11 — Ran the Hilbert 32 k sweep with `COVERTREEX_SCOPE_CHUNK_TARGET=8192` and `python -m cli.queries --metric residual --dimension 8 --tree-points 32768 --batch-size 512 --queries 256 --k 8 --seed 0 --batch-order hilbert --log-file artifacts/benchmarks/artifacts/residual_phase01_hilbert.jsonl`. The run completed (log at `artifacts/benchmarks/artifacts/benchmarks/artifacts/residual_phase01_hilbert.jsonl`, CSV summary at `artifacts/benchmarks/residual_phase01_hilbert.csv`), but traversal still averaged ~32 s per dominated batch and the global `whitened_to_kernel_pair_ratio` plateaued at 0.43× because `_residual_find_parents` continues to stream raw kernel tiles (≈15 M pairs/batch) before the SGEMM path takes over. Next validation should either (a) move parent search onto the whitened workspace so kernel pairs drop below the ≥80 % coverage target or (b) split telemetry to compare SGEMM vs kernel only within the scope streamer.
- 2025‑11‑11 — `_residual_find_parents` now routes every chunk through `compute_residual_distances_with_radius(..., force_whitened=True)` using the shared `ResidualWorkspace`. When Gate‑1 is enabled (`COVERTREEX_RESIDUAL_GATE1=1`), parent search only fetches kernel tiles for whitened survivors: on a synthetic 4 096×8 benchmark (512 queries, chunk size 256) we observed 1 124 864 whitened pairs vs **528** kernel pairs (99.95 % coverage), whereas the pre-change Hilbert sweep reported only 0.43× coverage because every parent scan streamed the dense kernel block.
- 2025‑11‑11 — Added a per-chunk fallback for Gate‑1 audits: if `compute_residual_distances_with_radius` raises the “Residual gate pruned…” error, we immediately rescan that chunk via `_residual_parent_kernel_block` so parents remain exact while the rest of the traversal stays on SGEMM. The Hilbert 32 k gate run (`artifacts/benchmarks/residual_phase01_hilbert_gate.jsonl`, run id `pcct-20251111-211123-2f711f`) demonstrates the end-to-end effect: `whitened_to_kernel_pair_ratio=0.9785`, `traversal_kernel_provider_pairs_sum≈7.78×10^8`, `traversal_whitened_block_pairs_sum≈7.61×10^8`, and audit stayed green (`traversal_gate1_pruned=0`).
- 2025‑11‑12 — Phase 4 float32 staging landed: `ResidualCorrHostData` now stores `v_matrix`, `p_diag`, `kernel_diag`, and `v_norm_sq` in float32, exposes `.v_matrix_view(dtype)` for audits/gates, and `compute_residual_pairwise_matrix` plus the residual traversal cache persist pairwise blocks in float32. Conflict graph now ingests these matrices without upcasting, `cli/queries` raises as soon as `conflict_pairwise_reused=0`, and regression tests assert both the dtype invariants and the Gate‑1 audit path (no false negatives recorded in `ResidualGateProfile`).

## Phase 2 — GEMM-Based Raw Kernel Provider (Exact Fallback)

**Current code.** `covertreex/metrics/residual/host_backend.py:14-112` still defines `_rbf_kernel` via full broadcasting (`x[:, None, :] - y[None, :, :]`) and wires it directly into `kernel_provider`. Every traversal chunk therefore allocates a (rows × cols × dim) float64 tensor under the GIL.

**Plan.**
- Introduce a helper (e.g., `make_rbf_provider(points_f32, gamma, *, workbuf=None)`) that keeps the dataset in `float32`, precomputes `row_norms = (points_f32 ** 2).sum(1)`, and computes tiles via `A @ B.T` (`SGEMM`).
- Extend `ResidualCorrHostData` (or a sibling struct) with optional `kernel_points_f32`, `kernel_row_norms`, and `kernel_workbuf` so traversal can reuse the buffers without reallocation.
- Update `build_residual_backend(...)` to normalise inputs once (`np.asarray(points, dtype=np.float32, order="C")`), stash the contiguous copy plus norms, and expose the GEMM provider. The provider should accept `out` buffers to avoid repeated allocations inside `_collect_residual_scopes_streaming_*`.
- Ensure `kernel_provider` always returns `float32` tiles; downstream callers (`compute_residual_distances_from_kernel`) can promote to `float64` only when necessary.
- Add unit tests in `tests/test_metrics.py` to confirm the provider matches the existing `_rbf_kernel` numerically (within 1e-6 relative error) and stays in float32.
- Acceptance criteria:
  * `kernel_provider_ms` collapses to a handful of large calls per batch (10–50× faster).
  * Survivors only trigger the provider, confirming Phase 1 masks eliminated most work.

**Key files:** [covertreex/metrics/residual/host_backend.py](../covertreex/metrics/residual/host_backend.py), [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [covertreex/metrics/_residual_numba.py](../covertreex/metrics/_residual_numba.py), [tests/test_metrics.py](../tests/test_metrics.py).

**Progress:**
- 2025‑11‑11 — `build_residual_backend` now materialises a float32, C-contiguous copy of the dataset plus squared norms and exposes an SGEMM-based RBF provider, so traversal no longer allocates the broadcasted `(rows × cols × dim)` tensor per chunk. `ResidualCorrHostData` stores the staged buffers (`kernel_points_f32`, `kernel_row_norms_f32`) for future reuse, and `tests/test_metrics.py::test_sgemm_kernel_provider_matches_reference` verifies numerical parity with the old float64 path.
- 2025‑11‑11 — Re-ran the Hilbert 32 k sweep with `COVERTREEX_SCOPE_CHUNK_TARGET=8192`, `MKL/OMP/OPENBLAS=1`, `NUMBA=32` (command: `python -m cli.queries --metric residual --dimension 8 --tree-points 32768 --batch-size 512 --queries 256 --k 8 --seed 0 --batch-order hilbert --log-file artifacts/benchmarks/artifacts/residual_phase01_hilbert_sgemm.jsonl`). Log lives at `artifacts/benchmarks/artifacts/benchmarks/artifacts/residual_phase01_hilbert_sgemm.jsonl`, CSV summary at `artifacts/benchmarks/residual_phase01_hilbert_sgemm.csv`. `traversal_kernel_provider_ms_sum` fell from 42.9 s (pre-SGEMM parent search) to **8.6 s** (≈5× reduction) even though we still stream the full parent-search matrix, and the traversal median dropped to ~20.2 s per dominated batch. `whitened_to_kernel_pair_ratio` remains ~0.43× because `_residual_find_parents` is still issuing raw kernel tiles; once that path moves to SGEMM (or we report streamer-only ratios) we should pass the ≥80 % coverage target.
- Follow-ups: document the gate-enabled Hilbert replay (`artifacts/benchmarks/residual_phase01_hilbert_gate.{jsonl,csv}`) inside `docs/residual_metric_notes.md` and continue monitoring the lookup (still `traversal_gate1_pruned=0`).
- 2025‑11‑11 — Conflict-graph fallbacks now share the same float32 backend: when the residual pairwise cache is unavailable we chunk the adjacency rows through `compute_residual_distances_with_radius` (with `force_whitened=True`), so the dense filter only requests SGEMM kernel tiles for candidates that survive the whitened mask. This replaces the previous `_rbf_kernel` broadcast path in `build_conflict_graph` and keeps the telemetry counters consistent with traversal.

## Phase 3 — Enforce Residual Pairwise Reuse & Strategy Selection

**Current code.** `build_conflict_graph` (`covertreex/algo/conflict/runner.py:55-205`) already raises `ResidualPairwiseCacheError` when a residual batch arrives without cached pairwise distances, and `_collect_residual(...)` (`covertreex/algo/traverse/strategies.py:1204-1399`) constructs a `ResidualTraversalCache` every time it runs. The weak point is the selection predicate (`strategies.py:1405-1426`): residual traversal is only chosen when both `runtime.enable_sparse_traversal` and `runtime.enable_numba` are true. If a user invokes the residual metric with those toggles off (“dense” regressions often do), the scheduler falls back to the Euclidean strategies, no cache is produced, and conflict graph either recomputes kernels or aborts.

**Plan.**
- Relax the traversal predicate so `_ResidualTraversal` is selected for every NumPy residual run, independent of the sparse toggle; emulate “dense vs sparse” solely via `runtime.scope_chunk_target` and related caps.
- Make the Euclidean strategies fail fast when `runtime.metric == "residual_correlation"` so misconfigurations are immediately visible in tests.
- Teach `cli/queries` to auto-enable Numba + residual traversal (or print a clear error) whenever `--metric residual` is requested, ensuring telemetry consistently records `conflict_pairwise_reused=1`.
- Keep the existing `ResidualPairwiseCacheError` but add regression tests in `tests/test_conflict_graph.py` that confirm cache-less traversals raise, and that both scope-cap=0 (“dense”) and scope-cap>0 (“sparse”) modes reuse the cached block.
- Plumb the reuse flag through `covertreex/algo/batch/insert.py` and `covertreex/telemetry/logs.py` so diagnostics can assert every residual batch set `pairwise_reused=1`; expand `tools/export_benchmark_diagnostics.py` to treat any zero as a hard failure.
- Acceptance criteria:
  * Conflict graph timings stay ≤50 ms per dominated batch in both dense/sparse recipes.
  * Every residual batch logs `conflict_pairwise_reused=1`; CI fails otherwise.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/conflict/runner.py](../covertreex/algo/conflict/runner.py), [covertreex/algo/conflict/strategies.py](../covertreex/algo/conflict/strategies.py), [covertreex/algo/batch/insert.py](../covertreex/algo/batch/insert.py), [covertreex/telemetry/logs.py](../covertreex/telemetry/logs.py), [tools/export_benchmark_diagnostics.py](../tools/export_benchmark_diagnostics.py), [tests/test_conflict_graph.py](../tests/test_conflict_graph.py), [cli/queries.py](../cli/queries.py).

**Progress:**
- 2025‑11‑12 — `_ResidualTraversal` is now selected for every NumPy residual build regardless of `enable_sparse_traversal`. The Euclidean dense fallback only registers when `runtime.metric == "euclidean"`, so mis-configured residual runs fail fast instead of silently falling back to Euclidean traversal.
- 2025‑11‑12 — `cli/queries` hard-stops when `--metric residual` is paired with `backend!=numpy` or `enable_numba=0`, ensuring residual batch telemetry always records `conflict_pairwise_reused=1` once the cache plumbing lands.
- 2025‑11‑12 — `tools/export_benchmark_diagnostics.py` now inspects `conflict_pairwise_reused` and raises when any residual batch reports `0`, and regression tests cover the helper so future residual runs cannot regress without failing CI.
- 2025‑11‑12 — `cli/queries` residual telemetry now prints a per-run `conflict_pairwise_reuse` summary and raises immediately when any batch reports `conflict_pairwise_reused=0`, so CI and local runs fail in place without waiting for CSV diagnostics.

## Phase 4 — Canonical Float32 Staging & Accuracy Guardrails

**Current code.** `ResidualCorrHostData` stores `v_matrix`, `p_diag`, and `kernel_diag` as float64 (`covertreex/metrics/residual/core.py:52-123`), and `configure_residual_correlation` re-computes `v_norm_sq` by re-casting the matrix back to float64. The host backend builder (`host_backend.py:86-109`) also emits float64 everywhere, so traversal drags twice the bandwidth it needs.

**Plan.**
- Switch `ResidualCorrHostData` to accept float32 matrices by default. Keep float64 buffers only where numerically necessary (e.g., when running Cholesky inside `build_residual_backend` or for auditing radii) and downcast once the stable factors are computed.
- Materialise and cache both `v_matrix_f32` and `v_matrix_f64` if Gate‑1 or audits still require float64 whitened vectors. Expose convenience properties (e.g., `.v_matrix_view(dtype)`) so Numba kernels (`metrics/_residual_numba.py`) can consume float32 without extra copies.
- Update `_compute_residual_distances_from_kernel` and `distance_block_no_gate` invocations to accept float32 inputs and cast internally only for intermediate numerics.
- Guard all `np.asarray(..., dtype=np.float64)` calls in `configure_residual_correlation` and `compute_*` helpers; only promote when the downstream API truly requires double precision.
- Add regression tests that load a float32 backend, run traversal, and confirm no implicit float64 upcasts occur (inspect `ndarray.dtype` on Cached arrays inside `ResidualTraversalCache`).
- Document a clear accuracy policy: e.g., “factorisation + audit remain float64; traversal/conflict run in float32 with tolerances ≤1e-4”. Add a test that audit mode never flags false positives after the dtype change.
- Acceptance criteria: memory footprint drops measurably (≈2× on caches), and audit passes remain clean.

**Key files:** [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [covertreex/metrics/residual/host_backend.py](../covertreex/metrics/residual/host_backend.py), [covertreex/metrics/_residual_numba.py](../covertreex/metrics/_residual_numba.py), [covertreex/algo/conflict/runner.py](../covertreex/algo/conflict/runner.py), [docs/residual_metric_notes.md](residual_metric_notes.md), [tests/test_metrics.py](../tests/test_metrics.py), [tests/test_conflict_graph.py](../tests/test_conflict_graph.py).

**Progress:**
- 2025‑11‑12 — `ResidualCorrHostData` now stages every traversal buffer in float32, exposes `.v_matrix_view(dtype)` / `.p_diag_view(dtype)` helpers for audits, and caches float64 views on demand. Cached pairwise tiles and `ResidualTraversalCache` entries stay in float32, slashing cache size ~2× without tripping audit tolerances.
- 2025‑11‑12 — The SGEMM kernel provider in `build_residual_backend` returns float32 tiles backed by a reusable workset (`kernel_points_f32`, `kernel_row_norms_f32`), and the conflict-graph fallbacks route through the same whitened workspace so diagnostics see consistent telemetry regardless of cache reuse.
- Follow-up: document the accuracy policy (float64 factorisation + audit, float32 traversal) inside `docs/residual_metric_notes.md` / `docs/CORE_IMPLEMENTATIONS.md` and add an audit-mode regression that exercises the mixed-precision path.

## Phase 5 — Selection with Deterministic Tie-Breaks

**Current code.** Dense traversal still sorts entire CSR masks using `np.lexsort` (`covertreex/algo/traverse/strategies.py:184-199`). `covertreex/algo/semisort.py` likewise sorts every `(key, value)` pair even though the downstream MIS only needs the first `scope_limit` entries per query.

**Plan.**
- ✅ Introduce a utility (now `select_topk_by_level`) that uses `np.argpartition` + deterministic tie breaks, and wire it into the Euclidean dense fallback as well as the residual streaming helpers so every path respects `scope_chunk_target` without a full-row sort.
- ✅ Within `_collect_residual_scopes_streaming_*`, stop appending entire `tree_positions` ranges when scopes already exceed `scope_limit`; the streaming code now feeds the temporary buffers through `select_topk_by_level` (with parent re-insertion) before emitting CSR chunks, so telemetry/caches see the capped order immediately.
- Extend the Numba CSR builder (`covertreex/algo/_scope_numba.py`) with a selection-based path when `scope_chunk_target` / `scope_limit` is non-zero; this keeps parity between dense and sparse configurations.
- After any `argpartition`, run a stable sort on the selected slice using `(level, index)` tie breakers so CSR output stays deterministic across runs (covered by the helper, but the Numba builder still needs the analogous ordering rule).
- Update regression tests (e.g., `tests/test_traverse.py::test_residual_scope_limit`) to assert that the new selection path yields the same neighbor set as the full sort when `k` >= actual scope size, and that telemetry counters (`scope_chunk_max_members`) reflect the capped selection.
- Acceptance criteria: `semisort_seconds` approaches zero and MIS regressions stay absent due to deterministic ordering.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/_scope_numba.py](../covertreex/algo/_scope_numba.py), [covertreex/algo/semisort.py](../covertreex/algo/semisort.py), [tests/test_traverse.py](../tests/test_traverse.py), [docs/CORE_IMPLEMENTATIONS.md](CORE_IMPLEMENTATIONS.md).

**Progress:**
- 2025‑11‑12 — Added `select_topk_by_level` in `covertreex/algo/semisort.py`, updated exports, and covered it with unit tests so both dense Euclidean and residual traversals can cap scopes via `np.argpartition` instead of `np.lexsort` on the full row.
- 2025‑11‑12 — Residual traversal (serial + parallel streamers) now orders/limits scope vectors through the new helper, re-inserts the parent when necessary, and reports capped scopes in the CSR builder + cache telemetry. Euclidean dense traversal gained the same limiting behavior, and the sparse traversal path now trims CSR rows after the Numba builder when `scope_chunk_target>0`.
- 2025‑11‑12 — `build_scope_csr_from_pairs` (Numba CSR builder) now supports deterministic level-aware ordering and enforces `scope_limit` directly (including the parent re-insertion pass). New regression tests (`tests/test_scope_numba.py`) cover the limit/ordering semantics so future sparse-traversal changes can lean on the Numba path without Python-side trimming.
- 2025‑11‑12 — Completed the Phase 5 cleanup: both residual streamers now let `build_scope_csr_from_pairs` handle ordering/limits (conflict scopes are rebuilt from the CSR results and `scope_chunk_emitted` still tracks caps), and Euclidean sparse traversal funnels every scope through the same builder instead of the removed `_limit_scope_csr` helper. Signal fidelity is verified by rerunning `python -m pytest tests/test_traverse.py tests/test_scope_numba.py`.
- 2025‑11‑12 — Replayed the Hilbert 32 k residual telemetry sweep with `COVERTREEX_SCOPE_CHUNK_TARGET=8192` (outputs under `artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert.jsonl`). Aggregate coverage stayed at `whitened_block_pairs_sum=1.475e9`, `kernel_provider_pairs_sum=1.507e9` → **0.979×** coverage, with whitened work accounting for **92.7 %** of traversal wall time even though pairs are split ≈50/50 across the cold-start batch. Every later batch reported `conflict_pairwise_reused=1` (123 batches total; the first conflict graph is empty so there’s no cache to reuse). Semisort timing is now the remaining lever: median `traversal_semisort_ms≈28.1 s`, p90 ≈63 s, max ≈74.7 s, providing the “before” numbers for the builder-only follow-up.
- 2025‑11‑12 — That same 32 k replay (`artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert.jsonl`) stretched past an hour end-to-end, so Phase 5 validation now mandates a short 4 k-point shakedown (identical flags, smaller `--tree-points`) before each 32 k rerun. Capture both telemetry files to compare against the November 11 “before” run and fail fast if `traversal_semisort_ms` or coverage regresses.
- Next up: rerun the phase‑5 telemetry sweep (Hilbert/coverage) so `traversal_semisort_ms` can be reported as near-zero and the before/after scope/semisort dashboards capture the impact of the builder-only path; stash the CSV/json artifacts for the later Phase 6 write-ups.

## Phase 6 — Remove Python Callouts from Hot Numba Loops

**Current code.** The streamer functions (`_collect_residual_scopes_streaming_serial/parallel` in `covertreex/algo/traverse/strategies.py:295-744`) repeatedly call the Python-level `kernel_provider` inside tight loops, and gate telemetry is updated via Python dataclasses on every chunk. Even though `_residual_scope_numba.py` provides Numba helpers, the surrounding loops stay in Python.

**Plan.**
- After landing the GEMM provider, restructure `_collect_residual_scopes_streaming_parallel` so that each block batches kernel requests: gather candidate indices into a contiguous array, pass them to the provider once per chunk, and immediately feed the resulting `float32` tile into a Numba `@njit` routine that filters survivors and updates telemetry structs stored in plain `np.ndarray`s rather than Python objects.
- Move gate telemetry accumulation into a lightweight struct of NumPy scalars; pass it by reference into the Numba kernel and only convert it back to `ResidualGateTelemetry` at the end of traversal.
- Ensure level-cache prefetching (`level_scope_cache` MAP) happens outside the Numba loop; once we enter `_distance_chunk`, no Python callbacks should be needed.
- Document the new “no Python in chunk loop” rule inside `docs/residual_metric_notes.md` and add a micro-benchmark under `benchmarks/` to assert we can stream 512×8 192 tiles without hitting the GIL.
- Acceptance criteria: nopython mode stays engaged; profiler shows negligible Python overhead in traversal once the preceding phases land.

**Key files:** [covertreex/algo/traverse/strategies.py](../covertreex/algo/traverse/strategies.py), [covertreex/algo/_residual_scope_numba.py](../covertreex/algo/_residual_scope_numba.py), [covertreex/metrics/residual/core.py](../covertreex/metrics/residual/core.py), [benchmarks/](../benchmarks).

**Progress:** Not started.

## Phase 7 — Threading & Runtime Guardrails

**Current code.** Neither `cli/queries.py` nor the residual builder constrains OpenMP / MKL / BLAS thread counts. When NumPy calls into MKL for the GEMM provider, it competes with Numba’s parallel loops, hurting determinism.

**Plan.**
- Inside `cli/queries.py` (around `main()` at lines 40-80) add a guard that sets `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and `NUMBA_NUM_THREADS` to 1 (or a user-provided override) unless the environment already specifies values. Document the defaults in the CLI help text.
- For library consumers, expose `covertreex.runtime.configure_threading(max_threads)` so non-CLI entrypoints can opt in; when defaults need to be applied, log a short warning (“defaulting MKL_NUM_THREADS=1; override via env”) instead of silently overriding user settings.
- Update `docs/CORE_IMPLEMENTATIONS.md` with a note that residual builds assume single-threaded BLAS for kernel tiles, and mention how to raise the limit safely.
- Acceptance criteria: rerunning the same config on a quiet machine stays within ±3 % wall clock, and users retain explicit control via env or the new API.

**Key files:** [cli/queries.py](../cli/queries.py), [covertreex/runtime/__init__.py](../covertreex/runtime/__init__.py), [covertreex/runtime/config.py](../covertreex/runtime/config.py), [docs/CORE_IMPLEMENTATIONS.md](CORE_IMPLEMENTATIONS.md).

**Progress:** Not started.

---

Once these sections land we can revisit the lower-priority gate experiments outlined in `docs/RESIDUAL_PCCT_PLAN.md`, confident that the core traversal cost structure matches the audit recommendations.
