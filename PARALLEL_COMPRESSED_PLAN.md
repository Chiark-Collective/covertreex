# Parallel Compressed Cover Tree Implementation Plan

## Context & Goal

This library is being designed as the neighbour-selection backbone for the Vecchia variational Gaussian process codebase (JAX, GPU-centric). At inference/training time we need:

- **Fast, persistent neighbour graphs** that can be swapped atomically during LBFGS iterations (async rebuild worklets).
- **GPU-friendly kernels** that play nicely with JAX (no host ↔ device ping-pong), yet keep CPU fallbacks for preprocessing.
- **Compressed cover-tree semantics** so we reuse state across epochs and avoid memory blow-up.
- **Parallel batch updates** to hide rebuild latency during Vecchia state refresh and to accelerate offline preparation.

Deliver a reusable “parallel compressed cover tree” (PCCT) library that combines:

- **Compressed cover tree** representation from Elkin–Kurlin (one node per point, distinctive descendant caches, O(1) `Children`/`Next` lookups).
- **Parallel batch updates** from Gu–Napier–Sun–Wang (prefix-doubling batches, conflict-graph MIS for insertion/deletion, path-copying persistence).
- **Unified backend** built on `jax.numpy`, with optional Numba acceleration for tight loops.

## Modules & Responsibilities

| Module | Purpose |
| ------ | ------- |
| `covertreex/config.py` | Centralised configuration surface: parse env vars, choose backend/device, set precision flags, expose `RuntimeConfig` used by tree/algo modules. |
| `covertreex/core/tree.py` | Immutable `PCCTree` dataclass holding points, per-point top levels, parent pointers, compressed child tables, per-level cover sets, cached `S_i` and `Next`, log counters, backend reference. |
| `covertreex/core/persistence.py` | Copy-on-write helpers to clone level slices and child segments when applying updates (path-copying). |
| `covertreex/core/metrics.py` | Backend registry with `jax.numpy` default; Numba kernels exposed behind feature flag; metric abstraction (pairwise & pointwise distances). |
| `covertreex/algo/traverse.py` | Batched traversal (Alg. 4 lines 5–9) returning `(parent, level)` decisions and conflict scopes `Π_parent`. |
| `covertreex/algo/conflict_graph.py` | Builds CSR adjacency for `L_k` using restricted annuli `(Π_{P_i} ∩ L_k)`; optional Euclidean grid/binning. |
| `covertreex/algo/mis.py` | Luby MIS on CSR (pure `jax.numpy` version, plus Numba acceleration path). |
| `covertreex/algo/batch_insert.py` | Prefix-doubling orchestration, per-level MIS selection, redistribution (ceil-log₂ distance), tree updates via persistence helpers. |
| `covertreex/algo/batch_delete.py` | Bottom-up uncovered set handling, MIS-driven promotions, new-root handling (Alg. 5). |
| `covertreex/queries/knn.py` | Corrected CCT k-NN (Alg. F.2) using cached `S_i` / `Next` tables; convenience 1-NN. |
| `tests/` | Pytest suite for invariants, traversal scopes, conflict graphs, MIS, batch ops, k-NN, persistence, backend parity. |
| `benchmarks/` | CLI scripts for insert/delete throughput, k-NN latency, distance-count scaling. |

## Backend Strategy

- **Primary API:** `jax.numpy` (`jnp`) for all core math; ensures eventual JIT/HLO portability.
- **Acceleration:** optional Numba wrappers for hotspots (distance kernels, MIS) with seamless fallback.
- **Future:** pure NumPy shim if needed for environments without JAX.

## Working Plan — 2025-11-05

**Focus:** Drive the PCCT NumPy+Numba build path towards sequential-baseline parity by removing the dense traversal bottleneck and tightening the conflict-graph staging pipeline.

**Current metrics (NumPy backend, diagnostics off):**
- 2 048 tree points / 512 queries / *k*=8 ⇒ build 0.366 s, query 0.097 s (5 261 q/s).
- 8 192 tree points / 1 024 queries / *k*=16 ⇒ build 4.014 s, query 0.931 s (1 099 q/s).

**Milestones:**
1. **Sparse traversal kernel** (owner: Liam) — Replace the dense mask creation in `traverse_collect_scopes` with a tree-guided Numba kernel that walks parent chains and emits CSR scopes directly. Target: cut traversal time per dominated batch from ~20 ms to ≤5 ms and drop peak memory pressure. *Status:* design locked, prototype scaffolding in progress (scheduled for 2025-11-07 branch cut).
2. **Traversal ↔ conflict-graph plumbing** (owner: Liam) — Adapt `build_conflict_graph` to consume the sparse scopes, reusing existing Numba adjacency path. Ensure parity tests (`test_traverse.py`, Tier-B integration) cover both dense and sparse implementations before flipping the default. *Status:* pending the sparse kernel output schema.
3. **Scope chunk limiter** (owner: Priya) — Wire `RuntimeConfig.scope_chunk_target` into `_scope_numba` so very wide scopes stream through fixed-size segments; fall back to current behaviour when the limit is zero. Adds backpressure for 32 k+ builds and prepares the ground for pooled scratch buffers. *Status:* design reviewed, implementation queued behind milestone 1.
4. **Diagnostics + benchmarks refresh** (owner: Priya) — Once milestones 1–3 land, regenerate `runtime_breakdown_*` artefacts (NumPy + JAX paths) and update benchmark tables/docstrings so auditors can trace performance deltas. *Status:* blocked on earlier milestones; target date 2025-11-10.

## Configuration & Environment

- **Env surface (`COVERTREEX_*`):** `BACKEND` (`jax`, `numpy`, future), `DEVICE` (e.g. `gpu:0,cpu:0`), `PRECISION` (`float64` default, `float32` alt), `ENABLE_NUMBA` (`0/1`), `LOG_LEVEL`.
- **Randomness:** `COVERTREEX_MIS_SEED` seeds Luby MIS for deterministic runs.
- **JAX flags:** gate `jax_enable_x64` via either `COVERTREEX_PRECISION` or inherited `JAX_ENABLE_X64`; respect downstream overrides instead of hard-coding.
- **Device detection:** inspect `jax.devices()` on import, honour `COVERTREEX_DEVICE` filters, fall back to CPU with debug log if GPUs unavailable.
- **Backend registry:** expose `RuntimeConfig` singleton that modules pull from to obtain `TreeBackend` objects (JAX primary, NumPy shim, optional Numba kernels keyed off `ENABLE_NUMBA`).
- **DX expectations:** deterministic configuration (cache results, avoid implicit JAX global state), environment warnings routed through logger (respect `LOG_LEVEL`).

## Implementation Tasks

1. **Configuration bootstrap**
   - Implement `covertreex/config.py` with `RuntimeConfig.from_env()` and unit tests for env parsing and device selection fallbacks.
   - Ensure JAX precision/device flags are applied exactly once and support explicit overrides from downstream libraries.
   - Provide helpers for obtaining `TreeBackend` instances and reporting the active configuration.

2. **Core structure & persistence**
   - Implement `PCCTree`, including cover-node caches, child tables, and `S_i` / `Next`.
   - Add copy-on-write utilities and tests verifying version isolation.

3. **Traversal & scopes**
   - Implement `traverse_collect_scopes` (vectorised over batch points).
   - Normalise ragged scopes with semisort/group-by utilities.
   - Tests: compare to sequential compressed insertion decisions.
   - Status: traversal emits semisorted scopes + CSR buffers (`scope_indptr`, `scope_indices`) with cached-radius floor and `Next` chain expansion, and these buffers now feed the batch-insert redistribution path exercised in `tests/integration/test_parallel_update.py`.

4. **Conflict graph**
   - Build CSR from `Π_{P_i} ∩ L_k` with distance filtering (`<= 2^k`).
   - Add optional grid/binning for Euclidean spaces; fallback to pure scope scan.
   - Tests: adjacency matches brute-force edge detection.
   - Status: CSR builder live with distance-aware pruning, semisort buffers, and integer annulus bins (`annulus_bin_{indptr,indices,ids}`) ready for GPU kernels.

5. **MIS kernels**
   - Implement Luby MIS with `jax.numpy`.
   - Provide Numba-accelerated version sharing the same interface, gated by runtime config.
   - Tests: correctness (independence + maximality) on random graphs; parity checks across backends.
   - Status: JAX and Numba MIS paths now share the same entrypoint (`run_mis`), seed batching plumbed via `batch_mis_seeds`, and `tests/test_mis.py` toggles `COVERTREEX_ENABLE_NUMBA` to assert parity.

6. **Batch insert (Alg. 4)**
   - Prefix-doubling grouping, per-level MIS loop, persistence-backed updates.
   - Redistribution: update `(P_j, ℓ_j)` via ceil-log₂ distance to new anchors.
   - Tests: invariants, distance counts, parity with sequential compressed build.
   - Status: Level offsets now regenerate after inserts with copy-on-write coverage; child adjacency maintenance (`children`/`next_cache`) and per-level semisort redistribution are validated in `tests/integration/test_parallel_update.py`, with dominated points reattached via analytic separation checks and no host-side fallbacks.

7. **Batch delete (Alg. 5)**
   - Implement uncovered-set processing, MIS promotion, new-root creation.
   - Tests similar to batch insert; verify node counts and invariants.
   - Status: Prefix-doubling delete path promotes uncovered nodes, reattaches descendants, and preserves prior tree versions; covered in `tests/integration/test_parallel_delete.py`.

8. **k-NN queries**
   - Implement corrected CCT k-NN loop with cached `S_i` and `Next`.
   - Tests: compare against brute-force distances; handle ties deterministically.
   - Status: Priority-queue traversal with cached radii is live; `tests/test_knn.py` locks deterministic tie handling, batching semantics, and dense fallback parity.

9. **Diagnostics & benchmarks**
- Add logging hooks (conflict edges, MIS iterations, cache hit rates).
- Create benchmark scripts for insert/delete throughput and query latency.
  - Added `benchmarks/batch_ops.py` and `benchmarks/queries.py` (2025-11-03).
  - Recommended quick-run settings (finish <1 min on dev GPU):
    - `uv run python -m benchmarks.batch_ops insert --dimension 8 --batch-size 64 --batches 4 --seed 42`
    - `uv run python -m benchmarks.batch_ops delete --dimension 8 --batch-size 64 --batches 4 --seed 42 --bootstrap-batches 8`
    - `uv run python -m benchmarks.queries --dimension 8 --tree-points 2048 --batch-size 128 --queries 256 --k 8 --seed 42`
  - Sample outputs (Nov 3 build, A100-40G, CUDA13 wheel):
    - Insert throughput ≈ 45 pts/s; delete ≈ 33 pts/s; k-NN latency ≈ 6.2 ms/query.
  - Status: Benchmarks wired into the CLI with smoke coverage (`tests/test_benchmarks.py`) and quick-run presets documented here. Batch/knn paths now emit resource usage telemetry (wall time, CPU, RSS, GPU memory) via the shared diagnostics logger for evidence-based tuning (toggle with `COVERTREEX_ENABLE_DIAGNOSTICS`). The k-NN CLI exposes `--baseline {sequential|external|both}` for side-by-side comparisons against the in-repo baseline tree and the optional `covertree` adapter (install via `pip install -e '.[baseline]'`). Recent benchmark snapshots (dimension = 8):
    - 2 048 tree pts, 512 queries, k = 8: PCCT build ≈ 58.8 s, query throughput ≈ 214 q/s (4.67 ms/query); sequential baseline ≈ 21 000 q/s; external CoverTree ≈ 409 q/s (pre-GPBoost snapshot).
- 2025-11-04: Added GPBoost-style Numba baseline. With logging suppressed (`COVERTREEX_LOG_LEVEL=warning`) the quick run reports PCCT build 25.90 s / query 2.12 s (241 q/s), sequential 2.21 s / 0.026 s (19 615 q/s), GPBoost baseline 0.49 s / 0.614 s (834 q/s), external 1.04 s / 1.26 s (406 q/s). Command: `UV_CACHE_DIR=$PWD/.uv-cache COVERTREEX_LOG_LEVEL=warning uv run python -m benchmarks.queries --dimension 8 --tree-points 2048 --batch-size 128 --queries 512 --k 8 --seed 42 --baseline all`.
- 2025-11-04 (later): Memoised child chains and distance reuse in the PCCT k-NN traversal (`queries/knn.py`). This removes repeated chain decoding during heap walks and caches per-node distances across heap pushes. Quick-run query latency currently ~2.29 s (223 q/s); the structural change sets the stage for plugging in a residual-correlation provider and a dense fallback threshold so the traversal behaves more like the GPBoost baseline.
    - 8 192 tree pts, 1 024 queries, k = 16: PCCT build ≈ 1 173 s, query throughput ≈ 41 q/s; sequential ≈ 5 300 q/s; external ≈ 125 q/s.
    - Diagnostics off (toggle disabled) yields ~1–2% change, so current overhead is negligible relative to compute cost. These results underscore that the present PCCT path is still 10–100× slower than baselines, making profiling of conflict-graph/MIS orchestration the next priority.

### Immediate Optimisation Actions (2025-11-04)

1. **Segmented conflict-graph adjacency:** Rebuild `build_conflict_graph` around the traversal semisort buffers so we emit candidate pairs per scope, dedupe once, and build CSR directly (no dense membership matmul). Keep the entire pass on-device and reuse cached pairwise distances so the per-batch cost scales with actual conflicts, not `batch_size²`. Current prototype still spends most time shuttling data to the host and re-instantiating pairwise/radius buffers (`build ≈ 166 s` vs. 24.5 s), so we need to eliminate those host round-trips before enabling it by default.
2. **Degenerate-batch fast path:** When MIS would leave ≤1 survivor (e.g. dominated prefixes), bypass the conflict-graph/MIS pipeline and fall back to the sequential insertion kernel. This prevents the 16×128 quick-run from paying the full O(b²) price for batches that collapse anyway.
3. **Validation cadence:** After wiring the segmented path and fast-path guard, rerun the 2 048×512 benchmark with diagnostics on/off, regenerate `runtime_breakdown.png`, then attempt the 8 192×1 024 scenario. Keep the existing dense builder behind a flag until the new path is proven faster.

#### Benchmark Snapshot — 2025-11-04 (quick run: 2 048 tree pts, 512 queries, k=8)

- Diagnostics enabled (`COVERTREEX_ENABLE_DIAGNOSTICS=1`): build 46.69 s; query batch 2.09 s → 245.2 q/s (4.08 ms/query). Per-batch timings: scope grouping ≈1.72 s, adjacency ≈0.18 s, MIS ≈0.20 s, annulus ≈3 ms.
- Diagnostics disabled (`COVERTREEX_ENABLE_DIAGNOSTICS=0`): build 46.31 s; query batch 2.08 s → 245.6 q/s. Logging overhead is now <0.4 s on the build and <0.01 s on queries.
- Conflict-graph assembly now averages ≈1.94 s per batch (scope ≈78 %, adjacency ≈9 %, annulus ≈0.2 %), down from the previous 3.13 s snapshot. The total build shrank by ~12.8 s versus the earlier 60.13 s run while maintaining the same prefix-batch structure (16×128).
- Tiered breakdowns are captured in `runtime_breakdown.png` (JAX vs Numba vs sequential). Numba MIS still builds in ≈2.8 s with identical query latency, so the gap is dominated by scope-group aggregation.
- Next milestone: push scope grouping below 0.8 s/batch, then re-run the 8 192-point scenario to confirm the improvements hold at larger scales.
###### Proposed scope-group fusion (design sketch)

1. Replace the current dense scope mask with segmented buffers derived from the grouped traversal indices:
   - Compute `group_counts = group_indptr[1:] - group_indptr[:-1]` and `group_ids = xp.repeat(xp.arange(num_groups), group_counts)` directly on the backend.
   - Build a padded `(num_groups, max_group)` table with `jax.lax.scatter` (or equivalent) so each row holds the member indices contiguously; retain a boolean mask for valid columns. This avoids repeated host conversions while giving us a compact representation for downstream operations.
2. Generate adjacency edges via segmented cartesian products without forming a full `batch×batch` matrix:
   - For each group, broadcast the row to `(m, m)` and drop the diagonal by masking with the validity grid. Implement this in bulk by broadcasting the padded table and mask to `(num_groups, max_group, max_group)`, applying the mask, and slicing with `xp.reshape` so the result stays on-device. Because `max_group ≤ batch_size` (128 in the benchmark), the temporary tensor remains tractable.
   - Alternatively (preferred if memory spikes), derive directed edges using per-member repeats: `sources = xp.repeat(values, counts-1)`; `targets` assembled by rotating the padded table and using masked gathers so each source receives the remaining members. Both approaches operate on backend primitives and eliminate per-group Python loops.
3. Fold the annulus binning into the same segmented pass: reuse the padded table and group masks to accumulate annulus metadata, so we touch the grouped buffers only once.
4. Once adjacency indices are produced, reuse the existing CSR builder (lexsort + bincount) and feed the results into MIS. Profile the new `group_by_int` + fusion path to ensure scope grouping time drops below the ≤1 s/batch target before moving to MIS tweaks.

#### Profiling Notes — 2025-11-04 (Euclidean) / 2025-11-06 (Residual)

- Added `BatchInsertTimings` instrumentation so `log_operation` now reports `traversal_ms`, `conflict_graph_ms`, and `mis_ms` per prefix batch. Latest Euclidean run (same benchmark, diagnostics enabled) highlights traversal as the dominant cost: mean traversal 5.53 s (median 4.54 s, max 13.4 s) versus conflict-graph 1.33 s (median 1.24 s) and MIS 0.39 s (median 0.35 s). Aggregated over 16 batches, traversal consumed 88.5 s of the 135 s build (65 %), conflict graph 21.3 s (16 %), and MIS 6.3 s (5 %).
- Residual benchmark telemetry (32 k pts / 512 batch). The earlier k = 16 / 2 048-query profile logged dominated batches at 0.90 s wall (`traversal_ms` 0.79 s, `conflict_graph_ms` 0.092 s, `mis_ms` 0.0002 s) with ~16 k-member scopes driving the RSS spikes. After the Nov 6 journal/builder split and a rerun with k = 8 / 1 024 queries, total build now lands at 56.09 s with steady-state query latency 0.229 s (4 469 q/s); we still need fresh per-phase telemetry on this lighter workload, but the outstanding work remains the same: prune scopes before they hit conflict graph and reuse cached residual pairwise blocks inside the filter.
- The inflated wall times relative to the earlier snapshot stem from the device being shared; GPU utilisation stayed flat while CPU user+system scaled with traversal_ms growth, confirming that host-side traversal/scoping is the first optimisation target.
- Next action: spike `traverse_collect_scopes` to identify why per-batch traversal cost increases with batch index (e.g., parent cache misses, redundant distance recomputation, or semisort buffer churn) before touching MIS or CSR code.
- Instrumented `TraversalResult` with `TraversalTimings` to expose `pairwise`, `mask`, and `semisort` slices. On the 48.97 s build, semisort (Python-side scope assembly) alone consumed 20.7 s, versus 2.7 s for mask formation and 1.4 s for pairwise distance evaluation; conflict-graph construction added 11.6 s and MIS only 3.4 s. These numbers confirm that the semisort/chain stitching path is the highest-leverage optimisation target.
- Profiling confirms MIS is secondary for now: even a 2× speedup would save <2 s on this workload. Focus on eliminating Python looping in `semisort_scope`, reusing chain caches, and pushing scope aggregation to the backend before revisiting MIS kernels.

#### Traversal Optimisation Results — 2025-11-04

- Reworked `traverse_collect_scopes` to keep data on-device longer, convert `next_cache` once per batch, and materialise conflict scopes via vectorised `np.nonzero`/`lexsort` without per-row Python loops. Chain expansion now uses a cached numpy view of `next_cache`, removing thousands of device round-trips.
- New timing fields (`traversal_chain_ms`, `traversal_nonzero_ms`, `traversal_sort_ms`, `traversal_assemble_ms`) reveal the chain splice is now O(1 ms) per batch, nonzero/sort/assemble stay below 5 ms, and overall semisort time dropped from ~20.7 s to <0.3 s across the full build.
- Quick benchmark rerun (2 048 tree pts / 512 queries / k=8, diagnostics on, sequential baseline enabled) reports:
  - PCCT build 28.27 s (down from 45.95 s) and query batch 3.12 s (164 q/s, unchanged). Sequential baseline unaffected (build 2.17 s, 21 239 q/s).
  - Per-batch traversal metrics: `traversal_semisort_ms` now 7–28 ms, `traversal_chain_ms` ≈0.2–4 ms, while conflict-graph construction remains 0.3–1.4 s and MIS ≈0.19–0.21 s.
  - Overall traversal contribution fell to 4.0 s per build (14 %), leaving conflict-graph assembly (~19 s, 67 %) and pairwise mask formation (~3.1 s, 11 %) as the dominant hotspots.
- Conflict-graph construction now streams residual tiles through the Numba builder (~75 ms per dominated batch on the 32 k workload after the 2025-11-06 updates); residual traversal still spends ~430 ms in the pairwise phase and ~330 ms assembling CSR scopes, so traversal/mask assembly remain the limiting factor in residual mode.
- Immediate optimisation focus (Nov 6): ✅ landed. Residual conflict filtering now reuses cached pairwise data; early-reject pruning keeps traversal chunk computations lean; the 32 k residual benchmark dropped from 290 s build / 156 s query to 59 s / 0.42 s (see `docs/CORE_IMPLEMENTATIONS.md`).
- Next optimisation focus (post-Nov 6):
  1. Introduce residual scope segmentation / caps so dominated batches stop materialising ~16 k members (still near-dense even after early rejects). Explore parent-level chunking or segmented builder reuse.
  2. Prototype tile-based traversal pair enumeration (upper-triangle tiles + guided scheduling) to balance work and cut `traversal_pairwise_ms` / `traversal_assemble_ms` on near-cliques.
  3. Optionally materialise residual pairwise matrices in float32 (with epsilon guard) and enforce scope caps to reduce the ~275 MB RSS spike per dominated batch.
  4. Add a telemetry export helper that parses benchmark logs into CSV summaries so auditors can diff before/after runs without manual parsing.

#### Visual Diagnostics

- Added `benchmarks/runtime_breakdown.py`, a CLI that runs the quick benchmark for PCCT (JAX + optional Numba), the in-repo sequential baseline, and the external `covertree` baseline (when installed) and produces a stacked runtime bar chart (`pairwise`, `mask`, `next_chain`, `nonzero/sort/assemble`, `conflict_graph`, `mis`, `other`). Enable with:
  - `uv run python -m benchmarks.runtime_breakdown --output runtime_breakdown.png`
  - Install plotting deps via `pip install -e '.[viz]'` (wraps `matplotlib>=3.8`). Pass `--skip-numba` or `--skip-external` if the corresponding backends are unavailable.
- The CLI also prints per-implementation build/query timings and honours the existing `--dimension/--tree-points/...` knobs for consistency with the textual diagnostics.

#### Conflict-Graph Optimisation — 2025-11-04

- Instrumented `build_conflict_graph` with `ConflictGraphTimings` (pairwise, scope grouping, adjacency, annulus) and rewired the adjacency builder: the per-scope membership is now assembled into a CSR-like mask, and clique formation is handled via boolean matrix products instead of Python set churn. This keeps all heavy lifting in numpy/JAX and eliminates the quadratic Python loops.
- Earlier pass (rolled back before the latest fusion) clocked an 18.17 s build (diagnostics on) with query time 3.11 s. Aggregate timings for that version:
  - Traversal: 4.28 s (≈23 %), conflict-graph: 1.33 s (≈7 %), MIS: 3.50 s (≈19 %), everything else: 9.06 s.
  - Conflict-graph sub-breakdown averages: pairwise 7.2 ms, scope grouping 20.3 ms, adjacency 12.2 ms, annulus 10.9 ms — trimming adjacency time from ~3.6 s per batch to ~12 ms.
- Batch logs now expose `conflict_pairwise_ms`, `conflict_scope_group_ms`, `conflict_adjacency_ms`, and `conflict_annulus_ms`, making it easy to monitor any regressions as we iterate (Numba toggle included).
- In that iteration, conflict-graph costs fell below 10 % of the build, shifting the hotspot back to traversal pairwise/mask formation and MIS orchestration; those notes remain relevant once we restore the faster adjacency path.
- 2025-11-04 (late) — `group_by_int` bypass & device-native scope membership: replaced the host-side pair-id accumulator with an on-device scatter that marks batch × node membership directly into a backend boolean matrix. Conflict adjacency now comes from an integer matmul + `nonzero`, keeping every step JAX-native. The same quick benchmark (`COVERTREEX_ENABLE_DIAGNOSTICS=1`, 2 048 tree pts, 512 queries, k=8) finishes in **24.51 s build / 2.07 s query** (247.7 q/s). Post-warmup batch means: scope grouping **0.16 s**, adjacency **0.24 s**, annulus 2.9 ms, MIS 0.19 s. Overall conflict-graph construction dropped to ≈0.43 s per batch, halving total build time relative to the previous 46.69 s snapshot. Diagnostics show the worst-case scope batch (first dominated run) at 0.25 s, comfortably below the 0.8 s target.
- Runtime breakdown snapshot after the refactor (diagnostics on, 2 048 / 512 / k=8, CPU only): **PCCT (Numba)** build 10.34 s / query 0.10 s, **Sequential baseline** 2.22 s / 0.024 s, **GPBoost** 0.31 s / 0.57 s, **External CoverTree** 1.10 s / 1.22 s.

###### Scope/adjacency outlook — 2025-11-04

- Dense membership uses the full tree column width; on 2 048-point trees this is cheap, but we should validate 8 192- and 32 768-point builds to ensure matmul cost scales acceptably. If it does not, fall back to a compressed column mapping (unique node ids per batch) before the larger benchmarks.
- With scope grouping no longer dominating, adjacency now sits around 0.23–0.28 s. If we need further reductions, consider switching to a tiled matmul or chunked batch processing, but defer until larger-scale profiling justifies it.
- Next actions: rerun the diagnostics-off quick benchmark, regenerate `runtime_breakdown.png` with the new timings (JAX vs Numba vs sequential), and schedule the 8 192×1 024, k=16 benchmark to confirm the improvements persist at higher scales.

###### Radius pruning integration — 2025-11-05

- Folded the radius check into `build_conflict_graph_numba_dense`, so the Numba kernel now prunes candidates while expanding pairs and returns both surviving and candidate counts (`candidate_pairs`). `ConflictGraphTimings`/logging gained `conflict_adj_candidates` to expose the ratio.
- Quick benchmark (diagnostics = 1, 2 048 × 512, k = 8, `enable_numba=1`) now reports `conflict_adj_filter_ms=0` on dominated batches. The cold-start batch still pays ~2.4 s for Numba compilation and ~0.59 s in CSR sorting, but steady-state adjacency drops to 2.5–3.4 ms with scatter ≤0.7 ms.
- Regenerated `runtime_breakdown.png` and refreshed `docs/CORE_IMPLEMENTATIONS.md` and `AUDIT.md` with the filtered kernel. Observed that the CSR argsort dominates the remaining adjacency time and should be the next optimisation target before scaling past 8 k.

###### Residual metric plumbing — 2025-11-05

- Added `COVERTREEX_METRIC` to the runtime config (default `euclidean`) and taught `get_metric()` to honour it. Traversal/conflict-graph paths now resolve the active metric on demand so swapping metrics no longer requires re-imports.
- Registered a `residual_correlation` metric stub in the registry and exposed `configure_residual_metric` / `reset_residual_metric` so the downstream Vecchia code can wire its residual-correlation provider without touching internal modules. Until configured, the wrapper raises with a descriptive error to avoid silent fallbacks.
- Updated the metric tests to cover the new hooks and document the expected usage pattern. CLI + docs will pick this up once we publish the Vecchia provider sketch.

###### CPU-only configuration — 2025-11-05

- Default backend now points to the NumPy implementation; JAX is no longer initialised unless explicitly requested. `TreeBackend.numpy()` feeds the entire pipeline, and persistence helpers fall back to host-side writes when `.at[]` is unavailable.
- Runtime warnings about missing CPU devices disappeared: `_resolve_jax_devices` is bypassed unless the user opts back into the JAX backend. Docs and README call out the CPU-first focus for the optimisation sprint.
- Benchmarks/diagnostics generation (`benchmarks.queries`, `benchmarks.runtime_breakdown`) no longer rely on `jax.random`; datasets are sampled with NumPy RNGs so we avoid accidental JAX dependencies during runs.

CPU benchmark highlights (diagnostics on unless stated):

| Workload | PCCT build/query | Sequential | GPBoost | External |
|----------|-----------------|------------|---------|----------|
| 2 048 pts / 512 queries / k=8 | 0.382 s / 0.098 s (5 216 q/s) | 2.25 s / 0.024 s (21 001 q/s) | 0.292 s / 0.519 s (987 q/s) | 1.00 s / 1.22 s (421 q/s) |
| 8 192 pts / 1 024 queries / k=16 | 4.03 s / 0.934 s (1 096 q/s) | 33.65 s / 0.192 s (5 327 q/s) | 0.569 s / 3.35 s (306 q/s) | 14.14 s / 8.40 s (122 q/s) |
| 32 768 pts / 2 048 queries / k=16 | 66.06 s / 8.28 s (248 q/s) | — | 2.41 s / 20.05 s (102 q/s) | — |

Latest logs: `runtime_breakdown_output_2048_numba_baselines.txt`, `runtime_breakdown_output_8192_numba_baselines.txt`, and `runtime_breakdown_output_32768_numba_gpboost.txt` alongside the diagnostics-on/off PCCT traces. The first dominated batch still pays the Numba compilation penalty (~20 ms build, <0.2 ms query warm-up), after which steady-state adjacency per batch holds at 12–18 ms.

###### Next steps — 2025-11-05

1. Trim the remaining `conflict_adj_scatter_ms` hotspot (12–18 ms on dominated batches) by chunking the directed pair expansion or fusing the scatter with the CSR write.
2. Add regression coverage for the NumPy backend (config defaults, persistence updates) and ensure the test suite exercises both NumPy-only and NumPy+Numba paths.
3. Extend the runtime breakdown CLI to emit warm-up vs steady-state metrics directly (CSV artefact) so we can track improvements as we iterate on the adjacency builder.

###### Implementation plan — 2025-11-06

- **Scope segmentation & caps** — Implement real segmentation once the sparse traversal kernel lands. Reuse `_chunk_ranges_from_indptr` inside `build_conflict_graph_numba_dense` so `chunk_target` produces filtered CSR shards rather than a no-op. Feed per-segment radii into the filter and accumulate telemetry for chunk hits, capped memberships, and backpressure bytes; surface them via `ConflictGraphTimings` and the `batch_insert` logger. On the traversal side, add level- or parent-chain splits in `_collect_residual_scopes_streaming` so we prune memberships before the conflict graph sees 16 k-entry scopes.
- **Traversal work balancing** — Prototype tile-based pair assembly around `collect_sparse_scopes` and gate it behind `RuntimeConfig.enable_sparse_traversal`. The kernel should walk parent chains, emit CSR scopes directly, and schedule tiles keyed by the new chunk ranges so `traversal_pairwise_ms` and `traversal_assemble_ms` fall from the current ~200 ms / ~300 ms hotspots. Instrument traversal timings to compare dense versus tiled paths batch-by-batch.
- **Memory pressure options** — After segmentation is wired, allow residual pairwise caches to downcast to float32 with a tolerance guard (raise if relative error exceeds the configured bound) before entering traversal or the conflict graph. Record peak RSS deltas and float32 hit counts in diagnostics so auditors can confirm the ~275 MB spikes shrink. Explore hard scope member caps that fall back to queued work when chunks still exceed the guardrails.
- **Telemetry export helper** — Standardise on the refreshed `benchmarks.runtime_breakdown --csv-output` artefacts and add a lightweight script/CLI to stitch per-run logs into a schema auditors can diff. Include the new scope-chunk, float32 fallback, and traversal tile counters so side-by-side comparisons no longer require manual parsing.
- Default runtime now leaves chunked traversal disabled (`scope_chunk_target=0`). Chunking must be enabled explicitly via `COVERTREEX_SCOPE_CHUNK_TARGET>0`, ensuring dense traversal remains the reference behaviour until the tile path is optimised.
- **Next optimisation** — Design a tile-level radius precheck (or hard width cap) so the initial chunk per dominated batch prunes members before hand-off to the conflict graph. Success criteria: first dominated batch traversal under 500 ms and `conflict_adj_scatter_ms` within 5 % of the dense baseline.

###### Sparse traversal log analysis — 2025-11-06

- Baseline dense scopes (`scope_chunk_target=0`) on the 8 192×1 024×k=16 benchmark recorded a first dominated batch with `traversal_semisort_ms≈157.6 ms`, `traversal_pairwise_ms≈4.0 ms`, and `conflict_adj_scatter_ms≈0.73 ms` (build steady-state ≈0.76 s, `dense_run.csv`). Later dominated batches settle near 12–20 ms of semisort work with conflict scatter under 1 ms.
- Enabling chunked traversal (`scope_chunk_target=8 192`) removes semisort entirely but pushes work into the tiled path: dominated batches show `traversal_ms≈20–87 ms` with `conflict_adj_scatter_ms` jumping to 5.7–1.9 ms. Build steady-state inflates to ≈1.44 s (`chunked_run.csv`). Tile telemetry (`tile_seconds`, `scope_chunk_segments`) confirms segmentation is active, but conflict scatter needs additional pruning before the chunked mode is competitive.
- Switching `_collect_scopes_csr` to emit memberships in a single pass trims chunked traversal to **0.78 s** on average (first dominated batch still ~1.66 s vs. 12 ms dense). Scatter remains higher (≈7.1 ms vs. 6.0 ms) because the first tile still loads the full 256-member scope before pruning; next optimisation should cap tile width or apply a radius pre-check so the initial chunk does not overfeed the conflict stage.

###### Follow-up — 2025-11-05 afternoon

- Regression coverage now enforces the NumPy backend path (`tests/test_config.py`, `tests/test_persistence.py`) and keeps the benchmark smoke tests working against both sequential and GPBoost baselines.
- `benchmarks.runtime_breakdown` learned `--csv-output`, recording build/query warm-up versus steady-state timings; docs point to the new artefact.
- Tried tightening `COVERTREEX_SCOPE_CHUNK_TARGET` (65 536 → 8 192) to shave `conflict_adj_scatter_ms`, but dominated batches still log ~13 ms. The bottleneck is the directed pair expansion itself; next attempt should focus on per-node offsets + compaction instead of chunk sizing alone.

##### Backend-aligned conflict graph (2025-11-04)

- Reworked `group_by_int` and `build_conflict_graph` to stay entirely within the active backend (`backend.xp`), keeping CSR buffers on-device while retaining the existing timing split.
- Scope co-occurrence, separation checks, and annulus binning now use backend primitives (`xp.repeat`, `xp.matmul`, `xp.nonzero`, `xp.bincount`) with `_block_until_ready` barriers so the recorded timings reflect device execution.
- Next up: rerun the 2 048×512 benchmark (diagnostics on/off) to confirm the on-device refactor preserves the recent speedups and refresh the runtime breakdown plots if the timing mix changes.

##### Numba lookup migration sketch — 2025-11-04

- Current query path (`queries/knn.py`) always materialises DeviceArrays via `backend.to_numpy` before performing a Python/NumPy traversal. Even when `COVERTREEX_ENABLE_NUMBA=1`, the hot loop still lives in Python, so MIS sees a 10× drop while queries remain ≈2.07 s.
- Goal: introduce a Numba-native lookup pipeline so the `enable_numba` toggle covers *both* MIS and k-NN. That prevents JAX from being a hard runtime dependency for lookup-heavy workloads and should bring CPU throughput closer to the sequential baseline (≤50 ms per 512-query batch).
- Proposed structure:
  1. **Tree materialisation helper**: a lightweight view (`NumbaTreeView`) holding `points`, `si_cache`, `children`, `next_cache`, and `root_ids` as `np.ndarray` buffers (int32/float64). It will be produced lazily via `tree.materialise_for_numba()` or a helper in `queries._numba_view`.
  2. **JIT kernels**: add `covertreex/queries/_knn_numba.py` with `@njit(cache=True)` helpers for batched Euclidean distances, candidate pruning, and heap maintenance. Start with a dense distance path (lexsorted on `(dist^2, index)` to keep deterministic ties) then incrementally port the cover-tree walk (using `numba.typed.List` for heaps) once validated.
  3. **Dispatcher**: inside `queries.knn`, gate on `runtime.enable_numba and NUMBA_QUERY_AVAILABLE`. If the Numba kernels are ready, reuse the cached view and skip the Python walker; otherwise, fall back to today’s code. Metadata logging stays unchanged.
  4. **Config surface**: reuse `COVERTREEX_ENABLE_NUMBA` initially; if we need finer control later, introduce `COVERTREEX_ENABLE_NUMBA_LOOKUP`.
- Testing plan:
  - Mirror existing tie-handling, shape, and multi-query tests under a `numba_enabled` context (skip if Numba missing).
  - Add regression timing harness in `benchmarks/runtime_breakdown.py` to dump Numba build/query numbers explicitly once the kernels land.
- Open questions before implementation: cache invalidation strategy for the host mirror (likely keyed by `tree.stats.num_batches`), and whether we need async-safe guards when accessing the cached buffers from concurrent threads.
- Reference sketches: the notebook notes (`notes/cover_tree_parallel_paper_sketch.md`, `notes/parallel_compressed_cover_tree.md`) already contain Numba-based cover-tree walkers and brute-force fallbacks; adapt their priority-queue traversal and deterministic tie-breaking when porting Algorithm F.2 to `@njit`.
- 2025-11-04 (later) — landed the first Numba lookup path: `queries.knn` now materialises a `NumbaTreeView` and delegates to `_knn_numba` when `COVERTREEX_ENABLE_NUMBA=1`. The initial kernel uses a dense distance pass with deterministic tie-breaking; quick benchmark (`diagnostics=1`, 2 048 / 512 / k=8, external baseline skipped) reports **PCCT (Numba)** build 1.70 s / query **30 ms**, while **PCCT (JAX)** remains 25.82 s / 2.10 s and the sequential baseline 2.22 s / 24 ms.
- 2025-11-04 (evening) — swapped the dense pass for an `@njit` cover-tree walk mirroring Algorithm F.2. Follow-up work introduced a Numba-compatible binary heap, cached node distances, and precomputed child lists, bringing quick-run latency down to **118 ms** (build 1.60 s). The prototype still trails the earlier 30 ms dense placeholder but already beats the external `covertree` baseline (≈1.18 s query) by ~10×.
- 2025-11-05 — Eliminated the global `within` mask by filtering candidate edges with squared distances (`covertreex/algo/conflict_graph.py`), and gated the device→host conversions so the dense path no longer materialises `pairwise_np`/`radii_np`. Quick benchmark snapshots:
  - Diagnostics **on**: build **20.66 s**, query **2.28 s** (512 queries, k=8, throughput 224 q/s).
  - Diagnostics **off**: build **20.31 s**, query **2.28 s** (throughput 225 q/s).
  - Medium benchmark (8 192 tree pts / 1 024 queries / k=16, diagnostics on): build **45.84 s**, query **18.35 s** (55.8 q/s). Scope grouping still dominates conflict-graph cost (~150–250 ms on dominated batches), while annulus settles near 3 ms and adjacency stays <20 ms.
- 2025-11-05 (late) — Replaced the dense membership matmul with a host-side CSR grouping of `scope_indices` → query ids, padding memberships to a rectangular matrix so we can deduplicate identical scopes via `np.unique`. Quick benchmark now lands at **20.31 s build / 2.28 s query** with diagnostics off (20.66 s / 2.28 s with diagnostics on), and `conflict_adj_pairs` collapses to 65 280 even on dominated batches (`conflict_adj_max_group=256`). The 8 192×1 024 × k=16 benchmark completes in **45.84 s build / 18.35 s query**; the remaining hotspot is `conflict_scope_group_ms` (150–250 ms) plus MIS (~195 ms). We now emit `adjacency_total_pairs` and `adjacency_max_group_size` in `ConflictGraphTimings`, and the diagnostic logger forwards them as `conflict_adj_pairs` / `conflict_adj_max_group`.
- 2025-11-05 (night) — Landed a Numba scope-adjacency builder with segment-level hashing before clique expansion (`covertreex/algo/_scope_numba.py`) and plumbed it into the dense conflict-graph path when `COVERTREEX_ENABLE_NUMBA=1`. The helper reuses counting-sort grouping, dedupes identical memberships upfront, and returns compact int32 edge lists; the dense builder now falls back to this path automatically. `ConflictGraphTimings` gained `scope_bytes_{d2h,h2d}`, `scope_groups`, `scope_groups_unique`, `scope_domination_ratio`, and `mis_seconds` (placeholder) so logs capture data movement and domination ratios. Exposed `COVERTREEX_SCOPE_SEGMENT_DEDUP` to toggle segment dedupe end-to-end and teach config/tests about it.
- 2025-11-05 (late, Numba focus) — Re-ran the quick benchmark (2 048×512×k=8) with the Numba path only. Diagnostics **on** now reports **build 25.45 s / query 135 ms** (0.27 ms latency, 3.77 k q/s) while diagnostics **off** drops query latency to **0.22 ms** (4.51 k q/s) at roughly the same build time (25.34 s). Dominated batches still show `conflict_scope_group_ms` ≈ 290–305 ms and MIS stays sub-millisecond. Logging now emits `conflict_scope_*` byte/group counters so the remaining hotspot is plainly attributed to the CPU scope grouping.
- 2025-11-05 (night, iteration) — Removed per-segment sorting in the Numba builder and enabled parallel hashing/expansion. Quick run remains **build 25.4 s / query 0.136 s** with dominated batches now avoiding the initial 3 s compile stall; steady-state `conflict_scope_group_ms` sits at ~150 ms and `conflict_adj_scatter_ms` drops below 2 ms after warm-up. The 8 192×1 024×k=16 run clocks in at **build 58.2 s / query 0.99 s**, reporting `conflict_scope_group_ms` in the 160–270 ms range with `scope_domination_ratio` ≈ 0, underscoring the need for a faster scope dedupe path.
- 2025-11-05 (late, chunked scope grouping) — Added a chunked Numba scope-grouping pipeline controlled by `COVERTREEX_SCOPE_CHUNK_TARGET` (default 65 536 memberships), sorted memberships in-place before hashing, and duplicated directed edges on-device so adjacency stays symmetric without host dedupe. Dominated batches now report `conflict_scope_group_ms ≈ 0.12–0.15 ms` (down from ~150 ms) while `conflict_adj_filter_ms` remains the dominant cost (~320 ms for the worst batch). Quick benchmark (2 048×512×k=8) lands at **build 14.61 s / query 0.123 s** with diagnostics on and **14.48 s / 0.102 s** with diagnostics off. The 8 192×1 024×k=16 run finishes in **32.10 s / 0.951 s** (diagnostics on) and **31.67 s / 0.936 s** (diagnostics off). Regenerated `runtime_breakdown.png` with the new timings and recorded `conflict_scope_groups_unique=1` across dominated batches, confirming segment dedupe works end-to-end.

###### Parallelisation effectiveness audit — 2025-11-05 evening

- **Instrumentation:** `benchmarks/runtime_breakdown.py` now snapshots process CPU time and RSS before/after each build/query run, emitting the results in the console and the CSV (`build_cpu_seconds`, `build_cpu_utilisation`, `build_rss_delta_bytes`, …). Updated `tests/test_benchmarks.py::test_runtime_breakdown_csv_output` ensures the new columns stay wired.
- **Theoretical takeaways:**
  - Build pipeline parallelism is concentrated in two places: the vectorised pairwise distance kernel (NumPy/BLAS can saturate multiple cores) and the Numba scope grouping helpers (`_sort_segments_inplace`, `_hash_segments`, `_expand_pairs` use `nb.prange`). The remaining stages—chunked pair expansion (`_expand_pairs_directed_into`), MIS (`_run_mis_numba_impl`), and redistribution—are still serial, limiting headroom even when the parallel segments scale well.
  - Traversal still bounces through NumPy for mask assembly and Next-chain stitching, so per-point work is effectively single-threaded. Expect <3× speedups until traversal is migrated to a parallel NumPy/Numba hybrid.
  - The Numba k-NN path caches distances aggressively but `_knn_batch_cover` iterates queries sequentially; we currently only exploit vectorisation within a single query. A straightforward `nb.prange` over queries (with per-thread heaps) should unlock near-linear scaling for large batches.
  - Memory pressure during build is dominated by materialising the `NumbaTreeView` (children lists + caches). Peak RSS climbs by ~85 MB regardless of backend because the snapshots hold onto that view for subsequent queries.
- **Empirical snapshot (CPU, dim = 8, 2 048 insertions / 512 queries / k = 8, diagnostics off, warm compilation amortised):**

  | Implementation     | Build wall (s) | Build CPU (s) | Build util (×) | Build ΔRSS | Query wall (s) | Query CPU (s) | Query util (×) | Query ΔRSS |
  |--------------------|----------------|---------------|----------------|------------|----------------|---------------|----------------|------------|
  | PCCT (Numba)       | 0.350          | 0.795         | 2.27           | 85.19MB    | 0.092          | 0.092         | 1.00           | 0B         |
  | Sequential baseline| 2.360          | 2.320         | 0.98           | 0.08MB     | 0.024          | 0.024         | 0.99           | 0B         |
  | GPBoost CoverTree  | 0.428          | 1.365         | 3.19           | 5.64MB     | 0.646          | 12.367        | 19.15          | 0.03MB     |
  | External CoverTree | 1.094          | 1.091         | 1.00           | 0B         | 1.188          | 1.185         | 1.00           | 0B         |

  Command: `COVERTREEX_BACKEND=numpy COVERTREEX_ENABLE_NUMBA=1 COVERTREEX_ENABLE_DIAGNOSTICS=0 UV_CACHE_DIR=$PWD/.uv-cache uv run python -m benchmarks.runtime_breakdown --output '' --skip-jax`.
  All runs share the same process high-water RSS (≈404 MB); the table reports incremental changes per implementation.
- **Observations / next actions:**
  1. PCCT build keeps ~2.3 effective cores busy yet still trails the sequential builder by ≈6× in wall time; the serial traversal + MIS sections dominate the budget. Prioritise migrating traversal mask assembly and MIS selection to parallel Numba kernels before revisiting the 8 k/32 k benchmarks.
  2. The GPBoost baseline’s query path fans out across ~19 cores, explaining its high CPU time despite modest wall-latency improvements. Mirroring that approach (`nb.prange` over queries with thread-local heaps) should be our next optimisation step for `_knn_batch_cover`.
  3. Memory deltas confirm the tree-view materialisation as the primary resident cost. Document this in `docs/CORE_IMPLEMENTATIONS.md` and explore recycling the buffers across builds to avoid repeated 80 MB spikes in long-running processes.
  4. Schedule a follow-up benchmark with diagnostics enabled to line up resource metrics against the fine-grained batch timings (pairwise vs scope vs MIS) so we can apply Amdahl-style budgeting before diving into MIS refactors.
- 2025-11-05 (late night) — Ported traversal scope assembly to a Numba helper (`build_scopes_numba`) so parent-chain stitching and CSR emission stay in native code. The dominated prefix batch that previously spent ≈618 ms in `traversal_semisort` now compiles away after warm-up (steady ≈2.0–2.1 ms; first batch still pays JIT cost). The k-NN kernel now parallelises across queries via `prange`, yielding **query_steady_seconds ≈ 6.2 ms** for the 512-query benchmark and pushing CPU utilisation to ~22× during lookups.
  - Fresh metrics (`COVERTREEX_BACKEND=numpy COVERTREEX_ENABLE_NUMBA=1 COVERTREEX_ENABLE_DIAGNOSTICS=0 uv run python -m benchmarks.runtime_breakdown --skip-jax --csv-output benchmark_metrics.csv`) show **PCCT (Numba)** build **0.305 s** (CPU 0.777 s, util 2.55×, ΔRSS ≈ 74 MB) and query **0.0062 s** (CPU 0.141 s, util 22.35×). Sequential build/query remain 2.27 s / 24 ms, while GPBoost reports 0.307 s / 0.499 s with 23× query utilisation. These numbers replace the earlier 0.350 s / 0.092 s baseline for PCCT.
  - With traversal chains handled inside Numba, the diagnostic traces confirm dominated batches no longer allocate minutes to mask assembly; `traversal_chain_ms` plus `traversal_semisort_ms` settle near 2 ms and 0.1 ms respectively after warm-up. The remaining host-bound hotspot for build is now the adjacency scatter (~12 ms) inside the conflict-graph path.
- 2025-11-05 (late night, adjacency scatter pass) — Reworked `_scope_numba.py` so only deduped scope groups participate in pair expansion (no per-node zero work) and the directed edge emission runs with `nb.prange`. Candidate capacity is reserved per surviving group, and a compact pass trims unused slots after radius pruning. A module-level warm-up now compiles the kernel at import time so the runtime path no longer pays a multi-second cold start.
  - Quick benchmark (2 048×512×k=8, diagnostics off) now lands at **build 0.242 s / query 0.0063 s** (build CPU 0.81 s, util 3.35×; query CPU 0.145 s, util 22.8×). Dominated batches report `conflict_adj_scatter_ms ≈ 0.26–0.40 ms` straight away—no JIT spike—and the overall build drops from ~0.30 s to ~0.24 s.
  - 8 k benchmark (`--tree-points 8192 --batch-size 256 --queries 1024 --k 16`, diagnostics off) records **build 3.04 s / query 0.051 s** (build CPU 5.06 s, util 1.66×; query CPU 1.17 s, util 23.0×). Dominated batches stay under 1 ms scatter, and the import-time warm-up keeps the totals consistent run-to-run.
  - 32 k benchmark (`--tree-points 32768 --batch-size 512 --queries 2048 --k 16`, diagnostics off) completes in **build 46.38 s / query 0.405 s** (build CPU 54.97 s, util 1.19×; query CPU 11.19 s, util 27.6×) with peak RSS ≈ 3.3 GiB. Dominated batches show `conflict_adj_scatter_ms ≈ 2.4–6.3 ms`, and `conflict_adj_csr_ms` now dominates the adjacency block—next target for large-scale tuning.
  - Follow-up: recycle the directed buffers (avoid the 230–950 MB host spikes), and push CSR assembly (`adjacency_csr_ms`) into a parallel kernel so the large benchmarks scale beyond 32 k without the current 60 ms per batch bottleneck.
- 2025-11-05 (CSR assembly kernel, 32 k follow-up) — Extended the Numba scope builder to emit CSR `indptr/indices` directly and taught the dense conflict-graph path to consume them, removing the Python-side CSR pass.
  - Dominated batches on the 32 k workload now log `conflict_adj_csr_ms ≈ 0.003–0.008 ms` (down from ~59 ms). Host-side CSR assembly disappears; adjacency wall is essentially the scatter cost.
  - Fresh 8 k run (`--tree-points 8192 --batch-size 256 --queries 1024 --k 16`, diagnostics off) produces **build 2.63 s / query 0.049 s** with `conflict_adj_csr_ms ≈ 0.002 ms` and unchanged CPU utilisation (≈1.78×). Scope H2D traffic now only moves the CSR buffers (~257 KiB per dominated batch).
  - Added `--skip-sequential` / `--skip-external` toggles to `benchmarks.runtime_breakdown` so the heavy baselines can be disabled on large runs. With the latest in-kernel CSR compaction, the 32 k benchmark (skipping sequential/external, diagnostics off) lands at **build 42.5 s / query 0.459 s** (build CPU 51.1 s, util 1.20×; query CPU 10.9 s, util 23.8×) while keeping `conflict_adj_csr_ms` sub‑0.01 ms and peak RSS ≈ 3.32 GiB.
  - Directed scatter buffers stay inside the Numba kernel now, eliminating the per-batch 1 MiB Python copies; remaining allocations come from the kernel’s scratch arrays and are candidates for chunk pooling if we still want further trims.

## Status Snapshot — 2025-11-05

- **Completed recently**
  - Conflict-graph CSR emission now fully resident in Numba; host CSR build is gone.
  - Benchmark harness supports skipping sequential/external baselines for heavyweight runs.
  - 32 k / 8 k diagnostics refreshed with the CSR improvements (see bullet above).

- **Top TODOs**
  1. Recycle / pool the temporary directed-edge scratch buffers inside `_scope_numba` to curb per-batch allocations on large inserts.
  2. Revisit MIS timing at scale (Luby loop still serial) and prototype a parallel path once the conflict graph settles.
  3. Re-measure the end-to-end pipeline with diagnostics enabled after buffer pooling to confirm no new hotspots emerge.

- **Working hypotheses**
  - Most of the remaining build gap vs GPBoost is MIS + traversal book-keeping; trimming scratch allocations should buy stability but not remove the gap entirely.
  - Keeping diagnostics off by default is still the right trade-off; enable them in controlled runs only (they double the knn wall time at 32 k).

### Adjacency filter fusion sketch (retained for reference)

- **Bring radii into the builder:** Pass the per-point minimum radii array (or its square) alongside `scope_indptr/indices` so Numba can reject pairs while expanding, eliminating the 300 ms GPU-side mask.
- **Chunk-aware distance reuse:** Precompute per-chunk squared norms (or reuse the existing pairwise matrix in host memory) so each chunk evaluates `‖x_i - x_j‖²` exactly once; store keep flags in a compact uint8 buffer that can be streamed back to the device.
- **Symmetric write-back:** When a pair passes the radius test, write both orientations directly into the output buffers to avoid post-filter duplications.
- **Optional H2D bypass:** If the filtered chunk is empty, skip copying anything back to the device; otherwise, reuse a small staging buffer sized by `scope_chunk_target` to keep host→device traffic under control.
- **Telemetry:** Extend `ConflictGraphTimings` with `adjacency_pruned_pairs` so benchmarks can confirm the pruning ratio and spot regressions quickly.

## Testing & Validation

- **Invariant checks:** nesting, covering, separation, distinctive descendant consistency.
- **Cross-checks:** sequential compressed tree comparisons; brute-force k-NN.
- **Property tests:** persistence (old versions intact), MIS separation per level.
- **Complexity sanity:** track distance operations versus `O(n log n)` trend.
- **Backend parity:** ensure NumPy+jax vs. Numba give identical results.
- **Pipeline smoke tests:** traversal → conflict graph → MIS placeholders stitched together; to be upgraded to real invariants once kernels land.
  - Status: pipeline harness active; will evolve alongside semisort scopes and final MIS kernels.

## Milestones

1. `PCCTree` skeleton with traversal + invariants.
2. Batch insert end-to-end parity with sequential builder.
3. Batch delete parity and persistence validation.
4. k-NN queries with correctness suite.
5. Backend acceleration (Numba) + benchmarks.

Each milestone must land with tests and documentation referencing the relevant algorithms/lemmas from both source papers.

## Dependencies & Performance Considerations

- **Core dependencies**
  - `jax[cuda13] >= 0.5.3`: primary array backend; aligns with downstream `survi` environment. CUDA 13 wheel required for GPU support.
  - `jaxlib` matching build (auto-resolved via `jax` metapackage).
  - `numpy`: host-side manipulation (light usage).
  - `numba >= 0.61.2`: optional CPU acceleration for distance kernels/MIS; respect availability checks.
  - `optax`, `tfp-nightly[jax]`: upstream dependencies already in consumer app (no direct use here but avoid conflicts).
- **Optional adapters**
  - `scikit-learn >= 1.3`: only needed when `tree.as_sklearn()` is invoked.
  - `plotly`, `matplotlib`: for visualisation helpers; keep optional.
- **Environment alignment**
  - Python `~=3.12` (matching `survi`).
  - GPU runtime assumes CUDA ≥13 (as in downstream project); document fallback for CPU-only installs.
  - Ensure `jax.config.update("jax_enable_x64", True)` compatibility since Vecchia code relies on float64.
- **Performance targets**
  - Batch insert/delete: within 1.2× sequential compressed build time for n ≤ 32k, with scaling trend matching O(n log n).
  - Async rebuild: overlap ≥80 % of insert compute with simulated LBFGS workload (Tier C test).
  - GPU path: keep host-device traffic <5 % of total time (measured via `jax.profiler`).
  - Memory footprint: double-buffer overhead ≤ 1.5× active neighbour graph (points × k × 8 bytes).
- **Validation milestones**
  - Tier A/B integration tests before exposing APIs.
  - Tier C async + GPU tests gate release candidate.
  - Tier D (`Vecchia Refresh Loop Mini`, adapter compatibility) required before tagging v0.1.

## Integration Test Ladder

To keep the pipeline tight while hitting meaningful milestones, introduce three tiers of integration checks between unit tests and full end-to-end runs:

### Tier A – Structural Core
1. **Traversal + Cache Sanity**
   - After building the initial PCCTree, run batched `traverse_collect_scopes` and verify `(parent, level)` plus scope contents against the sequential traversal and the cached `S_i/Next` tables.
   - Ensures traversal logic, cover sets, and caches are aligned before MIS is involved.
   - Status: `tests/integration/test_structural_core.py::test_traversal_matches_naive_computation` plus randomized fixtures (`test_randomized_structural_invariants`) cover deterministic and stochastic trees.
2. **Scoped Conflict Graph**
   - For a fixed level `k`, construct `Π_{P_i} ∩ L_k` and resulting CSR edges, then compare against a brute-force “check all pairs” routine.
   - Confirms annulus restriction and edge generation behave before entering Luby MIS.
   - Status: `tests/integration/test_structural_core.py::test_conflict_graph_matches_bruteforce_edges` and randomized checks assert CSR edges and annulus bin metadata.

### Tier B – Parallel Update Mechanics
3. **Level-wise MIS Update**
   - Feed a controlled batch through `batch_insert`, capturing intermediate MIS selections; compare against a reference MIS (deterministic seed) and ensure post-update levels satisfy separation.
   - Verifies orchestration of prefix-doubling, per-level MIS, and redistribution without yet touching Vecchia.
   - Status: `plan_batch_insert` + `batch_insert` wired into integration tests (`tests/integration/test_parallel_update.py`), verifying per-level separation, cached-radius pruning, and deterministic MIS selections while keeping originals untouched.
4. **Persistence Path Copy**
   - Execute successive updates, then diff consecutive tree versions to confirm only the intended level slices/child ranges changed and earlier versions remain queryable.
   - Status: `tests/integration/test_parallel_update.py::test_batch_insert_persistence_across_versions` and associated level-count delta checks assert copy-on-write behaviour and automated level-offset diffing. As of 2025-11-06 `batch_insert` routes through `PersistenceJournal` + `apply_persistence_journal`, so parents/levels/children/Next caches are cloned exactly once per batch (NumPy journal fallback retained for non-Numba backends).

### Tier C – Application Hooks
5. **Async Refresh Harness**
   - Spin two threads: main thread repeatedly queries neighbours; worker thread builds `tree.update(...)`. Validate that swap occurs only on `future.done()` and that queries never see partially updated state.
   - Direct precursor to plugging into `maybe_refresh_state`.
6. **GPU Builder Smoke**
   - Run build → `add` → `remove` → `knn` entirely on JAX device arrays; assert no implicit host round-trips and results match CPU baseline.

### Tier D – End-to-End Confidence
7. **Vecchia Refresh Loop Mini**
   - Mock LBFGS epochs with synthetic data: at each epoch, call `prefetch_scopes`, schedule async `update`, and check neighbour indices/dists against legacy output.
   - Validates caching, async scheduling, and Vecchia-specific tolerances.
8. **Adapter Compatibility**
   - Roundtrip through `as_sklearn()` and ANN-style adapters to ensure `add/update/query` sequences stay in sync with the underlying tree.

Each tier unlocks the next milestone; we only move forward once the relevant integration tests are green, ensuring issues are caught closer to their source rather than in full application runs.

## API & Developer Experience Sketch

### Core surface

```python
import jax.numpy as jnp
from covertreex import PCCTree

# hello world: build directly from data, Euclidean metric by default
tree = PCCTree.from_points(jnp.asarray(train_points))

# persistent updates return a fresh tree
tree2 = tree.update(insert=jnp.asarray(new_pts), delete=jnp.asarray(old_ids))

# queries
idx, dist = tree2.knn(jnp.asarray(query_pts), k=16, return_distances=True)

# optional mutable transaction for batched streaming updates
with tree2.writer() as txn:
    txn.insert(batch_pts)
    txn.delete(batch_ids)
tree3 = txn.commit()
```

### Key design points

- **Immutable default:** every mutating call (`update`, `add`, `remove`) returns a fresh `PCCTree`, keeping async rebuilds and concurrent analytics safe. Developers opt into mutation explicitly via `writer()`.
- **Ergonomic aliases:** `.add(points)`, `.remove(ids)`, and `.update(insert=..., delete=...)` all share the same engine; `.extend()` accepts iterables/iterators for streaming ingestion.
- **Backend-agnostic arrays:** accept `jax.Array`, `numpy.ndarray`, `cupy`, or `torch.Tensor`; data is coerced once through a lightweight registry (defaults auto-populated) so existing pipelines drop in without ceremony.
- **Metric hooks:** `"euclidean"`, `"manhattan"`, `"cosine"` ship ready; advanced users can register custom distance pairings via a single helper without touching core APIs.
- **Batch pipeline hooks:** `tree.prefetch_scopes(batch)` exposes the traversal/mutation plan so upstream schedulers (e.g., Ray, Dask) can partition work; `tree.stats()` surfaces conflict-graph size, MIS iterations, cache hits.
- **Interoperability adapters:**
  - `tree.as_sklearn()` → `sklearn.neighbors.KNeighborsMixin` wrapper for scikit-learn pipelines.
  - `tree.as_ann_index()` → `build/add/query` style adapter for ANN libraries.
  - `tree.export(level=k)` → JSON snapshot for tooling that expects explicit nodes.
- **Iterator UX:** `tree.walk(level=None)` yields `(node_id, level, point_index, children)` for debugging/inspection.
- **Configuration profiles:** `PCCTree.build(..., profile="low-latency")` toggles defaults (e.g., disable distance caching, shrink prefix batches) while `"throughput"` enables more aggressive caching and Numba kernels.

### Ergonomic helpers

- `covertreex.pipeline.BuildJob`: orchestrates staged ingest (chunked reading, prefix-doubling scheduling) with progress callbacks.
- `covertreex.inspect.compare(tree_a, tree_b)`: diff of invariants, overlap metrics, enabling regression tests for pipelines migrating from existing cover-tree libs.
- `covertreex.visualize.radial(tree, level=k)`: quick scatter/graph plotting for exploratory analysis (matplotlib-backed).

### DX enhancements to prioritise early

- Friendly error messages (`ValueError` with hints) when invariants would be violated (e.g., custom metric returning asymmetric distances).
- Autocomplete-friendly signatures (`update`, `add`, `remove`, `knn`, `radius`) with type hints and docstrings referencing algorithms.
- Lightweight progress hooks (`callbacks={"progress": fn}`) for batch builds—wired before heavy parallel work begins.
- Quickstart notebook showcasing build → update → query → adapter integration, runnable in Colab with minimal boilerplate.
- `tree.profile()` returning human-readable summary (levels, branching stats, cache sizes) for debugging.

### Documentation UX

- **Playbook-style guides:** “Replace your existing cover tree build with PCCTree”, “Batch updates during streaming ingestion”, “Hooking PCCTree into scikit-learn pipelines”.
- **Snippets-first:** every major method documented with minimal runnable code (works in REPL, Colab, or plain Python).
- **Reference to theory:** docstrings link back to algorithm/lemma identifiers so practitioners can align implementation with the papers when auditing correctness.

## Living Journal

- 2025-11-06: Journal-based persistence landed. `PersistenceJournal`, pooled scratch buffers, and the backend-aware `apply_persistence_journal` now power all `batch_insert` updates; NumPy/Numba sweeps share the same planner and `tests/test_persistence.py` asserts the journal head/next deltas alongside the legacy copy-on-write helpers.
- 2025-11-06: Conflict graph builders were split into `covertreex/algo/conflict_graph_builders.py` (dense / segmented / residual). `build_conflict_graph` is now orchestration-only, and `ConflictGraphTimings` surfaces per-builder telemetry so segmented runs and residual filters can be profiled independently.
- 2025-11-06: Reran the 32 768-point PCCT benchmarks (dimension 8, batch 512, 1 024 queries, k=8). Euclidean metric: PCCT build **42.13 s** / query **0.217 s** (4 724 q/s) vs GPBoost baseline build **2.55 s** / query **10.75 s** (95.2 q/s, 49.6× slower). Residual metric: PCCT build **56.09 s** / query **0.229 s** (4 469 q/s) vs GPBoost **2.51 s** / **11.18 s** (91.6 q/s). Commands and raw logs live in `bench_*.log`.
- 2025-11-03: Added level-offset recomputation to `batch_insert` (counts per level descending) and extended Tier-B integration tests to check copy-on-write behaviour and empty-tree bootstrap. Full suite (`uv run pytest`) passing. Outstanding work: wire child adjacency/Next updates during insert and implement redistribution for dominated batch points.
- 2025-11-03: `BatchInsertPlan` now exposes dominated indices alongside MIS selections, and Tier-B tests assert coverage; redistribution uses these summaries downstream.
- 2025-11-03: Sketched child-adjacency strategy — treat `children[p]` and `next_cache[p]` as the head of a per-parent singly linked list, inserting new anchors at the front while chaining the previous head behind the inserted node. New nodes keep `next_cache[new] = -1` when they lack descendants; dominated points will later splice under winning parents at lower levels.
- 2025-11-03: Child adjacency wiring landed: `batch_insert` now updates parent head pointers and stitches the prior child chain behind the inserted anchors, with integration tests covering existing-child and leaf-parent cases.
- 2025-11-03: Redistribution pass implemented—dominated points drop one level (host-side separation guard ensures >2^ℓ spacing unless already at level 0), and Tier-B invariants check ordering/level adjustments. Next iteration: enforce separation against new anchors and diff persistence snapshots.
- 2025-11-03: Prefix-doubling orchestrator + semisort utility added; redistribution now targets MIS-selected anchors via annulus scopes. Remaining work: plug semisort into conflict-graph construction and replace separation fallbacks with invariant-based checks.
- 2025-11-03: Persistence diff test added plus level-0 collision logging; back-to-back inserts keep previous versions intact while emitting debug breadcrumbs when separation falls back at leaf scale.
- 2025-11-03: Configuration module now tolerates missing JAX installations by stubbing CPU devices and deferring flag application; `test_config`/`test_logging` run green without optional deps.
- 2025-11-03: Introduced `RuntimeConfig.from_env` + `describe_runtime` helpers with accompanying tests, keeping cached config in sync with environment snapshots.
- 2025-11-03: Re-validated full suite via `uv run pytest`; reminder to execute tests through `uv` so the JAX-enabled environment is active.
- 2025-11-03: Added `core.metrics` registry with default Euclidean kernels, refactored traversal/conflict-graph distance calls, and landed `tests/test_metrics.py`; full suite (`uv run pytest`) now at 48 passing checks.
- 2025-11-03: Enabled runtime-selectable Numba MIS path; `run_mis` defers to `_mis_numba` when `COVERTREEX_ENABLE_NUMBA=1`, with parity locked in `tests/test_mis.py::test_run_mis_numba_matches_jax`.
- 2025-11-03: Introduced resource diagnostics logging (CPU/GPU/Wall RSS) via `covertreex.diagnostics.log_operation`, instrumented batch insert/delete and k-NN, and added `tests/test_logging_diagnostics.py`.
- 2025-11-03: Landed `BaselineCoverTree` for sequential comparisons (`covertreex/baseline.py`) with coverage in `tests/test_baseline_tree.py` for nearest and k-NN parity against brute force.
- 2025-11-03: Added `ExternalCoverTreeBaseline` adapter for the upstream `covertree` implementation with optional dependency group (`[baseline]`), plus parity checks in `tests/test_external_baseline.py` (skipped when the package is unavailable).
- 2025-11-04: Captured benchmark baselines (PCCT vs sequential vs external) showing a 10–100× performance gap; next steps involve profiling MIS/conflict-graph hot spots and reducing Python-control overhead before revisiting large-scale (≥32 k) builds.
- 2025-11-04: Reran the 2 048×512×k=8 benchmark after the backend-native conflict-graph refactor. Diagnostics on: build 68.51 s, queries 3.68 s (139.2 q/s); diagnostics off: build 57.10 s, queries 3.44 s (149.0 q/s). Conflict-graph assembly still dominates (~1.7–3.6 s per batch), confirming the next optimisation push stays focused on CSR/annulus fusion.
- 2025-11-04: Rewrote `group_by_int` and the conflict-graph adjacency path to run fully on `jax.numpy`, removing NumPy detours and preserving timing instrumentation. Validated via `UV_CACHE_DIR=$PWD/.uv-cache uv run pytest tests/test_semisort.py tests/test_conflict_graph.py tests/integration/test_structural_core.py`; follow-up is to rerun the 2 048×512 benchmark and refresh the runtime breakdown plots with the backend-native timings.
- 2025-11-04: Audit housekeeping: fixed the child/sibling insertion bug in `batch_insert`, added invariants around the singly linked child chains, exported `PCCTree`/`TreeBackend`/metrics helpers at package root, and updated `TreeBackend.to_numpy` to return real NumPy arrays. Targeted tests (`tests/integration/test_parallel_update.py`, `tests/test_conflict_graph.py`, `tests/test_semisort.py`) pass.
- 2025-11-04: Implemented the backend-native scope/adjacency fusion (padded membership + sorted dedup) in `build_conflict_graph` and re-ran targeted tests (`tests/test_conflict_graph.py`, `tests/test_semisort.py`, `tests/integration/test_structural_core.py`). Quick benchmark (`COVERTREEX_ENABLE_DIAGNOSTICS=1`, 2 048 tree pts, 512 queries, k=8) now reports a 151.41 s build and 3.30 s query batch (155 q/s). Conflict-graph averages per batch: `conflict_graph_ms` ≈ 7.44 s, with `scope_group_ms` ≈ 2.78 s, `adjacency_ms` ≈ 4.48 s, and `annulus_ms` ≈ 0.09 s (first batch spikes to ≈ 1.39 s). Next step: replace the lexsort/boolean dedup with scatter-based accumulation so adjacency drops back below the earlier ≈12 ms goal before attempting larger benchmarks.
- 2025-11-04: Added fine-grained adjacency instrumentation (membership/targets/CSR) and replaced the scatter/dedup stage with pair-id accumulation. Tests (`tests/test_conflict_graph.py`, `tests/test_semisort.py`, `tests/integration/test_structural_core.py`) stay green. Quick benchmark now builds in 46.69 s, queries in 2.09 s (245.2 q/s); conflict graph averages 1.94 s per batch with scope grouping ≈1.72 s and adjacency ≈0.18 s. Follow-ups: move scope grouping off the host path and validate the pair-id accumulator on the 8 192-point run.
- Implementation experiments are behind a toggle: `COVERTREEX_CONFLICT_GRAPH_IMPL={dense|segmented|auto}` (default `dense`). Current segmented prototype preserves correctness but is slower (quick run: `build ≈ 80.6 s` vs `24.5 s` dense), so dense remains default until the segmented path gets the expected win.
