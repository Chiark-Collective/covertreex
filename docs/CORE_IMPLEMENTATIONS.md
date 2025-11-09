# Core Implementations & Benchmark Snapshot (2025-11-06)

_This file is a to-be-maintained reference. Keep the benchmark table and the code listings in sync with the repository state whenever core algorithms change._

## Performance Summary (CPU, Numba-enabled)

> GPU/JAX execution is currently disabled. All timings below use the NumPy backend with `COVERTREEX_BACKEND=numpy` and `COVERTREEX_ENABLE_NUMBA=1`.

### Quick Benchmark — 2 048 tree pts / 512 queries / k=8

| Implementation               | Build Time (s) | Query Time (s) | Throughput (q/s) | Notes |
|------------------------------|----------------|----------------|------------------|-------|
| PCCT (Numba, diagnostics off)| 0.366          | 0.097          | 5 261            | `COVERTREEX_ENABLE_DIAGNOSTICS=0`; diagnostics-on run: 0.373 s / 0.098 s |
| Sequential baseline          | 2.25           | 0.024          | 21 001           | In-repo compressed cover tree |
| GPBoost Numba baseline       | 0.292          | 0.519          | 987              | Numba port of the GPBoost cover tree |
| External CoverTree baseline  | 1.00           | 1.215          | 421              | `pip install -e '.[baseline]'` |

_Command:_
```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=0 \
UV_CACHE_DIR=$PWD/.uv-cache \
uv run python -m cli.queries \
  --dimension 8 --tree-points 2048 \
  --batch-size 128 --queries 512 --k 8 \
  --seed 42 --baseline gpboost
```

_Set `COVERTREEX_ENABLE_DIAGNOSTICS=1` to collect the instrumentation counters (adds ~7 ms to the build in this configuration)._

### Telemetry artefacts (default output paths)

All benchmark entrypoints now stamp a unique `run_id` and write structured telemetry under `artifacts/` unless you explicitly opt out:

- `cli.queries` creates `artifacts/benchmarks/queries_<run_id>.jsonl` automatically. Use `--log-file custom.jsonl` to override the location or `--no-log-file` to suppress JSONL output altogether.
- `cli.runtime_breakdown` writes CSV summaries to `artifacts/benchmarks/runtime_breakdown_<run_id>.csv` unless you pass `--no-csv-output`. Override with `--csv-output /path/to/file.csv` if you need a fixed path (the `cli.runtime_breakdown` shim remains for backwards compatibility).
- `benchmarks.batch_ops` emits a JSON summary (`artifacts/benchmarks/batch_ops_<run_id>.json`) by default. Pass `--log-json custom.json` to change the destination or `--no-log-json` to skip it.

These files include the runtime configuration snapshot (`runtime_*` keys) so you can correlate logs back to the exact backend/precision/strategy selections that were active for that run.

### Scaling Snapshot — CPU builds (diagnostics on)

| Workload (tree pts / queries / k) | PCCT Build (s) | PCCT Query (s) | PCCT q/s | Sequential Build (s) | Sequential q/s | GPBoost Build (s) | GPBoost q/s | External Build (s) | External q/s |
|-----------------------------------|----------------|----------------|----------|----------------------|----------------|-------------------|-------------|--------------------|---------------|
| 8 192 / 1 024 / 16                | 4.15           | 0.018          | 57 660   | 33.65               | 5 327         | 0.75              | 285         | 14.14              | 122           |
| 32 768 / 1 024 / 8 (Euclidean)    | 16.75          | 0.039          | 25 973   | —                   | —             | 3.10              | 65.1        | —                  | —             |
| 32 768 / 1 024 / 8 (Residual)*    | 66.25          | 0.305          | 3 358    | —                   | —             | 2.51              | 91.6        | —                  | —             |

_*GPBoost remains Euclidean-only; the baseline numbers in the residual row are provided for throughput context only._

The 32 768-point run currently logs PCCT and the GPBoost baseline; sequential/external baselines are still pending optimisations to keep runtime manageable at that scale.

### Gold-standard residual benchmark (default path)

Fresh artefacts: `benchmark_grid_32768_baseline_20251108.jsonl` + `_run2.jsonl` (paired with `bench_euclidean_grid_32768_20251108*.log`) and `benchmark_residual_32768_default.jsonl` (with `run_residual_32768_default.txt`) capture the PCCT rows above using the Euclidean defaults (`COVERTREEX_BATCH_ORDER=hilbert`, `COVERTREEX_PREFIX_SCHEDULE=adaptive`, `COVERTREEX_ENABLE_NUMBA=1`, `COVERTREEX_SCOPE_CHUNK_TARGET=0`, `COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256`, `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`). Residual runs automatically flip to `COVERTREEX_PREFIX_SCHEDULE=doubling` and enable Gate‑1 lookup (`docs/data/residual_gate_profile_diag0.json`) while keeping chunking disabled.

The historical **56.09 s / 0.229 s (4 469 q/s)** residual run (captured 2025‑11‑06) predates the grid/batch-order refresh but remains our “gold standard” regression target. The maintained harness now pins the closest available configuration—Numba enabled, natural batch order, doubling prefix schedule, diagnostics off, and no scope chunking—and currently measures **≈71.8 s build / 0.272 s query (3 762 q/s)**. To regenerate the artefact (and keep the environment consistent) run:

```
./benchmarks/run_residual_gold_standard.sh [optional_log_path]
```

By default the script writes `bench_residual.log` in the repo root and sets `COVERTREEX_ENABLE_NUMBA=1`, `COVERTREEX_BATCH_ORDER=natural`, `COVERTREEX_PREFIX_SCHEDULE=doubling`, `COVERTREEX_SCOPE_CHUNK_TARGET=0`, and `COVERTREEX_ENABLE_DIAGNOSTICS=0` so the output stays comparable across machines. Treat this log as the reference artefact when auditing regressions or publishing updated numbers (refer back to the 2025‑11‑06 56.09 s run when you need the historical pre-grid baseline).

To reproduce the clamped adjacency run captured on 2025‑11‑07 (matching `benchmark_residual_clamped_20251107_fix_run2.jsonl`), invoke:

```
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
UV_CACHE_DIR=$PWD/.uv-cache \
uv run python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --metric residual --baseline none
```

The JSONL log now lands in `artifacts/benchmarks/queries_<run_id>.jsonl` automatically; pass `--log-file benchmark_residual_clamped_20251107_fix_run2.jsonl` if you need to mirror the historical file layout.

> **Baseline note:** The `_diag0` artefacts from 2025‑11‑07 (`benchmark_residual_clamped_20251107_diag0.jsonl`, `benchmark_residual_chunked_20251107_diag0.jsonl`) are the preferred reference logs whenever you need to compare against GPBoost or older “*_fix_run2” runs. They disable diagnostics to match the baseline settings but still include the new `gate1_*` and `traversal_scope_chunk_{scans,points,dedupe,saturated}` counters, so keep using them for apples-to-apples regressions going forward.

_Command (8 192 row):_
```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
python -m cli.queries \
  --dimension 8 --tree-points 8192 \
  --batch-size 256 --queries 1024 --k 16 \
  --seed 12345 --baseline gpboost \
  --log-file benchmark_grid_8192_baseline_20251108.jsonl

# 32k Euclidean vs GPBoost
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file benchmark_grid_32768_baseline_20251108.jsonl

# 32k residual vs GPBoost (baseline remains Euclidean-only)
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --metric residual
```
Residual commands automatically run with `COVERTREEX_PREFIX_SCHEDULE=doubling`, Gate‑1 lookup enabled, and chunking disabled unless you override those env vars.

To capture warm-up versus steady-state timings for plotting, append `--csv-output runtime_breakdown_metrics.csv` when running `cli.runtime_breakdown`.

## Current Observations & Hypotheses

### Euclidean metric (NumPy backend)

- Fresh 32 768-point Hilbert+grid reruns (dimension 8, batch 512, 1 024 queries, k = 8, seed 42) now land at **16.75 s build / 0.039 s query (~26 k q/s)** for PCCT with NumPy+Numba (`bench_euclidean_grid_32768_20251108*.log` + `benchmark_grid_32768_baseline_20251108{,_run2}.jsonl`). The paired GPBoost baseline clocks in at **≈3.10 s build / 15.72 s query (65 q/s)**, so PCCT sustains ~400× higher steady-state throughput in this configuration.
- The 8 192-point suite (`--batch-size 256`, `k=16`, seed 12345) records **4.15 s build / 0.018 s query (57.7 k q/s)** for PCCT versus GPBoost’s **0.75 s / 3.59 s (285 q/s)**, captured in `bench_euclidean_grid_8192_20251108.log` + `benchmark_grid_8192_baseline_20251108.jsonl`.
- Batch logs show `traversal_ms` clustering between ≈0.25–0.49 s on dominated 32 k batches while `conflict_graph_ms` remains within 7–19 ms and MIS stays sub-0.2 ms. With the journal pipeline eliminating repeated array slices, traversal/mask assembly is once again the limiting phase at scale even under the grid builder.
- Conflict-graph builders now live in `conflict_graph_builders.py`; dense vs segmented vs residual paths report distinct telemetry. Dense remains the default until scope chunk limits (see Next Steps) are dialled in.
- GPBoost’s cover tree baseline is sequential and Euclidean-only (no conflict graphs, no MIS). We continue to keep comparisons on the CPU/NumPy backend so wall-clock deltas stay reproducible without JAX/GPU variability.

These November 8 artefacts supersede the 37.7 s / 0.262 s Hilbert+grid numbers from earlier in the week; treat them as the current Euclidean “gold standard” for PCCT until another build drops below ~15 s while keeping telemetry comparable.

### Residual-correlation metric (synthetic RBF caches, 2025-11-09)

- **Current best (clamped, scope chunking off).** Re-running the 32 768×1 024×k=8 workload with Hilbert ordering, the grid conflict builder, diagnostics on, and no scope chunking now lands at **57.80 s build / 0.028 s query (36.2 k q/s)** for PCCT while the GPBoost baseline remains at **≈2.68 s build / 10.50 s query (97.5 q/s)**. Artefacts: `benchmark_residual_clamped_20251109.log` and `artifacts/benchmarks/benchmark_residual_clamped_20251109.jsonl`.
- **Chunked traversal (scope cap 8 192).** Enabling sparse traversal with `COVERTREEX_SCOPE_CHUNK_TARGET=8192` keeps query time identical (0.028 s) but increases build time to **727.44 s** because each dominated batch now scans ≈4.19 M points in 512 chunks before conflict filtering; adjacency scatter drops to ≈18 ms. Logs: `benchmark_residual_scope8192_20251109.log` + `artifacts/benchmarks/benchmark_residual_scope8192_20251109.jsonl`.
- The historical unclamped Hilbert run from 2025‑11‑07 (`benchmark_residual_32768_default.jsonl` / `run_residual_32768_default.txt`) is still useful for regression tracking (66.25 s build / 0.305 s query). Likewise, the earlier chunked-with-prefilter sweeps (`benchmark_residual_cache_prefilter_20251108.jsonl`, `benchmark_residual_scopecap_20251108.jsonl`) describe how lookup-driven prefilters impact traversal telemetry.
- Per-batch telemetry for the new clamped run shows dominated batches averaging **~0.76 s traversal / 93 ms conflict graph / 70 ms adjacency scatter** while leaving Gate‑1 counters at zero; conflict scopes regularly swell past 16 M members because chunking is disabled. The chunked run trades build time for tighter memory bounds (max scope shard 8 192 members, 512 segments per dominated batch) and keeps adjacency scatter bounded.
- The residual adjacency filter still recomputes pairwise kernels even when `residual_pairwise` is cached from traversal; linking those surfaces remains the most obvious follow-up before we attempt to rely on Gate‑1 lookups.

**Commands to reproduce**

Current best (clamped, diagnostics on, `scope_chunk_target=0`):

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
UV_CACHE_DIR=$PWD/.uv-cache \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file benchmark_residual_clamped_20251109.jsonl
```

Chunked traversal (sparse traversal + scan cap):

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_TARGET=8192 \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
UV_CACHE_DIR=$PWD/.uv-cache \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file benchmark_residual_scope8192_20251109.jsonl
```

These commands emit the same JSONL telemetry referenced above (see `artifacts/benchmarks/`) and print the console summaries captured in the paired `.log` files.

## Residual-Correlation Metric Benchmark Status (2025-11-06)

**Source overview**
- `covertreex/metrics/residual.py`, `covertreex/metrics/_residual_numba.py`, and the hooks in `covertreex/algo/{traverse,conflict_graph}.py` implement the host-side residual metric, streaming scopes, and adjacency reuse.
- `benchmarks/queries.py` builds synthetic RBF caches (lines 77–170) and exposes `--metric residual` plus the `--residual-*` tuning flags. The script now forces `JAX_PLATFORM_NAME=cpu` / `XLA_PYTHON_CLIENT_PREALLOCATE=false` so the MIS helper cannot reserve GPU memory when we are on the NumPy backend.
- Host cache construction mirrors the VIF pipeline: RBF kernel factors, `ResidualCorrHostData` configuration, and a hashed `point_decoder` that maps tree payloads back to dataset indices.

**Benchmark recipe**

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_METRIC=residual_correlation \
UV_CACHE_DIR=$PWD/.uv-cache \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --residual-inducing 512 \
  --residual-lengthscale 1.0 --residual-variance 1.0
```

The Euclidean-visible telemetry table from earlier runs is intentionally omitted until we re-run `cli.runtime_breakdown` with the residual metric enabled; the new builder split exposes `scope_chunk_segments`, `scope_chunk_emitted`, and `scope_chunk_max_members`, so refreshing that CSV after the scope-chunk work will give us defensible per-phase averages again.

### Residual-correlation machinery — key source excerpts

```python
# covertreex/metrics/residual.py (excerpt)
@dataclass(frozen=True)
class ResidualCorrHostData:
    v_matrix: np.ndarray
    p_diag: np.ndarray
    kernel_diag: np.ndarray
    kernel_provider: KernelProvider
    point_decoder: PointDecoder = _default_point_decoder
    chunk_size: int = 512
    v_norm_sq: np.ndarray = None  # type: ignore[misc]
    gate1_enabled: bool | None = None
    gate1_alpha: float | None = None
    gate1_margin: float | None = None
    gate1_eps: float | None = None
    gate1_audit: bool | None = None
    gate_v32: np.ndarray | None = None
    gate_norm32: np.ndarray | None = None
    gate_stats: ResidualGateTelemetry = field(default_factory=ResidualGateTelemetry)

def compute_residual_distances_with_radius(
    backend: ResidualCorrHostData,
    query_index: int,
    chunk_indices: np.ndarray,
    kernel_row: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(chunk_indices, dtype=np.int64)
    kernel_vals = np.asarray(kernel_row, dtype=np.float64)
    enabled, alpha, margin, eps, audit = _resolve_gate1_config(backend)
    gate_mask: np.ndarray | None = None
    if enabled and backend.gate_v32 is not None:
        threshold = alpha * float(radius) + margin
        keep_uint8 = gate1_whitened_mask(
            backend.gate_v32[query_index],
            backend.gate_v32[indices],
            threshold,
        )
        gate_mask = keep_uint8.astype(bool, copy=False)
        backend.gate_stats.candidates += int(indices.size)
        backend.gate_stats.kept += int(np.count_nonzero(gate_mask))
        backend.gate_stats.pruned += int(indices.size - np.count_nonzero(gate_mask))
    if gate_mask is None or gate_mask.all():
        return compute_distance_chunk(
            v_query=backend.v_matrix[query_index],
            v_chunk=backend.v_matrix[indices],
            kernel_chunk=kernel_vals,
            p_i=float(backend.p_diag[query_index]),
            p_chunk=backend.p_diag[indices],
            norm_query=float(backend.v_norm_sq[query_index]),
            norm_chunk=backend.v_norm_sq[indices],
            radius=float(radius),
            eps=_EPS,
        )
    survivors = np.nonzero(gate_mask)[0]
    if survivors.size == 0:
        empty_dist = np.full(indices.size, float(radius) + eps, dtype=np.float64)
        empty_mask = np.zeros(indices.size, dtype=np.uint8)
        return empty_dist, empty_mask
    distances = np.full(indices.size, float(radius) + eps, dtype=np.float64)
    mask = np.zeros(indices.size, dtype=np.uint8)
    sub_dist, sub_mask = compute_distance_chunk(
        v_query=backend.v_matrix[query_index],
        v_chunk=backend.v_matrix[indices[survivors]],
        kernel_chunk=kernel_vals[survivors],
        p_i=float(backend.p_diag[query_index]),
        p_chunk=backend.p_diag[indices[survivors]],
        norm_query=float(backend.v_norm_sq[query_index]),
        norm_chunk=backend.v_norm_sq[indices[survivors]],
        radius=float(radius),
        eps=_EPS,
    )
    distances[survivors] = sub_dist
    mask[survivors] = sub_mask
    if audit and np.any(~gate_mask):
        _audit_gate1_pruned(...)
    return distances, mask

def configure_residual_correlation(backend: ResidualCorrHostData) -> None:
    if backend.v_norm_sq is None:
        object.__setattr__(backend, "v_norm_sq", np.sum(backend.v_matrix * backend.v_matrix, axis=1))
    set_residual_backend(backend)
    def pairwise_kernel(tree_backend, lhs, rhs):
        host_backend = get_residual_backend()
        lhs_idx = decode_indices(host_backend, lhs)
        rhs_idx = decode_indices(host_backend, rhs)
        distances = compute_residual_distances(host_backend, lhs_idx, rhs_idx)
        return tree_backend.asarray(distances, dtype=tree_backend.default_float)
    def pointwise_kernel(tree_backend, lhs, rhs):
        host_backend = get_residual_backend()
        lhs_idx = decode_indices(host_backend, lhs)
        rhs_idx = decode_indices(host_backend, rhs)
        value = compute_residual_distance_single(host_backend, int(lhs_idx[0]), int(rhs_idx[0]))
        return tree_backend.asarray(value, dtype=tree_backend.default_float)
    configure_residual_metric(pairwise=pairwise_kernel, pointwise=pointwise_kernel)
```

```python
# covertreex/metrics/_residual_numba.py (excerpt)
@njit(cache=True, fastmath=True, parallel=True)
def _distance_chunk(
    v_query: np.ndarray,
    v_chunk: np.ndarray,
    kernel_chunk: np.ndarray,
    p_i: float,
    p_chunk: np.ndarray,
    norm_query: float,
    norm_chunk: np.ndarray,
    radius: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    threshold = 1.0 - radius * radius
    for j in prange(v_chunk.shape[0]):
        denom = math.sqrt(max(p_i * p_chunk[j], eps * eps))
        partial = 0.0
        qi_sq = 0.0
        qj_sq = 0.0
        pruned = False
        for d in range(v_query.shape[0]):
            vq = v_query[d]
            vc = v_chunk[j, d]
            partial += vq * vc
            qi_sq += vq * vq
            qj_sq += vc * vc
            tail = math.sqrt(max(norm_query - qi_sq, 0.0) * max(norm_chunk[j] - qj_sq, 0.0))
            if denom > 0.0 and threshold > 0.0:
                base = kernel_chunk[j] - partial
                max_abs = abs(base + tail) if tail > 0.0 else abs(base)
                if max_abs / denom + eps < threshold:
                    distances[j] = radius + eps
                    within[j] = 0
                    pruned = True
                    break
        if pruned:
            continue
        rho = (kernel_chunk[j] - partial) / denom if denom > 0.0 else 0.0
        rho = max(min(rho, 1.0), -1.0)
        dist = math.sqrt(max(0.0, 1.0 - abs(rho)))
        distances[j] = dist
        within[j] = 1 if dist <= radius + eps else 0
    return distances, within

def compute_distance_chunk(...):
    if NUMBA_RESIDUAL_AVAILABLE:
        return _distance_chunk(...)
    # NumPy fallback matches the Numba kernel semantics
```

Environment knobs

- `COVERTREEX_RESIDUAL_GATE1` (default `0`) enables the whitened float32 gate globally.
- `COVERTREEX_RESIDUAL_GATE1_ALPHA`, `..._MARGIN`, and `..._EPS` control the linear threshold mapping and the sentinel used for pruned entries.
- `COVERTREEX_RESIDUAL_GATE1_AUDIT=1` reruns the precise kernel on pruned entries and raises if any lie within the requested residual radius; this is wired into the regression tests and benchmark telemetry as `traversal_gate1_*` counters.
- `COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH` points at an empirical lookup table (for example `docs/data/residual_gate_profile_diag0.json`, generated by `tools/build_residual_gate_profile.py`). When present, the lookup supplies per-radius thresholds directly, so the linear α/margin knobs become safety fallbacks rather than the primary tuneable.

### Persistence journal (NumPy backend)

- With `COVERTREEX_BACKEND=numpy` and `COVERTREEX_ENABLE_NUMBA=1`, batch inserts now route through a Numba “journal” kernel that clones the parent/child/next arrays once per batch and applies updates in a single sweep. When the flag is disabled or a non-NumPy backend is selected, the legacy SliceUpdate path is used automatically.
- The journal path is metric-agnostic: it operates on the structural arrays only, so custom metrics (e.g. residual correlation) remain supported without additional guards.
- Scratch buffers for head/next chains are pooled across batches, eliminating the small-but-frequent NumPy allocations that previously showed up as RSS spikes in large builds. Enable diagnostics to monitor pool growth via the existing logging hooks.
- Rerun `cli.runtime_breakdown` before/after enabling the journal to record wall-clock deltas for auditors; the 32 768-point configuration is the most illustrative workload.

### Warm-up & JIT compilation

- The first PCCT build in any fresh Python session pays the full Numba compilation cost plus the actual 32 k build (≈40 s in the latest 8-dim profile). Subsequent builds reuse the cached kernels and drop to the steady-state numbers shown above (sub-second per batch).
- To hide the one-time hit in production pipelines, kick off a tiny “warm-up” batch during process start-up—for example, build a 64-point tree with the same runtime flags before ingesting real data. This compiles all hot kernels without noticeably touching RSS or wall time.
- To quantify the difference in practice, run `cli.runtime_breakdown` with `--runs 5` (or similar). The CSV now includes a `run` column, letting you compare the first-build warm-up against subsequent steady-state runs directly.
- Quick sanity runs still default to `COVERTREEX_METRIC=euclidean`. Use `cli.queries --metric residual` (which auto-builds synthetic caches) when you need the residual path; the script now forces JAX onto the CPU so the MIS helper cannot grab GPU memory.

### Threading defaults

- On import we clamp `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1 so NumPy/BLAS libraries do not oversubscribe the worker pool. We also set `NUMBA_THREADING_LAYER=tbb` and default `NUMBA_NUM_THREADS` to the host CPU count (override any of these via environment variables before importing `covertreex`).
- For heavier boxes increase `NUMBA_NUM_THREADS` and, if needed, batch size (`--batch-size`) and `COVERTREEX_SCOPE_CHUNK_TARGET` to keep each worker busy. The TBB threading layer provides work-stealing, so threads redistribute skewed batches automatically once these pools are configured.

### Grid conflict builder + batch ordering knobs (2025‑11‑07)

- Setting `COVERTREEX_CONFLICT_GRAPH_IMPL=grid` switches the dominated-batch path to a hash-grid leader election that bypasses CSR + MIS. Leaders are selected per cell via mixed 64-bit priorities and greedily filtered before insertion; MIS still runs but sees a trivially independent set. Telemetry published via `conflict_grid_cells`, `*_leaders_*`, and `grid_local_edges` counters in diagnostics/JSONL logs.
- Benchmark snapshot (NumPy backend, diagnostics off, hilbert disabled):
  - **8 192 / 1 024 / k=16**: build **5.20 s**, query **0.085 s** (12.0 k q/s) with `grid` vs 4.03 s/0.934 s on the legacy dense path.
  - **32 768 / 1 024 / k=8**: build **37.7 s**, query **0.262 s** (3.9 k q/s). The grid path removes MIS time entirely (0 edges) while keeping per-batch scatter ≤6.7 ms once Hilbert ordering is enabled (see `benchmark_grid_32768_default_run2.jsonl`).
- Batch-order controls ship as both env vars and CLI overrides in `benchmarks/queries.py`:
  - `COVERTREEX_BATCH_ORDER={natural|random|hilbert}` or `--batch-order …` sets the per-batch permutation; `hilbert` compacts domination-friendly prefixes and logs spread via `batch_order_hilbert_code_spread`.
  - `COVERTREEX_PREFIX_SCHEDULE={doubling|adaptive}` / `--prefix-schedule …` toggles density-aware prefix growth (see `COVERTREEX_PREFIX_DENSITY_*` + `COVERTREEX_PREFIX_GROWTH_*` envs for thresholds).
  - Example CLI: `uv run python -m cli.queries … --batch-order hilbert --prefix-schedule adaptive` (Hilbert run above yielded the same 5.20 s build with steadier per-batch scatter and domination ratios logged inline).
- **Hilbert becomes the default batch order (2025‑11‑07).** Fresh 32 768-point logs (`benchmark_grid_32768_natural.jsonl`, `benchmark_grid_32768_default_run2.jsonl`) show the first dominated batch’s scatter dropping from **3 951 ms → 6.6 ms**, average scatter falling **62 ms → 0.55 ms**, and `conflict_graph_ms` shrinking **83.7 ms → 22.4 ms** when Hilbert ordering is enabled alongside the grid builder. Because the domination ratio and leader counts stay unchanged (≈0.984 and ≈202 leaders/batch), we now default `COVERTREEX_BATCH_ORDER=hilbert` and updated the scaling table above (Euclidean build **37.7 s**, query **0.262 s**, `~3.9 k q/s`).
- **Adaptive prefix growth defaults tuned + exposed via `--build-mode prefix`.** Use `python -m cli.queries … --build-mode prefix` to route construction through `batch_insert_prefix_doubling` and emit per-prefix telemetry (`prefix_factor`, `prefix_domination_ratio`, `prefix_schedule`) into the JSONL log. The 32 k Hilbert+grid run in this mode (`benchmark_grid_32768_prefix.jsonl`) produced **16 385 adaptive groups** with scatter averaging **0.047 ms** (median 0.043 ms, max 12.2 ms) and domination ratio ≈1.0 while keeping the prefix-factor blend at 1.25/2.25. These numbers justify the new `_DEFAULT_PREFIX_*` constants and give auditors a structured artefact when analysing prefix shaping; the residual variant still needs follow-up because the clamped run exceeds 20 minutes under this schedule (see plan §2).
- **Residual batches now use the grid builder whenever `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`.** The forced-leader path consumes the Gate‑1 whitened vectors (`gate_v32`) so `conflict_grid_*` counters finally light up in residual JSONL logs; MIS sees an empty graph and per-batch adjacency collapses accordingly.
- **Scope chunking uses an adaptive target in highly dominated residual runs.** When <1 % of scopes contain survivors the runtime picks a chunk target inside `[8 192, 262 144]` so per-batch pair volume stays ≈128 k directed edges; the new target is surfaced via `conflict_graph.timings.scope_chunk_pair_cap`.

## PCCT Source & Execution Flow (2025-11-09)

### Source listing — `covertreex/api/pcct.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

from covertreex.algo.batch import BatchInsertPlan, batch_insert
from covertreex.algo.batch_delete import BatchDeletePlan, batch_delete
from covertreex.api.runtime import Runtime
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.queries.knn import knn as knn_query


def _ensure_points(backend: TreeBackend, value: Any) -> Any:
    arr = backend.asarray(value, dtype=backend.default_float)
    if arr.ndim == 0:
        arr = backend.xp.reshape(arr, (1, 1))
    elif arr.ndim == 1:
        length = int(arr.shape[0])
        if length == 0:
            arr = backend.xp.reshape(arr, (0, 0))
        else:
            arr = backend.xp.reshape(arr, (1, length))
    return backend.device_put(arr)


def _ensure_indices(backend: TreeBackend, value: Any) -> Any:
    arr = backend.asarray(value, dtype=backend.default_int)
    return backend.device_put(arr)


def _convert_tree(tree: PCCTree, backend: TreeBackend) -> PCCTree:
    if tree.backend is backend:
        return tree
    same_backend = (
        tree.backend.name == backend.name
        and tree.backend.default_float == backend.default_float
        and tree.backend.default_int == backend.default_int
    )
    if same_backend:
        return tree
    return tree.to_backend(backend)


@dataclass(frozen=True)
class PCCT:
    """Thin façade around batch insert/delete + query helpers."""

    runtime: Runtime = field(default_factory=Runtime)
    tree: PCCTree | None = None

    def fit(
        self,
        points: Any,
        *,
        apply_batch_order: bool = True,
        mis_seed: int | None = None,
        return_plan: bool = False,
    ) -> PCCTree | Tuple[PCCTree, BatchInsertPlan]:
        context = self.runtime.activate()
        backend = context.get_backend()
        batch = _ensure_points(backend, points)
        dimension = int(batch.shape[1])
        base_tree = PCCTree.empty(dimension=dimension, backend=backend)
        new_tree, plan = batch_insert(
            base_tree,
            batch,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
        )
        return (new_tree, plan) if return_plan else new_tree

    def insert(
        self,
        batch_points: Any,
        *,
        mis_seed: int | None = None,
        apply_batch_order: bool | None = None,
        return_plan: bool = False,
    ) -> PCCTree | Tuple[PCCTree, BatchInsertPlan]:
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        batch = _ensure_points(backend, batch_points)
        tree_backend = _convert_tree(tree, backend)
        new_tree, plan = batch_insert(
            tree_backend,
            batch,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
        )
        return (new_tree, plan) if return_plan else new_tree

    def delete(
        self,
        indices: Any,
        *,
        return_plan: bool = False,
    ) -> PCCTree | Tuple[PCCTree, BatchDeletePlan]:
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        remove = _ensure_indices(backend, indices)
        tree_backend = _convert_tree(tree, backend)
        new_tree, plan = batch_delete(tree_backend, remove, backend=backend)
        return (new_tree, plan) if return_plan else new_tree

    def knn(
        self,
        query_points: Any,
        *,
        k: int,
        return_distances: bool = False,
    ) -> Any:
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        tree_backend = _convert_tree(tree, backend)
        queries = _ensure_points(tree_backend.backend, query_points)
        return knn_query(
            tree_backend,
            queries,
            k=k,
            return_distances=return_distances,
            backend=tree_backend.backend,
        )

    def nearest(self, query_points: Any, *, return_distances: bool = False) -> Any:
        return self.knn(query_points, k=1, return_distances=return_distances)

    def _require_tree(self) -> PCCTree:
        if self.tree is None:
            raise ValueError("PCCT requires an existing tree; call fit() first.")
        return self.tree
```

### Execution flow overview

- **Runtime handshake.** `PCCT` owns a `covertreex.api.runtime.Runtime` instance; every public method calls `Runtime.activate()` to materialise a `RuntimeContext`, configure logging/JAX flags, and retrieve the backend (`covertreex/runtime/config.py`). This guarantees that downstream kernels see the same dtype/backend settings as the CLI entrypoints and tests.
- **Batch ingestion.** `.fit()` builds a fresh `PCCTree` via `PCCTree.empty()` and forwards the data into `covertreex/algo/batch/insert.py::batch_insert`. `.insert()` uses `_convert_tree()` to move an existing tree to the active backend before running the same pipeline. `_ensure_points()` standardises shapes/precision so the traversal/training kernels can assume contiguous 2-D arrays.
- **Plan + persistence.** `batch_insert()` calls `plan_batch_insert()` → `traverse_collect_scopes()` → `build_conflict_graph()` → `run_mis()` before applying the `covertreex.core.persistence` journal (`build_persistence_journal()` + `apply_persistence_journal()`). `BatchInsertPlan` keeps traversal, conflict, MIS, and ordering telemetry so diagnostics/CLI logs can be emitted verbatim from CLI scripts.
- **Deletion pipeline.** `.delete()` hands indices to `covertreex/algo/batch_delete.py::batch_delete`, which mirrors the journal update flow (recomputes parents/children and rewrites compressed arrays) while returning a `BatchDeletePlan` for instrumentation.
- **Queries.** `.knn()` (and `.nearest()`) bridge into `covertreex/queries/knn.py::knn`, which wraps the Numba-accelerated `knn_numba` path when available. The helper shares the same backend so query buffers stay on-device for either NumPy or JAX runs.

## Optional Features & Terminology Reference

| Term / Feature | Source entry points | Why it exists | How to enable / tune |
| --- | --- | --- | --- |
| Batches & batch ordering | `covertreex/algo/order/helpers.py::prepare_batch_points`, `covertreex/algo/batch/insert.py::plan_batch_insert` | Permutes each ingestion batch (Hilbert, random, natural) to minimise scope scatter before traversal/conflict building; records permutation + Hilbert metrics in `BatchInsertPlan`. | `COVERTREEX_BATCH_ORDER`, `COVERTREEX_BATCH_ORDER_SEED`, CLI `--batch-order/--batch-order-seed` (default `hilbert`). |
| Prefix schedules & groups | `covertreex/algo/batch/insert.py::batch_insert_prefix_doubling`, `covertreex/algo/order/helpers.py::choose_prefix_factor` | Splits a large ingestion into prefixes (doubling or adaptive) so domination ratios stay ≈1.0 and MIS never sees pathological superscopes; residual runs now default to `doubling` while Euclidean stays on `adaptive`. | `COVERTREEX_PREFIX_SCHEDULE`, `COVERTREEX_PREFIX_DENSITY_*`, `COVERTREEX_PREFIX_GROWTH_*`, CLI `--build-mode prefix`. |
| Conflict graph (dense / segmented / residual) | `covertreex/algo/conflict/strategies.py`, `covertreex/algo/conflict/builders.py` | Switches between CSR-from-mask (`dense`), point-ID segmented builders (`segmented`), and residual-aware pruning (`residual`) so MIS operates on the smallest safe adjacency. | `COVERTREEX_CONFLICT_GRAPH_IMPL={auto,dense,segmented,grid}`, `COVERTREEX_ENABLE_NUMBA`, `COVERTREEX_SCOPE_SEGMENT_DEDUP`. |
| Grid leader selection | `covertreex/algo/conflict/builders.py::build_grid_adjacency`, `covertreex/algo/_grid_numba.py::grid_select_leaders_numba` | Hash-grid allocator that forces leaders/dominated nodes per cell, eliminating MIS edges for highly dominated batches while publishing `grid_*` telemetry; residual runs now always feed the Gate‑1 whitened vectors (scaled via `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE`) into the same grid so they stay in lock-step with Euclidean runs. | `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`, `COVERTREEX_ENABLE_NUMBA=1`, tweak `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` (default `1.0`) to widen/narrow residual cells, Hilbert ordering recommended. |
| Scope chunks | `covertreex/algo/_scope_numba.py::_chunk_ranges_from_indptr`, `ScopeAdjacencyResult` | Splits oversubscribed scopes into bounded "chunks" so conflict-pair generation stays linear and exposes counters (`scope_chunk_*`) for auditors; residual runs auto-raise the target when domination falls below 1 %. | `COVERTREEX_SCOPE_CHUNK_TARGET`, `COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS`. |
| MIS seeding & solver | `covertreex/algo/mis.py::batch_mis_seeds`, `::run_mis`, `covertreex/algo/_mis_numba.py::run_mis_numba` | Deterministic seed fan-out plus Numba-accelerated MIS keep batch-to-batch behaviour reproducible and fast; falls back to pure JAX when Numba is unavailable. | `COVERTREEX_MIS_SEED`, CLI `--mis-seed`, `COVERTREEX_ENABLE_NUMBA`. |
| Residual gating & prefilter | `covertreex/metrics/residual.py` (gate1, scope radii, telemetry), `covertreex/metrics/residual/scope_caps.py` | Whitened float32 gates and lookup tables drop expensive residual chunk evaluations early, enforce radius floors, and populate `gate_stats`; lookup mode now activates automatically whenever `--metric residual`. | `COVERTREEX_METRIC=residual_correlation`, plus `COVERTREEX_RESIDUAL_*` knobs (gate1, lookup, prefilter, scope caps). |
| Sparse traversal | `covertreex/algo/_traverse_sparse_numba.py::collect_sparse_scopes`, `covertreex/algo/traverse/strategies.py` | Builds CSR scopes directly for large trees (instead of dense masks) to reduce memory pressure before conflict assembly. | `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1`, CLI `--sparse-traversal`. |
| Diagnostics & telemetry | `covertreex/diagnostics.py::log_operation`, `covertreex/telemetry/*`, CLI writers under `cli/` | Emits per-stage timings (`conflict_graph_ms`, `scope_chunk_*`, `grid_*`) and writes JSONL/CSV artefacts for the benchmarking harness. | `COVERTREEX_ENABLE_DIAGNOSTICS`, CLI `--log-file/--no-log-file`, `cli.runtime_breakdown --no-csv-output`. |

### Chunks (scope chunking & telemetry)

`_chunk_ranges_from_indptr()` and the `ScopeAdjacencyResult` dataclass in `covertreex/algo/_scope_numba.py` compute bounded (start, end) windows over `scope_indptr` so no worker ever tries to emit more than `scope_chunk_target` pairs. The helper also tracks `chunk_count`, `chunk_emitted`, `chunk_pair_cap`, and before/after pair counts; these surface in `BatchInsertPlan.conflict_graph.timings.scope_chunk_*`. Raising `COVERTREEX_SCOPE_CHUNK_TARGET` increases per-worker work, lowering it forces more segments (and higher scheduler overhead) but caps peak memory better.

### Grids & forced leader selection

`build_grid_adjacency()` (and its Numba twin `grid_select_leaders_numba()`) hash each batch point into a shifted grid keyed by level-derived radii, then picks a deterministic leader per cell via mixed 64-bit priorities. Survivors land in `forced_selected`, others in `forced_dominated`; MIS sees the pre-coloured arrays and can skip adjacency construction entirely. This mode exists for "dominated" regimes where almost every candidate is within someone else’s annulus—think Hilbert-ordered 32 k batches—so the builder can collapse conflict graphs to zero edges while still logging `grid_cells`, `grid_leaders_*`, and `grid_local_edges` counters for regressions.

### Batches & prefix schedules

`prepare_batch_points()` enforces the runtime permutation (default Hilbert), while `batch_insert_prefix_doubling()` in `covertreex/algo/batch/insert.py` slices the permuted data into prefixes dictated by `prefix_slices()` or the adaptive factor from `choose_prefix_factor()`. Every sub-batch carries its own MIS seed (via `batch_mis_seeds()`), conflict graph, and domination ratio so JSONL artefacts can expose where throughput stalls. Use `--build-mode prefix --prefix-schedule adaptive` when reproducing the 32 k benchmarks cited earlier—the CLI writes `prefix_factor`, `prefix_domination_ratio`, and permutation metadata directly from the plan objects.

### Conflict graphs & MIS

`covertreex/algo/conflict/base.py::ConflictGraph` represents adjacency in CSR form together with radii, annulus bins, and optional `forced_*` masks. `strategies.py` selects `dense`, `segmented`, `grid`, or `residual` builders based on runtime flags and cached residual distances; each `AdjacencyBuild` feeds `run_mis()` so the maximal independent set always reflects the exact scopes that survived pruning. The MIS solver accepts those forced masks from the grid builder (or radius filters) so dominated candidates never enter the random priority loop, and exposes its own iteration counter plus indicator arrays for logging.

### Additional optional controls

Other runtime-only features live alongside the terms above: enable JIT kernels with `COVERTREEX_ENABLE_NUMBA=1` (covering traversal, scope chunking, grid selection, and MIS), opt into residual scope caps via `COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH`, or mirror the benchmarking defaults by setting `COVERTREEX_ENABLE_SPARSE_TRAVERSAL`/`COVERTREEX_ENABLE_DIAGNOSTICS`. All knobs resolve through `covertreex/runtime/config.py::RuntimeConfig.from_env()`, so CLI wrappers and tests inherit the same behaviour without duplicating flag parsing.

### Next steps

- Capture before/after artefacts for the residual grid leader width using the new `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` knob so `grid_*` telemetry can prove parity with Euclidean runs.
- Promote the adaptive chunk-target heuristic into telemetry (`scope_chunk_pair_cap`) and write a short how-to on interpreting it so audits can link pair budgets to traversal drops.
- Sparse traversal now uses a bucketed level-ordering pass (implemented inside `covertreex/algo/_traverse_sparse_numba.py::_collect_scope_single_into`), so `traversal_semisort_ms` no longer scales quadratically with scope size; residual runs can build on this once the chunked streamer lands.
- Add regression tests that assert Gate‑1 lookup activation (and pruning) when `COVERTREEX_METRIC=residual_correlation`, ensuring the CLI knobs and docs stay in sync.

## Parallel Compressed Cover Tree — Conflict Graph Builder (dense + segmented)

`ScopeAdjacencyResult` in the implementation now also tracks chunk telemetry (`chunk_count`, `chunk_emitted`, `chunk_max_members`) so diagnostics reflect how `_chunk_ranges_from_indptr` splits oversized scopes.

```python
# covertreex/algo/_scope_numba.py (excerpt)
@nb.njit(cache=True, parallel=True)
def _expand_pairs_directed(
    values: np.ndarray,
    indptr: np.ndarray,
    kept_nodes: np.ndarray,
    offsets: np.ndarray,
    pairwise: np.ndarray,
    radii: np.ndarray,
):
    k = kept_nodes.size
    capacity = offsets[-1]
    sources = np.empty(capacity, dtype=I32)
    targets = np.empty(capacity, dtype=I32)
    used = np.zeros(k, dtype=I64)

    for idx in nb.prange(k):
        node = int(kept_nodes[idx])
        s = indptr[node]
        e = indptr[node + 1]
        c = e - s
        if c <= 1:
            continue
        base = offsets[idx]
        write = 0
        for a in range(c - 1):
            pa = values[s + a]
            ra = radii[pa]
            for b in range(a + 1, c):
                pb = values[s + b]
                rb = radii[pb]
                bound = ra if ra < rb else rb
                if pairwise[pa, pb] <= bound:
                    sources[base + write] = pa
                    targets[base + write] = pb
                    write += 1
                    sources[base + write] = pb
                    targets[base + write] = pa
                    write += 1
        used[idx] = write

    return sources, targets, used


@nb.njit(cache=True)
def _pairs_to_csr(
    sources: np.ndarray,
    targets: np.ndarray,
    offsets: np.ndarray,
    used: np.ndarray,
    batch_size: int,
):
    total_used = I64(0)
    for node in range(used.size):
        total_used += used[node]

    total_used_int = int(total_used)
    if total_used_int == 0:
        indptr = np.zeros(batch_size + 1, dtype=I64)
        indices = np.empty(0, dtype=I32)
        return indptr, indices, total_used_int

    trimmed_src = np.empty(total_used_int, dtype=I32)
    trimmed_dst = np.empty(total_used_int, dtype=I32)
    counts = np.zeros(batch_size, dtype=I64)
    cursor = I64(0)
    for node in range(used.size):
        count = used[node]
        if count == 0:
            continue
        start_in = offsets[node]
        for j in range(count):
            src = int(sources[start_in + j])
            tgt = targets[start_in + j]
            trimmed_src[cursor] = src
            trimmed_dst[cursor] = tgt
            counts[src] += 1
            cursor += 1
    indptr = np.empty(batch_size + 1, dtype=I64)
    acc = I64(0)
    indptr[0] = 0
    for i in range(batch_size):
        acc += counts[i]
        indptr[i + 1] = acc

    indices = np.empty(total_used_int, dtype=I32)
    heads = indptr[:-1].copy()
    for i in range(total_used_int):
        src = int(trimmed_src[i])
        pos = heads[src]
        indices[pos] = trimmed_dst[i]
        heads[src] = pos + 1

    return indptr, indices, total_used_int


def build_conflict_graph_numba_dense(
    scope_indptr: np.ndarray,
    scope_indices: np.ndarray,
    batch_size: int,
    *,
    segment_dedupe: bool = True,
    chunk_target: int = 0,
    pairwise: np.ndarray | None = None,
    radii: np.ndarray | None = None,
) -> ScopeAdjacencyResult:
    if scope_indices.size == 0:
        return ScopeAdjacencyResult(
            sources=np.empty(0, dtype=I32),
            targets=np.empty(0, dtype=I32),
            csr_indptr=np.zeros(batch_size + 1, dtype=I64),
            csr_indices=np.empty(0, dtype=I32),
            max_group_size=0,
            total_pairs=0,
            candidate_pairs=0,
            num_groups=0,
            num_unique_groups=0,
        )

    if pairwise is None or radii is None:
        raise ValueError(
            "pairwise distances and radii arrays are required for the Numba "
            "conflict-graph builder"
        )

    pairwise_arr = np.ascontiguousarray(np.asarray(pairwise, dtype=np.float64))
    radii_arr = np.asarray(radii, dtype=np.float64)

    total = scope_indices.size
    point_ids = _membership_point_ids_from_indptr(scope_indptr.astype(I64), total)
    num_nodes = int(scope_indices.max()) + 1 if scope_indices.size else 0
    indptr_nodes, node_members = _group_by_key_counting(
        scope_indices.astype(I32),
        point_ids,
        num_nodes,
    )
    _sort_segments_inplace(node_members, indptr_nodes)

    if segment_dedupe:
        hashes = _hash_segments(node_members, indptr_nodes)
        keep = _dedupe_segments_by_hash(node_members, indptr_nodes, hashes)
    else:
        keep = np.ones(indptr_nodes.size - 1, dtype=np.bool_)

    pair_counts, total_pairs, max_group_size = _compute_pair_counts(
        indptr_nodes, keep
    )
    num_groups = int(indptr_nodes.size - 1)
    num_unique_groups = int(np.count_nonzero(keep))
    if total_pairs == 0:
        return ScopeAdjacencyResult(
            sources=np.empty(0, dtype=I32),
            targets=np.empty(0, dtype=I32),
            csr_indptr=np.zeros(batch_size + 1, dtype=I64),
            csr_indices=np.empty(0, dtype=I32),
            max_group_size=int(max_group_size),
            total_pairs=0,
            candidate_pairs=0,
            num_groups=num_groups,
            num_unique_groups=num_unique_groups,
        )

    directed_total_pairs = int(total_pairs * 2)
    candidate_pairs = directed_total_pairs
    kept_nodes = np.nonzero(keep)[0].astype(I64)
    if kept_nodes.size == 0:
        return ScopeAdjacencyResult(
            sources=np.empty(0, dtype=I32),
            targets=np.empty(0, dtype=I32),
            csr_indptr=np.zeros(batch_size + 1, dtype=I64),
            csr_indices=np.empty(0, dtype=I32),
            max_group_size=int(max_group_size),
            total_pairs=0,
            candidate_pairs=candidate_pairs,
            num_groups=num_groups,
            num_unique_groups=num_unique_groups,
        )
    pair_counts_kept = pair_counts[keep]
    directed_counts = pair_counts_kept * 2
    offsets = _prefix_sum(directed_counts)
    capacity = int(offsets[-1])
    if capacity == 0:
        return ScopeAdjacencyResult(
            sources=np.empty(0, dtype=I32),
            targets=np.empty(0, dtype=I32),
            csr_indptr=np.zeros(batch_size + 1, dtype=I64),
            csr_indices=np.empty(0, dtype=I32),
            max_group_size=int(max_group_size),
            total_pairs=0,
            candidate_pairs=candidate_pairs,
            num_groups=num_groups,
            num_unique_groups=num_unique_groups,
        )

    sources, targets, used_counts = _expand_pairs_directed(
        node_members,
        indptr_nodes,
        kept_nodes,
        offsets,
        pairwise_arr,
        radii_arr,
    )
    csr_indptr, csr_indices, actual_pairs = _pairs_to_csr(
        sources,
        targets,
        offsets,
        used_counts,
        batch_size,
    )
    if actual_pairs == 0:
        return ScopeAdjacencyResult(
            sources=np.empty(0, dtype=I32),
            targets=np.empty(0, dtype=I32),
            csr_indptr=np.zeros(batch_size + 1, dtype=I64),
            csr_indices=np.empty(0, dtype=I32),
            max_group_size=int(max_group_size),
            total_pairs=0,
            candidate_pairs=candidate_pairs,
            num_groups=num_groups,
            num_unique_groups=num_unique_groups,
        )
    return ScopeAdjacencyResult(
        sources=np.empty(0, dtype=I32),
        targets=np.empty(0, dtype=I32),
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        max_group_size=int(max_group_size),
        total_pairs=actual_pairs,
        candidate_pairs=candidate_pairs,
        num_groups=num_groups,
        num_unique_groups=num_unique_groups,
    )
```
```

## Baseline Implementations

### Sequential compressed cover tree (reference)

```python
# covertreex/baseline.py (excerpt)

@dataclass
class BaselineCoverTree:
    coords: np.ndarray
    root: int
    children: Dict[int, List[int]] = field(default_factory=dict)
    parents: Dict[int, int] = field(default_factory=dict)
    levels: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> "BaselineCoverTree":
        coords = np.asarray(points, dtype=np.float64)
        if coords.ndim != 2:
            raise ValueError("Points must be a 2-D array-like structure")
        if coords.size == 0:
            raise ValueError("At least one point is required")

        tree = cls(coords=coords, root=0)
        tree.children[0] = [0]
        tree.parents[0] = -1
        tree.levels[0] = 0

        for idx in range(1, coords.shape[0]):
            tree._insert(idx)

        return tree

    def _insert(self, idx: int) -> None:
        current = self.root
        level = self.levels[current]

        while True:
            best_child = current
            best_dist = np.linalg.norm(self.coords[idx] - self.coords[current])
            for child in self.children.get(current, []):
                dist = np.linalg.norm(self.coords[idx] - self.coords[child])
                if dist < best_dist:
                    best_dist = dist
                    best_child = child

            if best_child == current:
                break

            current = best_child
            level = self.levels[current]

        self.parents[idx] = current
        self.levels[idx] = level
        self.children.setdefault(current, []).append(idx)
        self.children.setdefault(idx, [idx])

```

### GPBoost cover tree baseline (Numba)

```python
# covertreex/baseline.py (excerpt)

@njit(parallel=True, fastmath=True)
def _gpboost_euclidean_distances(coords: np.ndarray, i: int, cand_idx: np.ndarray) -> np.ndarray:
    d = coords.shape[1]
    m = cand_idx.shape[0]
    out = np.empty(m, dtype=np.float64)
    xi = coords[i, :]
    for t in prange(m):
        j = cand_idx[t]
        acc = 0.0
        for p in range(d):
            diff = coords[j, p] - xi[p]
            acc += diff * diff
        out[t] = np.sqrt(acc)
    return out


def _gpboost_cover_tree_knn(
    coords_mat: np.ndarray,
    *,
    start: int = 0,
    max_radius: float,
) -> Tuple[Dict[int, List[int]], int]:
    n_local = int(coords_mat.shape[0])
    cover_tree: Dict[int, List[int]] = {-1: [int(start)]}
    if n_local == 0:
        return cover_tree, 0

    R_max = max(1.0, float(max_radius))
    base = 2.0
    level = 0

    all_indices = list(range(1, n_local))
    covert_points_old: Dict[int, List[int]] = {0: all_indices}

    while (len(cover_tree) - 1) != n_local:
        level += 1
        if base == 2.0:
            R_l = math.ldexp(R_max, -level)
        else:
            R_l = R_max / (base ** level)
        covert_points: Dict[int, List[int]] = {}

        for key, cov_old in list(covert_points_old.items()):
            cov_list = list(cov_old)
            not_all_covered = len(cov_list) > 0

            cover_tree[key + start] = [key + start]

            while not_all_covered:
                sample_ind = cov_list[0]
                cover_tree[key + start].append(sample_ind + start)

                up = [j for j in cov_list if j > sample_ind]

                if up:
                    dists = _gpboost_euclidean_distances(coords_mat, sample_ind, np.asarray(up, dtype=np.int64))
                else:
                    dists = np.empty(0, dtype=np.float64)

                covered = {up[idx] for idx, value in enumerate(dists) if value <= R_l}

                cov_list = [j for j in cov_list[1:] if j not in covered]
                not_all_covered = len(cov_list) > 0

                if covered:
                    covert_points.setdefault(sample_ind, []).extend(sorted(covered))

        if not covert_points:
            parent_key = start
            if parent_key not in cover_tree:
                cover_tree[parent_key] = [parent_key]
            existing = {node for node in cover_tree if node >= start}
            for idx in range(n_local):
                node_id = idx + start
                if node_id not in existing:
                    cover_tree.setdefault(node_id, [node_id])
                    if node_id not in cover_tree[parent_key]:
                        cover_tree[parent_key].append(node_id)
            break

        covert_points_old = covert_points

    return cover_tree, level


def _gpboost_find_knn(
    *,
    query_index: int,
    k: int,
    levels: int,
    coords: np.ndarray,
    cover_tree: Dict[int, List[int]],
) -> Tuple[List[int], List[float]]:
    root = cover_tree[-1][0]
    Q: List[int] = []
    Q_dist: List[float] = []
    diff_rev: List[int] = [root]

    max_dist = 1.0
    dist_k_Q_cor = max_dist
    k_scaled = int(k)
    Q_before_size = 1
    base = 2.0

    for ii in range(1, int(levels)):
        diff_rev_interim: List[int] = []
        if ii == 1:
            Q.append(root)
            diff_rev_interim.append(root)

        for j in diff_rev:
            children = cover_tree.get(j, [])
            for jj in children:
                if jj != j:
                    Q.append(jj)
                    diff_rev_interim.append(jj)

        diff_rev = []
        early_stop = (len(diff_rev_interim) == 0) or (ii == (levels - 1))

        if diff_rev_interim:
            dvec = _gpboost_euclidean_distances(coords, query_index, np.asarray(diff_rev_interim, dtype=np.int64))
            Q_dist.extend(dvec.tolist())

        if ii > 1:
            if len(Q_dist) < k_scaled:
                dist_k_Q_cor = max(Q_dist) if Q_dist else max_dist
            else:
                arr = np.asarray(Q_dist, dtype=np.float64)
                dist_k_Q_cor = float(np.partition(arr, k_scaled - 1)[k_scaled - 1])
            dist_k_Q_cor += 1.0 / (base ** (ii - 1))

        if dist_k_Q_cor >= max_dist:
            if not early_stop:
                diff_rev = diff_rev_interim.copy()
                if ii == 1 and diff_rev:
                    diff_rev = diff_rev[1:]
        else:
            Q_interim: List[int] = []
            Q_dist_interim: List[float] = []
            count = 0
            for xi, yi in zip(Q_dist, Q):
                if xi <= dist_k_Q_cor:
                    Q_dist_interim.append(xi)
                    Q_interim.append(yi)
                    if count >= Q_before_size:
                        diff_rev.append(yi)
                count += 1
            Q = Q_interim
            Q_dist = Q_dist_interim

        Q_before_size = len(Q)
        if early_stop:
            break

    neighbors_i: List[int] = [-1] * k
    nn_dist: List[float] = [float("inf")] * k
    return neighbors_i, nn_dist

```
```

```python
# covertreex/algo/conflict_graph.py (excerpt)
runtime = cx_config.runtime_config()
scope_group_start = time.perf_counter()
scope_indptr = backend.asarray(traversal.scope_indptr, dtype=backend.default_int)
scope_indices = backend.asarray(traversal.scope_indices, dtype=backend.default_int)
enable_numba = runtime.enable_numba and NUMBA_SCOPE_AVAILABLE
need_point_ids = bool(scope_indices.size) and runtime.conflict_graph_impl == "segmented"
if need_point_ids:
    counts = scope_indptr[1:] - scope_indptr[:-1]
    point_ids = xp.repeat(
        xp.arange(batch_size, dtype=backend.default_int),
        counts,
    )
    _block_until_ready(point_ids)
else:
    point_ids = xp.zeros((0,), dtype=backend.default_int)
scope_group_seconds = time.perf_counter() - scope_group_start

...

adjacency_build = _build_dense_adjacency(
    backend=backend,
    batch_size=batch_size,
    scope_indptr=scope_indptr,
    scope_indices=scope_indices,
)
```
