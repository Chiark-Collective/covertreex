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
uv run python -m benchmarks.queries \
  --dimension 8 --tree-points 2048 \
  --batch-size 128 --queries 512 --k 8 \
  --seed 42 --baseline gpboost
```

_Set `COVERTREEX_ENABLE_DIAGNOSTICS=1` to collect the instrumentation counters (adds ~7 ms to the build in this configuration)._

### Scaling Snapshot — CPU builds (diagnostics on)

| Workload (tree pts / queries / k) | PCCT Build (s) | PCCT Query (s) | PCCT q/s | Sequential Build (s) | Sequential q/s | GPBoost Build (s) | GPBoost q/s | External Build (s) | External q/s |
|-----------------------------------|----------------|----------------|----------|----------------------|----------------|-------------------|-------------|--------------------|---------------|
| 8 192 / 1 024 / 16                | 4.03           | 0.934          | 1 096    | 33.65               | 5 327         | 0.569             | 306         | 14.14              | 122           |
| 32 768 / 2 048 / 16               | 41.01          | 0.419          | 4 889    | —                   | —             | 2.80              | 98          | —                  | —             |

The 32 768-point run currently logs PCCT and the GPBoost baseline; sequential/external baselines are still pending optimisations to keep runtime manageable at that scale.

_Command (8 192 row):_
```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
UV_CACHE_DIR=$PWD/.uv-cache \
uv run python -m benchmarks.queries \
  --dimension 8 --tree-points 8192 \
  --batch-size 256 --queries 1024 --k 16 \
  --seed 12345 --baseline gpboost

# GPBoost-only 32k run
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
UV_CACHE_DIR=$PWD/.uv-cache \
uv run python -m benchmarks.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 256 --queries 2048 --k 16 \
  --seed 1337 --baseline gpboost
```

To capture warm-up versus steady-state timings for plotting, append `--csv-output runtime_breakdown_metrics.csv` when running `benchmarks.runtime_breakdown`.

## Current Observations & Hypotheses

### Euclidean metric (NumPy backend)

- Fresh 32 768-point benchmark on 2025-11-06 lands at **41.01 s build / 0.42 s query (4 889 q/s)** for PCCT with NumPy+Numba, while the GPBoost baseline stays at 2.80 s build / 20.94 s query (98 q/s).
- Dominated batches still spend ~200 ms in `traversal_pairwise_ms` and ~300 ms in `traversal_assemble_ms` at this scale; conflict-graph construction remains ~59 ms per dominated batch, so traversal continues to be the limiting factor for the Euclidean path.
- The Numba conflict-graph builder (segment dedupe + directed pair expansion) holds steady at 3–4 ms per dominated batch with `conflict_scope_group_ms < 0.01 ms` and `conflict_adj_scatter_ms ≈ 1.0–1.3 ms`.
- MIS is effectively free (`mis_ms ≤ 0.2 ms`) once the initial JIT completes. Remaining variation comes from the first-build warm-up.
- Running strictly on the CPU (NumPy backend, diagnostics optional) keeps comparisons against the GPBoost baseline reproducible while avoiding the slower sequential/external references that we now skip by default.
- GPBoost’s cover tree baseline is sequential and Euclidean-only: it inserts points one at a time, mutates its structure in-place, and never builds a conflict graph or MIS. PCCT’s batch design (exact Euclidean checks + MIS + immutable journals) is inherently heavier; the optimisation path is to trim traversal/mask overhead, not to regress to per-point mutation.
- Sparse traversal now exposes `tile_seconds` along with `scope_chunk_segments`, `scope_chunk_emitted`, and `scope_chunk_max_members` inside `TraversalTimings` whenever the CSR chunking path is active. This makes it obvious when traversal chunking is actually engaged before we enter the conflict-graph builder.
- Chunked traversal is opt-in: the default runtime keeps `COVERTREEX_SCOPE_CHUNK_TARGET=0`, so dense traversal remains the baseline until the chunked pipeline is tuned. Set the env var (or pass `chunk_target` through `RuntimeConfig`) to activate segmentation for experiments.

### Residual-correlation metric (synthetic RBF caches, 2025-11-06)

- First end-to-end synthetic run (32 768 tree points / 2 048 queries / k=16, seed 42) with `--metric residual` reports **290.64 s build / 156.51 s query (13.1 q/s)** using the NumPy backend and streamed residual caches.
- Average dominated batch wall time is 4.33 s with the following split: `conflict_graph_ms` 72.5 %, `traversal_ms` 24.7 %, `mis_ms` 2.4 %. Despite streaming, each batch still materialises the full 512×511 edge set (261 632 pairs).
- Conflict graph time is consumed almost entirely by residual adjacency filtering (`conflict_adjacency_ms` ≈ 3.12 s, 99.5 % of the conflict phase). The per-source residual distance recomputation costs ~0.59 s inside that filter even though the dense residual matrix is already in memory.
- Traversal remains costly: chunked residual pairwise work averages 0.39 s (36 % of traversal), with semisort (~0.36 s) and CSR assembly (~0.32 s) indicating we are still emitting near-dense scopes. Residual scope groups average 16 384 entries with a single unique parent chain, so pruning is effectively inactive.
- Residual telemetry shows sustained RSS deltas of ~288 MB per dominated batch (peaking ~703 MB) alongside repeated 64–132 MB host→CPU kernel transfers (`conflict_scope_bytes_d2h`).
- Priority fixes:
  1. Reuse the precomputed `residual_pairwise` matrix inside `_build_dense_adjacency` / filter to eliminate redundant kernel-provider calls and their 0.6 s per-batch cost.
  2. Tighten `_collect_residual_scopes_streaming` (radius guards / partial dot-product bounds) to cap scope sizes and reduce the 3.1 s adjacency workload.
  3. Land the partial dot-product early-exit in `compute_residual_distances_with_radius` so `traversal_pairwise_ms` stops drifting toward 0.7–0.8 s on the later batches.
- The scope adjacency builder now respects `_chunk_ranges_from_indptr(scope_indptr, scope_chunk_target)`: each chunk emits its CSR page independently, and `ConflictGraphTimings` reports `scope_chunk_segments`, `scope_chunk_emitted`, and `scope_chunk_max_members` so large residual batches show when backpressure or segmentation kicks in. This trimmed the scratch RSS spikes during dominated batches and exposes whether chunk caps are actually being exercised.

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
python -m benchmarks.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 2048 --k 16 \
  --seed 42 --residual-inducing 512 \
  --residual-lengthscale 1.0 --residual-variance 1.0
```

**Telemetry snapshot (dominated batches, averages over 63 batches)**

| Metric | Mean | Share of phase |
|--------|------|----------------|
| Wall time per dominated batch | 0.90 s | — |
| `traversal_ms` | 0.79 s | 87.4 % of wall |
| • `traversal_pairwise_ms` | 0.43 s | 54.4 % of traversal |
| • `traversal_semisort_ms` | 0.027 s | 3.4 % of traversal |
| • `traversal_assemble_ms` | 0.33 s | 42.2 % of traversal |
| `conflict_graph_ms` | 0.092 s | 10.2 % of wall |
| • `conflict_adjacency_ms` | 0.075 s | 81.4 % of conflict graph |
| • `conflict_adj_filter_ms` | ~0.000 s | ≈0 % of conflict graph |
| `mis_ms` | 0.0002 s (0.20 ms) | <0.1 % of wall |
| Residual scope size (`conflict_scope_groups`) | 16 384 | — |
| Host transfer (`conflict_scope_bytes_d2h`) | 67 MB | — |
| RSS delta per batch | 288 MB | — |

**Status & follow-up**
- Reusing the dense residual matrix for CSR filtering eliminates the 0.6 s per-batch recomputation; `conflict_adj_filter_ms` now vanishes and conflict graph time drops to ~0.09 s.
- Early-reject bounds trim residual pairwise work (now ~0.43 s of the 0.79 s traversal block), but scopes are still near-dense (average `conflict_scope_groups` ≈ 16 k). Further pruning must focus on radius tightening or scope segmentation.
- Memory churn is lower but still sizable (≈ 275 MB RSS delta per dominated batch). Explore segment caps or tile-based adjacency to reduce temporary buffers once the above pruning lands.

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

def compute_residual_distances_with_radius(
    backend: ResidualCorrHostData,
    query_index: int,
    chunk_indices: np.ndarray,
    kernel_row: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    distances, mask = compute_distance_chunk(
        v_query=backend.v_matrix[query_index],
        v_chunk=backend.v_matrix[chunk_indices],
        kernel_chunk=kernel_row,
        p_i=float(backend.p_diag[query_index]),
        p_chunk=backend.p_diag[chunk_indices],
        norm_query=float(backend.v_norm_sq[query_index]),
        norm_chunk=backend.v_norm_sq[chunk_indices],
        radius=float(radius),
        eps=_EPS,
    )
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

### Persistence journal (NumPy backend)

- With `COVERTREEX_BACKEND=numpy` and `COVERTREEX_ENABLE_NUMBA=1`, batch inserts now route through a Numba “journal” kernel that clones the parent/child/next arrays once per batch and applies updates in a single sweep. When the flag is disabled or a non-NumPy backend is selected, the legacy SliceUpdate path is used automatically.
- The journal path is metric-agnostic: it operates on the structural arrays only, so custom metrics (e.g. residual correlation) remain supported without additional guards.
- Scratch buffers for head/next chains are pooled across batches, eliminating the small-but-frequent NumPy allocations that previously showed up as RSS spikes in large builds. Enable diagnostics to monitor pool growth via the existing logging hooks.
- Rerun `benchmarks.runtime_breakdown` before/after enabling the journal to record wall-clock deltas for auditors; the 32 768-point configuration is the most illustrative workload.

### Warm-up & JIT compilation

- The first PCCT build in any fresh Python session pays the full Numba compilation cost plus the actual 32 k build (≈40 s in the latest 8-dim profile). Subsequent builds reuse the cached kernels and drop to the steady-state numbers shown above (sub-second per batch).
- To hide the one-time hit in production pipelines, kick off a tiny “warm-up” batch during process start-up—for example, build a 64-point tree with the same runtime flags before ingesting real data. This compiles all hot kernels without noticeably touching RSS or wall time.
- To quantify the difference in practice, run `benchmarks.runtime_breakdown` with `--runs 5` (or similar). The CSV now includes a `run` column, letting you compare the first-build warm-up against subsequent steady-state runs directly.
- Quick sanity runs still default to `COVERTREEX_METRIC=euclidean`. Use `benchmarks.queries --metric residual` (which auto-builds synthetic caches) when you need the residual path; the script now forces JAX onto the CPU so the MIS helper cannot grab GPU memory.

### Threading defaults

- On import we clamp `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1 so NumPy/BLAS libraries do not oversubscribe the worker pool. We also set `NUMBA_THREADING_LAYER=tbb` and default `NUMBA_NUM_THREADS` to the host CPU count (override any of these via environment variables before importing `covertreex`).
- For heavier boxes increase `NUMBA_NUM_THREADS` and, if needed, batch size (`--batch-size`) and `COVERTREEX_SCOPE_CHUNK_TARGET` to keep each worker busy. The TBB threading layer provides work-stealing, so threads redistribute skewed batches automatically once these pools are configured.

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
