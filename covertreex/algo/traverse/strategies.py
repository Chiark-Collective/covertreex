from __future__ import annotations

from typing import Any, Dict, Tuple

import time
import numpy as np

from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.logging import get_logger
from covertreex.metrics import residual as residual_metrics
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    ResidualGateTelemetry,
    compute_residual_distances_from_kernel,
    compute_residual_distances_with_radius,
    decode_indices,
    get_residual_backend,
)
from covertreex.metrics.residual.scope_caps import get_scope_cap_table
from .._traverse_numba import NUMBA_TRAVERSAL_AVAILABLE, build_scopes_numba
from .._traverse_sparse_numba import (
    NUMBA_SPARSE_TRAVERSAL_AVAILABLE,
    collect_sparse_scopes,
    collect_sparse_scopes_csr,
)
from covertreex.queries._knn_numba import (
    knn_numba,
    materialise_tree_view_cached,
)
from .base import (
    ResidualTraversalCache,
    TraversalResult,
    TraversalTimings,
    TraversalStrategy,
    block_until_ready,
    collect_distances,
    empty_result,
)

LOGGER = get_logger("algo.traverse")
_RESIDUAL_SCOPE_EPS = 1e-9
_RESIDUAL_SCOPE_DEFAULT_LIMIT = 16_384


class _EuclideanDenseTraversal(TraversalStrategy):
    def collect(
        self,
        tree: PCCTree,
        batch: Any,
        *,
        backend: TreeBackend,
        runtime: Any,
    ) -> TraversalResult:
        return _collect_euclidean_dense(tree, batch, backend=backend, runtime=runtime)


class _EuclideanSparseTraversal(TraversalStrategy):
    def collect(
        self,
        tree: PCCTree,
        batch: Any,
        *,
        backend: TreeBackend,
        runtime: Any,
    ) -> TraversalResult:
        return _collect_euclidean_sparse(tree, batch, backend=backend, runtime=runtime)


class _ResidualTraversal(TraversalStrategy):
    def collect(
        self,
        tree: PCCTree,
        batch: Any,
        *,
        backend: TreeBackend,
        runtime: Any,
    ) -> TraversalResult:
        return _collect_residual(tree, batch, backend=backend, runtime=runtime)


def _collect_euclidean_dense(
    tree: PCCTree,
    batch: Any,
    *,
    backend: TreeBackend,
    runtime: Any,
) -> TraversalResult:
    xp = backend.xp
    batch_size = int(batch.shape[0]) if batch.size else 0

    start = time.perf_counter()
    distances = collect_distances(tree, batch, backend)
    block_until_ready(distances)
    pairwise_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parents = xp.argmin(distances, axis=1).astype(backend.default_int)
    levels = tree.top_levels[parents]

    base_radius = xp.power(2.0, levels.astype(backend.default_float) + 1.0)
    si_values = tree.si_cache[parents]
    radius = xp.maximum(base_radius, si_values)
    node_indices = xp.arange(tree.num_points, dtype=backend.default_int)
    parent_mask = node_indices[None, :] == parents[:, None]
    within_radius = distances <= radius[:, None]
    mask = xp.logical_or(within_radius, parent_mask)
    block_until_ready(mask)
    mask_seconds = time.perf_counter() - start

    parents_np = np.asarray(parents, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)

    mask_np = np.asarray(backend.to_numpy(mask), dtype=bool)
    if mask_np.ndim != 2:
        mask_np = np.reshape(mask_np, (batch_size, tree.num_points))

    use_numba = runtime.enable_numba and NUMBA_TRAVERSAL_AVAILABLE

    if use_numba:
        numba_start = time.perf_counter()
        scope_indptr_np, scope_indices_np = build_scopes_numba(
            mask_np,
            parents_np,
            next_cache_np,
            top_levels_np,
        )
        numba_end = time.perf_counter()
        semisort_seconds = numba_end - start
        chain_seconds = numba_end - numba_start
        nonzero_seconds = 0.0
        sort_seconds = 0.0
        assemble_seconds = 0.0
    else:
        unique_parents = {int(p) for p in parents_np if int(p) >= 0}
        chain_update_start = time.perf_counter()
        chain_map = {
            parent: _collect_next_chain(tree, parent, next_cache=next_cache_np)
            for parent in unique_parents
        }
        if chain_map:
            for idx, parent in enumerate(parents_np):
                if parent >= 0:
                    chain = chain_map.get(int(parent))
                    if chain:
                        mask_np[idx, list(chain)] = True
        chain_seconds = time.perf_counter() - chain_update_start

        nonzero_start = time.perf_counter()
        row_ids, col_ids = np.nonzero(mask_np)
        nonzero_seconds = time.perf_counter() - nonzero_start

        sort_start = time.perf_counter()
        if row_ids.size:
            level_vals = top_levels_np[col_ids]
            order = np.lexsort((col_ids, -level_vals, row_ids))
            row_sorted = row_ids[order]
            col_sorted = col_ids[order]
            counts = np.bincount(row_sorted, minlength=batch_size)
            scope_indptr_np = np.concatenate(
                ([0], np.cumsum(counts, dtype=np.int64))
            )
            scope_indices_np = col_sorted.astype(np.int64, copy=False)
        else:
            scope_indptr_np = np.zeros(batch_size + 1, dtype=np.int64)
            scope_indices_np = np.zeros(0, dtype=np.int64)
        sort_seconds = time.perf_counter() - sort_start

        semisort_seconds = time.perf_counter() - start
        assemble_seconds = 0.0

    assemble_start = time.perf_counter()
    conflict_scopes = tuple(
        tuple(int(x) for x in scope_indices_np[scope_indptr_np[i] : scope_indptr_np[i + 1]])
        for i in range(batch_size)
    )
    scope_indptr_arr = backend.asarray(scope_indptr_np, dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices_np, dtype=backend.default_int)
    assemble_seconds += time.perf_counter() - assemble_start

    LOGGER.debug(
        "Traversal assigned parents %s at levels %s",
        backend.to_numpy(parents),
        backend.to_numpy(levels),
    )

    return TraversalResult(
        parents=backend.device_put(parents),
        levels=backend.device_put(levels),
        conflict_scopes=conflict_scopes,
        scope_indptr=scope_indptr_arr,
        scope_indices=scope_indices_arr,
        timings=TraversalTimings(
            pairwise_seconds=pairwise_seconds,
            mask_seconds=mask_seconds,
            semisort_seconds=semisort_seconds,
            chain_seconds=chain_seconds,
            nonzero_seconds=nonzero_seconds,
            sort_seconds=sort_seconds,
            assemble_seconds=assemble_seconds,
        ),
    )


def _collect_next_chain(
    tree: PCCTree,
    start: int,
    *,
    next_cache: np.ndarray | None = None,
) -> Tuple[int, ...]:
    cache = next_cache if next_cache is not None else np.asarray(tree.next_cache, dtype=np.int64)
    num_points = cache.shape[0]
    if start < 0 or start >= num_points:
        return ()
    chain: list[int] = []
    visited: set[int] = set()
    current = start
    while 0 <= current < num_points and current not in visited:
        chain.append(current)
        visited.add(current)
        if cache.size == 0:
            break
        nxt = int(cache[current])
        if nxt < 0:
            break
        current = nxt
    return tuple(chain)


def _residual_find_parents(
    *,
    host_backend: ResidualCorrHostData,
    query_indices: np.ndarray,
    tree_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(query_indices.shape[0])
    best_dist = np.full(batch_size, np.inf, dtype=np.float64)
    best_idx = np.full(batch_size, -1, dtype=np.int64)

    chunk = int(host_backend.chunk_size or 512)
    total = int(tree_indices.shape[0])
    query_arr = np.asarray(query_indices, dtype=np.int64)

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        chunk_ids = tree_indices[start:stop]
        kernel_block = host_backend.kernel_provider(query_arr, chunk_ids)
        distances_block = compute_residual_distances_from_kernel(
            host_backend,
            query_arr,
            chunk_ids,
            kernel_block,
        )
        row_min_idx = np.argmin(distances_block, axis=1)
        row_min_val = distances_block[np.arange(batch_size), row_min_idx]
        improved = row_min_val < best_dist
        if np.any(improved):
            best_dist[improved] = row_min_val[improved]
            best_idx[improved] = chunk_ids[row_min_idx[improved]]

    return best_idx, best_dist


def _collect_residual_scopes_streaming(
    *,
    tree: PCCTree,
    host_backend: ResidualCorrHostData,
    query_indices: np.ndarray,
    tree_indices: np.ndarray,
    parent_positions: np.ndarray,
    radii: np.ndarray,
    scope_limit: int | None = None,
    scan_cap: int | None = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[Tuple[int, ...], ...],
    int,
    int,
    ResidualGateTelemetry,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    batch_size = int(query_indices.shape[0])
    total_points = int(tree_indices.shape[0])
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    tree_id_to_pos = {int(tree_indices[i]): int(i) for i in range(total_points)}

    scopes: list[np.ndarray] = []
    scope_indptr = np.zeros(batch_size + 1, dtype=np.int64)
    chunk = int(host_backend.chunk_size or 512)
    trimmed_scopes = 0
    max_scope_members = 0
    max_scope_after = 0
    limit = scope_limit if scope_limit and scope_limit > 0 else None

    observed_radii = np.zeros(batch_size, dtype=np.float64)
    saturation_flags = np.zeros(batch_size, dtype=np.uint8)
    chunk_iterations = np.zeros(batch_size, dtype=np.int64)
    chunk_points = np.zeros(batch_size, dtype=np.int64)
    dedupe_hits = np.zeros(batch_size, dtype=np.int64)
    flags_length = max(total_points, 1)
    collected_flags = np.zeros(flags_length, dtype=np.uint8)
    scan_cap_value = scan_cap if scan_cap and scan_cap > 0 else None
    cache_limit = scope_limit if scope_limit and scope_limit > 0 else _RESIDUAL_SCOPE_DEFAULT_LIMIT
    level_scope_cache: Dict[int, np.ndarray] = {}
    cache_hits_total = 0
    cache_prefetch_total = 0

    gate_snapshot = host_backend.gate_stats.snapshot()

    for qi in range(batch_size):
        parent_pos = int(parent_positions[qi])
        if parent_pos < 0:
            scopes.append(np.empty(0, dtype=np.int64))
            scope_indptr[qi + 1] = scope_indptr[qi]
            continue

        parent_level = int(top_levels_np[parent_pos]) if 0 <= parent_pos < top_levels_np.shape[0] else -1
        query_id = int(query_indices[qi])
        radius = float(radii[qi])
        collected_positions: list[int] = []
        saturated = False
        fully_scanned = True
        cap_reached = False

        query_idx_arr = np.asarray([query_id], dtype=np.int64)
        cached_positions = (
            level_scope_cache.get(parent_level)
            if parent_level >= 0
            else None
        )
        if cached_positions is not None and cached_positions.size:
            valid_cached = cached_positions[
                (cached_positions >= 0) & (cached_positions < total_points)
            ]
            if valid_cached.size:
                cache_prefetch_total += int(valid_cached.size)
                cached_ids = tree_indices[valid_cached]
                cache_kernel = host_backend.kernel_provider(
                    query_idx_arr,
                    cached_ids,
                )[0]
                cache_distances, cache_mask = compute_residual_distances_with_radius(
                    backend=host_backend,
                    query_index=query_id,
                    chunk_indices=cached_ids,
                    kernel_row=cache_kernel,
                    radius=radius,
                )
                cache_include = np.nonzero(cache_mask)[0]
                cache_hits_total += int(cache_include.size)
                for idx_local in cache_include:
                    pos = int(valid_cached[idx_local])
                    if collected_flags[pos]:
                        dedupe_hits[qi] += 1
                        continue
                    collected_flags[pos] = 1
                    collected_positions.append(pos)
                    if limit and len(collected_positions) >= limit:
                        saturated = True
                        fully_scanned = False
                        break
                if saturated:
                    cap_reached = True
        if not saturated:
            for start in range(0, total_points, chunk):
                if scan_cap_value and chunk_points[qi] >= scan_cap_value:
                    saturated = True
                    fully_scanned = False
                    break
                stop = min(start + chunk, total_points)
                chunk_ids = tree_indices[start:stop]
                if chunk_ids.size == 0:
                    continue
                chunk_iterations[qi] += 1
                chunk_points[qi] += chunk_ids.size
                if scan_cap_value and chunk_points[qi] >= scan_cap_value:
                    cap_reached = True
                kernel_block = host_backend.kernel_provider(
                    query_idx_arr,
                    chunk_ids,
                )[0]
                distances, mask = compute_residual_distances_with_radius(
                    backend=host_backend,
                    query_index=query_id,
                    chunk_indices=chunk_ids,
                    kernel_row=kernel_block,
                    radius=radius,
                )
                if mask.size == 0:
                    continue
                include_idx = np.nonzero(mask)[0]
                if include_idx.size == 0:
                    continue
                max_chunk = float(np.max(distances[include_idx]))
                if max_chunk > observed_radii[qi]:
                    observed_radii[qi] = max_chunk
                for idx_local in include_idx:
                    cid = chunk_ids[idx_local]
                    pos = tree_id_to_pos.get(int(cid))
                    if pos is None or pos < 0:
                        continue
                    if collected_flags[pos]:
                        dedupe_hits[qi] += 1
                        continue
                    collected_flags[pos] = 1
                    collected_positions.append(int(pos))
                    if limit and len(collected_positions) >= limit:
                        saturated = True
                        break
                if saturated or cap_reached:
                    saturated = True
                    fully_scanned = False
                    break
        # `saturated` indicates we bailed early; keep track of full scans so
        # telemetry can distinguish true caps from naturally small scopes.

        def _ensure_collected(idx: int) -> None:
            if idx < 0 or idx >= total_points:
                return
            if not collected_flags[idx]:
                collected_flags[idx] = 1
                collected_positions.append(idx)

        _ensure_collected(parent_pos)
        chain = _collect_next_chain(tree, parent_pos, next_cache=next_cache_np)
        for node in chain:
            _ensure_collected(int(node))

        if collected_positions:
            scope_vec = np.asarray(collected_positions, dtype=np.int64)
            scope_vec.sort()
            order = np.lexsort((scope_vec, -top_levels_np[scope_vec]))
            scope_vec = scope_vec[order]
        else:
            scope_vec = np.empty(0, dtype=np.int64)

        original_size = scope_vec.size
        max_scope_members = max(max_scope_members, original_size)
        trimmed_flag = False

        if limit and original_size > limit:
            trimmed_scopes += 1
            trimmed_flag = True
            scope_vec = _trim_residual_scope_vector(
                scope_vec,
                parent_pos,
                top_levels_np,
                limit,
            )
        elif limit and saturated and not fully_scanned:
            trimmed_scopes += 1
            trimmed_flag = True

        if trimmed_flag:
            saturation_flags[qi] = 1
        elif saturated:
            saturation_flags[qi] = 1

        max_scope_after = max(max_scope_after, scope_vec.size)

        scopes.append(scope_vec)
        scope_indptr[qi + 1] = scope_indptr[qi] + scope_vec.size

        if parent_level >= 0 and scope_vec.size:
            cache_slice = scope_vec[: min(cache_limit, scope_vec.size)].copy()
            level_scope_cache[parent_level] = cache_slice

        for pos in collected_positions:
            if 0 <= pos < total_points:
                collected_flags[pos] = 0

    total_scope = int(scope_indptr[-1])
    scope_indices = np.empty(total_scope, dtype=np.int64)
    cursor = 0
    conflict_scopes: list[Tuple[int, ...]] = []
    for scope_vec in scopes:
        if scope_vec.size:
            scope_indices[cursor : cursor + scope_vec.size] = scope_vec
            conflict_scopes.append(tuple(int(x) for x in scope_vec.tolist()))
            cursor += scope_vec.size
        else:
            conflict_scopes.append(())

    gate_delta = host_backend.gate_stats.delta(gate_snapshot)
    return (
        scope_indptr,
        scope_indices,
        tuple(conflict_scopes),
        trimmed_scopes,
        max_scope_after,
        gate_delta,
        observed_radii,
        saturation_flags,
        chunk_iterations,
        chunk_points,
        dedupe_hits,
        cache_hits_total,
        cache_prefetch_total,
    )


def _trim_residual_scope_vector(
    scope_vec: np.ndarray,
    parent_pos: int,
    top_levels_np: np.ndarray,
    scope_limit: int,
) -> np.ndarray:
    trimmed = scope_vec[:scope_limit].copy()
    if parent_pos not in trimmed:
        trimmed = np.concatenate([trimmed, np.asarray([parent_pos], dtype=np.int64)])
    # Remove duplicates while preserving intent to keep highest-level nodes first.
    trimmed = np.unique(trimmed)
    order = np.lexsort((trimmed, -top_levels_np[trimmed]))
    trimmed = trimmed[order]
    if trimmed.size > scope_limit:
        trimmed = trimmed[:scope_limit]
    return trimmed


def _collect_euclidean_sparse(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
    runtime: Any,
) -> TraversalResult:
    queries_np = np.asarray(backend.to_numpy(batch_points), dtype=np.float64)
    if queries_np.ndim == 1:
        queries_np = queries_np[None, :]

    batch_size = int(queries_np.shape[0])
    if batch_size == 0:
        return empty_result(backend, 0)

    view = materialise_tree_view_cached(tree)

    parent_start = time.perf_counter()
    indices, _distances = knn_numba(view, queries_np, k=1, return_distances=True)
    pairwise_seconds = time.perf_counter() - parent_start

    parents_np = np.asarray(indices, dtype=np.int64).reshape(batch_size)
    top_levels_np = view.top_levels
    levels_np = np.full(batch_size, -1, dtype=np.int64)
    valid_mask = parents_np >= 0
    if np.any(valid_mask):
        levels_np[valid_mask] = top_levels_np[parents_np[valid_mask]]

    base_radii = np.zeros(batch_size, dtype=np.float64)
    base_radii[valid_mask] = np.power(2.0, levels_np[valid_mask].astype(np.float64) + 1.0)

    si_values = np.zeros(batch_size, dtype=np.float64)
    if view.si_cache.size:
        within_si = np.logical_and(valid_mask, parents_np < view.si_cache.shape[0])
        si_values[within_si] = view.si_cache[parents_np[within_si]]

    radii_np = np.maximum(base_radii, si_values)

    chunk_target = int(runtime.scope_chunk_target)

    scope_start = time.perf_counter()
    if chunk_target > 0:
        scope_indptr_np, scope_indices_np, chunk_segments, chunk_emitted, chunk_max_members = collect_sparse_scopes_csr(
            view,
            queries_np,
            parents_np,
            radii_np,
            chunk_target,
        )
        scope_seconds = time.perf_counter() - scope_start
        conflict_scopes = []
        for idx in range(batch_size):
            start = int(scope_indptr_np[idx])
            end = int(scope_indptr_np[idx + 1])
            if start == end:
                conflict_scopes.append(())
            else:
                conflict_scopes.append(
                    tuple(int(x) for x in scope_indices_np[start:end])
                )
        conflict_scopes_tuple = tuple(conflict_scopes)
        semisort_seconds = 0.0
        tile_seconds = scope_seconds
    else:
        scopes = collect_sparse_scopes(view, queries_np, parents_np, radii_np)
        scope_seconds = time.perf_counter() - scope_start

        scope_lengths = [scope.shape[0] for scope in scopes]
        scope_indptr_np = np.empty(batch_size + 1, dtype=np.int64)
        scope_indptr_np[0] = 0
        for idx in range(batch_size):
            scope_indptr_np[idx + 1] = scope_indptr_np[idx] + scope_lengths[idx]
        total_scope = int(scope_indptr_np[-1])
        scope_indices_np = np.empty(total_scope, dtype=np.int64)
        cursor = 0
        conflict_scopes = []
        for scope in scopes:
            count = scope.shape[0]
            if count:
                scope_indices_np[cursor : cursor + count] = scope
                conflict_scopes.append(tuple(int(x) for x in scope.tolist()))
            else:
                conflict_scopes.append(())
            cursor += count

        conflict_scopes_tuple = tuple(conflict_scopes)
        chunk_segments = batch_size
        chunk_emitted = 0
        chunk_max_members = int(max(scope_lengths)) if scope_lengths else 0
        semisort_seconds = scope_seconds
        tile_seconds = 0.0

    if chunk_target > 0:
        # ensure chunk metrics have sensible defaults when scopes are empty
        if chunk_max_members == 0 and scope_indices_np.size:
            chunk_max_members = int(np.max(scope_indptr_np[1:] - scope_indptr_np[:-1]))

    parents_arr = backend.asarray(parents_np.astype(np.int64), dtype=backend.default_int)
    levels_arr = backend.asarray(levels_np.astype(np.int64), dtype=backend.default_int)
    scope_indptr_arr = backend.asarray(scope_indptr_np.astype(np.int64), dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices_np.astype(np.int64), dtype=backend.default_int)

    return TraversalResult(
        parents=backend.device_put(parents_arr),
        levels=backend.device_put(levels_arr),
        conflict_scopes=conflict_scopes_tuple,
        scope_indptr=backend.device_put(scope_indptr_arr),
        scope_indices=backend.device_put(scope_indices_arr),
        timings=TraversalTimings(
            pairwise_seconds=pairwise_seconds,
            mask_seconds=0.0,
            semisort_seconds=semisort_seconds,
            chain_seconds=0.0,
            nonzero_seconds=0.0,
            sort_seconds=0.0,
            assemble_seconds=0.0,
            tile_seconds=tile_seconds,
            scope_chunk_segments=int(chunk_segments),
            scope_chunk_emitted=int(chunk_emitted),
            scope_chunk_max_members=int(chunk_max_members),
        ),
    )

def _collect_residual(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
    runtime: Any,
) -> TraversalResult:
    queries_np = np.asarray(backend.to_numpy(batch_points), dtype=np.float64)
    if queries_np.ndim == 1:
        queries_np = queries_np[None, :]

    batch_size = int(queries_np.shape[0])
    if batch_size == 0:
        return empty_result(backend, 0)

    host_backend = get_residual_backend()
    query_indices = decode_indices(host_backend, queries_np)
    batch_indices_np = np.asarray(query_indices, dtype=np.int64)
    tree_points_np = np.asarray(backend.to_numpy(tree.points))
    tree_indices = decode_indices(host_backend, tree_points_np)
    if tree_indices.shape[0] != tree.num_points:
        raise ValueError(
            "Residual metric decoder produced inconsistent tree indices. "
            f"Expected {tree.num_points}, received {tree_indices.shape[0]}."
        )

    pairwise_start = time.perf_counter()
    parent_dataset_idx, _parent_distances = _residual_find_parents(
        host_backend=host_backend,
        query_indices=batch_indices_np,
        tree_indices=np.asarray(tree_indices, dtype=np.int64),
    )
    pairwise_seconds = time.perf_counter() - pairwise_start

    residual_pairwise_np = residual_metrics.compute_residual_pairwise_matrix(
        host_backend=host_backend,
        batch_indices=batch_indices_np,
    )
    residual_pairwise_np = np.ascontiguousarray(residual_pairwise_np, dtype=np.float64)

    tree_indices_np = np.asarray(tree_indices, dtype=np.int64)
    dataset_to_pos = {int(tree_indices_np[i]): int(i) for i in range(tree_indices_np.shape[0])}
    parents_np = np.array(
        [dataset_to_pos.get(int(idx), -1) for idx in parent_dataset_idx],
        dtype=np.int64,
    )

    if np.any(parents_np < 0):
        raise ValueError(
            "Residual traversal produced parent identifiers not present in the tree."
        )

    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    levels_np = np.full(batch_size, -1, dtype=np.int64)
    valid_mask = parents_np >= 0
    if np.any(valid_mask):
        levels_np[valid_mask] = top_levels_np[parents_np[valid_mask]]

    base_radii = np.zeros(batch_size, dtype=np.float64)
    base_radii[valid_mask] = np.power(
        2.0, levels_np[valid_mask].astype(np.float64) + 1.0
    )
    si_cache_np = np.asarray(tree.si_cache, dtype=np.float64)
    si_values = np.zeros(batch_size, dtype=np.float64)
    if si_cache_np.size:
        si_values[valid_mask] = si_cache_np[parents_np[valid_mask]]
    radii_np = np.maximum(base_radii, si_values)
    radii_initial_np = radii_np.copy()

    scope_cap_values: np.ndarray | None = None
    cap_table = get_scope_cap_table(runtime.residual_scope_cap_path)
    if cap_table is not None:
        scope_cap_values = cap_table.lookup(levels_np)
    cap_default = runtime.residual_scope_cap_default
    if cap_default is not None and cap_default > 0.0:
        if scope_cap_values is None:
            scope_cap_values = np.full(batch_size, float(cap_default), dtype=np.float64)
        else:
            missing = ~np.isfinite(scope_cap_values)
            scope_cap_values[missing] = float(cap_default)
    if scope_cap_values is not None:
        scope_cap_values = np.maximum(
            scope_cap_values,
            float(runtime.residual_radius_floor),
        )
        valid_caps = np.isfinite(scope_cap_values) & (scope_cap_values > 0.0)
        if np.any(valid_caps):
            cap_mask = np.logical_and(valid_caps, radii_np > scope_cap_values)
            if np.any(cap_mask):
                radii_np = radii_np.copy()
                radii_np[cap_mask] = scope_cap_values[cap_mask]
    radii_limits_np = radii_np.copy()

    scope_limit = int(runtime.scope_chunk_target) if int(runtime.scope_chunk_target) > 0 else _RESIDUAL_SCOPE_DEFAULT_LIMIT
    scan_cap = int(runtime.scope_chunk_target) if int(runtime.scope_chunk_target) > 0 else 0
    scope_start = time.perf_counter()
    (
        scope_indptr_np,
        scope_indices_np,
        conflict_scopes,
        trimmed_scopes,
        chunk_max_members,
        gate_delta,
        scope_radii_np,
        saturation_flags,
        chunk_iterations,
        chunk_points,
        dedupe_hits,
        cache_hits_total,
        cache_prefetch_total,
    ) = _collect_residual_scopes_streaming(
        tree=tree,
        host_backend=host_backend,
        query_indices=batch_indices_np,
        tree_indices=tree_indices_np,
        parent_positions=parents_np,
        radii=radii_np,
        scope_limit=scope_limit,
        scan_cap=scan_cap if scan_cap > 0 else None,
    )
    semisort_seconds = time.perf_counter() - scope_start

    parents_arr = backend.asarray(parents_np.astype(np.int64), dtype=backend.default_int)
    levels_arr = backend.asarray(levels_np.astype(np.int64), dtype=backend.default_int)
    scope_indptr_arr = backend.asarray(scope_indptr_np.astype(np.int64), dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices_np.astype(np.int64), dtype=backend.default_int)

    total_scope_scans = int(np.sum(chunk_iterations))
    total_scope_points = int(np.sum(chunk_points))
    total_dedupe_hits = int(np.sum(dedupe_hits))
    total_saturated = int(np.count_nonzero(saturation_flags))

    return TraversalResult(
        parents=backend.device_put(parents_arr),
        levels=backend.device_put(levels_arr),
        conflict_scopes=conflict_scopes,
        scope_indptr=backend.device_put(scope_indptr_arr),
        scope_indices=backend.device_put(scope_indices_arr),
        timings=TraversalTimings(
            pairwise_seconds=pairwise_seconds,
            mask_seconds=0.0,
            semisort_seconds=semisort_seconds,
            chain_seconds=0.0,
            nonzero_seconds=0.0,
            sort_seconds=0.0,
            assemble_seconds=0.0,
            scope_chunk_segments=batch_size,
            scope_chunk_emitted=int(trimmed_scopes),
            scope_chunk_max_members=int(chunk_max_members),
            scope_chunk_scans=total_scope_scans,
            scope_chunk_points=total_scope_points,
            scope_chunk_dedupe=total_dedupe_hits,
            scope_chunk_saturated=total_saturated,
            scope_cache_hits=int(cache_hits_total),
            scope_cache_prefetch=int(cache_prefetch_total),
            gate1_candidates=int(gate_delta.candidates),
            gate1_kept=int(gate_delta.kept),
            gate1_pruned=int(gate_delta.pruned),
            gate1_seconds=float(gate_delta.seconds),
        ),
        residual_cache=ResidualTraversalCache(
            batch_indices=batch_indices_np,
            pairwise=residual_pairwise_np,
            scope_radii=scope_radii_np,
            scope_saturated=saturation_flags,
            scope_chunk_iterations=chunk_iterations,
            scope_chunk_points=chunk_points,
            scope_dedupe_hits=dedupe_hits,
            scope_radius_initial=radii_initial_np,
            scope_radius_limits=radii_limits_np,
            scope_radius_caps=scope_cap_values,
        ),
    )

def select_traversal_strategy(runtime: Any, backend: TreeBackend) -> TraversalStrategy:
    if (
        runtime.metric == "residual_correlation"
        and runtime.enable_sparse_traversal
        and runtime.enable_numba
        and backend.name == "numpy"
    ):
        return _ResidualTraversal()
    if (
        runtime.enable_sparse_traversal
        and runtime.enable_numba
        and runtime.metric == "euclidean"
        and NUMBA_SPARSE_TRAVERSAL_AVAILABLE
        and backend.name == "numpy"
    ):
        return _EuclideanSparseTraversal()
    return _EuclideanDenseTraversal()
