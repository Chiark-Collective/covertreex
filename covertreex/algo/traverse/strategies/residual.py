from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import time

from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics import residual as residual_metrics
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    ResidualDistanceTelemetry,
    ResidualWorkspace,
    compute_residual_distances_block_no_gate,
    compute_residual_distances_from_kernel,
    compute_residual_distances_with_radius,
    decode_indices,
    get_residual_backend,
)
from covertreex.metrics.residual.scope_caps import get_scope_cap_table

from ..._residual_scope_numba import (
    residual_scope_append,
    residual_scope_append_bitset,
    residual_scope_append_masked,
    residual_scope_append_masked_bitset,
    residual_scope_reset,
    residual_scope_dynamic_tile_stride,
    residual_scope_update_budget_state,
    residual_collect_next_chain,
    NUMBA_RESIDUAL_SCOPE_AVAILABLE,
)
from .._residual_parent_numba import find_parents_numba, NUMBA_AVAILABLE as NUMBA_PARENT_AVAILABLE
from ..._scope_numba import build_scope_csr_from_pairs
from ...semisort import select_topk_by_level
from ..base import (
    ResidualTraversalCache,
    TraversalResult,
    TraversalTimings,
    TraversalStrategy,
    empty_result,
)
from .common import _collect_next_chain
from .registry import register_traversal_strategy

_RESIDUAL_SCOPE_EPS = 1e-9
_RESIDUAL_SCOPE_DEFAULT_LIMIT = 16_384
_RESIDUAL_SCOPE_BUDGET_DEFAULT = (32, 64, 96)
_RESIDUAL_SCOPE_DENSE_FALLBACK_LIMIT = 64


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


def _residual_find_parents(
    *,
    host_backend: ResidualCorrHostData,
    query_indices: np.ndarray,
    tree: PCCTree,
    telemetry: ResidualDistanceTelemetry | None = None,
    runtime: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(query_indices.shape[0])
    if batch_size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        
    if tree.num_points == 0:
        return (
            np.full(batch_size, -1, dtype=np.int64),
            np.full(batch_size, np.inf, dtype=np.float64),
        )

    # Fast path: Numba-accelerated Single-Tree Traversal with Pruning
    # This avoids the O(N) flat scan by using the tree structure.
    if NUMBA_PARENT_AVAILABLE and runtime is not None:
        # Only RBF kernel is supported in the Numba path currently
        # We infer this from the presence of lengthscale/variance in runtime config
        var = float(getattr(runtime, "residual_variance", 1.0))
        ls = float(getattr(runtime, "residual_lengthscale", 1.0))
        
        # We need the raw point coordinates (float64) to compute distances in Numba.
        # host_backend stores them in `kernel_points_f32` if available (from `build_residual_backend`).
        # If not available (e.g. custom kernel provider), we cannot use this optimization.
        points_f32 = getattr(host_backend, "kernel_points_f32", None)
        
        if points_f32 is not None:
            # Map tree indices (0..N-1) to dataset indices.
            # tree.points is (N, 1) containing the dataset index for each node.
            t_points_idx = tree.backend.to_numpy(tree.points)[:, 0].astype(np.int64)
            
            # Gather tree topology
            t_children = tree.backend.to_numpy(tree.children).astype(np.int32)
            t_next = tree.backend.to_numpy(tree.next_cache).astype(np.int32)
            t_si = tree.backend.to_numpy(tree.si_cache).astype(np.float64)
            
            # Materialize coordinates for Tree Nodes and Queries
            # Note: This copy might be expensive O(N*D), but D is small (8-128).
            # Compared to O(N^2) distance comps, O(N) copy is negligible.
            # We cast to float64 for precision in the kernel.
            t_coords = points_f32[t_points_idx].astype(np.float64)
            q_coords = points_f32[query_indices].astype(np.float64)
            
            start_numba = time.perf_counter()
            # Assume root is always at index 0 of the tree arrays
            idx, dist = find_parents_numba(
                points=t_coords,
                queries=q_coords,
                children=t_children,
                next_nodes=t_next,
                si_cache=t_si,
                variance=var,
                lengthscale=ls,
                root_idx=0,
            )
            elapsed_numba = time.perf_counter() - start_numba
            
            if telemetry is not None:
                # Record approximate "kernel pairs" for consistency?
                # Or just the time. Recording 0 pairs might look weird but is accurate (no block kernel used).
                # Let's record effective pairs = batch * N to allow comparison of throughput,
                # or maybe 0 to indicate "smart traversal".
                # Let's record 0 pairs but the time, so "seconds per call" is high but "pairs" is low.
                telemetry.record_kernel(0, 0, elapsed_numba)
                
            return idx, dist

    chunk = int(host_backend.chunk_size or 512)
    total = int(tree.num_points)
    query_arr = np.asarray(query_indices, dtype=np.int64)
    # tree_indices is implicit 0..total-1
    tree_arr = np.arange(total, dtype=np.int64)

    best_dist = np.full(batch_size, np.inf, dtype=np.float64)
    best_idx = np.full(batch_size, -1, dtype=np.int64)

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        chunk_ids = tree_arr[start:stop]
        if chunk_ids.size == 0:
            continue
        
        kernel_start = time.perf_counter()
        kernel_block = host_backend.kernel_provider(query_arr, chunk_ids)
        distances_block = compute_residual_distances_from_kernel(
            host_backend,
            query_arr,
            chunk_ids,
            kernel_block,
        )
        kernel_elapsed = time.perf_counter() - kernel_start
        if telemetry is not None:
            telemetry.record_kernel(batch_size, int(chunk_ids.size), kernel_elapsed)
            
        row_min_idx = np.argmin(distances_block, axis=1)
        row_min_val = distances_block[np.arange(batch_size), row_min_idx]
        improved = row_min_val < best_dist
        if np.any(improved):
            best_dist[improved] = row_min_val[improved]
            best_idx[improved] = chunk_ids[row_min_idx[improved]]

    return best_idx, best_dist


def _process_level_cache_hits(
    *,
    cache_jobs: Dict[int, List[int]],
    level_scope_cache: Dict[int, np.ndarray],
    total_points: int,
    tree_indices_np: np.ndarray,
    query_indices: np.ndarray,
    radii: np.ndarray,
    host_backend: ResidualCorrHostData,
    distance_telemetry: ResidualDistanceTelemetry,
    limit_value: int,
    use_masked_append: bool,
    bitset_enabled: bool,
    scope_buffers: np.ndarray | None,
    scope_counts: np.ndarray,
    scope_bitsets: np.ndarray | None,
    flags_matrix: np.ndarray,
    budget_applied: np.ndarray,
    budget_survivors: np.ndarray,
    trimmed_flags: np.ndarray,
    saturated: np.ndarray,
    saturated_flags: np.ndarray,
    dedupe_hits: np.ndarray,
    observed_radii: np.ndarray,
    shared_workspace: ResidualWorkspace,
) -> tuple[int, int]:
    """Apply cached level scopes to a set of queries, batching kernel fetches."""

    if not cache_jobs:
        return 0, 0

    total_prefetch = 0
    total_hits = 0

    for parent_level, qi_list in cache_jobs.items():
        if parent_level < 0:
            continue
        cached_positions = level_scope_cache.get(parent_level)
        if cached_positions is None or cached_positions.size == 0:
            continue
        valid_mask = (cached_positions >= 0) & (cached_positions < total_points)
        valid_cached = np.asarray(cached_positions[valid_mask], dtype=np.int64)
        if valid_cached.size == 0:
            continue
        qi_arr = np.asarray(qi_list, dtype=np.int64)
        if qi_arr.size == 0:
            continue
        cached_ids = tree_indices_np[valid_cached]
        query_id_arr = np.asarray(query_indices[qi_arr], dtype=np.int64)
        radii_arr = np.asarray(radii[qi_arr], dtype=np.float64)
        total_prefetch += int(valid_cached.size) * int(qi_arr.size)
        kernel_start = time.perf_counter()
        kernel_block = host_backend.kernel_provider(query_id_arr, cached_ids)
        kernel_elapsed = time.perf_counter() - kernel_start
        distance_telemetry.record_kernel(
            int(query_id_arr.size),
            int(cached_ids.size),
            kernel_elapsed,
        )
        
        dist_block, mask_block = compute_residual_distances_block_no_gate(
            backend=host_backend,
            query_indices=query_id_arr,
            chunk_indices=cached_ids,
            kernel_block=kernel_block,
            radii=radii_arr,
        )
        
        for local_idx, qi in enumerate(qi_arr):
            if saturated[qi]:
                continue
            flags_row = flags_matrix[qi]
            bitset_row = scope_bitsets[qi] if (bitset_enabled and scope_bitsets is not None) else None
            buffer_row = scope_buffers[qi] if scope_buffers is not None else None
            mask_row = mask_block[local_idx]
            distances_row = dist_block[local_idx]
            cache_included = int(np.count_nonzero(mask_row))
            dedupe_delta = 0
            trimmed_flag = False
            added = 0
            observed_val = 0.0
            if use_masked_append:
                (
                    scope_counts[qi],
                    dedupe_delta,
                    trimmed_flag,
                    added,
                    observed_val,
                ) = _append_scope_positions_masked(
                    flags_row=flags_row,
                    bitset_row=bitset_row,
                    mask_row=mask_row,
                    distances_row=distances_row,
                    tile_positions=valid_cached,
                    limit_value=limit_value,
                    scope_count=int(scope_counts[qi]),
                    buffer_row=buffer_row,
                )
            else:
                include_idx = np.nonzero(mask_row)[0]
                if include_idx.size:
                    include_positions = valid_cached[include_idx]
                    scope_counts[qi], dedupe_delta, trimmed_flag, added = _append_scope_positions(
                        flags_row,
                        bitset_row,
                        include_positions,
                        limit_value,
                        int(scope_counts[qi]),
                        buffer_row=buffer_row,
                    )
                    if added:
                        observed_val = float(np.max(distances_row[include_idx]))
            if cache_included:
                total_hits += cache_included
            dedupe_hits[qi] += dedupe_delta
            if added:
                if observed_val > observed_radii[qi]:
                    observed_radii[qi] = observed_val
                if budget_applied[qi]:
                    budget_survivors[qi] += int(added)
                if limit_value and scope_counts[qi] >= limit_value:
                    trimmed_flags[qi] = True
                    saturated[qi] = True
                    saturated_flags[qi] = 1
            if trimmed_flag:
                trimmed_flags[qi] = True
                saturated[qi] = True
                saturated_flags[qi] = 1

    return total_prefetch, total_hits


def _collect_residual_scopes(
    *,
    tree: PCCTree,
    host_backend: ResidualCorrHostData,
    query_indices: np.ndarray,
    tree_indices: np.ndarray,
    parent_positions: np.ndarray,
    radii: np.ndarray,
    scope_limit: int | None = None,
    scan_cap: int | None = None,
    scope_budget_schedule: Tuple[int, ...] | None = None,
    scope_budget_up_thresh: float | None = None,
    scope_budget_down_thresh: float | None = None,
    stream_tile: int | None = None,
    workspace: ResidualWorkspace | None = None,
    telemetry: ResidualDistanceTelemetry | None = None,
    bitset_enabled: bool = False,
    dynamic_query_block: bool = False,
    dense_scope_streamer: bool = False,
    masked_scope_append: bool = True,
    level_cache_batching: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[Tuple[int, ...], ...],
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    distance_telemetry = telemetry or ResidualDistanceTelemetry()
    batch_size = int(query_indices.shape[0])
    total_points = int(tree_indices.shape[0])
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    parent_positions_np = np.asarray(parent_positions, dtype=np.int64)
    tree_indices_np = np.asarray(tree_indices, dtype=np.int64)
    
    limit_value = int(scope_limit) if scope_limit and scope_limit > 0 else 0
    max_points = max(total_points, 1)
    flags_matrix = np.zeros((batch_size, max_points), dtype=np.uint8)
    bitset_words = (max_points + 63) // 64 if bitset_enabled else 0
    scope_bitsets = (
        np.zeros((batch_size, bitset_words), dtype=np.uint64) if bitset_enabled else None
    )
    scope_counts = np.zeros(batch_size, dtype=np.int64)
    scope_lengths = np.zeros(batch_size, dtype=np.int64)
    parents_valid = parent_positions >= 0
    saturated = np.zeros(batch_size, dtype=bool)
    trimmed_flags = np.zeros(batch_size, dtype=bool)
    saturated_flags = np.zeros(batch_size, dtype=np.uint8)
    observed_radii = np.zeros(batch_size, dtype=np.float64)
    chunk_iterations = np.zeros(batch_size, dtype=np.int64)
    chunk_points = np.zeros(batch_size, dtype=np.int64)
    dedupe_hits = np.zeros(batch_size, dtype=np.int64)
    chain_capacity = max(int(next_cache_np.shape[0]), 1)
    chain_flags = np.zeros(chain_capacity, dtype=np.uint8)
    chain_buffer = np.empty(chain_capacity, dtype=np.int64)
    cache_hits_total = 0
    cache_prefetch_total = 0
    level_scope_cache: Dict[int, np.ndarray] = {}
    cache_limit = scope_limit if scope_limit and scope_limit > 0 else _RESIDUAL_SCOPE_DEFAULT_LIMIT
    chunk = int(host_backend.chunk_size or 512)
    stream_tile_value = None
    if isinstance(stream_tile, int) and stream_tile > 0:
        stream_tile_value = stream_tile
    tile_size = chunk
    if stream_tile_value is not None:
        tile_size = min(tile_size, stream_tile_value)
    if limit_value > 0:
        tile_size = min(tile_size, limit_value)
    tile_size = max(1, tile_size)
    query_block = max(1, min(batch_size, max(4, chunk // 4)))
    if dense_scope_streamer:
        query_block = batch_size
        dynamic_query_block = False
    elif batch_size <= 32:
        query_block = 1
    shared_workspace = workspace or ResidualWorkspace(
        max_queries=max(1, query_block),
        max_chunk=chunk,
    )

    scope_owner_chunks: list[np.ndarray] = []
    scope_member_chunks: list[np.ndarray] = []
    trimmed_scopes = 0
    max_scope_members = 0
    max_scope_after = 0
    saturation_flags = np.zeros(batch_size, dtype=np.uint8)

    scope_buffers: np.ndarray | None = None
    if not bitset_enabled and limit_value > 0:
        buffer_capacity = min(limit_value, max_points)
        if buffer_capacity > 0:
            scope_buffers = np.empty((batch_size, buffer_capacity), dtype=np.int64)

    use_masked_append = bool(
        masked_scope_append
        and (
            (scope_buffers is not None)
            or (bitset_enabled and scope_bitsets is not None)
        )
    )

    scan_cap_value = scan_cap if scan_cap and scan_cap > 0 else None

    tree_positions_np = np.arange(total_points, dtype=np.int64)
    budget_schedule = tuple(scope_budget_schedule or ())
    budget_schedule_arr = np.asarray(budget_schedule, dtype=np.int64)
    budget_up = float(scope_budget_up_thresh) if scope_budget_up_thresh is not None else 0.0
    budget_down = float(scope_budget_down_thresh) if scope_budget_down_thresh is not None else 0.0
    budget_enabled = bool(budget_schedule)
    budget_indices = np.zeros(batch_size, dtype=np.int64)
    budget_limits = np.zeros(batch_size, dtype=np.int64)
    budget_final_limits = np.zeros(batch_size, dtype=np.int64)
    budget_start_limits = np.zeros(batch_size, dtype=np.int64)
    budget_escalations = np.zeros(batch_size, dtype=np.int64)
    budget_low_streak = np.zeros(batch_size, dtype=np.int64)
    budget_survivors = np.zeros(batch_size, dtype=np.int64)
    budget_applied = np.zeros(batch_size, dtype=bool)
    budget_early_flags = np.zeros(batch_size, dtype=np.uint8)

    use_dynamic_blocks = dynamic_query_block

    def _block_ranges():
        if not use_dynamic_blocks:
            for start in range(0, batch_size, query_block):
                yield start, min(batch_size, start + query_block)
            return
        block_start = 0
        while block_start < batch_size:
            remaining_active = int(np.count_nonzero(parents_valid[block_start:]))
            if remaining_active <= 0:
                break
            block_size = _resolve_query_block_size(query_block, remaining_active)
            if block_size <= 0:
                break
            block_end = min(batch_size, block_start + block_size)
            yield block_start, block_end
            block_start = block_end
    for block_start, block_end in _block_ranges():
        block_range = range(block_start, block_end)
        block_valid: list[int] = []
        cache_jobs_block: Dict[int, List[int]] = {}
        for qi in block_range:
            if not parents_valid[qi]:
                continue
            parent_pos = int(parent_positions[qi])
            parent_level = int(top_levels_np[parent_pos]) if 0 <= parent_pos < top_levels_np.shape[0] else -1
            query_id = int(query_indices[qi])
            radius = float(radii[qi])
            flags_row = flags_matrix[qi]
            bitset_row = scope_bitsets[qi] if bitset_enabled else None
            buffer_row = scope_buffers[qi] if scope_buffers is not None else None
            if budget_enabled:
                initial_limit = budget_schedule[0] if budget_schedule else 0
                if scan_cap_value is not None:
                    initial_limit = min(initial_limit, scan_cap_value)
                if initial_limit > 0:
                    budget_applied[qi] = True
                    budget_limits[qi] = initial_limit
                    budget_final_limits[qi] = initial_limit
                    budget_start_limits[qi] = initial_limit
                    budget_indices[qi] = 0
                    budget_low_streak[qi] = 0
                    budget_survivors[qi] = 0
                else:
                    budget_applied[qi] = False
            cached_positions = level_scope_cache.get(parent_level)
            if cached_positions is not None and cached_positions.size:
                valid_cached = cached_positions[
                    (cached_positions >= 0) & (cached_positions < total_points)
                ]
                if valid_cached.size:
                    if level_cache_batching:
                        cache_jobs_block.setdefault(parent_level, []).append(qi)
                    else:
                        prefetch_delta, hit_delta = _process_level_cache_hits(
                            cache_jobs={parent_level: [qi]},
                            level_scope_cache=level_scope_cache,
                            total_points=total_points,
                            tree_indices_np=tree_indices_np,
                            query_indices=query_indices,
                            radii=radii,
                            host_backend=host_backend,
                            distance_telemetry=distance_telemetry,
                            limit_value=limit_value,
                            use_masked_append=use_masked_append,
                            bitset_enabled=bitset_enabled,
                            scope_buffers=scope_buffers,
                            scope_counts=scope_counts,
                            scope_bitsets=scope_bitsets,
                            flags_matrix=flags_matrix,
                            budget_applied=budget_applied,
                            budget_survivors=budget_survivors,
                            trimmed_flags=trimmed_flags,
                            saturated=saturated,
                            saturated_flags=saturated_flags,
                            dedupe_hits=dedupe_hits,
                            observed_radii=observed_radii,
                            shared_workspace=shared_workspace,
                        )
                        cache_prefetch_total += prefetch_delta
                        cache_hits_total += hit_delta
            block_valid.append(qi)

        if level_cache_batching and cache_jobs_block:
            prefetch_delta, hit_delta = _process_level_cache_hits(
                cache_jobs=cache_jobs_block,
                level_scope_cache=level_scope_cache,
                total_points=total_points,
                tree_indices_np=tree_indices_np,
                query_indices=query_indices,
                radii=radii,
                host_backend=host_backend,
                distance_telemetry=distance_telemetry,
                limit_value=limit_value,
                use_masked_append=use_masked_append,
                bitset_enabled=bitset_enabled,
                scope_buffers=scope_buffers,
                scope_counts=scope_counts,
                scope_bitsets=scope_bitsets,
                flags_matrix=flags_matrix,
                budget_applied=budget_applied,
                budget_survivors=budget_survivors,
                trimmed_flags=trimmed_flags,
                saturated=saturated,
                saturated_flags=saturated_flags,
                dedupe_hits=dedupe_hits,
                observed_radii=observed_radii,
                shared_workspace=shared_workspace,
            )
            cache_prefetch_total += prefetch_delta
            cache_hits_total += hit_delta

        block_idx_arr = np.asarray(block_valid, dtype=np.int64)
        if block_idx_arr.size == 0:
            continue
        block_query_ids_np = query_indices[block_idx_arr]
        block_radii_np = radii[block_idx_arr]
        for start in range(0, total_points, chunk):
            active_mask = np.logical_not(saturated[block_idx_arr])
            if not np.any(active_mask):
                break
            stop = min(start + chunk, total_points)
            chunk_ids = tree_indices_np[start:stop]
            if chunk_ids.size == 0:
                continue
            chunk_positions = tree_positions_np[start:stop]
            base_stride = tile_size if tile_size > 0 else chunk_ids.size
            base_stride = max(1, base_stride)
            tile_start = 0
            while tile_start < chunk_ids.size:
                active_idx = np.nonzero(active_mask)[0]
                if active_idx.size == 0:
                    break
                stride = _compute_dynamic_tile_stride(
                    base_stride,
                    active_idx,
                    block_idx_arr,
                    scope_counts,
                    limit_value,
                    budget_enabled,
                    budget_applied,
                    budget_limits,
                )
                tile_stop = min(tile_start + stride, chunk_ids.size)
                tile_ids = chunk_ids[tile_start:tile_stop]
                if tile_ids.size == 0:
                    break
                tile_positions = chunk_positions[tile_start:tile_stop]
                active_qis = block_idx_arr[active_idx]
                chunk_iterations[active_qis] += 1
                chunk_points[active_qis] += tile_ids.size
                tile_query_ids = block_query_ids_np[active_idx]
                tile_radii = block_radii_np[active_idx]
                kernel_start = time.perf_counter()
                kernel_block = host_backend.kernel_provider(tile_query_ids, tile_ids)
                kernel_elapsed = time.perf_counter() - kernel_start
                distance_telemetry.record_kernel(
                    int(tile_query_ids.size),
                    int(tile_ids.size),
                    kernel_elapsed,
                )
                
                dist_block, mask_block = compute_residual_distances_block_no_gate(
                    backend=host_backend,
                    query_indices=tile_query_ids,
                    chunk_indices=tile_ids,
                    kernel_block=kernel_block,
                    radii=tile_radii,
                )
                
                for local_idx, block_slot in enumerate(active_idx):
                    qi = int(block_idx_arr[block_slot])
                    if saturated[qi]:
                        continue
                    mask_row = mask_block[local_idx]
                    distances_row = dist_block[local_idx]
                    if use_masked_append:
                        buffer_row = scope_buffers[qi] if scope_buffers is not None else None
                        (
                            scope_counts[qi],
                            dedupe_delta,
                            trimmed_flag,
                            added,
                            observed_val,
                        ) = _append_scope_positions_masked(
                            flags_row=flags_matrix[qi],
                            bitset_row=scope_bitsets[qi] if bitset_enabled else None,
                            mask_row=mask_row,
                            distances_row=distances_row,
                            tile_positions=tile_positions,
                            limit_value=limit_value,
                            scope_count=int(scope_counts[qi]),
                            buffer_row=buffer_row,
                        )
                    else:
                        include_idx = np.nonzero(mask_row)[0]
                        dedupe_delta = 0
                        trimmed_flag = False
                        added = 0
                        observed_val = 0.0
                        if include_idx.size:
                            include_positions = tile_positions[include_idx]
                            scope_counts[qi], dedupe_delta, trimmed_flag, added = _append_scope_positions(
                                flags_matrix[qi],
                                scope_bitsets[qi] if bitset_enabled else None,
                                include_positions,
                                limit_value,
                                int(scope_counts[qi]),
                                buffer_row=scope_buffers[qi] if scope_buffers is not None else None,
                            )
                            if added:
                                observed_val = float(np.max(distances_row[include_idx]))
                    dedupe_hits[qi] += dedupe_delta
                    if added:
                        if observed_val > observed_radii[qi]:
                            observed_radii[qi] = observed_val
                        if budget_applied[qi]:
                            budget_survivors[qi] += int(added)
                        if limit_value and scope_counts[qi] >= limit_value:
                            trimmed_flags[qi] = True
                            saturated[qi] = True
                            saturated_flags[qi] = 1
                    if trimmed_flag:
                        trimmed_flags[qi] = True
                        saturated[qi] = True
                        saturated_flags[qi] = 1
                    _update_scope_budget_state(
                        qi,
                        chunk_points,
                        scan_cap_value,
                        budget_applied,
                        budget_up,
                        budget_down,
                        budget_schedule_arr,
                        budget_indices,
                        budget_limits,
                        budget_final_limits,
                        budget_escalations,
                        budget_low_streak,
                        budget_survivors,
                        budget_early_flags,
                        saturated,
                        saturated_flags,
                    )
                if active_qis.size:
                    active_mask[active_idx] = np.logical_not(saturated[active_qis])
                tile_start = tile_stop
            if not np.any(active_mask):
                break

        for qi in block_valid:
            if not parents_valid[qi]:
                continue
            parent_pos = int(parent_positions[qi])
            flags_row = flags_matrix[qi]
            if level_cache_batching:
                chain_len = residual_collect_next_chain(
                    next_cache_np,
                    parent_pos,
                    chain_flags,
                    chain_buffer,
                )
                if chain_len > 0:
                    chain_positions = chain_buffer[:chain_len]
                    scope_counts[qi], dedupe_delta, trimmed_flag, _ = _append_scope_positions(
                        flags_row,
                        scope_bitsets[qi] if bitset_enabled else None,
                        chain_positions,
                        limit_value,
                        int(scope_counts[qi]),
                        buffer_row=scope_buffers[qi] if scope_buffers is not None else None,
                    )
                    dedupe_hits[qi] += dedupe_delta
                    if limit_value and scope_counts[qi] >= limit_value:
                        trimmed_flags[qi] = True
                        saturated_flags[qi] = 1
                    if trimmed_flag:
                        trimmed_flags[qi] = True
                        saturated_flags[qi] = 1
            else:
                parent_arr = np.asarray([parent_pos], dtype=np.int64)
                scope_counts[qi], dedupe_delta, trimmed_flag, _ = _append_scope_positions(
                    flags_row,
                    scope_bitsets[qi] if bitset_enabled else None,
                    parent_arr,
                    limit_value,
                    int(scope_counts[qi]),
                    buffer_row=scope_buffers[qi] if scope_buffers is not None else None,
                )
                dedupe_hits[qi] += dedupe_delta
                if limit_value and scope_counts[qi] >= limit_value:
                    trimmed_flags[qi] = True
                    saturated_flags[qi] = 1
                if trimmed_flag:
                    trimmed_flags[qi] = True
                    saturated_flags[qi] = 1
                chain = _collect_next_chain(tree, parent_pos, next_cache=next_cache_np)
                if chain:
                    chain_positions = np.asarray([int(pos) for pos in chain], dtype=np.int64)
                    scope_counts[qi], dedupe_delta, trimmed_flag, _ = _append_scope_positions(
                        flags_row,
                        scope_bitsets[qi] if bitset_enabled else None,
                        chain_positions,
                        limit_value,
                        int(scope_counts[qi]),
                        buffer_row=scope_buffers[qi] if scope_buffers is not None else None,
                    )
                    dedupe_hits[qi] += dedupe_delta
                    if limit_value and scope_counts[qi] >= limit_value:
                        trimmed_flags[qi] = True
                        saturated_flags[qi] = 1
                    if trimmed_flag:
                        trimmed_flags[qi] = True
                        saturated_flags[qi] = 1

            scope_vec = np.nonzero(flags_row)[0]
            if limit_value and scope_vec.size > limit_value:
                level_slice = top_levels_np[scope_vec]
                scope_vec = select_topk_by_level(scope_vec, level_slice, limit_value)
            if scope_buffers is not None:
                residual_scope_reset(
                    flags_row,
                    scope_buffers[qi],
                    int(scope_counts[qi]),
                )
            else:
                flags_row[: total_points] = 0
            if bitset_enabled and scope_bitsets is not None:
                scope_bitsets[qi].fill(0)
            original_size = scope_vec.size
            max_scope_members = max(max_scope_members, original_size)
            exceeded_limit = bool(limit_value) and original_size > limit_value
            if exceeded_limit:
                trimmed_flags[qi] = True
                saturation_flags[qi] = 1
            elif saturated[qi]:
                saturation_flags[qi] = 1
            if scope_vec.size:
                scope_owner_chunks.append(np.full(scope_vec.size, qi, dtype=np.int64))
                scope_member_chunks.append(scope_vec.astype(np.int64, copy=False))
            else:
                scope_vec = np.empty(0, dtype=np.int64)
                saturated_flags[qi] = 0
            if limit_value:
                effective_length = min(original_size, limit_value)
            else:
                effective_length = original_size
            scope_lengths[qi] = int(effective_length)
            max_scope_after = max(max_scope_after, effective_length)
            if scope_vec.size:
                cache_slice = scope_vec[: min(cache_limit, scope_vec.size)].copy()
                parent_level = int(top_levels_np[parent_pos]) if 0 <= parent_pos < top_levels_np.shape[0] else -1
                level_scope_cache[parent_level] = cache_slice

    trimmed_scopes = int(np.count_nonzero(trimmed_flags))

    if scope_member_chunks:
        if len(scope_owner_chunks) == 1:
            owners_arr = scope_owner_chunks[0]
            members_arr = scope_member_chunks[0]
        else:
            owners_arr = np.concatenate(scope_owner_chunks)
            members_arr = np.concatenate(scope_member_chunks)
        scope_indptr, scope_indices_i32 = build_scope_csr_from_pairs(
            owners_arr.astype(np.int64, copy=False),
            members_arr.astype(np.int64, copy=False),
            batch_size,
            limit=limit_value,
            top_levels=top_levels_np,
            parents=parent_positions_np,
        )
        scope_indices = scope_indices_i32.astype(np.int64, copy=False)
    else:
        scope_indptr = np.zeros(batch_size + 1, dtype=np.int64)
        scope_indices = np.empty(0, dtype=np.int64)

    conflict_scopes_tuple = tuple(
        tuple(int(x) for x in scope_indices[scope_indptr[i] : scope_indptr[i + 1]])
        for i in range(batch_size)
    )

    total_scope_points = int(np.sum(chunk_points))
    total_scope_scans = int(np.sum(chunk_iterations))
    total_dedupe_hits = int(np.sum(dedupe_hits))
    budget_start_total = int(np.sum(budget_start_limits)) if budget_enabled else 0
    budget_final_total = int(np.sum(budget_final_limits)) if budget_enabled else 0
    budget_escalations_total = int(np.sum(budget_escalations)) if budget_enabled else 0
    budget_early_total = int(np.count_nonzero(budget_early_flags)) if budget_enabled else 0

    return (
        scope_indptr,
        scope_indices,
        conflict_scopes_tuple,
        trimmed_scopes,
        max_scope_after,
        observed_radii,
        saturated_flags,
        chunk_iterations,
        chunk_points,
        dedupe_hits,
        cache_hits_total,
        cache_prefetch_total,
        budget_start_total,
        budget_final_total,
        budget_escalations_total,
        budget_early_total,
    )

def _resolve_scope_limits(runtime: Any) -> Tuple[int, int]:
    """Derive scope/scan caps with a dense fallback for gate-off runs."""

    chunk_target = int(getattr(runtime, "scope_chunk_target", 0) or 0)
    if chunk_target > 0:
        scope_limit = chunk_target
        scan_cap = chunk_target
    else:
        scope_limit = _RESIDUAL_SCOPE_DEFAULT_LIMIT
        scan_cap = 0

    member_limit_raw = getattr(runtime, "residual_scope_member_limit", None)
    member_limit: int | None
    
    if member_limit_raw is None:
        member_limit = _RESIDUAL_SCOPE_DENSE_FALLBACK_LIMIT
    elif isinstance(member_limit_raw, int) and member_limit_raw > 0:
        member_limit = member_limit_raw
    else:
        member_limit = None

    if member_limit is not None and member_limit > 0:
        scope_limit = min(scope_limit, member_limit) if scope_limit > 0 else member_limit

    return scope_limit, scan_cap


def _update_scope_budget_state(
    qi: int,
    chunk_points: np.ndarray,
    scan_cap_value: int | None,
    budget_applied: np.ndarray,
    budget_up: float,
    budget_down: float,
    budget_schedule_arr: np.ndarray,
    budget_indices: np.ndarray,
    budget_limits: np.ndarray,
    budget_final_limits: np.ndarray,
    budget_escalations: np.ndarray,
    budget_low_streak: np.ndarray,
    budget_survivors: np.ndarray,
    budget_early_flags: np.ndarray,
    saturated: np.ndarray,
    saturated_flags: np.ndarray,
) -> None:
    if NUMBA_RESIDUAL_SCOPE_AVAILABLE:
        scan_value = int(scan_cap_value) if scan_cap_value is not None else -1
        residual_scope_update_budget_state(
            qi,
            chunk_points,
            scan_value,
            budget_applied,
            budget_up,
            budget_down,
            budget_schedule_arr,
            budget_indices,
            budget_limits,
            budget_final_limits,
            budget_escalations,
            budget_low_streak,
            budget_survivors,
            budget_early_flags,
            saturated,
            saturated_flags,
        )
        return

    schedule_tuple = tuple(int(x) for x in np.asarray(budget_schedule_arr, dtype=np.int64))
    if scan_cap_value and chunk_points[qi] >= scan_cap_value:
        saturated[qi] = True
        saturated_flags[qi] = 1
        return
    if budget_applied[qi] and not saturated[qi] and chunk_points[qi] > 0:
        ratio = budget_survivors[qi] / float(chunk_points[qi])
        if (
            ratio >= budget_up
            and schedule_tuple
            and budget_indices[qi] + 1 < len(schedule_tuple)
        ):
            next_limit = schedule_tuple[budget_indices[qi] + 1]
            if scan_cap_value is not None:
                next_limit = min(next_limit, scan_cap_value)
            if next_limit > budget_limits[qi]:
                budget_indices[qi] += 1
                budget_limits[qi] = next_limit
                budget_final_limits[qi] = next_limit
                budget_escalations[qi] += 1
                budget_low_streak[qi] = 0
        elif ratio < budget_down:
            budget_low_streak[qi] += 1
            if budget_low_streak[qi] >= 2:
                budget_early_flags[qi] = 1
                saturated[qi] = True
                saturated_flags[qi] = 1
                return
        else:
            budget_low_streak[qi] = 0
    if (
        budget_applied[qi]
        and not saturated[qi]
        and budget_limits[qi] > 0
        and budget_survivors[qi] >= budget_limits[qi]
    ):
        saturated[qi] = True
        saturated_flags[qi] = 1


def _append_scope_positions(
    flags_row: np.ndarray,
    bitset_row: np.ndarray | None,
    positions: np.ndarray,
    limit_value: int,
    scope_count: int,
    *,
    buffer_row: np.ndarray | None = None,
) -> tuple[int, int, bool, int]:
    if bitset_row is not None:
        return _append_scope_positions_bitset(
            flags_row,
            bitset_row,
            positions,
            limit_value,
            scope_count,
        )
    if buffer_row is not None:
        return _append_scope_positions_numba(
            flags_row,
            buffer_row,
            positions,
            limit_value,
            scope_count,
        )
    return _append_scope_positions_dense(flags_row, positions, limit_value, scope_count)


def _append_scope_positions_masked(
    *,
    flags_row: np.ndarray,
    bitset_row: np.ndarray | None,
    mask_row: np.ndarray,
    distances_row: np.ndarray,
    tile_positions: np.ndarray,
    limit_value: int,
    scope_count: int,
    buffer_row: np.ndarray | None = None,
) -> tuple[int, int, bool, int, float]:
    mask_arr = np.asarray(mask_row, dtype=np.uint8)
    if mask_arr.size == 0 or not np.any(mask_arr):
        return scope_count, 0, False, 0, 0.0
    if bitset_row is not None:
        mask_u8 = np.ascontiguousarray(mask_arr, dtype=np.uint8)
        distances_arr = np.ascontiguousarray(distances_row, dtype=np.float64)
        tile_arr = np.ascontiguousarray(tile_positions, dtype=np.int64)
        respect_limit = bool(limit_value and limit_value > 0)
        limit_arg = int(limit_value) if respect_limit else 0
        return residual_scope_append_masked_bitset(
            flags_row,
            bitset_row,
            mask_u8,
            distances_arr,
            tile_arr,
            int(scope_count),
            limit_arg,
            respect_limit=respect_limit,
        )
    if buffer_row is not None:
        mask_u8 = np.ascontiguousarray(mask_arr, dtype=np.uint8)
        distances_arr = np.ascontiguousarray(distances_row, dtype=np.float64)
        tile_arr = np.ascontiguousarray(tile_positions, dtype=np.int64)
        respect_limit = bool(limit_value and limit_value > 0)
        if respect_limit:
            limit_arg = min(int(limit_value), int(buffer_row.shape[0]))
        else:
            limit_arg = int(buffer_row.shape[0])
        new_count, dedupe_hits, trimmed, added, observed = residual_scope_append_masked(
            flags_row,
            buffer_row,
            mask_u8,
            distances_arr,
            tile_arr,
            int(scope_count),
            limit_arg,
            respect_limit=respect_limit,
        )
        return new_count, dedupe_hits, trimmed, added, observed

    include_idx = np.nonzero(mask_arr)[0]
    if include_idx.size == 0:
        return scope_count, 0, False, 0, 0.0
    positions_arr = np.asarray(tile_positions, dtype=np.int64)[include_idx]
    distances_arr = np.asarray(distances_row, dtype=np.float64)
    observed = float(np.max(distances_arr[include_idx])) if include_idx.size else 0.0
    new_count, dedupe_hits, trimmed, added = _append_scope_positions(
        flags_row,
        bitset_row,
        positions_arr,
        limit_value,
        scope_count,
        buffer_row=buffer_row,
    )
    return new_count, dedupe_hits, trimmed, added, observed


def _append_scope_positions_numba(
    flags_row: np.ndarray,
    buffer_row: np.ndarray,
    positions: np.ndarray,
    limit_value: int,
    scope_count: int,
) -> tuple[int, int, bool, int]:
    positions_arr = np.asarray(positions, dtype=np.int64)
    if positions_arr.size == 0:
        return scope_count, 0, False, 0
    respect_limit = limit_value > 0
    if respect_limit:
        limit_arg = int(min(limit_value, buffer_row.shape[0]))
    else:
        limit_arg = int(buffer_row.shape[0])
    prev_count = int(scope_count)
    new_count, dedupe_hits, saturated = residual_scope_append(
        flags_row,
        positions_arr,
        buffer_row,
        prev_count,
        limit_arg,
        respect_limit=bool(respect_limit),
    )
    added = int(new_count - prev_count)
    trimmed = bool(saturated)
    return int(new_count), int(dedupe_hits), trimmed, added


def _append_scope_positions_dense(
    flags_row: np.ndarray,
    positions: np.ndarray,
    limit_value: int,
    scope_count: int,
) -> tuple[int, int, bool, int]:
    positions_arr = np.asarray(positions, dtype=np.int64)
    if positions_arr.size == 0:
        return scope_count, 0, False, 0
    max_index = flags_row.shape[0]
    valid_mask = (positions_arr >= 0) & (positions_arr < max_index)
    if not np.any(valid_mask):
        return scope_count, 0, False, 0
    positions_arr = positions_arr[valid_mask]
    if positions_arr.size == 0:
        return scope_count, 0, False, 0
    existing_mask = flags_row[positions_arr] != 0
    dedupe = int(np.count_nonzero(existing_mask))
    new_positions = positions_arr[~existing_mask]
    if new_positions.size == 0:
        return scope_count, dedupe, False, 0
    trimmed = False
    if limit_value > 0:
        available = max(limit_value - scope_count, 0)
        if available <= 0:
            return scope_count, dedupe, True, 0
        if new_positions.size > available:
            trimmed = True
            new_positions = new_positions[:available]
    flags_row[new_positions] = 1
    scope_count += int(new_positions.size)
    if limit_value > 0 and scope_count >= limit_value:
        trimmed = True
    return scope_count, dedupe, trimmed, int(new_positions.size)


def _append_scope_positions_bitset(
    flags_row: np.ndarray,
    bitset_row: np.ndarray,
    positions: np.ndarray,
    limit_value: int,
    scope_count: int,
) -> tuple[int, int, bool, int]:
    positions_arr = np.asarray(positions, dtype=np.int64)
    if positions_arr.size == 0:
        return scope_count, 0, False, 0
    respect_limit = bool(limit_value and limit_value > 0)
    limit_arg = int(limit_value) if respect_limit else 0
    return residual_scope_append_bitset(
        flags_row,
        bitset_row,
        positions_arr,
        int(scope_count),
        limit_arg,
        respect_limit=respect_limit,
    )


def _compute_dynamic_tile_stride(
    base_stride: int,
    active_idx: np.ndarray,
    block_idx_arr: np.ndarray,
    scope_counts: np.ndarray,
    limit_value: int,
    budget_enabled: bool,
    budget_applied: np.ndarray,
    budget_limits: np.ndarray,
) -> int:
    """Shrink tile length once surviving budgets are nearly saturated."""

    if NUMBA_RESIDUAL_SCOPE_AVAILABLE:
        return residual_scope_dynamic_tile_stride(
            int(base_stride),
            active_idx,
            block_idx_arr,
            scope_counts,
            int(limit_value),
            bool(budget_enabled),
            budget_applied,
            budget_limits,
        )

    stride = max(1, base_stride)
    if active_idx.size == 0 or (limit_value <= 0 and not budget_enabled):
        return stride

    max_remaining = 0
    for local_slot in active_idx:
        qi = int(block_idx_arr[int(local_slot)])
        cap = limit_value if limit_value > 0 else 0
        if budget_enabled and budget_applied[qi]:
            budget_cap = int(budget_limits[qi])
            if budget_cap > 0:
                cap = budget_cap if cap <= 0 else min(cap, budget_cap)
        if cap <= 0:
            continue
        remaining = cap - int(scope_counts[qi])
        if remaining > max_remaining:
            max_remaining = remaining
            if max_remaining >= stride:
                return stride
    if max_remaining <= 0:
        return stride
    return max(1, min(stride, max_remaining))


def _resolve_query_block_size(base_block: int, remaining_active: int) -> int:
    if remaining_active <= 0:
        return 0
    if remaining_active <= 8:
        return remaining_active
    half_block = max(8, base_block // 2)
    if remaining_active <= half_block:
        return remaining_active
    return min(base_block, remaining_active)


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

    build_wall_start = time.perf_counter()
    host_backend = get_residual_backend()
    distance_telemetry = ResidualDistanceTelemetry()
    workspace = ResidualWorkspace(
        max_queries=1,
        max_chunk=max(1, int(host_backend.chunk_size or 512)),
    )
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
        tree=tree,
        telemetry=distance_telemetry,
        runtime=runtime,
    )
    pairwise_seconds = time.perf_counter() - pairwise_start

    residual_pairwise_np = residual_metrics.compute_residual_pairwise_matrix(
        host_backend=host_backend,
        batch_indices=batch_indices_np,
        telemetry=distance_telemetry,
    )
    residual_pairwise_np = np.ascontiguousarray(residual_pairwise_np, dtype=np.float32)

    tree_indices_np = np.asarray(tree_indices, dtype=np.int64)
    dataset_to_pos = {int(tree_indices_np[i]): int(i) for i in range(tree_indices_np.shape[0])}
    parents_np = np.array(
        [dataset_to_pos.get(int(idx), -1) for idx in parent_dataset_idx],
        dtype=np.int64,
    )

    if np.any(parents_np < 0) and tree_indices_np.size > 0:
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
    radius_floor = float(getattr(runtime, "residual_radius_floor", 0.0) or 0.0)
    if radius_floor > 0.0:
        # Ensure the first scope probe always has a radius large enough for the gate
        radii_np = np.maximum(radii_np, radius_floor)
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

    budget_schedule: Tuple[int, ...] = ()
    runtime_schedule = tuple(getattr(runtime, "scope_budget_schedule", ()) or ())
    runtime_metric = str(getattr(runtime, "metric", "") or "").lower()
    fallback_budget = False
    if not runtime_schedule and runtime_metric in {"residual", "residual_correlation"}:
        runtime_schedule = _RESIDUAL_SCOPE_BUDGET_DEFAULT
        fallback_budget = True
    if runtime_schedule:
        sanitized_levels: list[int] = []
        max_points = int(tree_indices_np.shape[0])
        for level in runtime_schedule:
            capped = min(int(level), max_points)
            if capped <= 0:
                continue
            if sanitized_levels and capped <= sanitized_levels[-1]:
                continue
            sanitized_levels.append(capped)
        budget_schedule = tuple(sanitized_levels)
    budget_up = getattr(runtime, "scope_budget_up_thresh", None)
    budget_down = getattr(runtime, "scope_budget_down_thresh", None)
    if fallback_budget:
        if budget_up is None or budget_up < 1.01:
            budget_up = 1.01
        if budget_down is None:
            budget_down = 0.01
    scope_start = time.perf_counter()
    
    scope_limit, scan_cap = _resolve_scope_limits(runtime)
    stream_tile_override = getattr(runtime, "residual_stream_tile", None)
    if isinstance(stream_tile_override, int) and stream_tile_override <= 0:
        stream_tile_override = None
    
    bitset_enabled = bool(getattr(runtime, "residual_scope_bitset", False))
    dynamic_query_block = bool(getattr(runtime, "residual_dynamic_query_block", False))
    dense_scope_streamer = bool(getattr(runtime, "residual_dense_scope_streamer", False))
    masked_scope_append = bool(getattr(runtime, "residual_masked_scope_append", True))
    level_cache_batching = bool(getattr(runtime, "residual_level_cache_batching", True))
    engine_label = "residual_parallel"
    (
        scope_indptr_np,
        scope_indices_np,
        conflict_scopes,
        trimmed_scopes,
        chunk_max_members,
        scope_radii_np,
        saturation_flags,
        chunk_iterations,
        chunk_points,
        dedupe_hits,
        cache_hits_total,
        cache_prefetch_total,
        budget_start_total,
        budget_final_total,
        budget_escalations_total,
        budget_early_total,
    ) = _collect_residual_scopes(
        tree=tree,
        host_backend=host_backend,
        query_indices=batch_indices_np,
        tree_indices=tree_indices_np,
        parent_positions=parents_np,
        radii=radii_np,
        scope_limit=scope_limit,
        scan_cap=scan_cap if scan_cap > 0 else None,
        scope_budget_schedule=budget_schedule if budget_schedule else None,
        scope_budget_up_thresh=budget_up if budget_schedule else None,
        scope_budget_down_thresh=budget_down if budget_schedule else None,
        stream_tile=stream_tile_override,
        workspace=workspace,
        telemetry=distance_telemetry,
        bitset_enabled=bitset_enabled,
        dynamic_query_block=dynamic_query_block,
        dense_scope_streamer=dense_scope_streamer,
        masked_scope_append=masked_scope_append,
        level_cache_batching=level_cache_batching,
    )
    semisort_seconds = time.perf_counter() - scope_start
    whitened_pairs_total = 0
    whitened_seconds_total = 0.0
    whitened_calls_total = 0
    kernel_pairs_total = int(distance_telemetry.kernel_pairs)
    kernel_seconds_total = float(distance_telemetry.kernel_seconds)
    kernel_calls_total = int(distance_telemetry.kernel_calls)

    parents_arr = backend.asarray(parents_np.astype(np.int64), dtype=backend.default_int)
    levels_arr = backend.asarray(levels_np.astype(np.int64), dtype=backend.default_int)
    scope_indptr_arr = backend.asarray(scope_indptr_np.astype(np.int64), dtype=backend.default_int)
    scope_indices_arr = backend.asarray(scope_indices_np.astype(np.int64), dtype=backend.default_int)

    total_scope_scans = int(np.sum(chunk_iterations))
    total_scope_points = int(np.sum(chunk_points))
    total_dedupe_hits = int(np.sum(dedupe_hits))
    total_saturated = int(np.count_nonzero(saturation_flags))
    build_wall_seconds = time.perf_counter() - build_wall_start

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
            build_wall_seconds=build_wall_seconds,
            scope_chunk_segments=batch_size,
            scope_chunk_emitted=int(trimmed_scopes),
            scope_chunk_max_members=int(chunk_max_members),
            scope_chunk_scans=total_scope_scans,
            scope_chunk_points=total_scope_points,
            scope_chunk_dedupe=total_dedupe_hits,
            scope_chunk_saturated=total_saturated,
            scope_cache_hits=int(cache_hits_total),
            scope_cache_prefetch=int(cache_prefetch_total),
            scope_budget_start=int(budget_start_total),
            scope_budget_final=int(budget_final_total),
            scope_budget_escalations=int(budget_escalations_total),
            scope_budget_early_terminate=int(budget_early_total),
            whitened_block_pairs=int(whitened_pairs_total),
            whitened_block_seconds=float(whitened_seconds_total),
            whitened_block_calls=int(whitened_calls_total),
            kernel_provider_pairs=int(kernel_pairs_total),
            kernel_provider_seconds=float(kernel_seconds_total),
            kernel_provider_calls=int(kernel_calls_total),
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
        engine=engine_label,
        gate_active=False,
    )


register_traversal_strategy(
    "residual_sparse",
    predicate=lambda runtime, backend: (
        runtime.metric == "residual_correlation"
        and backend.name == "numpy"
    ),
    factory=_ResidualTraversal,
)


__all__ = ["_ResidualTraversal"]