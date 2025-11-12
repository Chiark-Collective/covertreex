from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import time

from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics import residual as residual_metrics
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    ResidualGateTelemetry,
    ResidualDistanceTelemetry,
    ResidualWorkspace,
    compute_residual_distances_from_kernel,
    compute_residual_distances_with_radius,
    decode_indices,
    get_residual_backend,
)
from covertreex.metrics.residual.scope_caps import get_scope_cap_table

from ..._residual_scope_numba import residual_scope_append, residual_scope_reset
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
    tree_indices: np.ndarray,
    telemetry: ResidualDistanceTelemetry | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(query_indices.shape[0])
    if batch_size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    chunk = int(host_backend.chunk_size or 512)
    total = int(tree_indices.shape[0])
    query_arr = np.asarray(query_indices, dtype=np.int64)
    tree_arr = np.asarray(tree_indices, dtype=np.int64)

    best_dist = np.full(batch_size, np.inf, dtype=np.float64)
    best_idx = np.full(batch_size, -1, dtype=np.int64)

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        chunk_ids = tree_arr[start:stop]
        if chunk_ids.size == 0:
            continue
        kernel_start = time.perf_counter()
        kernel_block = host_backend.kernel_provider(query_arr, chunk_ids)
        kernel_elapsed = time.perf_counter() - kernel_start
        if telemetry is not None:
            telemetry.record_kernel(batch_size, int(chunk_ids.size), kernel_elapsed)
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

def _collect_residual_scopes_streaming_serial(
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
    workspace: ResidualWorkspace | None = None,
    telemetry: ResidualDistanceTelemetry | None = None,
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
    int,
    int,
    int,
    int,
    int,
]:
    batch_size = int(query_indices.shape[0])
    total_points = int(tree_indices.shape[0])
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    parent_positions_np = np.asarray(parent_positions, dtype=np.int64)
    tree_positions = np.arange(total_points, dtype=np.int64)

    scope_owner_chunks: list[np.ndarray] = []
    scope_member_chunks: list[np.ndarray] = []
    scope_lengths = np.zeros(batch_size, dtype=np.int64)
    chunk = int(host_backend.chunk_size or 512)
    trimmed_scopes = 0
    max_scope_members = 0
    max_scope_after = 0
    limit_value = int(scope_limit) if scope_limit and scope_limit > 0 else 0

    shared_workspace = workspace or ResidualWorkspace(max_queries=1, max_chunk=chunk)
    distance_telemetry = telemetry or ResidualDistanceTelemetry()
    observed_radii = np.zeros(batch_size, dtype=np.float64)
    saturation_flags = np.zeros(batch_size, dtype=np.uint8)
    chunk_iterations = np.zeros(batch_size, dtype=np.int64)
    chunk_points = np.zeros(batch_size, dtype=np.int64)
    dedupe_hits = np.zeros(batch_size, dtype=np.int64)
    flags_length = max(total_points, 1)
    collected_flags = np.zeros(flags_length, dtype=np.uint8)
    scope_buffer = np.empty(flags_length, dtype=np.int64)
    scan_cap_value = scan_cap if scan_cap and scan_cap > 0 else None
    cache_limit = scope_limit if scope_limit and scope_limit > 0 else _RESIDUAL_SCOPE_DEFAULT_LIMIT
    level_scope_cache: Dict[int, np.ndarray] = {}
    cache_hits_total = 0
    cache_prefetch_total = 0
    budget_schedule = tuple(scope_budget_schedule or ())
    budget_up = float(scope_budget_up_thresh) if scope_budget_up_thresh is not None else 0.0
    budget_down = float(scope_budget_down_thresh) if scope_budget_down_thresh is not None else 0.0
    budget_enabled = bool(budget_schedule) and scan_cap_value is not None
    budget_start_total = 0
    budget_final_total = 0
    budget_escalations_total = 0
    budget_early_total = 0

    gate_snapshot = host_backend.gate_stats.snapshot()

    for qi in range(batch_size):
        parent_pos = int(parent_positions_np[qi])
        if parent_pos < 0:
            scope_lengths[qi] = 0
            continue

        parent_level = int(top_levels_np[parent_pos]) if 0 <= parent_pos < top_levels_np.shape[0] else -1
        query_id = int(query_indices[qi])
        radius = float(radii[qi])
        scope_count = 0
        saturated = False
        fully_scanned = True
        cap_reached = False
        budget_active = budget_enabled and bool(budget_schedule)
        budget_current_idx = 0
        budget_current_limit = 0
        budget_final_val = 0
        budget_escalations = 0
        budget_low_streak = 0
        budget_survivors = 0
        budget_early = False
        if budget_active:
            budget_current_limit = budget_schedule[0]
            if scan_cap_value is not None:
                budget_current_limit = min(budget_current_limit, scan_cap_value)
            if budget_current_limit <= 0:
                budget_active = False
            else:
                budget_start_total += budget_current_limit
                budget_final_val = budget_current_limit

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
                cache_kernel_start = time.perf_counter()
                cache_kernel_full = host_backend.kernel_provider(
                    query_idx_arr,
                    cached_ids,
                )
                cache_kernel_elapsed = time.perf_counter() - cache_kernel_start
                distance_telemetry.record_kernel(
                    int(query_idx_arr.size), int(cached_ids.size), cache_kernel_elapsed
                )
                cache_kernel = cache_kernel_full[0]
                cache_distances, cache_mask = compute_residual_distances_with_radius(
                    backend=host_backend,
                    query_index=query_id,
                    chunk_indices=cached_ids,
                    kernel_row=cache_kernel,
                    radius=radius,
                    workspace=shared_workspace,
                    telemetry=distance_telemetry,
                )
                cache_include = np.nonzero(cache_mask)[0]
                if cache_include.size:
                    cache_hits_total += int(cache_include.size)
                    include_positions = valid_cached[cache_include]
                    prev_scope = scope_count
                    scope_count, dedupe_delta, hit_limit = residual_scope_append(
                        collected_flags,
                        include_positions,
                        scope_buffer,
                        scope_count,
                        limit_value,
                        respect_limit=True,
                    )
                    added = scope_count - prev_scope
                    if budget_active and added > 0:
                        budget_survivors += added
                    dedupe_hits[qi] += dedupe_delta
                    if hit_limit:
                        saturated = True
                        fully_scanned = False
                        cap_reached = True
        if not saturated:
            for start in range(0, total_points, chunk):
                if scan_cap_value and chunk_points[qi] >= scan_cap_value:
                    saturated = True
                    fully_scanned = False
                    break
                if budget_active and chunk_points[qi] >= budget_current_limit:
                    saturated = True
                    fully_scanned = False
                    cap_reached = True
                    break
                stop = min(start + chunk, total_points)
                chunk_ids = tree_indices[start:stop]
                if chunk_ids.size == 0:
                    continue
                chunk_positions = tree_positions[start:stop]
                chunk_iterations[qi] += 1
                chunk_points[qi] += chunk_ids.size
                if scan_cap_value and chunk_points[qi] >= scan_cap_value:
                    cap_reached = True
                distances, mask = compute_residual_distances_with_radius(
                    backend=host_backend,
                    query_index=query_id,
                    chunk_indices=chunk_ids,
                    kernel_row=None,
                    radius=radius,
                    workspace=shared_workspace,
                    telemetry=distance_telemetry,
                )
                if mask.size == 0:
                    continue
                include_idx = np.nonzero(mask)[0]
                if include_idx.size:
                    max_chunk = float(np.max(distances[include_idx]))
                    if max_chunk > observed_radii[qi]:
                        observed_radii[qi] = max_chunk
                    include_positions = chunk_positions[include_idx]
                    prev_scope = scope_count
                    scope_count, dedupe_delta, hit_limit = residual_scope_append(
                        collected_flags,
                        include_positions,
                        scope_buffer,
                        scope_count,
                        limit_value,
                        respect_limit=True,
                    )
                    added = scope_count - prev_scope
                    if budget_active and added > 0:
                        budget_survivors += added
                    dedupe_hits[qi] += dedupe_delta
                    if hit_limit:
                        saturated = True
                        cap_reached = True
                if budget_active and chunk_points[qi] > 0:
                    ratio = budget_survivors / float(chunk_points[qi]) if chunk_points[qi] else 0.0
                    if (
                        ratio >= budget_up
                        and budget_schedule
                        and budget_current_idx + 1 < len(budget_schedule)
                    ):
                        next_limit = budget_schedule[budget_current_idx + 1]
                        if scan_cap_value is not None:
                            next_limit = min(next_limit, scan_cap_value)
                        if next_limit > budget_current_limit:
                            budget_current_idx += 1
                            budget_current_limit = next_limit
                            budget_final_val = next_limit
                            budget_escalations += 1
                            budget_low_streak = 0
                    elif ratio < budget_down:
                        budget_low_streak += 1
                        if budget_low_streak >= 2:
                            budget_early = True
                            saturated = True
                            fully_scanned = False
                            break
                    else:
                        budget_low_streak = 0
                if budget_active and chunk_points[qi] >= budget_current_limit:
                    cap_reached = True
                if saturated or cap_reached:
                    saturated = True
                    fully_scanned = False
                    break
        # `saturated` indicates we bailed early; keep track of full scans so
        # telemetry can distinguish true caps from naturally small scopes.

        ensure_positions: list[int] = []
        if 0 <= parent_pos < total_points:
            ensure_positions.append(int(parent_pos))
        chain = _collect_next_chain(tree, parent_pos, next_cache=next_cache_np)
        for node in chain:
            pos = int(node)
            if 0 <= pos < total_points:
                ensure_positions.append(pos)
        if ensure_positions:
            ensure_arr = np.asarray(ensure_positions, dtype=np.int64)
            scope_count, _, _ = residual_scope_append(
                collected_flags,
                ensure_arr,
                scope_buffer,
                scope_count,
                limit_value,
                respect_limit=False,
            )

        if scope_count > 0:
            scope_vec = scope_buffer[:scope_count].copy()
        else:
            scope_vec = np.empty(0, dtype=np.int64)

        original_size = scope_vec.size
        max_scope_members = max(max_scope_members, original_size)

        over_limit = bool(limit_value) and original_size > limit_value
        if over_limit:
            trimmed_scopes += 1
            saturation_flags[qi] = 1
        elif limit_value and saturated and not fully_scanned:
            trimmed_scopes += 1
            saturation_flags[qi] = 1
        elif saturated:
            saturation_flags[qi] = 1

        scope_after_limit = original_size if not limit_value else min(original_size, limit_value)
        max_scope_after = max(max_scope_after, scope_after_limit)

        scope_lengths[qi] = int(scope_after_limit)
        if scope_vec.size:
            owner_chunk = np.full(scope_vec.size, qi, dtype=np.int64)
            scope_owner_chunks.append(owner_chunk)
            scope_member_chunks.append(scope_vec.astype(np.int64, copy=False))

        if budget_active:
            budget_final_total += budget_final_val
            budget_escalations_total += budget_escalations
            if budget_early:
                budget_early_total += 1

        if parent_level >= 0 and scope_vec.size:
            cache_slice = scope_vec[: min(cache_limit, scope_vec.size)].copy()
            level_scope_cache[parent_level] = cache_slice

        residual_scope_reset(collected_flags, scope_buffer, scope_count)

    if scope_member_chunks:
        if len(scope_owner_chunks) == 1:
            owners_arr = scope_owner_chunks[0]
            members_arr = scope_member_chunks[0]
        else:
            owners_arr = np.concatenate(scope_owner_chunks)
            members_arr = np.concatenate(scope_member_chunks)
        owners_arr = owners_arr.astype(np.int64, copy=False)
        members_arr = members_arr.astype(np.int64, copy=False)
        scope_indptr, scope_indices_i32 = build_scope_csr_from_pairs(
            owners_arr,
            members_arr,
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
    gate_delta = host_backend.gate_stats.delta(gate_snapshot)
    return (
        scope_indptr,
        scope_indices,
        conflict_scopes_tuple,
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
        budget_start_total,
        budget_final_total,
        budget_escalations_total,
        budget_early_total,
    )

def _collect_residual_scopes_streaming_parallel(
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
    workspace: ResidualWorkspace | None = None,
    telemetry: ResidualDistanceTelemetry | None = None,
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
    int,
    int,
    int,
    int,
    int,
    float,
]:
    distance_telemetry = telemetry or ResidualDistanceTelemetry()
    batch_size = int(query_indices.shape[0])
    total_points = int(tree_indices.shape[0])
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    parent_positions_np = np.asarray(parent_positions, dtype=np.int64)
    tree_indices_np = np.asarray(tree_indices, dtype=np.int64)
    dataset_to_pos = {int(tree_indices_np[i]): int(i) for i in range(total_points)}

    gate_snapshot = host_backend.gate_stats.snapshot()

    limit_value = int(scope_limit) if scope_limit and scope_limit > 0 else 0
    flags_matrix = np.zeros((batch_size, max(total_points, 1)), dtype=np.uint8)
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
    cache_hits_total = 0
    cache_prefetch_total = 0
    level_scope_cache: Dict[int, np.ndarray] = {}
    cache_limit = scope_limit if scope_limit and scope_limit > 0 else _RESIDUAL_SCOPE_DEFAULT_LIMIT
    chunk = int(host_backend.chunk_size or 512)
    query_block = max(1, min(batch_size, max(4, chunk // 4)))
    if batch_size <= 32:
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

    scan_cap_value = scan_cap if scan_cap and scan_cap > 0 else None

    tree_positions_np = np.arange(total_points, dtype=np.int64)
    budget_schedule = tuple(scope_budget_schedule or ())
    budget_up = float(scope_budget_up_thresh) if scope_budget_up_thresh is not None else 0.0
    budget_down = float(scope_budget_down_thresh) if scope_budget_down_thresh is not None else 0.0
    budget_enabled = bool(budget_schedule) and scan_cap_value is not None
    budget_indices = np.zeros(batch_size, dtype=np.int64)
    budget_limits = np.zeros(batch_size, dtype=np.int64)
    budget_final_limits = np.zeros(batch_size, dtype=np.int64)
    budget_start_limits = np.zeros(batch_size, dtype=np.int64)
    budget_escalations = np.zeros(batch_size, dtype=np.int64)
    budget_low_streak = np.zeros(batch_size, dtype=np.int64)
    budget_survivors = np.zeros(batch_size, dtype=np.int64)
    budget_applied = np.zeros(batch_size, dtype=bool)
    budget_early_flags = np.zeros(batch_size, dtype=np.uint8)

    for block_start in range(0, batch_size, query_block):
        block_range = range(block_start, min(batch_size, block_start + query_block))
        block_valid: list[int] = []
        for qi in block_range:
            if not parents_valid[qi]:
                continue
            parent_pos = int(parent_positions[qi])
            parent_level = int(top_levels_np[parent_pos]) if 0 <= parent_pos < top_levels_np.shape[0] else -1
            query_id = int(query_indices[qi])
            radius = float(radii[qi])
            flags_row = flags_matrix[qi]
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
                    cache_prefetch_total += int(valid_cached.size)
                    cached_ids = tree_indices_np[valid_cached]
                    cache_kernel_start = time.perf_counter()
                    cache_kernel_full = host_backend.kernel_provider(
                        np.asarray([query_id], dtype=np.int64),
                        cached_ids,
                    )
                    cache_kernel_elapsed = time.perf_counter() - cache_kernel_start
                    distance_telemetry.record_kernel(1, int(cached_ids.size), cache_kernel_elapsed)
                    cache_kernel = cache_kernel_full[0]
                    cache_distances, cache_mask = compute_residual_distances_with_radius(
                        backend=host_backend,
                        query_index=query_id,
                        chunk_indices=cached_ids,
                        kernel_row=cache_kernel,
                        radius=radius,
                        workspace=shared_workspace,
                        telemetry=distance_telemetry,
                        force_whitened=True,
                    )
                    cache_include = np.nonzero(cache_mask)[0]
                    if cache_include.size:
                        cache_hits_total += int(cache_include.size)
                        include_positions = valid_cached[cache_include]
                        scope_counts[qi], dedupe_delta, trimmed_flag, added = _append_scope_positions(
                            flags_row,
                            include_positions,
                            limit_value,
                            int(scope_counts[qi]),
                        )
                        if added and budget_applied[qi]:
                            budget_survivors[qi] += int(added)
                        dedupe_hits[qi] += dedupe_delta
                        if added:
                            obs = float(np.max(cache_distances[cache_include]))
                            if obs > observed_radii[qi]:
                                observed_radii[qi] = obs
                        if trimmed_flag:
                            trimmed_flags[qi] = True
                            saturated[qi] = True
                            saturated_flags[qi] = 1
            block_valid.append(qi)

        active_block = [qi for qi in block_valid if not saturated[qi]]
        for start in range(0, total_points, chunk):
            if not active_block:
                break
            stop = min(start + chunk, total_points)
            chunk_ids = tree_indices_np[start:stop]
            if chunk_ids.size == 0:
                continue
            chunk_positions = tree_positions_np[start:stop]
            block_arr = np.asarray(active_block, dtype=np.int64)
            block_query_ids = query_indices[block_arr]
            block_radii = radii[block_arr]
            kernel_start = time.perf_counter()
            kernel_block = host_backend.kernel_provider(block_query_ids, chunk_ids)
            kernel_elapsed = time.perf_counter() - kernel_start
            distance_telemetry.record_kernel(
                int(block_query_ids.size), int(chunk_ids.size), kernel_elapsed
            )
            next_active: list[int] = []
            for local_idx, qi in enumerate(active_block):
                if saturated[qi]:
                    continue
                chunk_iterations[qi] += 1
                chunk_points[qi] += chunk_ids.size
                query_dataset_idx = int(block_query_ids[local_idx])
                distances_row, mask_row = compute_residual_distances_with_radius(
                    backend=host_backend,
                    query_index=query_dataset_idx,
                    chunk_indices=chunk_ids,
                    kernel_row=kernel_block[local_idx],
                    radius=float(block_radii[local_idx]),
                    workspace=shared_workspace,
                    telemetry=distance_telemetry,
                    force_whitened=True,
                )
                include_idx = np.nonzero(mask_row)[0]
                if include_idx.size:
                    include_positions = chunk_positions[include_idx]
                    scope_counts[qi], dedupe_delta, trimmed_flag, added = _append_scope_positions(
                        flags_matrix[qi],
                        include_positions,
                        limit_value,
                        int(scope_counts[qi]),
                    )
                    dedupe_hits[qi] += dedupe_delta
                    if added:
                        obs = float(np.max(distances_row[include_idx]))
                        if obs > observed_radii[qi]:
                            observed_radii[qi] = obs
                        if budget_applied[qi]:
                            budget_survivors[qi] += int(added)
                    if trimmed_flag:
                        trimmed_flags[qi] = True
                        saturated[qi] = True
                        saturated_flags[qi] = 1
                if scan_cap_value and chunk_points[qi] >= scan_cap_value:
                    saturated[qi] = True
                    saturated_flags[qi] = 1
                if (
                    budget_applied[qi]
                    and not saturated[qi]
                    and chunk_points[qi] > 0
                ):
                    ratio = budget_survivors[qi] / float(chunk_points[qi])
                    if (
                        ratio >= budget_up
                        and budget_schedule
                        and budget_indices[qi] + 1 < len(budget_schedule)
                    ):
                        next_limit = budget_schedule[budget_indices[qi] + 1]
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
                    else:
                        budget_low_streak[qi] = 0
                if budget_applied[qi] and not saturated[qi] and chunk_points[qi] >= budget_limits[qi]:
                    saturated[qi] = True
                    saturated_flags[qi] = 1
                if not saturated[qi]:
                    next_active.append(qi)
            active_block = next_active

        for qi in block_valid:
            if not parents_valid[qi]:
                continue
            parent_pos = int(parent_positions[qi])
            flags_row = flags_matrix[qi]
            scope_counts[qi], dedupe_delta, trimmed_flag, _ = _append_scope_positions(
                flags_row,
                np.asarray([parent_pos], dtype=np.int64),
                limit_value,
                int(scope_counts[qi]),
            )
            dedupe_hits[qi] += dedupe_delta
            if trimmed_flag:
                trimmed_flags[qi] = True
                saturated_flags[qi] = 1
            chain = _collect_next_chain(tree, parent_pos, next_cache=next_cache_np)
            if chain:
                chain_positions = np.asarray([int(pos) for pos in chain], dtype=np.int64)
                scope_counts[qi], dedupe_delta, trimmed_flag, _ = _append_scope_positions(
                    flags_row,
                    chain_positions,
                    limit_value,
                    int(scope_counts[qi]),
                )
                dedupe_hits[qi] += dedupe_delta
                if trimmed_flag:
                    trimmed_flags[qi] = True
                    saturated_flags[qi] = 1

            scope_vec = np.nonzero(flags_row)[0]
            flags_row[: total_points] = 0
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
        host_backend.gate_stats.delta(gate_snapshot),
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

def _residual_gate_active(host_backend: ResidualCorrHostData) -> bool:
    enabled = bool(getattr(host_backend, "gate1_enabled", False))
    if not enabled:
        return False
    radius_cap = getattr(host_backend, "gate1_radius_cap", None)
    if radius_cap is not None and radius_cap <= 0.0:
        return False
    gate_vectors = getattr(host_backend, "gate_v32", None)
    lookup = getattr(host_backend, "gate_lookup", None)
    return gate_vectors is not None or lookup is not None

def _append_scope_positions(
    flags_row: np.ndarray,
    positions: np.ndarray,
    limit_value: int,
    scope_count: int,
) -> tuple[int, int, bool, int]:
    if positions.size == 0:
        return scope_count, 0, False, 0
    dedupe = int(np.count_nonzero(flags_row[positions]))
    new_mask = flags_row[positions] == 0
    new_positions = positions[new_mask]
    trimmed = False
    if limit_value > 0 and new_positions.size:
        available = max(limit_value - scope_count, 0)
        if available <= 0:
            trimmed = True
            new_positions = new_positions[:0]
        elif new_positions.size > available:
            trimmed = True
            new_positions = new_positions[:available]
    added = int(new_positions.size)
    if added:
        flags_row[new_positions] = 1
        scope_count += added
    return scope_count, dedupe, trimmed, added

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
        tree_indices=np.asarray(tree_indices, dtype=np.int64),
        telemetry=distance_telemetry,
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
    budget_schedule: Tuple[int, ...] = ()
    runtime_schedule = tuple(getattr(runtime, "scope_budget_schedule", ()) or ())
    if scan_cap > 0 and runtime_schedule:
        sanitized_levels: list[int] = []
        for level in runtime_schedule:
            capped = min(level, scan_cap)
            if capped <= 0:
                continue
            if sanitized_levels and capped <= sanitized_levels[-1]:
                continue
            sanitized_levels.append(capped)
        budget_schedule = tuple(sanitized_levels)
    budget_up = getattr(runtime, "scope_budget_up_thresh", None)
    budget_down = getattr(runtime, "scope_budget_down_thresh", None)
    scope_start = time.perf_counter()
    gate_active = _residual_gate_active(host_backend)
    streaming_fn = (
        _collect_residual_scopes_streaming_serial
        if gate_active
        else _collect_residual_scopes_streaming_parallel
    )
    engine_label = "residual_serial" if gate_active else "residual_parallel"
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
        budget_start_total,
        budget_final_total,
        budget_escalations_total,
        budget_early_total,
    ) = streaming_fn(
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
        workspace=workspace,
        telemetry=distance_telemetry,
    )
    semisort_seconds = time.perf_counter() - scope_start
    whitened_pairs_total = int(distance_telemetry.whitened_pairs)
    whitened_seconds_total = float(distance_telemetry.whitened_seconds)
    whitened_calls_total = int(distance_telemetry.whitened_calls)
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
            gate1_candidates=int(gate_delta.candidates),
            gate1_kept=int(gate_delta.kept),
            gate1_pruned=int(gate_delta.pruned),
            gate1_seconds=float(gate_delta.seconds),
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
        gate_active=gate_active,
    )


register_traversal_strategy(
    "residual_sparse",
    predicate=lambda runtime, backend: (
        runtime.metric == "residual_correlation"
        and runtime.enable_numba
        and backend.name == "numpy"
    ),
    factory=_ResidualTraversal,
)


__all__ = ["_ResidualTraversal"]
