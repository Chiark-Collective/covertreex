from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from covertreex import config as cx_config
from covertreex.algo._grid_numba import (
    NUMBA_GRID_AVAILABLE,
    grid_select_leaders_numba,
)
from covertreex.algo._scope_numba import (
    NUMBA_SCOPE_AVAILABLE,
    build_conflict_graph_numba_dense,
    warmup_scope_builder,
)
from covertreex.algo.semisort import group_by_int
from .arena import get_conflict_arena, get_scope_builder_arena
from covertreex.core.tree import TreeBackend


def block_until_ready(value: Any) -> None:
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


@dataclass(frozen=True)
class AdjacencyBuild:
    sources: Any
    targets: Any
    membership_seconds: float
    targets_seconds: float
    scatter_seconds: float
    dedup_seconds: float
    csr_indptr: Any | None = None
    csr_indices: Any | None = None
    total_pairs: int = 0
    candidate_pairs: int = 0
    max_group_size: int = 0
    scope_groups: int = 0
    scope_groups_unique: int = 0
    scope_domination_ratio: float = 0.0
    bytes_h2d: int = 0
    bytes_d2h: int = 0
    radius_pruned: bool = False
    scope_chunk_segments: int = 0
    scope_chunk_emitted: int = 0
    scope_chunk_max_members: int = 0
    scope_chunk_pair_cap: int = 0
    scope_chunk_pairs_before: int = 0
    scope_chunk_pairs_after: int = 0
    scope_chunk_pair_merges: int = 0
    forced_selected: Any | None = None
    forced_dominated: Any | None = None
    grid_cells: int = 0
    grid_leaders_raw: int = 0
    grid_leaders_after: int = 0
    grid_local_edges: int = 0
    arena_bytes: int = 0
    degree_cap: int = 0
    degree_pruned_pairs: int = 0


def build_dense_adjacency(
    *,
    backend: TreeBackend,
    batch_size: int,
    scope_indptr: Any,
    scope_indices: Any,
    pairwise: Any | None = None,
    radii: np.ndarray | None = None,
    residual_pairwise: np.ndarray | None = None,
    chunk_target_override: int | None = None,
) -> AdjacencyBuild:
    xp = backend.xp
    membership_seconds = 0.0
    scatter_seconds = 0.0
    targets_seconds = 0.0
    dedup_seconds = 0.0

    sources = xp.zeros((0,), dtype=backend.default_int)
    targets = xp.zeros((0,), dtype=backend.default_int)

    total_pairs = 0
    candidate_pairs = 0
    max_group_metric = 0
    scope_groups = 0
    scope_groups_unique = 0
    scope_domination_ratio = 0.0
    bytes_d2h = 0
    bytes_h2d = 0
    arena_bytes = 0
    radius_pruned = False
    csr_indptr_np: np.ndarray | None = None
    csr_indices_np: np.ndarray | None = None
    scope_chunk_segments = 1
    scope_chunk_emitted = 0
    scope_chunk_max_members = 0
    scope_chunk_pair_cap = 0
    scope_chunk_pairs_before = 0
    scope_chunk_pairs_after = 0
    scope_chunk_pair_merges = 0

    degree_cap_metric = 0
    degree_pruned_pairs = 0

    if batch_size and scope_indices.size:
        runtime = cx_config.runtime_config()
        chunk_target = runtime.scope_chunk_target
        if chunk_target_override is not None and chunk_target_override > 0:
            chunk_target = int(chunk_target_override)
        degree_cap = int(runtime.conflict_degree_cap or 0)
        if degree_cap < 0:
            degree_cap = 0
        degree_cap_metric = degree_cap
        degree_pruned_pairs = 0
        scope_arena = get_scope_builder_arena() if runtime.scope_conflict_buffer_reuse else None
        arena_bytes = 0
        membership_start = time.perf_counter()
        scope_indptr_np = np.asarray(backend.to_numpy(scope_indptr), dtype=np.int64)
        scope_indices_np = np.asarray(backend.to_numpy(scope_indices), dtype=np.int64)
        membership_seconds = time.perf_counter() - membership_start
        bytes_d2h = int(scope_indptr_np.nbytes + scope_indices_np.nbytes)

        if runtime.enable_numba and NUMBA_SCOPE_AVAILABLE:
            warmup_scope_builder()
            if pairwise is None and residual_pairwise is None or radii is None:
                raise ValueError(
                    "pairwise distances and radii must be provided when using the "
                    "Numba conflict-graph builder"
                )
            if residual_pairwise is not None:
                pairwise_np = residual_pairwise
            else:
                pairwise_np = (
                    pairwise
                    if isinstance(pairwise, np.ndarray)
                    else np.asarray(backend.to_numpy(pairwise), dtype=np.float64)
                )
            radii_np = (
                radii
                if isinstance(radii, np.ndarray)
                else np.asarray(backend.to_numpy(radii), dtype=np.float64)
            )
            pairwise_np = np.ascontiguousarray(pairwise_np)
            radii_np = np.asarray(radii_np, dtype=np.float64)
            numba_start = time.perf_counter()
            adjacency = build_conflict_graph_numba_dense(
                scope_indptr_np,
                scope_indices_np,
                batch_size,
                segment_dedupe=runtime.scope_segment_dedupe,
                chunk_target=chunk_target,
                chunk_max_segments=runtime.scope_chunk_max_segments,
                pairwise=pairwise_np,
                radii=radii_np,
                degree_cap=degree_cap,
                scratch_pool=scope_arena,
                pair_merge=runtime.scope_chunk_pair_merge,
            )
            numba_seconds = time.perf_counter() - numba_start
            sources = adjacency.sources.astype(np.int32, copy=False)
            targets = adjacency.targets.astype(np.int32, copy=False)
            csr_indptr_np = adjacency.csr_indptr.astype(np.int64, copy=False)
            csr_indices_np = adjacency.csr_indices.astype(np.int32, copy=False)
            scatter_seconds = numba_seconds
            total_pairs = int(adjacency.total_pairs)
            candidate_pairs = int(adjacency.candidate_pairs)
            max_group_metric = int(adjacency.max_group_size)
            scope_groups = int(adjacency.num_groups)
            scope_groups_unique = int(adjacency.num_unique_groups)
            scope_domination_ratio = (
                scope_groups_unique / scope_groups if scope_groups else 0.0
            )
            bytes_h2d = int(csr_indptr_np.nbytes + csr_indices_np.nbytes)
            radius_pruned = True
            scope_chunk_segments = int(adjacency.chunk_count)
            scope_chunk_emitted = int(adjacency.chunk_emitted)
            scope_chunk_max_members = int(adjacency.chunk_max_members)
            scope_chunk_pair_cap = int(adjacency.chunk_pairs_cap)
            scope_chunk_pairs_before = int(adjacency.chunk_pairs_before)
            scope_chunk_pairs_after = int(adjacency.chunk_pairs_after)
            scope_chunk_pair_merges = int(adjacency.chunk_pair_merges)
            if scope_arena is not None:
                arena_bytes = scope_arena.total_bytes
            degree_cap_metric = int(adjacency.degree_cap)
            degree_pruned_pairs = int(adjacency.degree_pruned_pairs)
        else:
            counts_np = np.diff(scope_indptr_np)
            if scope_indices_np.size == 0:
                return AdjacencyBuild(
                    sources=sources,
                    targets=targets,
                    membership_seconds=membership_seconds,
                    targets_seconds=targets_seconds,
                    scatter_seconds=scatter_seconds,
                    dedup_seconds=0.0,
                    total_pairs=0,
                    candidate_pairs=0,
                    max_group_size=0,
                    scope_groups=0,
                    scope_groups_unique=0,
                    scope_domination_ratio=0.0,
                    bytes_d2h=bytes_d2h,
                    scope_chunk_segments=scope_chunk_segments,
                    scope_chunk_emitted=scope_chunk_emitted,
                    scope_chunk_max_members=scope_chunk_max_members,
                    scope_chunk_pair_cap=scope_chunk_pair_cap,
                    scope_chunk_pairs_before=scope_chunk_pairs_before,
                    scope_chunk_pairs_after=scope_chunk_pairs_after,
                    scope_chunk_pair_merges=scope_chunk_pair_merges,
                )

            if counts_np.size and np.max(counts_np) <= 1:
                non_empty = int(np.count_nonzero(counts_np))
                scope_groups = int(counts_np.shape[0])
                return AdjacencyBuild(
                    sources=sources,
                    targets=targets,
                    membership_seconds=membership_seconds,
                    targets_seconds=targets_seconds,
                    scatter_seconds=scatter_seconds,
                    dedup_seconds=0.0,
                    total_pairs=0,
                    candidate_pairs=0,
                    max_group_size=int(np.max(counts_np)) if counts_np.size else 0,
                    scope_groups=scope_groups,
                    scope_groups_unique=non_empty,
                    scope_domination_ratio=(1.0 if scope_groups else 0.0),
                    bytes_d2h=bytes_d2h,
                    scope_chunk_segments=scope_chunk_segments,
                    scope_chunk_emitted=scope_chunk_emitted,
                    scope_chunk_max_members=int(counts_np.max()) if counts_np.size else scope_chunk_max_members,
                    scope_chunk_pair_cap=scope_chunk_pair_cap,
                    scope_chunk_pairs_before=scope_chunk_pairs_before,
                    scope_chunk_pairs_after=scope_chunk_pairs_after,
                    scope_chunk_pair_merges=scope_chunk_pair_merges,
                )

            point_ids_np = np.repeat(
                np.arange(batch_size, dtype=np.int64),
                counts_np,
            )
            node_ids_np = scope_indices_np.astype(np.int64, copy=False)
            max_node = int(node_ids_np.max()) if node_ids_np.size else -1
            if max_node < 0:
                return AdjacencyBuild(
                    sources=sources,
                    targets=targets,
                    membership_seconds=membership_seconds,
                    targets_seconds=targets_seconds,
                    scatter_seconds=scatter_seconds,
                    dedup_seconds=0.0,
                    total_pairs=0,
                    candidate_pairs=0,
                    max_group_size=0,
                    scope_groups=0,
                    scope_groups_unique=0,
                    scope_domination_ratio=0.0,
                    bytes_d2h=bytes_d2h,
                    scope_chunk_segments=scope_chunk_segments,
                    scope_chunk_emitted=scope_chunk_emitted,
                    scope_chunk_max_members=scope_chunk_max_members,
                    scope_chunk_pair_cap=scope_chunk_pair_cap,
                    scope_chunk_pairs_before=scope_chunk_pairs_before,
                    scope_chunk_pairs_after=scope_chunk_pairs_after,
                    scope_chunk_pair_merges=scope_chunk_pair_merges,
                )

            counts_by_node = np.bincount(node_ids_np, minlength=max_node + 1)
            scope_groups = int(np.count_nonzero(counts_by_node))
            if scope_groups == 0:
                return AdjacencyBuild(
                    sources=sources,
                    targets=targets,
                    membership_seconds=membership_seconds,
                    targets_seconds=targets_seconds,
                    scatter_seconds=scatter_seconds,
                    dedup_seconds=0.0,
                    total_pairs=0,
                    candidate_pairs=0,
                    max_group_size=0,
                    scope_groups=0,
                    scope_groups_unique=0,
                    scope_domination_ratio=0.0,
                    bytes_d2h=bytes_d2h,
                    scope_chunk_segments=scope_chunk_segments,
                    scope_chunk_emitted=scope_chunk_emitted,
                    scope_chunk_max_members=scope_chunk_max_members,
                    scope_chunk_pair_cap=scope_chunk_pair_cap,
                    scope_chunk_pairs_before=scope_chunk_pairs_before,
                    scope_chunk_pairs_after=scope_chunk_pairs_after,
                    scope_chunk_pair_merges=scope_chunk_pair_merges,
                )

            offsets = np.concatenate(
                ([0], np.cumsum(counts_by_node, dtype=np.int64))
            )
            grouped_points = np.empty_like(point_ids_np)
            write_pos = offsets[:-1].copy()
            for idx, node in enumerate(node_ids_np):
                pos = write_pos[node]
                grouped_points[pos] = point_ids_np[idx]
                write_pos[node] += 1

            unique_groups = []
            unique_counts_list = []
            seen_groups: dict[tuple[int, ...], int] = {}
            max_group_metric = 0
            for node in np.nonzero(counts_by_node)[0]:
                start = offsets[node]
                end = offsets[node + 1]
                members = grouped_points[start:end]
                if members.size == 0:
                    continue
                members_unique = np.unique(members.astype(np.int64, copy=False))
                group_len = int(members_unique.size)
                max_group_metric = max(max_group_metric, group_len)
                key = tuple(members_unique.tolist())
                if key in seen_groups:
                    continue
                seen_groups[key] = len(unique_groups)
                unique_groups.append(members_unique)
                unique_counts_list.append(group_len)

            scope_groups_unique = len(unique_groups)
            scope_domination_ratio = (
                scope_groups_unique / scope_groups if scope_groups else 0.0
            )

            if scope_groups_unique == 0:
                return AdjacencyBuild(
                    sources=sources,
                    targets=targets,
                    membership_seconds=membership_seconds,
                    targets_seconds=targets_seconds,
                    scatter_seconds=scatter_seconds,
                    dedup_seconds=0.0,
                    total_pairs=0,
                    candidate_pairs=0,
                    max_group_size=0,
                    scope_groups=scope_groups,
                    scope_groups_unique=scope_groups_unique,
                    scope_domination_ratio=scope_domination_ratio,
                    bytes_h2d=bytes_h2d,
                    bytes_d2h=bytes_d2h,
                    scope_chunk_segments=scope_chunk_segments,
                    scope_chunk_emitted=scope_chunk_emitted,
                    scope_chunk_max_members=scope_chunk_max_members,
                    scope_chunk_pair_cap=scope_chunk_pair_cap,
                    scope_chunk_pairs_before=scope_chunk_pairs_before,
                    scope_chunk_pairs_after=scope_chunk_pairs_after,
                    scope_chunk_pair_merges=scope_chunk_pair_merges,
                )

            unique_counts = np.asarray(unique_counts_list, dtype=np.int64)
            effective_mask = unique_counts > 1
            if not np.any(effective_mask):
                return AdjacencyBuild(
                    sources=sources,
                    targets=targets,
                    membership_seconds=membership_seconds,
                    targets_seconds=targets_seconds,
                    scatter_seconds=scatter_seconds,
                    dedup_seconds=0.0,
                    total_pairs=0,
                    candidate_pairs=0,
                    max_group_size=max_group_metric,
                    scope_groups=scope_groups,
                    scope_groups_unique=scope_groups_unique,
                    scope_domination_ratio=scope_domination_ratio,
                    bytes_h2d=bytes_h2d,
                    bytes_d2h=bytes_d2h,
                    scope_chunk_segments=scope_chunk_segments,
                    scope_chunk_emitted=scope_chunk_emitted,
                    scope_chunk_max_members=scope_chunk_max_members,
                    scope_chunk_pair_cap=scope_chunk_pair_cap,
                    scope_chunk_pairs_before=scope_chunk_pairs_before,
                    scope_chunk_pairs_after=scope_chunk_pairs_after,
                    scope_chunk_pair_merges=scope_chunk_pair_merges,
                )

            scatter_start = time.perf_counter()
            candidate_pairs = int(
                np.sum(unique_counts[effective_mask] * (unique_counts[effective_mask] - 1))
            )
            edge_sources: list[np.ndarray] = []
            edge_targets: list[np.ndarray] = []
            for members_unique, group_len in zip(unique_groups, unique_counts):
                if group_len <= 1:
                    continue
                members_unique = members_unique.astype(np.int64, copy=False)
                src = np.repeat(members_unique, group_len)
                tgt = np.tile(members_unique, group_len)
                mask = src != tgt
                edge_sources.append(src[mask])
                edge_targets.append(tgt[mask])

            arena = get_conflict_arena()
            total_edges = sum(int(arr.size) for arr in edge_sources)
            if total_edges:
                stacked_sources = arena.borrow_sources(total_edges)
                offset = 0
                for chunk in edge_sources:
                    size = int(chunk.size)
                    stacked_sources[offset : offset + size] = chunk
                    offset += size
                stacked_targets = arena.borrow_targets(total_edges)
                offset = 0
                for chunk in edge_targets:
                    size = int(chunk.size)
                    stacked_targets[offset : offset + size] = chunk
                    offset += size
                candidate_pairs = total_edges
            else:
                stacked_sources = arena.borrow_sources(0)
                stacked_targets = arena.borrow_targets(0)
                candidate_pairs = 0
            arena_bytes = arena.total_bytes

            sources = backend.asarray(stacked_sources, dtype=backend.default_int)
            targets = backend.asarray(stacked_targets, dtype=backend.default_int)
            block_until_ready(targets)
            scatter_seconds = time.perf_counter() - scatter_start
            targets_seconds = 0.0
            bytes_h2d = int(stacked_sources.nbytes + stacked_targets.nbytes)
            if max_group_metric == 0 and unique_counts.size:
                max_group_metric = int(unique_counts.max())
            total_pairs = int(stacked_sources.shape[0])

    return AdjacencyBuild(
        sources=sources,
        targets=targets,
        csr_indptr=csr_indptr_np,
        csr_indices=csr_indices_np,
        membership_seconds=membership_seconds,
        targets_seconds=targets_seconds,
        scatter_seconds=scatter_seconds,
        dedup_seconds=dedup_seconds,
        total_pairs=int(total_pairs),
        candidate_pairs=int(candidate_pairs),
        max_group_size=max_group_metric,
        scope_groups=scope_groups,
        scope_groups_unique=scope_groups_unique,
        scope_domination_ratio=scope_domination_ratio,
        bytes_h2d=bytes_h2d,
        bytes_d2h=bytes_d2h,
        radius_pruned=radius_pruned,
        scope_chunk_segments=scope_chunk_segments,
        scope_chunk_emitted=scope_chunk_emitted,
        scope_chunk_max_members=scope_chunk_max_members,
        scope_chunk_pair_cap=scope_chunk_pair_cap,
        scope_chunk_pairs_before=scope_chunk_pairs_before,
        scope_chunk_pairs_after=scope_chunk_pairs_after,
        scope_chunk_pair_merges=scope_chunk_pair_merges,
        arena_bytes=arena_bytes,
        degree_cap=degree_cap_metric,
        degree_pruned_pairs=degree_pruned_pairs,
    )


def build_segmented_adjacency(
    *,
    backend: TreeBackend,
    scope_indices: Any,
    point_ids: Any,
    pairwise_np: np.ndarray,
    radii_np: np.ndarray,
) -> AdjacencyBuild:
    membership_start = time.perf_counter()
    grouped = group_by_int(scope_indices, point_ids, backend=backend)
    block_until_ready(grouped.values)
    values_np = np.asarray(backend.to_numpy(grouped.values), dtype=np.int64)
    indptr_np = np.asarray(backend.to_numpy(grouped.indptr), dtype=np.int64)
    membership_seconds = time.perf_counter() - membership_start

    counts = grouped.indptr[1:] - grouped.indptr[:-1]
    counts_np = indptr_np[1:] - indptr_np[:-1]
    edges_np_list: list[np.ndarray] = []

    scatter_start = time.perf_counter()
    for group_idx in range(counts_np.size):
        start = indptr_np[group_idx]
        end = indptr_np[group_idx + 1]
        members = values_np[start:end]
        if members.size <= 1:
            continue
        sub_pairwise = pairwise_np[np.ix_(members, members)]
        radii_sub = radii_np[members]
        thresholds = np.minimum.outer(radii_sub, radii_sub)
        mask = np.triu(sub_pairwise <= thresholds, k=1)
        if not mask.any():
            continue
        src_idx, tgt_idx = np.nonzero(mask)
        src_vals = members[src_idx]
        tgt_vals = members[tgt_idx]
        pair_edges = np.stack((src_vals, tgt_vals), axis=1)
        edges_np_list.append(pair_edges)
        edges_np_list.append(pair_edges[:, ::-1])
    scatter_seconds = time.perf_counter() - scatter_start

    dedup_start = time.perf_counter()
    if edges_np_list:
        edges_np = np.concatenate(edges_np_list, axis=0)
        edges_np = np.unique(edges_np, axis=0)
        sources = backend.asarray(edges_np[:, 0], dtype=backend.default_int)
        targets = backend.asarray(edges_np[:, 1], dtype=backend.default_int)
        total_pairs = int(edges_np.shape[0])
        max_group_size = int(counts_np.max()) if counts_np.size else 0
    else:
        sources = backend.asarray([], dtype=backend.default_int)
        targets = backend.asarray([], dtype=backend.default_int)
        total_pairs = 0
        max_group_size = 0
    dedup_seconds = time.perf_counter() - dedup_start

    return AdjacencyBuild(
        sources=sources,
        targets=targets,
        membership_seconds=membership_seconds,
        targets_seconds=0.0,
        scatter_seconds=scatter_seconds,
        dedup_seconds=dedup_seconds,
        total_pairs=total_pairs,
        candidate_pairs=total_pairs,
        max_group_size=max_group_size,
        radius_pruned=True,
        scope_groups=int(counts_np.size),
        scope_groups_unique=int(np.count_nonzero(counts_np)),
        scope_domination_ratio=(
            float(np.count_nonzero(counts_np)) / float(counts_np.size)
            if counts_np.size
            else 0.0
        ),
        scope_chunk_segments=1,
        scope_chunk_emitted=int(np.count_nonzero(counts_np > 0)),
        scope_chunk_max_members=int(counts_np.max()) if counts_np.size else 0,
        scope_chunk_pair_cap=0,
        scope_chunk_pairs_before=0,
        scope_chunk_pairs_after=0,
        scope_chunk_pair_merges=0,
    )


def build_residual_adjacency(
    *,
    backend: TreeBackend,
    batch_size: int,
    scope_indptr: Any,
    scope_indices: Any,
    pairwise: Any | None,
    radii: np.ndarray | None,
    residual_pairwise: np.ndarray,
    chunk_target_override: int | None = None,
) -> AdjacencyBuild:
    return build_dense_adjacency(
        backend=backend,
        batch_size=batch_size,
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
        pairwise=pairwise,
        radii=radii,
        residual_pairwise=residual_pairwise,
        chunk_target_override=chunk_target_override,
    )


def build_grid_adjacency(
    *,
    backend: TreeBackend,
    batch_points: Any,
    batch_levels: Any,
    radii: Any,
    scope_indptr: Any,
    scope_indices: Any,
) -> AdjacencyBuild:
    xp = backend.xp
    batch_np = np.asarray(backend.to_numpy(batch_points), dtype=np.float64)
    batch_np = np.ascontiguousarray(batch_np)
    if batch_np.ndim == 1:
        batch_np = batch_np.reshape(batch_np.shape[0], 1)
    levels_np = np.asarray(backend.to_numpy(batch_levels), dtype=np.int64)
    levels_np = np.ascontiguousarray(levels_np)
    radii_np = np.asarray(backend.to_numpy(radii), dtype=np.float64)
    batch_size = batch_np.shape[0]

    membership_start = time.perf_counter()
    scope_indptr_np = np.asarray(backend.to_numpy(scope_indptr), dtype=np.int64)
    scope_indices_np = np.asarray(backend.to_numpy(scope_indices), dtype=np.int64)
    membership_seconds = time.perf_counter() - membership_start
    counts = scope_indptr_np[1:] - scope_indptr_np[:-1] if scope_indptr_np.size else np.asarray([], dtype=np.int64)
    scope_groups = int(counts.size)
    scope_groups_unique = int(np.count_nonzero(counts)) if counts.size else 0
    scope_domination_ratio = float(scope_groups_unique / scope_groups) if scope_groups else 0.0
    scope_chunk_segments = scope_groups
    scope_chunk_emitted = scope_groups_unique
    scope_chunk_max_members = int(counts.max()) if counts.size else 0

    runtime = cx_config.runtime_config()
    seed = runtime.mis_seed or 0

    grid_start = time.perf_counter()
    use_numba_grid = runtime.enable_numba and NUMBA_GRID_AVAILABLE

    if use_numba_grid:
        (
            forced_selected_np,
            forced_dominated_np,
            grid_stats_arr,
        ) = grid_select_leaders_numba(
            batch_np,
            levels_np,
            seed=seed,
            num_shifts=3,
        )
        grid_stats = {
            "cells": int(grid_stats_arr[0]),
            "leaders_raw": int(grid_stats_arr[1]),
            "leaders_final": int(grid_stats_arr[2]),
            "local_edges": int(grid_stats_arr[3]),
        }
    else:
        (
            forced_selected_np,
            forced_dominated_np,
            grid_stats,
        ) = _grid_select_leaders(
            points=batch_np,
            levels=levels_np,
            radii=radii_np,
            seed=seed,
        )
    scatter_seconds = time.perf_counter() - grid_start

    forced_selected = backend.asarray(forced_selected_np.astype(backend.default_int))
    forced_selected = backend.device_put(forced_selected)
    forced_dominated = backend.asarray(forced_dominated_np.astype(backend.default_int))
    forced_dominated = backend.device_put(forced_dominated)

    indptr_np = np.zeros(batch_size + 1, dtype=np.int64)
    indices_np = np.empty(0, dtype=np.int32)
    indptr = backend.asarray(indptr_np, dtype=backend.default_int)
    indices = backend.asarray(indices_np, dtype=backend.default_int)

    throughput_bytes = int(scope_indptr_np.nbytes + scope_indices_np.nbytes)

    sources = xp.zeros((0,), dtype=backend.default_int)
    targets = xp.zeros((0,), dtype=backend.default_int)
    sources = backend.device_put(sources)
    targets = backend.device_put(targets)

    return AdjacencyBuild(
        sources=sources,
        targets=targets,
        membership_seconds=membership_seconds,
        targets_seconds=0.0,
        scatter_seconds=scatter_seconds,
        dedup_seconds=0.0,
        csr_indptr=indptr_np,
        csr_indices=indices_np,
        total_pairs=0,
        candidate_pairs=0,
        max_group_size=scope_chunk_max_members,
        scope_groups=scope_groups,
        scope_groups_unique=scope_groups_unique,
        scope_domination_ratio=scope_domination_ratio,
        bytes_h2d=0,
        bytes_d2h=throughput_bytes,
        radius_pruned=True,
        scope_chunk_segments=scope_chunk_segments,
        scope_chunk_emitted=scope_chunk_emitted,
        scope_chunk_max_members=scope_chunk_max_members,
        forced_selected=forced_selected,
        forced_dominated=forced_dominated,
        grid_cells=grid_stats["cells"],
        grid_leaders_raw=grid_stats["leaders_raw"],
        grid_leaders_after=grid_stats["leaders_final"],
        grid_local_edges=grid_stats["local_edges"],
        scope_chunk_pair_cap=0,
        scope_chunk_pairs_before=0,
        scope_chunk_pairs_after=0,
        scope_chunk_pair_merges=0,
    )


def _grid_select_leaders(
    *,
    points: np.ndarray,
    levels: np.ndarray,
    radii: np.ndarray,
    seed: int,
    num_shifts: int = 3,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    batch_size = points.shape[0]
    forced_selected = np.zeros(batch_size, dtype=np.uint8)
    forced_dominated = np.ones(batch_size, dtype=np.uint8)
    stats = {
        "cells": 0,
        "leaders_raw": 0,
        "leaders_final": 0,
        "local_edges": 0,
    }
    if batch_size == 0:
        return forced_selected, forced_dominated, stats

    dim = points.shape[1]
    levels_clamped = np.maximum(levels, 0)
    base_radius = np.power(2.0, levels_clamped.astype(np.float64) + 1.0)
    width = np.maximum(np.power(2.0, levels_clamped.astype(np.float64)), 1e-6)

    leader_mask = np.zeros(batch_size, dtype=bool)
    leader_priorities = np.full(batch_size, np.iinfo(np.uint64).max, dtype=np.uint64)

    for shift in range(num_shifts):
        shift_vec = _shift_vector(dim, seed + shift)
        scaled = points / width[:, None]
        coords = np.floor(scaled + shift_vec).astype(np.int64, copy=False)
        contiguous = np.ascontiguousarray(coords)
        prior_base = np.arange(batch_size, dtype=np.uint64)
        level_hash = levels_clamped.astype(np.uint64, copy=False) << np.uint64(33)
        priorities = _mix_uint64(
            prior_base ^ level_hash ^ np.uint64(seed + shift + 1)
        )
        key_dtype = np.dtype(
            (np.void, contiguous.dtype.itemsize * contiguous.shape[1])
        )
        keys = contiguous.view(key_dtype).ravel()
        order = np.lexsort((priorities, keys))
        sorted_keys = keys[order]
        if order.size == 0:
            continue
        unique_mask = np.empty(order.size, dtype=bool)
        unique_mask[0] = True
        if order.size > 1:
            unique_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        chosen = order[unique_mask]
        stats["cells"] += int(unique_mask.sum())
        leader_mask[chosen] = True
        leader_priorities[chosen] = np.minimum(
            leader_priorities[chosen],
            priorities[chosen],
        )

    leader_indices = np.flatnonzero(leader_mask)
    stats["leaders_raw"] = int(leader_indices.size)

    if leader_indices.size == 0:
        forced_selected[:] = 1
        forced_dominated[:] = 0
        stats["leaders_final"] = batch_size
        return forced_selected, forced_dominated, stats

    leader_priorities_subset = leader_priorities[leader_indices]
    order = np.argsort(leader_priorities_subset, kind="mergesort")
    accepted: list[int] = []
    for rank in order:
        candidate = int(leader_indices[rank])
        candidate_point = points[candidate]
        candidate_radius = base_radius[candidate]
        keep = True
        for accepted_idx in accepted:
            stats["local_edges"] += 1
            diff = candidate_point - points[accepted_idx]
            dist = np.linalg.norm(diff)
            cutoff = min(candidate_radius, base_radius[accepted_idx])
            if dist <= cutoff:
                keep = False
                break
        if keep:
            accepted.append(candidate)

    if not accepted:
        accepted = [int(leader_indices[order[0]])]
    forced_selected[accepted] = 1
    forced_dominated[accepted] = 0
    stats["leaders_final"] = len(accepted)
    return forced_selected, forced_dominated, stats


def _shift_vector(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float64) - 0.5


def _mix_uint64(values: np.ndarray) -> np.ndarray:
    x = np.uint64(values)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xbf58476d1ce4e5b9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return x


__all__ = [
    "AdjacencyBuild",
    "block_until_ready",
    "build_dense_adjacency",
    "build_grid_adjacency",
    "build_residual_adjacency",
    "build_segmented_adjacency",
]
