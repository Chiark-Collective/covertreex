from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from covertreex import config as cx_config
from covertreex.algo._scope_numba import (
    NUMBA_SCOPE_AVAILABLE,
    build_conflict_graph_numba_dense,
    warmup_scope_builder,
)
from covertreex.algo.semisort import group_by_int
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


def build_dense_adjacency(
    *,
    backend: TreeBackend,
    batch_size: int,
    scope_indptr: Any,
    scope_indices: Any,
    pairwise: Any | None = None,
    radii: np.ndarray | None = None,
    residual_pairwise: np.ndarray | None = None,
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
    radius_pruned = False
    csr_indptr_np: np.ndarray | None = None
    csr_indices_np: np.ndarray | None = None
    scope_chunk_segments = 1
    scope_chunk_emitted = 0
    scope_chunk_max_members = 0

    if batch_size and scope_indices.size:
        runtime = cx_config.runtime_config()
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
                chunk_target=runtime.scope_chunk_target,
                chunk_max_segments=runtime.scope_chunk_max_segments,
                pairwise=pairwise_np,
                radii=radii_np,
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

            if edge_sources:
                stacked_sources = np.concatenate(edge_sources)
                stacked_targets = np.concatenate(edge_targets)
            else:
                stacked_sources = np.empty((0,), dtype=np.int64)
                stacked_targets = np.empty((0,), dtype=np.int64)
                candidate_pairs = 0

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
) -> AdjacencyBuild:
    return build_dense_adjacency(
        backend=backend,
        batch_size=batch_size,
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
        pairwise=pairwise,
        radii=radii,
        residual_pairwise=residual_pairwise,
    )


__all__ = [
    "AdjacencyBuild",
    "block_until_ready",
    "build_dense_adjacency",
    "build_residual_adjacency",
    "build_segmented_adjacency",
]
