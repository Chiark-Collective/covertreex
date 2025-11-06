from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import numpy as np

from covertreex import config as cx_config
from covertreex.algo.semisort import group_by_int
from covertreex.algo._scope_numba import (
    NUMBA_SCOPE_AVAILABLE,
    build_conflict_graph_numba_dense,
    warmup_scope_builder,
)
from covertreex.algo.traverse import TraversalResult
from covertreex.core.metrics import get_metric
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics.residual import (
    compute_residual_distances_from_kernel,
    decode_indices,
    get_residual_backend,
)


@dataclass(frozen=True)
class ConflictGraph:
    """Conflict graph encoded in CSR form."""

    indptr: Any
    indices: Any
    pairwise_distances: Any
    scope_indptr: Any
    scope_indices: Any
    radii: Any
    annulus_bounds: Any
    annulus_bins: Any
    annulus_bin_indptr: Any
    annulus_bin_indices: Any
    annulus_bin_ids: Any
    timings: "ConflictGraphTimings"

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def num_edges(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_scopes(self) -> int:
        return int(self.scope_indptr.shape[0] - 1)


@dataclass(frozen=True)
class ConflictGraphTimings:
    pairwise_seconds: float
    scope_group_seconds: float
    adjacency_seconds: float
    annulus_seconds: float
    adjacency_membership_seconds: float = 0.0
    adjacency_targets_seconds: float = 0.0
    adjacency_scatter_seconds: float = 0.0
    adjacency_filter_seconds: float = 0.0
    adjacency_sort_seconds: float = 0.0
    adjacency_dedup_seconds: float = 0.0
    adjacency_extract_seconds: float = 0.0
    adjacency_csr_seconds: float = 0.0
    adjacency_total_pairs: float = 0.0
    adjacency_candidate_pairs: float = 0.0
    adjacency_max_group_size: float = 0.0
    scope_bytes_h2d: int = 0
    scope_bytes_d2h: int = 0
    scope_groups: int = 0
    scope_groups_unique: int = 0
    scope_domination_ratio: float = 0.0
    mis_seconds: float = 0.0


@dataclass(frozen=True)
class _AdjacencyBuild:
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


def _block_until_ready(value: Any) -> None:
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


def _build_dense_adjacency(
    *,
    backend: TreeBackend,
    batch_size: int,
    scope_indptr: Any,
    scope_indices: Any,
    pairwise: Any | None = None,
    radii: Any | None = None,
    residual_pairwise: np.ndarray | None = None,
) -> _AdjacencyBuild:
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
        else:
            counts_np = np.diff(scope_indptr_np)
            if scope_indices_np.size == 0:
                return _AdjacencyBuild(
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
                )

            if counts_np.size and np.max(counts_np) <= 1:
                non_empty = int(np.count_nonzero(counts_np))
                scope_groups = int(counts_np.shape[0])
                return _AdjacencyBuild(
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
                )

            point_ids_np = np.repeat(
                np.arange(batch_size, dtype=np.int64),
                counts_np,
            )
            node_ids_np = scope_indices_np.astype(np.int64, copy=False)
            max_node = int(node_ids_np.max()) if node_ids_np.size else -1
            if max_node < 0:
                return _AdjacencyBuild(
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
                )

            counts_by_node = np.bincount(node_ids_np, minlength=max_node + 1)
            scope_groups = int(np.count_nonzero(counts_by_node))
            if scope_groups == 0:
                return _AdjacencyBuild(
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
                return _AdjacencyBuild(
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
                )

            unique_counts = np.asarray(unique_counts_list, dtype=np.int64)
            effective_mask = unique_counts > 1
            if not np.any(effective_mask):
                return _AdjacencyBuild(
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
            _block_until_ready(targets)
            scatter_seconds = time.perf_counter() - scatter_start
            targets_seconds = 0.0
            bytes_h2d = int(stacked_sources.nbytes + stacked_targets.nbytes)
            if max_group_metric == 0 and unique_counts.size:
                max_group_metric = int(unique_counts.max())
            total_pairs = int(stacked_sources.shape[0])

    return _AdjacencyBuild(
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
    )


def _compute_residual_pairwise_matrix(
    *,
    host_backend: "ResidualCorrHostData",
    batch_indices: np.ndarray,
) -> np.ndarray:
    total = batch_indices.shape[0]
    result = np.empty((total, total), dtype=np.float64)
    chunk = int(host_backend.chunk_size or 512)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        rows = batch_indices[start:stop]
        kernel_block = host_backend.kernel_provider(rows, batch_indices)
        distances = compute_residual_distances_from_kernel(
            host_backend,
            rows,
            batch_indices,
            kernel_block,
        )
        result[start:stop, :] = distances
    return result


def _build_segmented_adjacency(
    *,
    backend: TreeBackend,
    scope_indices: Any,
    point_ids: Any,
    pairwise_np: np.ndarray,
    radii_np: np.ndarray,
) -> _AdjacencyBuild:
    xp = backend.xp

    membership_start = time.perf_counter()
    grouped = group_by_int(scope_indices, point_ids, backend=backend)
    _block_until_ready(grouped.values)
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
        sources = xp.zeros((0,), dtype=backend.default_int)
        targets = xp.zeros((0,), dtype=backend.default_int)
        total_pairs = 0
        max_group_size = 0
    dedup_seconds = time.perf_counter() - dedup_start

    return _AdjacencyBuild(
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
    )
def build_conflict_graph(
    tree: PCCTree,
    traversal: TraversalResult,
    batch_points: Any,
    *,
    backend: TreeBackend | None = None,
) -> ConflictGraph:
    """Construct a conflict graph with distance-aware pruning."""

    backend = backend or tree.backend
    xp = backend.xp
    batch = backend.asarray(batch_points, dtype=backend.default_float)
    metric = get_metric()

    runtime = cx_config.runtime_config()
    residual_mode = (
        runtime.metric == "residual_correlation" and backend.name == "numpy"
    )

    residual_pairwise_np: np.ndarray | None = None
    batch_indices_np: np.ndarray | None = None

    pairwise_start = time.perf_counter()
    if residual_mode:
        host_backend = get_residual_backend()
        batch_np = np.asarray(backend.to_numpy(batch))
        batch_indices_np = decode_indices(host_backend, batch_np)
        residual_pairwise_np = _compute_residual_pairwise_matrix(
            host_backend=host_backend,
            batch_indices=batch_indices_np,
        )
        pairwise = backend.asarray(residual_pairwise_np, dtype=backend.default_float)
    else:
        pairwise = metric.pairwise(backend, batch, batch)
        _block_until_ready(pairwise)
    pairwise_seconds = time.perf_counter() - pairwise_start

    batch_size = int(traversal.parents.shape[0])

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

    parents = backend.asarray(traversal.parents, dtype=backend.default_int)
    levels = backend.asarray(traversal.levels, dtype=backend.default_int)
    level_float = levels.astype(backend.default_float)
    base = xp.power(
        xp.asarray(2.0, dtype=backend.default_float), level_float + 1.0
    )
    parent_valid = parents >= 0
    safe_parents = xp.where(parent_valid, parents, xp.zeros_like(parents))
    si_cache = backend.asarray(tree.si_cache, dtype=backend.default_float)
    if si_cache.size:
        si_values = xp.take(si_cache, safe_parents, mode="clip")
        si_values = xp.where(parent_valid, si_values, xp.zeros_like(si_values))
    else:
        si_values = xp.zeros(parents.shape, dtype=backend.default_float)
    radii = xp.where(
        parent_valid,
        xp.maximum(base, si_values),
        xp.full_like(base, float("inf")),
    )
    radii = backend.device_put(radii)
    impl = runtime.conflict_graph_impl

    pairwise_np: np.ndarray | None = None
    radii_np: np.ndarray | None = None
    need_numpy_buffers = enable_numba or impl == "segmented" or residual_mode
    if need_numpy_buffers:
        if residual_pairwise_np is not None:
            pairwise_np = residual_pairwise_np
        else:
            pairwise_np = np.asarray(backend.to_numpy(pairwise), dtype=float)
            pairwise_np = np.ascontiguousarray(pairwise_np)
        radii_np = np.asarray(backend.to_numpy(radii), dtype=float)

    adjacency_start = time.perf_counter()
    adjacency_filter_seconds = 0.0
    adjacency_extract_seconds = 0.0
    adjacency_sort_seconds = 0.0
    adjacency_csr_seconds = 0.0

    if impl == "segmented":
        adjacency_build = _build_segmented_adjacency(
            backend=backend,
            scope_indices=scope_indices,
            point_ids=point_ids,
            pairwise_np=pairwise_np,
            radii_np=radii_np,
        )
    else:
        adjacency_build = _build_dense_adjacency(
            backend=backend,
            batch_size=batch_size,
            scope_indptr=scope_indptr,
            scope_indices=scope_indices,
            pairwise=pairwise,
            radii=radii_np,
            residual_pairwise=residual_pairwise_np,
        )

    sources = adjacency_build.sources
    targets = adjacency_build.targets
    adjacency_membership_seconds = adjacency_build.membership_seconds
    adjacency_targets_seconds = adjacency_build.targets_seconds
    adjacency_scatter_seconds = adjacency_build.scatter_seconds
    adjacency_dedup_seconds = adjacency_build.dedup_seconds
    adjacency_total_pairs = adjacency_build.total_pairs
    adjacency_candidate_pairs = adjacency_build.candidate_pairs
    adjacency_max_group = adjacency_build.max_group_size
    scope_bytes_h2d = adjacency_build.bytes_h2d
    scope_bytes_d2h = adjacency_build.bytes_d2h
    scope_groups = adjacency_build.scope_groups
    scope_groups_unique = adjacency_build.scope_groups_unique
    scope_domination_ratio = adjacency_build.scope_domination_ratio
    radius_pruned = adjacency_build.radius_pruned

    if sources.size and not radius_pruned:
        filter_start = time.perf_counter()
        if residual_mode and batch_indices_np is not None:
            host_backend = get_residual_backend()
            sources_np = np.asarray(backend.to_numpy(sources), dtype=np.int64)
            targets_np = np.asarray(backend.to_numpy(targets), dtype=np.int64)
            keep_mask_np = np.zeros(sources_np.shape[0], dtype=np.bool_)
            unique_sources = np.unique(sources_np)
            for src in unique_sources:
                mask_indices = np.where(sources_np == src)[0]
                if mask_indices.size == 0:
                    continue
                src_dataset = int(batch_indices_np[src])
                tgt_batch_ids = targets_np[mask_indices]
                tgt_dataset = batch_indices_np[tgt_batch_ids]
                kernel_block = host_backend.kernel_provider(
                    np.asarray([src_dataset], dtype=np.int64),
                    tgt_dataset,
                )
                distances = compute_residual_distances_from_kernel(
                    host_backend,
                    np.asarray([src_dataset], dtype=np.int64),
                    tgt_dataset,
                    kernel_block,
                )[0]
                min_radii = np.minimum(radii_np[src], radii_np[tgt_batch_ids])
                keep = distances <= (min_radii + RESIDUAL_FILTER_EPS)
                keep_mask_np[mask_indices] = keep
            keep_mask = backend.asarray(keep_mask_np, dtype=backend.xp.bool_)
            sources = sources[keep_mask]
            targets = targets[keep_mask]
            adjacency_filter_seconds = time.perf_counter() - filter_start
        else:
            src_pts = xp.take(batch, sources, axis=0)
            tgt_pts = xp.take(batch, targets, axis=0)
            diff = src_pts - tgt_pts
            squared_distances = xp.sum(diff * diff, axis=1)
            min_radii = xp.minimum(radii[sources], radii[targets])
            squared_bounds = min_radii * min_radii
            keep_mask = squared_distances <= squared_bounds
            sources = sources[keep_mask]
            targets = targets[keep_mask]
            _block_until_ready(keep_mask)
            adjacency_filter_seconds = time.perf_counter() - filter_start
    else:
        adjacency_filter_seconds = 0.0

    adjacency_sort_seconds = 0.0
    adjacency_csr_seconds = 0.0

    csr_indptr_np = adjacency_build.csr_indptr
    csr_indices_np = adjacency_build.csr_indices

    if csr_indptr_np is not None and csr_indices_np is not None:
        csr_start = time.perf_counter()
        indptr = backend.asarray(
            np.asarray(csr_indptr_np, dtype=np.int64), dtype=backend.default_int
        )
        indices = backend.asarray(
            np.asarray(csr_indices_np, dtype=np.int32), dtype=backend.default_int
        )
        adjacency_csr_seconds = time.perf_counter() - csr_start

        indptr = backend.device_put(indptr)
        indices = backend.device_put(indices)
        _block_until_ready(indices)
    elif sources.size:
        csr_start = time.perf_counter()
        sources_np = np.asarray(backend.to_numpy(sources), dtype=np.int64)
        targets_np = np.asarray(backend.to_numpy(targets), dtype=np.int64)
        counts_np = np.bincount(sources_np, minlength=batch_size)
        indptr_np = np.empty(batch_size + 1, dtype=np.int64)
        indptr_np[0] = 0
        np.cumsum(counts_np, dtype=np.int64, out=indptr_np[1:])
        indices_np = np.empty_like(targets_np, dtype=np.int64)
        offsets_np = indptr_np[:-1].copy()
        for idx in range(sources_np.size):
            src = sources_np[idx]
            pos = offsets_np[src]
            indices_np[pos] = targets_np[idx]
            offsets_np[src] = pos + 1
        indptr = backend.asarray(indptr_np, dtype=backend.default_int)
        indices = backend.asarray(indices_np, dtype=backend.default_int)
        adjacency_csr_seconds = time.perf_counter() - csr_start

        indptr = backend.device_put(indptr)
        indices = backend.device_put(indices)
        _block_until_ready(indices)
    else:
        indptr = xp.zeros((batch_size + 1,), dtype=backend.default_int)
        indices = xp.zeros((0,), dtype=backend.default_int)
        indptr = backend.device_put(indptr)
        indices = backend.device_put(indices)

    adjacency_seconds = time.perf_counter() - adjacency_start

    scope_indptr_arr = traversal.scope_indptr
    scope_indices_arr = traversal.scope_indices

    annulus_start = time.perf_counter()
    lower_bounds = xp.zeros((batch_size, 1), dtype=backend.default_float)
    upper_bounds = radii[:, None]
    annulus_bounds = xp.concatenate((lower_bounds, upper_bounds), axis=1)
    log_r = xp.log2(xp.maximum(radii, xp.asarray(1.0, dtype=backend.default_float)))
    annulus_bins = xp.floor(log_r).astype(backend.default_int)
    annulus_bins = xp.where(
        xp.isinf(radii),
        xp.full_like(annulus_bins, -1),
        annulus_bins,
    )
    if annulus_bins.size:
        point_indices = xp.arange(batch_size, dtype=backend.default_int)
        grouped_bins = group_by_int(
            annulus_bins,
            point_indices,
            backend=backend,
        )
        annulus_bin_ids = grouped_bins.keys
        annulus_bin_indptr = grouped_bins.indptr
        annulus_bin_indices = grouped_bins.values.astype(backend.default_int)
    else:
        annulus_bin_ids = backend.asarray([], dtype=backend.default_int)
        annulus_bin_indptr = backend.asarray([0], dtype=backend.default_int)
        annulus_bin_indices = backend.asarray([], dtype=backend.default_int)
    _block_until_ready(annulus_bin_indices)
    annulus_seconds = time.perf_counter() - annulus_start

    return ConflictGraph(
        indptr=indptr,
        indices=indices,
        pairwise_distances=pairwise,
        scope_indptr=scope_indptr_arr,
        scope_indices=scope_indices_arr,
        radii=radii,
        annulus_bounds=backend.device_put(annulus_bounds),
        annulus_bins=backend.device_put(annulus_bins),
        annulus_bin_indptr=annulus_bin_indptr,
        annulus_bin_indices=annulus_bin_indices,
        annulus_bin_ids=annulus_bin_ids,
        timings=ConflictGraphTimings(
            pairwise_seconds=pairwise_seconds,
            scope_group_seconds=scope_group_seconds,
            adjacency_seconds=adjacency_seconds,
            annulus_seconds=annulus_seconds,
            adjacency_membership_seconds=adjacency_membership_seconds,
            adjacency_targets_seconds=adjacency_targets_seconds,
            adjacency_scatter_seconds=adjacency_scatter_seconds,
            adjacency_filter_seconds=adjacency_filter_seconds,
            adjacency_sort_seconds=adjacency_sort_seconds,
            adjacency_dedup_seconds=adjacency_dedup_seconds,
            adjacency_extract_seconds=adjacency_extract_seconds,
            adjacency_csr_seconds=adjacency_csr_seconds,
            adjacency_total_pairs=float(adjacency_total_pairs),
            adjacency_candidate_pairs=float(adjacency_candidate_pairs),
            adjacency_max_group_size=float(adjacency_max_group),
            scope_bytes_h2d=scope_bytes_h2d,
            scope_bytes_d2h=scope_bytes_d2h,
            scope_groups=scope_groups,
            scope_groups_unique=scope_groups_unique,
            scope_domination_ratio=scope_domination_ratio,
        ),
    )
RESIDUAL_FILTER_EPS = 1e-9
