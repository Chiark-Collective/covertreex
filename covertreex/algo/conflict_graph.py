from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import time
import numpy as np

from covertreex import config as cx_config
from covertreex.algo._scope_numba import (
    NUMBA_SCOPE_AVAILABLE,
    filter_csr_by_radii_from_pairwise,
)
from covertreex.algo.conflict_graph_builders import (
    AdjacencyBuild,
    block_until_ready,
    build_dense_adjacency,
    build_grid_adjacency,
    build_residual_adjacency,
    build_segmented_adjacency,
)
from covertreex.algo.semisort import group_by_int
from covertreex.algo.traverse import TraversalResult
from covertreex.core.metrics import get_metric
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics.residual import (
    compute_residual_distances_from_kernel,
    compute_residual_pairwise_matrix,
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
    forced_selected: Any | None = None
    forced_dominated: Any | None = None
    grid_cells: int = 0
    grid_leaders_raw: int = 0
    grid_leaders_after: int = 0
    grid_local_edges: int = 0

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
    scope_chunk_segments: int = 0
    scope_chunk_emitted: int = 0
    scope_chunk_max_members: int = 0
    scope_chunk_pair_cap: int = 0
    scope_chunk_pairs_before: int = 0
    scope_chunk_pairs_after: int = 0
    mis_seconds: float = 0.0
    grid_cells: int = 0
    grid_leaders_raw: int = 0
    grid_leaders_after: int = 0
    grid_local_edges: int = 0


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
    residual_cache = traversal.residual_cache if residual_mode else None

    pairwise_start = time.perf_counter()
    if residual_mode:
        host_backend = get_residual_backend()
        if residual_cache is not None:
            batch_indices_np = np.asarray(residual_cache.batch_indices, dtype=np.int64)
            residual_pairwise_np = np.asarray(residual_cache.pairwise, dtype=np.float64)
        else:
            batch_np = np.asarray(backend.to_numpy(batch))
            batch_indices_np = decode_indices(host_backend, batch_np)
            residual_pairwise_np = compute_residual_pairwise_matrix(
                host_backend,
                batch_indices_np,
            )
        pairwise = backend.asarray(residual_pairwise_np, dtype=backend.default_float)
    else:
        pairwise = metric.pairwise(backend, batch, batch)
        block_until_ready(pairwise)
    pairwise_seconds = time.perf_counter() - pairwise_start

    batch_size = int(traversal.parents.shape[0])
    residual_scope_radii_np: np.ndarray | None = None
    if (
        residual_mode
        and residual_cache is not None
        and getattr(residual_cache, "scope_radii", None) is not None
    ):
        scope_arr = np.asarray(residual_cache.scope_radii, dtype=np.float64)
        if scope_arr.shape[0] == batch_size:
            residual_scope_radii_np = scope_arr

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
        block_until_ready(point_ids)
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
        si_values = xp.where(
            xp.isfinite(si_values),
            si_values,
            base,
        )
    else:
        si_values = xp.zeros(parents.shape, dtype=backend.default_float)
    fallback_radii = xp.where(parent_valid, xp.maximum(base, si_values), base)
    if residual_scope_radii_np is not None:
        scope_radii_backend = backend.asarray(
            residual_scope_radii_np,
            dtype=backend.default_float,
        )
        min_floor = xp.asarray(runtime.residual_radius_floor, dtype=backend.default_float)
        scope_radii_backend = xp.maximum(scope_radii_backend, min_floor)
        finite_mask = xp.logical_and(parent_valid, xp.isfinite(scope_radii_backend))
        radii = xp.where(finite_mask, scope_radii_backend, fallback_radii)
    else:
        radii = fallback_radii
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
        adjacency_build = build_segmented_adjacency(
            backend=backend,
            scope_indices=scope_indices,
            point_ids=point_ids,
            pairwise_np=pairwise_np,
            radii_np=radii_np,
        )
    elif impl == "grid":
        adjacency_build = build_grid_adjacency(
            backend=backend,
            batch_points=batch,
            batch_levels=levels,
            radii=radii,
            scope_indptr=scope_indptr,
            scope_indices=scope_indices,
        )
    elif residual_mode and residual_pairwise_np is not None:
        adjacency_build = build_residual_adjacency(
            backend=backend,
            batch_size=batch_size,
            scope_indptr=scope_indptr,
            scope_indices=scope_indices,
            pairwise=pairwise,
            radii=radii_np,
            residual_pairwise=residual_pairwise_np,
        )
    else:
        adjacency_build = build_dense_adjacency(
            backend=backend,
            batch_size=batch_size,
            scope_indptr=scope_indptr,
            scope_indices=scope_indices,
            pairwise=pairwise,
            radii=radii_np,
        )

    if (
        residual_mode
        and residual_pairwise_np is not None
        and adjacency_build.csr_indptr is not None
        and adjacency_build.csr_indices is not None
    ):
        filtered_indptr, filtered_indices = filter_csr_by_radii_from_pairwise(
            np.asarray(adjacency_build.csr_indptr, dtype=np.int64),
            np.asarray(adjacency_build.csr_indices, dtype=np.int32),
            radii_np,
            residual_pairwise_np,
        )
        adjacency_build = replace(
            adjacency_build,
            csr_indptr=filtered_indptr,
            csr_indices=filtered_indices,
            total_pairs=int(filtered_indptr[-1]),
            radius_pruned=True,
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
    scope_chunk_segments = adjacency_build.scope_chunk_segments
    scope_chunk_emitted = adjacency_build.scope_chunk_emitted
    scope_chunk_max_members = adjacency_build.scope_chunk_max_members
    scope_chunk_pair_cap = adjacency_build.scope_chunk_pair_cap
    scope_chunk_pairs_before = adjacency_build.scope_chunk_pairs_before
    scope_chunk_pairs_after = adjacency_build.scope_chunk_pairs_after
    forced_selected = adjacency_build.forced_selected
    forced_dominated = adjacency_build.forced_dominated
    grid_cells = adjacency_build.grid_cells
    grid_leaders_raw = adjacency_build.grid_leaders_raw
    grid_leaders_after = adjacency_build.grid_leaders_after
    grid_local_edges = adjacency_build.grid_local_edges

    if sources.size and not radius_pruned:
        filter_start = time.perf_counter()
        if residual_mode and residual_pairwise_np is not None:
            sources_np = np.asarray(backend.to_numpy(sources), dtype=np.int64)
            targets_np = np.asarray(backend.to_numpy(targets), dtype=np.int64)
            keep_mask_np = np.zeros(sources_np.shape[0], dtype=np.bool_)
            unique_sources = np.unique(sources_np)
            for src in unique_sources:
                mask_indices = np.where(sources_np == src)[0]
                if mask_indices.size == 0:
                    continue
                tgt_batch_ids = targets_np[mask_indices]
                distances = residual_pairwise_np[src, tgt_batch_ids]
                min_radii = np.minimum(radii_np[src], radii_np[tgt_batch_ids])
                keep = distances <= (min_radii + RESIDUAL_FILTER_EPS)
                keep_mask_np[mask_indices] = keep
            keep_mask = backend.asarray(keep_mask_np, dtype=backend.xp.bool_)
            sources = sources[keep_mask]
            targets = targets[keep_mask]
            block_until_ready(keep_mask)
            adjacency_filter_seconds = time.perf_counter() - filter_start
        elif residual_mode and batch_indices_np is not None:
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
            block_until_ready(keep_mask)
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
        block_until_ready(indices)
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
        block_until_ready(indices)
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
    block_until_ready(annulus_bin_indices)
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
            scope_chunk_segments=scope_chunk_segments,
            scope_chunk_emitted=scope_chunk_emitted,
            scope_chunk_max_members=scope_chunk_max_members,
            scope_chunk_pair_cap=scope_chunk_pair_cap,
            scope_chunk_pairs_before=scope_chunk_pairs_before,
            scope_chunk_pairs_after=scope_chunk_pairs_after,
            grid_cells=grid_cells,
            grid_leaders_raw=grid_leaders_raw,
            grid_leaders_after=grid_leaders_after,
            grid_local_edges=grid_local_edges,
        ),
        forced_selected=forced_selected,
        forced_dominated=forced_dominated,
        grid_cells=grid_cells,
        grid_leaders_raw=grid_leaders_raw,
        grid_leaders_after=grid_leaders_after,
        grid_local_edges=grid_local_edges,
    )
RESIDUAL_FILTER_EPS = 1e-9
