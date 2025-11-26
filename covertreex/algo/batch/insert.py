from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math
import time

import numpy as np

from covertreex import config as cx_config
from covertreex.algo.conflict import ConflictGraph, build_conflict_graph
from covertreex.algo.mis import MISResult, batch_mis_seeds, run_mis
from covertreex.algo.order import (
    compute_batch_order,
    choose_prefix_factor,
    prepare_batch_points,
    prefix_slices,
)
from covertreex.algo.traverse import TraversalResult, traverse_collect_scopes
from covertreex.core.metrics import get_metric
from covertreex.core.persistence import (
    apply_persistence_journal,
    build_persistence_journal,
)
from covertreex.core.tree import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
)
from covertreex.diagnostics import log_operation
from covertreex.logging import get_logger
from .types import (
    BatchInsertPlan,
    BatchInsertTimings,
    LevelSummary,
    PrefixBatchGroup,
    PrefixBatchResult,
)




def plan_batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
    apply_batch_order: bool = True,
    context: cx_config.RuntimeContext | None = None,
) -> BatchInsertPlan:
    backend = backend or tree.backend
    context = context or cx_config.runtime_context()
    runtime = context.config
    batch_array, batch_permutation, batch_order_metrics = prepare_batch_points(
        backend=backend,
        batch_points=batch_points,
        runtime=runtime,
        apply_batch_order=apply_batch_order,
    )
    start = time.perf_counter()
    traversal = traverse_collect_scopes(tree, batch_array, backend=backend, context=context)
    traversal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    conflict_graph = build_conflict_graph(
        tree,
        traversal,
        batch_array,
        backend=backend,
        context=context,
    )
    conflict_graph_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mis_result = run_mis(backend, conflict_graph, seed=mis_seed, runtime=runtime)
    mis_seconds = time.perf_counter() - start
    xp = backend.xp
    independent = mis_result.independent_set
    indicator_bool = independent.astype(bool)
    forced_selected = conflict_graph.forced_selected
    if forced_selected is not None:
        forced_selected_bool = forced_selected.astype(bool)
        indicator_bool = xp.logical_or(indicator_bool, forced_selected_bool)
    forced_dominated = conflict_graph.forced_dominated
    if forced_dominated is not None:
        forced_dominated_bool = forced_dominated.astype(bool)
        indicator_bool = xp.logical_and(
            indicator_bool,
            xp.logical_not(forced_dominated_bool),
        )
    selected_indices = xp.where(indicator_bool)[0]
    dominated_indices = xp.where(xp.logical_not(indicator_bool))[0]

    levels_np = np.asarray(backend.to_numpy(traversal.levels), dtype=np.int64)
    selected_np = np.asarray(backend.to_numpy(selected_indices), dtype=np.int64)
    dominated_np = np.asarray(backend.to_numpy(dominated_indices), dtype=np.int64)
    clamped_levels = np.maximum(levels_np, 0)
    unique_levels = np.unique(clamped_levels)
    level_summaries = []
    for lvl in unique_levels:
        mask = clamped_levels == lvl
        candidate_idx = np.nonzero(mask)[0]
        if candidate_idx.size == 0:
            continue
        selected_mask = np.isin(candidate_idx, selected_np, assume_unique=False)
        selected_idx = candidate_idx[selected_mask]
        dominated_idx = candidate_idx[~selected_mask]
        level_summaries.append(
            LevelSummary(
                level=int(lvl),
                candidates=backend.asarray(candidate_idx, dtype=backend.default_int),
                selected=backend.asarray(selected_idx, dtype=backend.default_int),
                dominated=backend.asarray(dominated_idx, dtype=backend.default_int),
            )
        )

    return BatchInsertPlan(
        traversal=traversal,
        conflict_graph=conflict_graph,
        mis_result=mis_result,
        selected_indices=selected_indices,
        dominated_indices=dominated_indices,
        level_summaries=tuple(level_summaries),
        timings=BatchInsertTimings(
            traversal_seconds=traversal_seconds,
            conflict_graph_seconds=conflict_graph_seconds,
            mis_seconds=mis_seconds,
        ),
        batch_permutation=batch_permutation,
        batch_order_strategy=runtime.batch_order_strategy if apply_batch_order else "natural",
        batch_order_metrics=batch_order_metrics,
    )


def batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
    apply_batch_order: bool | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> tuple[PCCTree, BatchInsertPlan]:
    backend = backend or tree.backend
    resolved_context = context or cx_config.current_runtime_context()
    if resolved_context is None:
        resolved_context = cx_config.runtime_context()
        
    if resolved_context.config.enable_rust:
        try:
            return _rust_batch_insert(
                tree,
                batch_points,
                backend=backend,
                context=resolved_context,
            )
        except ImportError:
            # Fallback if rust not available despite config?
            # Or just fail. Config said enable_rust=True.
            # If default is auto, we should check avail.
            pass
            
    with log_operation(LOGGER, "batch_insert", context=resolved_context) as op_log:
        return _batch_insert_impl(
            op_log,
            tree,
            batch_points,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
            context=resolved_context,
        )

def _create_dummy_plan(backend: TreeBackend, batch_order_strategy: str = "rust") -> BatchInsertPlan:
    from covertreex.algo.batch.types import BatchInsertPlan, BatchInsertTimings
    from covertreex.algo.traverse import TraversalResult, TraversalTimings
    from covertreex.algo.conflict import ConflictGraph, ConflictGraphTimings
    from covertreex.algo.mis import MISResult

    xp = backend.xp
    empty_int = backend.asarray(xp.asarray([], dtype=backend.default_int))
    empty_float = backend.asarray(xp.asarray([], dtype=backend.default_float))
    indptr = backend.asarray(xp.asarray([0], dtype=backend.default_int))

    timings = BatchInsertTimings(0.0, 0.0, 0.0)
    t_timings = TraversalTimings(
        pairwise_seconds=0.0,
        mask_seconds=0.0,
        semisort_seconds=0.0,
    )
    c_timings = ConflictGraphTimings(
        pairwise_seconds=0.0,
        scope_group_seconds=0.0,
        adjacency_seconds=0.0,
        annulus_seconds=0.0,
    )

    traversal = TraversalResult(
        parents=empty_int,
        levels=empty_int,
        conflict_scopes=tuple(),
        scope_indptr=indptr,
        scope_indices=empty_int,
        timings=t_timings,
    )

    conflict_graph = ConflictGraph(
        indptr=indptr,
        indices=empty_int,
        pairwise_distances=empty_float,
        scope_indptr=indptr,
        scope_indices=empty_int,
        radii=empty_float,
        annulus_bounds=empty_float,
        annulus_bins=indptr,
        annulus_bin_indptr=indptr,
        annulus_bin_indices=empty_int,
        annulus_bin_ids=empty_int,
        timings=c_timings,
    )

    return BatchInsertPlan(
        traversal=traversal,
        conflict_graph=conflict_graph,
        mis_result=MISResult(empty_int, 0),
        selected_indices=empty_int,
        dominated_indices=empty_int,
        level_summaries=(),
        timings=timings,
        batch_permutation=None,
        batch_order_strategy=batch_order_strategy,
        batch_order_metrics={},
    )

def _rust_batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
    context: cx_config.RuntimeContext,
) -> tuple[PCCTree, BatchInsertPlan]:
    import covertreex_backend
    from covertreex.queries.utils import to_numpy_array
    from covertreex.metrics.residual.core import get_residual_backend, decode_indices
    
    batch_np = to_numpy_array(backend, batch_points, dtype=np.float32)
    points_np = to_numpy_array(backend, tree.points, dtype=np.float32)
    parents_np = to_numpy_array(backend, tree.parents, dtype=np.int64)
    children_np = to_numpy_array(backend, tree.children, dtype=np.int64)
    next_np = to_numpy_array(backend, tree.next_cache, dtype=np.int64)
    levels_np = to_numpy_array(backend, tree.top_levels, dtype=np.int32)
    
    min_level = int(tree.min_scale) if tree.min_scale is not None else -100
    max_level = int(tree.max_scale) if tree.max_scale is not None else 100
    
    wrapper = covertreex_backend.CoverTreeWrapper(
        points_np, parents_np, children_np, next_np, levels_np, min_level, max_level
    )
    
    if context.config.metric == "residual_correlation":
        host_backend = get_residual_backend()

        # Rust backend for Residual Metric expects INDICES, not coordinates.
        # Decode batch payload to dataset indices.
        batch_indices = decode_indices(host_backend, batch_np)
        v_matrix = host_backend.v_matrix
        p_diag = host_backend.p_diag
        coords = host_backend.kernel_points_f32
        rbf_var = float(getattr(host_backend, "rbf_variance", 1.0))
        rbf_ls = getattr(host_backend, "rbf_lengthscale", 1.0)
        if np.ndim(rbf_ls) == 0:
            dim = coords.shape[1]
            rbf_ls = np.full(dim, float(rbf_ls), dtype=np.float32)
        else:
            rbf_ls = np.asarray(rbf_ls, dtype=np.float32)

        chunk_size = int(getattr(context.config, "residual_chunk_size", batch_indices.shape[0]))

        # Fresh build: reuse the PCCT2 rust builder so si_cache and survivors are populated.
        if int(tree.num_points) == 0:
            batch_order = getattr(context.config, "batch_order_strategy", "hilbert") or "hilbert"
            wrapper, _node_to_dataset = covertreex_backend.build_pcct2_residual_tree(  # type: ignore
                v_matrix,
                p_diag,
                coords,
                rbf_var,
                rbf_ls,
                chunk_size,
                batch_order,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            # Incremental insert into the existing wrapper.
            batch_indices_f32 = batch_indices.astype(np.float32).reshape(-1, 1)
            wrapper.insert_residual(batch_indices_f32, v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size)
    else:
        wrapper.insert(batch_np)

    new_points = wrapper.get_points()
    new_parents = wrapper.get_parents()
    new_children = wrapper.get_children()
    new_next = wrapper.get_next_node()
    new_levels = wrapper.get_levels()
    new_min = wrapper.get_min_level()
    new_max = wrapper.get_max_level()
    new_si = np.power(2.0, new_levels.astype(np.float64) + 1.0)
    
    # Construct result
    new_tree = PCCTree(
        backend=backend,
        points=backend.asarray(new_points),
        parents=backend.asarray(new_parents),
        children=backend.asarray(new_children),
        top_levels=backend.asarray(new_levels),
        next_cache=backend.asarray(new_next),
        si_cache=backend.asarray(new_si),
        level_offsets=backend.asarray(np.zeros(1)), 
        min_scale=new_min,
        max_scale=new_max,
        stats=tree.stats, 
    )
    
    strategy = getattr(context.config, "batch_order_strategy", "rust")
    return new_tree, _create_dummy_plan(backend, batch_order_strategy=str(strategy))


def _batch_insert_impl(
    op_log: Any,
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
    mis_seed: int | None,
    apply_batch_order: bool | None,
    context: cx_config.RuntimeContext | None,
) -> tuple[PCCTree, BatchInsertPlan]:
    context = context or cx_config.runtime_context()
    runtime = context.config
    apply_order = True if apply_batch_order is None else bool(apply_batch_order)
    plan = plan_batch_insert(
        tree,
        batch_points,
        backend=backend,
        mis_seed=mis_seed,
        apply_batch_order=apply_order,
        context=context,
    )

    total_candidates = int(plan.traversal.parents.shape[0])
    selected_count = int(plan.selected_indices.size)
    dominated_count = int(plan.dominated_indices.size)
    edges = int(plan.conflict_graph.num_edges)
    mis_iterations = int(plan.mis_result.iterations)

    cache_hits = 0
    cache_denominator = 0
    if total_candidates:
        parents_np = np.asarray(
            backend.to_numpy(plan.traversal.parents), dtype=np.int64
        )
        levels_np = np.asarray(
            backend.to_numpy(plan.traversal.levels), dtype=np.int64
        )
        parent_si_np = np.asarray(backend.to_numpy(tree.si_cache), dtype=float)
        valid_parent_mask = parents_np >= 0
        if np.any(valid_parent_mask):
            cache_denominator = int(np.count_nonzero(valid_parent_mask))
            base_radii = np.power(2.0, levels_np.astype(float) + 1.0)
            si_vals = parent_si_np[parents_np[valid_parent_mask]]
            cache_hits = int(
                np.count_nonzero(
                    si_vals >= base_radii[valid_parent_mask] - 1e-12
                )
            )

    if op_log is not None:
        traversal_pairwise_ms = plan.traversal.timings.pairwise_seconds * 1e3
        traversal_mask_ms = plan.traversal.timings.mask_seconds * 1e3
        traversal_semisort_ms = plan.traversal.timings.semisort_seconds * 1e3
        conflict_timings = plan.conflict_graph.timings
        op_log.add_metadata(
            candidates=total_candidates,
            selected=selected_count,
            dominated=dominated_count,
            edges=edges,
            mis_iterations=mis_iterations,
            cache_hits=cache_hits,
            cache_total=cache_denominator,
            traversal_ms=plan.timings.traversal_seconds * 1e3,
            conflict_graph_ms=plan.timings.conflict_graph_seconds * 1e3,
            mis_ms=plan.timings.mis_seconds * 1e3,
            traversal_pairwise_ms=traversal_pairwise_ms,
            traversal_mask_ms=traversal_mask_ms,
            traversal_semisort_ms=traversal_semisort_ms,
            traversal_chain_ms=plan.traversal.timings.chain_seconds * 1e3,
            traversal_nonzero_ms=plan.traversal.timings.nonzero_seconds * 1e3,
            traversal_sort_ms=plan.traversal.timings.sort_seconds * 1e3,
            traversal_assemble_ms=plan.traversal.timings.assemble_seconds * 1e3,
            conflict_pairwise_ms=conflict_timings.pairwise_seconds * 1e3,
            conflict_scope_group_ms=conflict_timings.scope_group_seconds * 1e3,
            conflict_adjacency_ms=conflict_timings.adjacency_seconds * 1e3,
            conflict_annulus_ms=conflict_timings.annulus_seconds * 1e3,
            conflict_adj_membership_ms=conflict_timings.adjacency_membership_seconds * 1e3,
            conflict_adj_targets_ms=conflict_timings.adjacency_targets_seconds * 1e3,
            conflict_adj_scatter_ms=conflict_timings.adjacency_scatter_seconds * 1e3,
            conflict_adj_filter_ms=conflict_timings.adjacency_filter_seconds * 1e3,
            conflict_adj_sort_ms=conflict_timings.adjacency_sort_seconds * 1e3,
            conflict_adj_dedup_ms=conflict_timings.adjacency_dedup_seconds * 1e3,
            conflict_adj_extract_ms=conflict_timings.adjacency_extract_seconds * 1e3,
            conflict_adj_csr_ms=conflict_timings.adjacency_csr_seconds * 1e3,
            conflict_adj_pairs=int(conflict_timings.adjacency_total_pairs),
            conflict_adj_candidates=int(conflict_timings.adjacency_candidate_pairs),
            conflict_adj_max_group=int(conflict_timings.adjacency_max_group_size),
            conflict_scope_groups=int(conflict_timings.scope_groups),
            conflict_scope_groups_unique=int(conflict_timings.scope_groups_unique),
            conflict_scope_domination_ratio=float(conflict_timings.scope_domination_ratio),
            conflict_scope_bytes_d2h=int(conflict_timings.scope_bytes_d2h),
            conflict_scope_bytes_h2d=int(conflict_timings.scope_bytes_h2d),
            conflict_pairwise_reused=int(conflict_timings.pairwise_reused),
            traversal_scope_chunk_segments=int(plan.traversal.timings.scope_chunk_segments),
            traversal_scope_chunk_emitted=int(plan.traversal.timings.scope_chunk_emitted),
            traversal_scope_chunk_max_members=int(plan.traversal.timings.scope_chunk_max_members),
            conflict_scope_chunk_segments=int(conflict_timings.scope_chunk_segments),
            conflict_scope_chunk_emitted=int(conflict_timings.scope_chunk_emitted),
            conflict_scope_chunk_max_members=int(conflict_timings.scope_chunk_max_members),
            conflict_mis_ms=conflict_timings.mis_seconds * 1e3,
            conflict_grid_cells=int(plan.conflict_graph.grid_cells),
            conflict_grid_leaders_raw=int(plan.conflict_graph.grid_leaders_raw),
            conflict_grid_leaders_after=int(plan.conflict_graph.grid_leaders_after),
            conflict_grid_local_edges=int(plan.conflict_graph.grid_local_edges),
        )
        op_log.add_metadata(batch_order_strategy=plan.batch_order_strategy)
        if plan.batch_permutation is not None:
            op_log.add_metadata(
                batch_order_permuted=int(len(plan.batch_permutation))
            )
        for key, value in plan.batch_order_metrics.items():
            op_log.add_metadata(**{f"batch_order_{key}": float(value)})

    total_new_candidates = int(plan.selected_indices.size + plan.dominated_indices.size)
    if total_new_candidates == 0:
        return tree, plan

    xp = backend.xp
    batch = backend.asarray(batch_points, dtype=backend.default_float)
    batch = backend.device_put(batch)
    if plan.batch_permutation is not None:
        perm_backend = backend.asarray(plan.batch_permutation, dtype=backend.default_int)
        perm_backend = backend.device_put(perm_backend)
        batch = xp.take(batch, perm_backend, axis=0)
        batch = backend.device_put(batch)
    metric = get_metric()
    selected_points = batch[plan.selected_indices]
    selected_levels = plan.traversal.levels[plan.selected_indices]
    selected_levels = xp.maximum(
        selected_levels, xp.zeros_like(selected_levels, dtype=backend.default_int)
    )
    selected_parents = plan.traversal.parents[plan.selected_indices]
    selected_si = plan.conflict_graph.radii[plan.selected_indices]

    dominated_points = batch[plan.dominated_indices]
    dominated_levels = plan.traversal.levels[plan.dominated_indices]
    dominated_levels = xp.maximum(
        dominated_levels
        - xp.ones_like(dominated_levels, dtype=backend.default_int),
        xp.zeros_like(dominated_levels, dtype=backend.default_int),
    )
    dominated_parents = plan.traversal.parents[plan.dominated_indices]
    dominated_si = plan.conflict_graph.radii[plan.dominated_indices]

    inserted_points = xp.concatenate([selected_points, dominated_points], axis=0)
    inserted_levels = xp.concatenate([selected_levels, dominated_levels], axis=0)
    inserted_parents = xp.concatenate([selected_parents, dominated_parents], axis=0)
    inserted_si = xp.concatenate([selected_si, dominated_si], axis=0)

    dim = int(tree.dimension)
    selected_points_np = np.asarray(
        backend.to_numpy(selected_points), dtype=float
    )
    if selected_points_np.size:
        selected_points_np = selected_points_np.reshape(-1, dim)
    else:
        selected_points_np = np.empty((0, dim), dtype=float)
    dominated_points_np = np.asarray(
        backend.to_numpy(dominated_points), dtype=float
    )
    if dominated_points_np.size:
        dominated_points_np = dominated_points_np.reshape(-1, dim)
    else:
        dominated_points_np = np.empty((0, dim), dtype=float)
    if inserted_points.shape[0]:
        inserted_points_np = np.concatenate(
            [selected_points_np, dominated_points_np], axis=0
        )
    else:
        inserted_points_np = np.empty((0, tree.dimension), dtype=float)
    tree_points_np = np.asarray(backend.to_numpy(tree.points), dtype=float)

    selected_batch_indices = np.asarray(
        backend.to_numpy(plan.selected_indices), dtype=np.int64
    )
    dominated_batch_indices = np.asarray(
        backend.to_numpy(plan.dominated_indices), dtype=np.int64
    )
    inserted_parents_np = np.asarray(
        backend.to_numpy(inserted_parents), dtype=np.int64
    )
    batch_np = np.asarray(backend.to_numpy(batch), dtype=float)
    dominated_parent_dists_np = np.full(
        dominated_batch_indices.shape[0], np.inf, dtype=float
    )

    if dominated_batch_indices.size:
        selected_to_global: dict[int, int] = {
            int(batch_idx): int(tree.num_points + offset)
            for offset, batch_idx in enumerate(selected_batch_indices)
        }

        graph_indptr = np.asarray(
            backend.to_numpy(plan.conflict_graph.indptr), dtype=np.int64
        )
        graph_indices = np.asarray(
            backend.to_numpy(plan.conflict_graph.indices), dtype=np.int64
        )
        mis_mask = np.asarray(
            backend.to_numpy(plan.mis_result.independent_set), dtype=np.int8
        )
        pairwise_np = np.asarray(
            backend.to_numpy(plan.conflict_graph.pairwise_distances), dtype=float
        )

        num_selected = int(selected_batch_indices.size)
        for offset, batch_idx in enumerate(dominated_batch_indices):
            start = graph_indptr[batch_idx]
            end = graph_indptr[batch_idx + 1]
            neighbors = graph_indices[start:end]
            candidate: list[tuple[float, int]] = []
            for nb in neighbors:
                if mis_mask[nb] != 1:
                    continue
                parent_idx = selected_to_global.get(int(nb))
                if parent_idx is None:
                    continue
                dist = float(pairwise_np[batch_idx, int(nb)])
                candidate.append((dist, parent_idx))
            if candidate:
                candidate.sort(key=lambda item: item[0])
                inserted_parents_np[num_selected + offset] = candidate[0][1]
                dominated_parent_dists_np[offset] = candidate[0][0]

        if dominated_parent_dists_np.size:
            LOG_EPS = 1e-12
            for offset in range(dominated_parent_dists_np.shape[0]):
                if math.isfinite(dominated_parent_dists_np[offset]):
                    continue
                parent_idx = int(inserted_parents_np[num_selected + offset])
                batch_idx = int(dominated_batch_indices[offset])
                if parent_idx < 0:
                    dominated_parent_dists_np[offset] = 0.0
                    continue
                if parent_idx < tree.num_points:
                    parent_point = tree.points[parent_idx]
                    dom_point = batch[batch_idx]
                    dist_backend = metric.pointwise(backend, dom_point, parent_point)
                    dominated_parent_dists_np[offset] = float(
                        np.asarray(backend.to_numpy(dist_backend), dtype=float)
                    )
                else:
                    parent_offset = parent_idx - tree.num_points
                    if 0 <= parent_offset < num_selected:
                        parent_batch_idx = int(selected_batch_indices[parent_offset])
                        dominated_parent_dists_np[offset] = float(
                            pairwise_np[batch_idx, parent_batch_idx]
                        )
                    else:
                        dominated_parent_dists_np[offset] = 0.0

    inserted_levels_np = np.asarray(backend.to_numpy(inserted_levels), dtype=np.int64)
    parent_levels_np = np.empty_like(inserted_parents_np)
    dominated_levels_np = np.asarray(
        backend.to_numpy(dominated_levels), dtype=np.int64
    )

    total_inserted = inserted_levels_np.shape[0]
    for idx_parent, parent in enumerate(inserted_parents_np):
        if parent < 0:
            parent_levels_np[idx_parent] = 0
        elif parent < tree.num_points:
            parent_levels_np[idx_parent] = int(tree.top_levels[parent])
        else:
            offset = parent - tree.num_points
            if offset < total_inserted:
                parent_levels_np[idx_parent] = int(inserted_levels_np[offset])
            else:
                parent_levels_np[idx_parent] = 0

    selected_count = selected_batch_indices.size
    dominated_count = dominated_batch_indices.size
    if dominated_count:
        LOG_EPS = 1e-12
        for idx_dom in range(dominated_count):
            global_idx = selected_count + idx_dom
            parent_level = int(parent_levels_np[global_idx])
            candidate = int(dominated_levels_np[idx_dom])
            dist_value = (
                float(dominated_parent_dists_np[idx_dom])
                if dominated_parent_dists_np.size
                else 0.0
            )
            distance_level = 0
            if dist_value > LOG_EPS:
                log_val = math.log(dist_value, 2) - 1e-12
                distance_level = int(math.floor(log_val))
                if distance_level < 0:
                    distance_level = 0
            max_parent_level = parent_level - 1
            new_level = min(candidate, max_parent_level, distance_level)
            if new_level < 0:
                new_level = 0
            inserted_levels_np[selected_count + idx_dom] = new_level

    inserted_si_np = np.asarray(backend.to_numpy(inserted_si), dtype=float)
    if inserted_si_np.size:
        level_floor = np.maximum(inserted_levels_np.astype(np.float64), 0.0)
        base_radii_np = np.power(2.0, level_floor + 1.0)
        finite_mask = np.isfinite(inserted_si_np)
        inserted_si_np = np.where(finite_mask, inserted_si_np, base_radii_np)
        if runtime.metric == "residual_correlation":
            inserted_si_np = np.maximum(
                inserted_si_np,
                float(runtime.residual_radius_floor),
            )

    journal = build_persistence_journal(
        tree,
        backend=backend,
        inserted_points=inserted_points_np,
        inserted_levels=inserted_levels_np,
        inserted_parents=inserted_parents_np,
        inserted_si=inserted_si_np,
    )

    updated_tree = apply_persistence_journal(
        tree,
        journal,
        backend=backend,
        context=context,
    )

    stats = TreeLogStats(
        num_batches=tree.stats.num_batches + 1,
        num_insertions=tree.stats.num_insertions + total_inserted,
        num_deletions=tree.stats.num_deletions,
        num_conflicts_resolved=tree.stats.num_conflicts_resolved
        + int(plan.conflict_graph.num_edges // 2),
    )

    new_tree = updated_tree.replace(stats=stats)

    return new_tree, plan


def batch_insert_prefix_doubling(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
    shuffle_seed: int | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> tuple[PCCTree, PrefixBatchResult]:
    """Insert a batch using prefix-doubling sub-batches.

    Randomly permutes `batch_points`, then processes prefix groups of doubling
    size (1, 2, 4, …) to mirror Algorithm 4 from Gu et al. Returns the final
    tree together with the permutation metadata for downstream inspection."""

    backend = backend or tree.backend
    context = context or cx_config.runtime_context()
    runtime = context.config
    batch_np = np.asarray(backend.to_numpy(batch_points))
    batch_size = batch_np.shape[0]
    if batch_size == 0:
        empty_perm = backend.asarray([], dtype=backend.default_int)
        return tree, PrefixBatchResult(
            permutation=empty_perm,
            groups=tuple(),
            order_strategy=runtime.batch_order_strategy,
            order_metrics={},
        )
    if shuffle_seed is not None:
        order_seed = shuffle_seed
    else:
        seed_pack = runtime.seeds
        order_seed = seed_pack.resolved("batch_order", fallback=seed_pack.resolved("mis"))
    order_points = np.asarray(batch_np, dtype=np.float64)
    order_result = compute_batch_order(
        order_points,
        strategy=runtime.batch_order_strategy,
        seed=order_seed,
    )
    if order_result.permutation is None:
        permutation = np.arange(batch_size, dtype=np.int64)
        order_metrics = {}
    else:
        permutation = order_result.permutation
        order_metrics = dict(order_result.metrics)
    permuted = batch_np[permutation]

    current_tree = tree
    groups: list[PrefixBatchGroup] = []
    schedule = runtime.prefix_schedule
    if schedule == "doubling":
        slices = prefix_slices(batch_size)
        seeds: Tuple[int, ...] = batch_mis_seeds(
            len(slices),
            seed=mis_seed,
            runtime=runtime,
        )
        for idx, (start, end) in enumerate(slices):
            sub_batch = permuted[start:end]
            sub_seed: int | None
            if seeds:
                sub_seed = int(seeds[idx])
            else:
                sub_seed = None
            current_tree, plan = batch_insert(
                current_tree,
                sub_batch,
                backend=backend,
                mis_seed=sub_seed,
                apply_batch_order=False,
                context=context,
            )
            dom_ratio = float(plan.conflict_graph.timings.scope_domination_ratio)
            group_indices = permutation[start:end]
            groups.append(
                PrefixBatchGroup(
                    permutation_indices=backend.asarray(
                        group_indices, dtype=backend.default_int
                    ),
                    plan=plan,
                    prefix_factor=2.0,
                    domination_ratio=dom_ratio,
                )
            )
    else:
        seeds = batch_mis_seeds(
            batch_size,
            seed=mis_seed,
            runtime=runtime,
        )
        seed_iter = iter(seeds)
        start = 0
        current_size = 1
        while start < batch_size:
            end = min(batch_size, start + current_size)
            sub_batch = permuted[start:end]
            sub_seed = next(seed_iter, None)
            current_tree, plan = batch_insert(
                current_tree,
                sub_batch,
                backend=backend,
                mis_seed=sub_seed,
                apply_batch_order=False,
                context=context,
            )
            dom_ratio = float(plan.conflict_graph.timings.scope_domination_ratio)
            factor = choose_prefix_factor(runtime, dom_ratio)
            group_indices = permutation[start:end]
            groups.append(
                PrefixBatchGroup(
                    permutation_indices=backend.asarray(
                        group_indices, dtype=backend.default_int
                    ),
                    plan=plan,
                    prefix_factor=factor,
                    domination_ratio=dom_ratio,
                )
            )
            start = end
            remaining = batch_size - start
            if remaining <= 0:
                break
            next_size = int(round(current_size * max(factor, 0.1)))
            if next_size <= 0:
                next_size = 1
            current_size = min(next_size, remaining)

    return current_tree, PrefixBatchResult(
        permutation=backend.asarray(permutation, dtype=backend.default_int),
        groups=tuple(groups),
        order_strategy=runtime.batch_order_strategy,
        order_metrics=order_metrics,
    )
LOGGER = get_logger("algo.batch_insert")
