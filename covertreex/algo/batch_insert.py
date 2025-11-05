from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import math
import time

import numpy as np

from covertreex.algo.conflict_graph import ConflictGraph, build_conflict_graph
from covertreex.algo.mis import MISResult, batch_mis_seeds, run_mis
from covertreex.algo.traverse import TraversalResult, traverse_collect_scopes
from covertreex.core.metrics import get_metric
from covertreex.core.persistence import SliceUpdate, clone_tree_with_updates
from covertreex.core.tree import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
    compute_level_offsets,
)
from covertreex.logging import get_logger
from covertreex.diagnostics import log_operation


@dataclass(frozen=True)
class BatchInsertPlan:
    traversal: TraversalResult
    conflict_graph: ConflictGraph
    mis_result: MISResult
    selected_indices: Any
    dominated_indices: Any
    level_summaries: tuple["LevelSummary", ...]
    timings: "BatchInsertTimings"


@dataclass(frozen=True)
class LevelSummary:
    level: int
    candidates: Any
    selected: Any
    dominated: Any


@dataclass(frozen=True)
class PrefixBatchGroup:
    permutation_indices: Any
    plan: BatchInsertPlan


@dataclass(frozen=True)
class PrefixBatchResult:
    permutation: Any
    groups: Tuple[PrefixBatchGroup, ...]


@dataclass(frozen=True)
class BatchInsertTimings:
    traversal_seconds: float
    conflict_graph_seconds: float
    mis_seconds: float
def plan_batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
) -> BatchInsertPlan:
    backend = backend or tree.backend
    start = time.perf_counter()
    traversal = traverse_collect_scopes(tree, batch_points, backend=backend)
    traversal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    conflict_graph = build_conflict_graph(tree, traversal, batch_points, backend=backend)
    conflict_graph_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mis_result = run_mis(backend, conflict_graph, seed=mis_seed)
    mis_seconds = time.perf_counter() - start
    xp = backend.xp
    independent = mis_result.independent_set
    indicator_bool = independent.astype(bool)
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
    )


def batch_insert(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
) -> tuple[PCCTree, BatchInsertPlan]:
    backend = backend or tree.backend
    with log_operation(LOGGER, "batch_insert") as op_log:
        return _batch_insert_impl(
            op_log,
            tree,
            batch_points,
            backend=backend,
            mis_seed=mis_seed,
        )


def _batch_insert_impl(
    op_log: Any,
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
    mis_seed: int | None,
) -> tuple[PCCTree, BatchInsertPlan]:
    plan = plan_batch_insert(tree, batch_points, backend=backend, mis_seed=mis_seed)

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
            conflict_mis_ms=conflict_timings.mis_seconds * 1e3,
        )

    total_new_candidates = int(plan.selected_indices.size + plan.dominated_indices.size)
    if total_new_candidates == 0:
        return tree, plan

    xp = backend.xp
    batch = backend.asarray(batch_points, dtype=backend.default_float)
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

        inserted_parents = backend.asarray(
            inserted_parents_np, dtype=backend.default_int
        )

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

    inserted_levels = backend.asarray(inserted_levels_np, dtype=backend.default_int)

    base_index = tree.num_points
    append_slice = slice(base_index, base_index + total_inserted)

    points_updates = [
        SliceUpdate(index=(append_slice, slice(None)), values=inserted_points)
    ]
    top_level_updates = [SliceUpdate(index=(append_slice,), values=inserted_levels)]
    parent_updates: list[SliceUpdate] = [
        SliceUpdate(index=(append_slice,), values=inserted_parents)
    ]
    si_cache_updates = [SliceUpdate(index=(append_slice,), values=inserted_si)]

    default_child_block = xp.full((total_inserted,), -1, dtype=backend.default_int)
    child_updates: list[SliceUpdate] = [
        SliceUpdate(index=(append_slice,), values=default_child_block)
    ]
    default_next_block = xp.full((total_inserted,), -1, dtype=backend.default_int)
    next_updates: list[SliceUpdate] = [
        SliceUpdate(index=(append_slice,), values=default_next_block)
    ]

    tree_children_np = np.asarray(backend.to_numpy(tree.children), dtype=np.int64)

    current_children: dict[int, int] = {}

    def _parent_state(parent_idx: int) -> int:
        if parent_idx not in current_children:
            if parent_idx < tree.num_points:
                prev_child = int(tree_children_np[parent_idx])
            else:
                prev_child = -1
            current_children[parent_idx] = prev_child
        return current_children[parent_idx]

    for offset, parent in enumerate(inserted_parents_np):
        global_idx = base_index + offset
        current_children.setdefault(global_idx, -1)
        if parent < 0:
            continue

        prev_child = _parent_state(parent)
        child_updates.append(SliceUpdate(index=(parent,), values=int(global_idx)))
        next_updates.append(
            SliceUpdate(index=(global_idx,), values=int(prev_child if prev_child >= 0 else -1))
        )
        current_children[parent] = int(global_idx)

    combined_top_levels = backend.asarray(
        np.concatenate(
            [
                np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int64),
                inserted_levels_np,
            ]
        ),
        dtype=backend.default_int,
    )
    new_level_offsets = compute_level_offsets(backend, combined_top_levels)
    level_offset_updates = [
        SliceUpdate(index=(slice(0, new_level_offsets.shape[0]),), values=new_level_offsets)
    ]

    updated_tree = clone_tree_with_updates(
        tree,
        points_updates=points_updates,
        top_level_updates=top_level_updates,
        parent_updates=parent_updates,
        child_updates=child_updates,
        level_offset_updates=level_offset_updates,
        si_cache_updates=si_cache_updates,
        next_cache_updates=next_updates,
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


def _prefix_slices(length: int) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    size = 1
    start = 0
    while start < length:
        end = min(start + size, length)
        slices.append((start, end))
        start = end
        size = min(size * 2, length - start if length - start > 0 else size)
    return slices


def batch_insert_prefix_doubling(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: Optional[TreeBackend] = None,
    mis_seed: int | None = None,
    shuffle_seed: int | None = None,
) -> tuple[PCCTree, PrefixBatchResult]:
    """Insert a batch using prefix-doubling sub-batches.

    Randomly permutes `batch_points`, then processes prefix groups of doubling
    size (1, 2, 4, …) to mirror Algorithm 4 from Gu et al. Returns the final
    tree together with the permutation metadata for downstream inspection."""

    backend = backend or tree.backend
    batch_np = np.asarray(backend.to_numpy(batch_points))
    batch_size = batch_np.shape[0]
    if batch_size == 0:
        empty_perm = backend.asarray([], dtype=backend.default_int)
        return tree, PrefixBatchResult(permutation=empty_perm, groups=tuple())

    rng = np.random.default_rng(shuffle_seed)
    permutation = np.arange(batch_size, dtype=np.int64)
    rng.shuffle(permutation)

    permuted = batch_np[permutation]
    slices = _prefix_slices(batch_size)

    current_tree = tree
    groups: list[PrefixBatchGroup] = []
    seeds: Tuple[int, ...] = batch_mis_seeds(len(slices), seed=mis_seed)
    for idx, (start, end) in enumerate(slices):
        sub_batch = permuted[start:end]
        sub_seed: int | None
        if seeds:
            sub_seed = int(seeds[idx])
        else:
            sub_seed = None
        current_tree, plan = batch_insert(
            current_tree, sub_batch, backend=backend, mis_seed=sub_seed
        )
        group_indices = permutation[start:end]
        groups.append(
            PrefixBatchGroup(
                permutation_indices=backend.asarray(
                    group_indices, dtype=backend.default_int
                ),
                plan=plan,
            )
        )

    return current_tree, PrefixBatchResult(
        permutation=backend.asarray(permutation, dtype=backend.default_int),
        groups=tuple(groups),
    )
LOGGER = get_logger("algo.batch_insert")
