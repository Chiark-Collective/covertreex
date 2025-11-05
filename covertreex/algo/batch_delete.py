from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from covertreex import config as cx_config
from covertreex.algo.batch_insert import plan_batch_insert
from covertreex.algo.mis import batch_mis_seeds
from covertreex.core.tree import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
    compute_level_offsets,
)
from covertreex.logging import get_logger
from covertreex.diagnostics import log_operation


LOGGER = get_logger("algo.batch_delete")


@dataclass(frozen=True)
class DeleteLevelSummary:
    level: int
    uncovered: Any
    promoted: Any
    reattached: Any
    promoted_levels: Any
    reattached_levels: Any
    mis_iterations: int


@dataclass(frozen=True)
class BatchDeletePlan:
    removed_indices: Any
    retained_count: int
    reattach_indices: Any
    level_summaries: Tuple[DeleteLevelSummary, ...]
    reattached_order: Any
    created_new_root: bool


@dataclass
class _DeleteLevelSummaryData:
    level: int
    uncovered: np.ndarray
    promoted: np.ndarray
    reattached: np.ndarray
    promoted_levels: np.ndarray
    reattached_levels: np.ndarray
    mis_iterations: int


@dataclass
class _DeleteComputation:
    removed_indices: np.ndarray
    retained_count: int
    reattach_indices: np.ndarray
    level_summaries: Tuple[_DeleteLevelSummaryData, ...]
    reattached_order: np.ndarray
    points_final: np.ndarray
    top_levels_final: np.ndarray
    parents_final: np.ndarray
    si_final: np.ndarray
    children_final: np.ndarray
    next_final: np.ndarray
    level_offsets_final: np.ndarray
    conflicts_resolved: int
    created_new_root: bool


def _build_child_links_from_parents(parents_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = parents_np.shape[0]
    children = np.full(n, -1, dtype=np.int64)
    next_cache = np.full(n, -1, dtype=np.int64)
    for child in range(n - 1, -1, -1):
        parent = int(parents_np[child])
        if parent >= 0:
            next_cache[child] = children[parent]
            children[parent] = child
    return children, next_cache


def _ensure_2d(points: np.ndarray, dimension: int) -> np.ndarray:
    if points.ndim == 2:
        return points
    if dimension == 0:
        return points.reshape(-1)
    return points.reshape(points.shape[0], dimension)


def _collect_descendants(parents_np: np.ndarray) -> List[List[int]]:
    num_points = parents_np.shape[0]
    children: List[List[int]] = [list() for _ in range(num_points)]
    for child_idx in range(num_points):
        parent = int(parents_np[child_idx])
        if 0 <= parent < num_points:
            children[parent].append(child_idx)
    return children


def _compute_delete_computation(
    tree: PCCTree,
    remove_indices: Any,
    *,
    backend: Optional[TreeBackend] = None,
) -> _DeleteComputation:
    backend = backend or tree.backend
    runtime = cx_config.runtime_config()

    indices = backend.asarray(remove_indices, dtype=backend.default_int)
    indices_np = np.sort(np.asarray(backend.to_numpy(indices), dtype=np.int64))

    if indices_np.size == 0:
        empty = np.asarray([], dtype=np.int64)
        points_np = np.asarray(backend.to_numpy(tree.points), dtype=float)
        top_levels_np = np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int64)
        parents_np = np.asarray(backend.to_numpy(tree.parents), dtype=np.int64)
        si_np = np.asarray(backend.to_numpy(tree.si_cache), dtype=float)
        children_np = np.asarray(backend.to_numpy(tree.children), dtype=np.int64)
        next_np = np.asarray(backend.to_numpy(tree.next_cache), dtype=np.int64)
        level_offsets_np = np.asarray(backend.to_numpy(tree.level_offsets), dtype=np.int64)
        return _DeleteComputation(
            removed_indices=empty,
            retained_count=tree.num_points,
            reattach_indices=empty,
            level_summaries=tuple(),
            reattached_order=empty,
            points_final=points_np,
            top_levels_final=top_levels_np,
            parents_final=parents_np,
            si_final=si_np,
            children_final=children_np,
            next_final=next_np,
            level_offsets_final=level_offsets_np,
            conflicts_resolved=0,
            created_new_root=False,
        )

    if np.unique(indices_np).shape[0] != indices_np.shape[0]:
        raise ValueError("Duplicate deletion indices are not allowed.")
    if np.any(indices_np < 0) or np.any(indices_np >= tree.num_points):
        raise ValueError("Deletion index out of range.")

    points_np = np.asarray(backend.to_numpy(tree.points), dtype=float)
    top_levels_np = np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int64)
    parents_np = np.asarray(backend.to_numpy(tree.parents), dtype=np.int64)
    si_np = np.asarray(backend.to_numpy(tree.si_cache), dtype=float)

    num_points = tree.num_points
    removed_mask = np.zeros(num_points, dtype=bool)
    removed_mask[indices_np] = True

    children_lists = _collect_descendants(parents_np)

    reattach_mask = np.zeros(num_points, dtype=bool)
    queue: List[int] = indices_np.tolist()
    while queue:
        node = queue.pop()
        for child in children_lists[node]:
            if removed_mask[child] or reattach_mask[child]:
                continue
            reattach_mask[child] = True
            queue.append(child)

    reattach_nodes = np.sort(np.where(reattach_mask)[0])
    base_mask = (~removed_mask) & (~reattach_mask)
    base_nodes = np.sort(np.where(base_mask)[0])
    retained_count = int(base_nodes.shape[0])

    base_index_map = -np.ones(num_points, dtype=np.int64)
    for new_idx, old_idx in enumerate(base_nodes):
        base_index_map[old_idx] = new_idx

    dimension = tree.dimension
    if dimension == 0 and points_np.ndim >= 2:
        dimension = int(points_np.shape[1])
    if retained_count:
        base_points_np = points_np[base_nodes]
        base_points_np = _ensure_2d(base_points_np, dimension)
        base_top_levels_np = top_levels_np[base_nodes]
        base_si_np = si_np[base_nodes]
        base_parents_np = np.full(retained_count, -1, dtype=np.int64)
        for new_idx, old_idx in enumerate(base_nodes):
            parent = int(parents_np[old_idx])
            if parent >= 0:
                mapped = base_index_map[parent]
                base_parents_np[new_idx] = int(mapped) if mapped >= 0 else -1
    else:
        base_points_np = np.empty((0, dimension), dtype=float)
        base_top_levels_np = np.empty((0,), dtype=np.int64)
        base_si_np = np.empty((0,), dtype=float)
        base_parents_np = np.empty((0,), dtype=np.int64)

    base_children_np, base_next_np = _build_child_links_from_parents(base_parents_np)

    if retained_count:
        base_level_offsets_backend = compute_level_offsets(
            backend, backend.asarray(base_top_levels_np, dtype=backend.default_int)
        )
        base_level_offsets_np = np.asarray(
            backend.to_numpy(base_level_offsets_backend), dtype=np.int64
        )
    else:
        base_level_offsets_np = np.asarray([0], dtype=np.int64)
        base_level_offsets_backend = backend.asarray(base_level_offsets_np, dtype=backend.default_int)

    current_points_np = base_points_np
    current_top_levels_np = base_top_levels_np
    current_parents_np = base_parents_np
    current_si_np = base_si_np
    current_children_np = base_children_np
    current_next_np = base_next_np
    current_level_offsets_np = base_level_offsets_np

    current_tree = PCCTree(
        points=backend.asarray(current_points_np, dtype=backend.default_float),
        top_levels=backend.asarray(current_top_levels_np, dtype=backend.default_int),
        parents=backend.asarray(current_parents_np, dtype=backend.default_int),
        children=backend.asarray(current_children_np, dtype=backend.default_int),
        level_offsets=backend.asarray(current_level_offsets_np, dtype=backend.default_int),
        si_cache=backend.asarray(current_si_np, dtype=backend.default_float),
        next_cache=backend.asarray(current_next_np, dtype=backend.default_int),
        stats=tree.stats,
        backend=backend,
    )

    level_nodes: Dict[int, List[int]] = {}
    for node in reattach_nodes:
        lvl = int(max(top_levels_np[node], 0))
        level_nodes.setdefault(lvl, []).append(int(node))

    if level_nodes:
        unique_levels = sorted(level_nodes.keys())
    else:
        unique_levels = []

    seeds = batch_mis_seeds(len(unique_levels), seed=runtime.mis_seed)

    level_summaries: List[_DeleteLevelSummaryData] = []
    ordered_original: List[int] = []
    total_conflicts_resolved = 0
    created_new_root = False

    for level_idx, level in enumerate(unique_levels):
        group_nodes = np.asarray(level_nodes[level], dtype=np.int64)
        if group_nodes.size == 0:
            continue

        group_points = points_np[group_nodes]
        group_points = _ensure_2d(group_points, dimension)
        batch_points = backend.asarray(group_points, dtype=backend.default_float)

        mis_seed = seeds[level_idx] if level_idx < len(seeds) else None
        plan = plan_batch_insert(current_tree, batch_points, backend=backend, mis_seed=mis_seed)

        traversal_parents_np = np.asarray(
            backend.to_numpy(plan.traversal.parents), dtype=np.int64
        )
        traversal_levels_np = np.asarray(
            backend.to_numpy(plan.traversal.levels), dtype=np.int64
        )
        conflict_radii_np = np.asarray(
            backend.to_numpy(plan.conflict_graph.radii), dtype=float
        )
        selected_batch_indices = np.asarray(
            backend.to_numpy(plan.selected_indices), dtype=np.int64
        )
        dominated_batch_indices = np.asarray(
            backend.to_numpy(plan.dominated_indices), dtype=np.int64
        )
        graph_indptr = np.asarray(
            backend.to_numpy(plan.conflict_graph.indptr), dtype=np.int64
        )
        graph_indices = np.asarray(
            backend.to_numpy(plan.conflict_graph.indices), dtype=np.int64
        )
        mis_mask = np.asarray(
            backend.to_numpy(plan.mis_result.independent_set), dtype=np.int64
        )

        group_np = np.asarray(group_points, dtype=float)
        current_size = current_points_np.shape[0]

        selected_points_np = group_np[selected_batch_indices] if selected_batch_indices.size else np.empty((0, dimension), dtype=float)
        dominated_points_np = group_np[dominated_batch_indices] if dominated_batch_indices.size else np.empty((0, dimension), dtype=float)

        selected_levels_np = traversal_levels_np[selected_batch_indices] if selected_batch_indices.size else np.empty((0,), dtype=np.int64)
        dominated_levels_np = traversal_levels_np[dominated_batch_indices] if dominated_batch_indices.size else np.empty((0,), dtype=np.int64)

        if selected_levels_np.size:
            selected_levels_np = np.maximum(selected_levels_np, 0)
        if dominated_levels_np.size:
            dominated_levels_np = np.maximum(dominated_levels_np - 1, 0)

        selected_parents_np = traversal_parents_np[selected_batch_indices] if selected_batch_indices.size else np.empty((0,), dtype=np.int64)
        dominated_parents_np = traversal_parents_np[dominated_batch_indices] if dominated_batch_indices.size else np.empty((0,), dtype=np.int64)

        selected_si_np = conflict_radii_np[selected_batch_indices] if selected_batch_indices.size else np.empty((0,), dtype=float)
        dominated_si_np = conflict_radii_np[dominated_batch_indices] if dominated_batch_indices.size else np.empty((0,), dtype=float)

        inserted_points_np = np.concatenate([selected_points_np, dominated_points_np], axis=0)
        inserted_levels_np = np.concatenate([selected_levels_np, dominated_levels_np], axis=0)
        inserted_parents_np = np.concatenate([selected_parents_np, dominated_parents_np], axis=0)
        inserted_si_np = np.concatenate([selected_si_np, dominated_si_np], axis=0)

        num_selected = selected_batch_indices.size
        num_dominated = dominated_batch_indices.size

        if num_dominated:
            selected_to_global: Dict[int, int] = {
                int(batch_idx): current_size + offset
                for offset, batch_idx in enumerate(selected_batch_indices)
            }
            dominated_parent_dists_np = np.full(num_dominated, np.inf, dtype=float)
            for offset, batch_idx in enumerate(dominated_batch_indices):
                start = int(graph_indptr[batch_idx])
                end = int(graph_indptr[batch_idx + 1])
                neighbors = graph_indices[start:end]
                candidates: List[Tuple[float, int]] = []
                for nb in neighbors:
                    if mis_mask[nb] != 1:
                        continue
                    parent_idx = selected_to_global.get(int(nb))
                    if parent_idx is None:
                        continue
                    dist = float(np.linalg.norm(group_np[batch_idx] - group_np[int(nb)]))
                    candidates.append((dist, parent_idx))
                if candidates:
                    candidates.sort(key=lambda item: item[0])
                    inserted_parents_np[num_selected + offset] = candidates[0][1]
                    dominated_parent_dists_np[offset] = candidates[0][0]

            for offset in range(num_dominated):
                if math.isfinite(float(dominated_parent_dists_np[offset])):
                    continue
                parent_idx = int(inserted_parents_np[num_selected + offset])
                batch_idx = int(dominated_batch_indices[offset])
                if parent_idx < 0:
                    dominated_parent_dists_np[offset] = 0.0
                    continue
                if parent_idx < current_size:
                    dominated_parent_dists_np[offset] = float(
                        np.linalg.norm(group_np[batch_idx] - current_points_np[parent_idx])
                    )
                else:
                    parent_offset = parent_idx - current_size
                    if 0 <= parent_offset < inserted_points_np.shape[0]:
                        dominated_parent_dists_np[offset] = float(
                            np.linalg.norm(group_np[batch_idx] - inserted_points_np[parent_offset])
                        )
                    else:
                        dominated_parent_dists_np[offset] = 0.0
        else:
            dominated_parent_dists_np = np.empty((0,), dtype=float)

        parent_levels_np = np.zeros(inserted_parents_np.shape[0], dtype=np.int64)
        prior_top_levels_np = current_top_levels_np
        for idx_parent, parent in enumerate(inserted_parents_np):
            if parent < 0:
                parent_levels_np[idx_parent] = 0
            elif parent < current_size:
                parent_levels_np[idx_parent] = int(prior_top_levels_np[parent])
            else:
                offset = parent - current_size
                if 0 <= offset < inserted_levels_np.shape[0]:
                    parent_levels_np[idx_parent] = int(inserted_levels_np[offset])
                else:
                    parent_levels_np[idx_parent] = 0

        if num_dominated:
            LOG_EPS = 1e-12
            for idx_dom in range(num_dominated):
                global_idx = num_selected + idx_dom
                parent_level = int(parent_levels_np[global_idx])
                candidate = int(inserted_levels_np[global_idx])
                dist_value = float(dominated_parent_dists_np[idx_dom])
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
                inserted_levels_np[global_idx] = new_level

        if inserted_points_np.size:
            current_points_np = (
                np.concatenate([current_points_np, inserted_points_np], axis=0)
                if current_points_np.size
                else inserted_points_np
            )
            current_top_levels_np = (
                np.concatenate([current_top_levels_np, inserted_levels_np], axis=0)
                if current_top_levels_np.size
                else inserted_levels_np
            )
            current_parents_np = (
                np.concatenate([current_parents_np, inserted_parents_np], axis=0)
                if current_parents_np.size
                else inserted_parents_np
            )
            current_si_np = (
                np.concatenate([current_si_np, inserted_si_np], axis=0)
                if current_si_np.size
                else inserted_si_np
            )

            current_children_np, current_next_np = _build_child_links_from_parents(
                current_parents_np
            )
            level_offsets_backend = compute_level_offsets(
                backend, backend.asarray(current_top_levels_np, dtype=backend.default_int)
            )
            current_level_offsets_np = np.asarray(
                backend.to_numpy(level_offsets_backend), dtype=np.int64
            )

            current_tree = PCCTree(
                points=backend.asarray(current_points_np, dtype=backend.default_float),
                top_levels=backend.asarray(current_top_levels_np, dtype=backend.default_int),
                parents=backend.asarray(current_parents_np, dtype=backend.default_int),
                children=backend.asarray(current_children_np, dtype=backend.default_int),
                level_offsets=backend.asarray(current_level_offsets_np, dtype=backend.default_int),
                si_cache=backend.asarray(current_si_np, dtype=backend.default_float),
                next_cache=backend.asarray(current_next_np, dtype=backend.default_int),
                stats=tree.stats,
                backend=backend,
            )

        selected_original = group_nodes[selected_batch_indices] if num_selected else np.empty((0,), dtype=np.int64)
        dominated_original = group_nodes[dominated_batch_indices] if num_dominated else np.empty((0,), dtype=np.int64)
        ordered_batch = np.concatenate([selected_original, dominated_original], axis=0)
        ordered_original.extend(int(idx) for idx in ordered_batch.tolist())

        promoted_levels_final = inserted_levels_np[:num_selected] if num_selected else np.empty((0,), dtype=np.int64)
        reattached_levels_final = inserted_levels_np[num_selected:] if num_dominated else np.empty((0,), dtype=np.int64)

        level_summaries.append(
            _DeleteLevelSummaryData(
                level=int(level),
                uncovered=group_nodes,
                promoted=selected_original,
                reattached=dominated_original,
                promoted_levels=promoted_levels_final,
                reattached_levels=reattached_levels_final,
                mis_iterations=int(plan.mis_result.iterations),
            )
        )

        total_conflicts_resolved += int(plan.conflict_graph.num_edges // 2)
        if inserted_parents_np.size and np.any(inserted_parents_np < 0):
            created_new_root = True

    final_level_offsets_backend = compute_level_offsets(
        backend, backend.asarray(current_top_levels_np, dtype=backend.default_int)
    )
    final_level_offsets_np = np.asarray(
        backend.to_numpy(final_level_offsets_backend), dtype=np.int64
    )

    reattached_order_np = np.asarray(ordered_original, dtype=np.int64)

    return _DeleteComputation(
        removed_indices=indices_np,
        retained_count=retained_count,
        reattach_indices=reattach_nodes,
        level_summaries=tuple(level_summaries),
        reattached_order=reattached_order_np,
        points_final=current_points_np,
        top_levels_final=current_top_levels_np,
        parents_final=current_parents_np,
        si_final=current_si_np,
        children_final=current_children_np,
        next_final=current_next_np,
        level_offsets_final=final_level_offsets_np,
        conflicts_resolved=total_conflicts_resolved,
        created_new_root=created_new_root,
    )


def _build_plan_from_computation(
    backend: TreeBackend,
    computation: _DeleteComputation,
) -> BatchDeletePlan:
    removed_backend = backend.asarray(
        computation.removed_indices, dtype=backend.default_int
    )
    reattach_backend = backend.asarray(
        computation.reattach_indices, dtype=backend.default_int
    )
    reattached_order_backend = backend.asarray(
        computation.reattached_order, dtype=backend.default_int
    )

    summaries: List[DeleteLevelSummary] = []
    for summary in computation.level_summaries:
        uncovered = backend.asarray(summary.uncovered, dtype=backend.default_int)
        promoted = backend.asarray(summary.promoted, dtype=backend.default_int)
        reattached = backend.asarray(summary.reattached, dtype=backend.default_int)
        promoted_levels = backend.asarray(summary.promoted_levels, dtype=backend.default_int)
        reattached_levels = backend.asarray(summary.reattached_levels, dtype=backend.default_int)
        summaries.append(
            DeleteLevelSummary(
                level=summary.level,
                uncovered=uncovered,
                promoted=promoted,
                reattached=reattached,
                promoted_levels=promoted_levels,
                reattached_levels=reattached_levels,
                mis_iterations=summary.mis_iterations,
            )
        )

    return BatchDeletePlan(
        removed_indices=removed_backend,
        retained_count=computation.retained_count,
        reattach_indices=reattach_backend,
        level_summaries=tuple(summaries),
        reattached_order=reattached_order_backend,
        created_new_root=computation.created_new_root,
    )


def plan_batch_delete(
    tree: PCCTree,
    remove_indices: Any,
    *,
    backend: Optional[TreeBackend] = None,
) -> BatchDeletePlan:
    backend = backend or tree.backend
    computation = _compute_delete_computation(tree, remove_indices, backend=backend)
    return _build_plan_from_computation(backend, computation)


def _materialise_tree_from_computation(
    tree: PCCTree,
    backend: TreeBackend,
    computation: _DeleteComputation,
) -> PCCTree:
    points_final = backend.asarray(
        computation.points_final, dtype=backend.default_float
    )
    top_levels_final = backend.asarray(
        computation.top_levels_final, dtype=backend.default_int
    )
    parents_final = backend.asarray(
        computation.parents_final, dtype=backend.default_int
    )
    children_final = backend.asarray(
        computation.children_final, dtype=backend.default_int
    )
    next_final = backend.asarray(
        computation.next_final, dtype=backend.default_int
    )
    si_final = backend.asarray(
        computation.si_final, dtype=backend.default_float
    )
    level_offsets_final = backend.asarray(
        computation.level_offsets_final, dtype=backend.default_int
    )

    stats = TreeLogStats(
        num_batches=tree.stats.num_batches + 1,
        num_insertions=tree.stats.num_insertions,
        num_deletions=tree.stats.num_deletions + int(computation.removed_indices.shape[0]),
        num_conflicts_resolved=tree.stats.num_conflicts_resolved + computation.conflicts_resolved,
    )

    return PCCTree(
        points=points_final,
        top_levels=top_levels_final,
        parents=parents_final,
        children=children_final,
        level_offsets=level_offsets_final,
        si_cache=si_final,
        next_cache=next_final,
        stats=stats,
        backend=backend,
    )


def batch_delete(
    tree: PCCTree,
    remove_indices: Any,
    *,
    backend: Optional[TreeBackend] = None,
) -> Tuple[PCCTree, BatchDeletePlan]:
    backend = backend or tree.backend
    with log_operation(LOGGER, "batch_delete") as op_log:
        return _batch_delete_impl(
            op_log,
            tree,
            remove_indices,
            backend=backend,
        )


def _batch_delete_impl(
    op_log: Any,
    tree: PCCTree,
    remove_indices: Any,
    *,
    backend: TreeBackend,
) -> Tuple[PCCTree, BatchDeletePlan]:
    computation = _compute_delete_computation(tree, remove_indices, backend=backend)
    plan = _build_plan_from_computation(backend, computation)

    total_removed = int(plan.removed_indices.shape[0])
    promoted = sum(int(summary.promoted.shape[0]) for summary in plan.level_summaries)
    reattached = sum(
        int(summary.reattached.shape[0]) for summary in plan.level_summaries
    )
    total_levels = len(plan.level_summaries)
    max_iterations = max(
        (summary.mis_iterations for summary in plan.level_summaries), default=0
    )

    if op_log is not None:
        op_log.add_metadata(
            removed=total_removed,
            promoted=promoted,
            reattached=reattached,
            levels=total_levels,
            mis_iterations_max=max_iterations,
            conflicts=computation.conflicts_resolved,
            new_root=bool(computation.created_new_root),
        )

    if computation.removed_indices.size == 0:
        return tree, plan
    new_tree = _materialise_tree_from_computation(tree, backend, computation)
    return new_tree, plan
