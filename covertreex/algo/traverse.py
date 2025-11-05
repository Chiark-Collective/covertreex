from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import time
import numpy as np

from covertreex import config as cx_config
from covertreex.core.metrics import get_metric
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.logging import get_logger
from ._traverse_numba import NUMBA_TRAVERSAL_AVAILABLE, build_scopes_numba
from ._traverse_sparse_numba import (
    NUMBA_SPARSE_TRAVERSAL_AVAILABLE,
    collect_sparse_scopes,
)
from covertreex.queries._knn_numba import (
    knn_numba,
    materialise_tree_view_cached,
)

LOGGER = get_logger("algo.traverse")


@dataclass(frozen=True)
class TraversalResult:
    """Structured output produced by batched traversal."""

    parents: Any
    levels: Any
    conflict_scopes: Tuple[Tuple[int, ...], ...]
    scope_indptr: Any
    scope_indices: Any
    timings: "TraversalTimings"


@dataclass(frozen=True)
class TraversalTimings:
    pairwise_seconds: float
    mask_seconds: float
    semisort_seconds: float
    chain_seconds: float = 0.0
    nonzero_seconds: float = 0.0
    sort_seconds: float = 0.0
    assemble_seconds: float = 0.0


def _block_until_ready(value: Any) -> None:
    """Best-effort barrier for asynchronous backends (e.g., JAX)."""

    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


def _broadcast_batch(backend: TreeBackend, batch_points: Any) -> Any:
    return backend.asarray(batch_points, dtype=backend.default_float)


def _collect_distances(tree: PCCTree, batch: Any, backend: TreeBackend) -> Any:
    metric = get_metric()
    return metric.pairwise(backend, batch, tree.points)


def _empty_result(backend: TreeBackend, batch_size: int) -> TraversalResult:
    xp = backend.xp
    parents = backend.asarray(
        xp.full((batch_size,), -1), dtype=backend.default_int
    )
    levels = backend.asarray(
        xp.full((batch_size,), -1), dtype=backend.default_int
    )
    conflict_scopes: Tuple[Tuple[int, ...], ...] = tuple(() for _ in range(batch_size))
    scope_indptr = backend.asarray([0] * (batch_size + 1), dtype=backend.default_int)
    scope_indices = backend.asarray([], dtype=backend.default_int)
    return TraversalResult(
        parents=parents,
        levels=levels,
        conflict_scopes=conflict_scopes,
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
        timings=TraversalTimings(
            pairwise_seconds=0.0,
            mask_seconds=0.0,
            semisort_seconds=0.0,
            chain_seconds=0.0,
            nonzero_seconds=0.0,
            sort_seconds=0.0,
            assemble_seconds=0.0,
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


def _traverse_collect_sparse(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend,
) -> TraversalResult:
    queries_np = np.asarray(backend.to_numpy(batch_points), dtype=np.float64)
    if queries_np.ndim == 1:
        queries_np = queries_np[None, :]

    batch_size = int(queries_np.shape[0])
    if batch_size == 0:
        return _empty_result(backend, 0)

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

    scope_start = time.perf_counter()
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
            semisort_seconds=scope_seconds,
            chain_seconds=0.0,
            nonzero_seconds=0.0,
            sort_seconds=0.0,
            assemble_seconds=0.0,
        ),
    )


def traverse_collect_scopes(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend | None = None,
) -> TraversalResult:
    """Compute parent assignments and conflict scopes for a batch of points."""

    backend = backend or tree.backend
    batch = _broadcast_batch(backend, batch_points)
    batch_size = int(batch.shape[0]) if batch.size else 0

    if batch_size == 0:
        return _empty_result(backend, 0)

    if tree.is_empty():
        return _empty_result(backend, batch_size)

    runtime = cx_config.runtime_config()

    if (
        runtime.enable_sparse_traversal
        and runtime.enable_numba
        and runtime.metric == "euclidean"
        and NUMBA_SPARSE_TRAVERSAL_AVAILABLE
        and backend.name == "numpy"
    ):
        return _traverse_collect_sparse(tree, batch, backend=backend)

    xp = backend.xp
    start = time.perf_counter()
    distances = _collect_distances(tree, batch, backend)
    _block_until_ready(distances)
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
    _block_until_ready(mask)
    mask_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parents_np = np.asarray(parents, dtype=np.int64)
    top_levels_np = np.asarray(tree.top_levels, dtype=np.int64)
    mask_np = np.asarray(backend.to_numpy(mask), dtype=bool)
    if mask_np.ndim != 2:
        mask_np = np.reshape(mask_np, (batch_size, tree.num_points))
    next_cache_np = np.asarray(tree.next_cache, dtype=np.int64)

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
    assemble_seconds = time.perf_counter() - assemble_start

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
