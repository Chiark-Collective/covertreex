from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from covertreex import config as cx_config
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.diagnostics import log_operation
from covertreex.logging import get_logger
from covertreex.queries._knn_numba import (
    NUMBA_QUERY_AVAILABLE,
    materialise_tree_view_cached,
    knn_numba as _knn_numba,
)


LOGGER = get_logger("queries.knn")


def _to_numpy_array(backend: TreeBackend, array: Any, dtype: Any) -> np.ndarray:
    """Materialise a backend array as a NumPy array with the desired dtype."""

    return np.asarray(backend.to_numpy(array), dtype=dtype)


class _ChildChainCache:
    """Memoise decoded child chains from the compressed representation."""

    __slots__ = ("_children", "_next", "_cache", "_empty")

    def __init__(self, children: np.ndarray, next_cache: np.ndarray) -> None:
        self._children = children
        self._next = next_cache
        self._cache: Dict[int, np.ndarray] = {}
        self._empty = np.empty(0, dtype=np.int64)

    def get(self, parent: int) -> np.ndarray:
        if parent < 0 or parent >= self._children.shape[0]:
            return self._empty
        cached = self._cache.get(parent)
        if cached is not None:
            return cached

        head = int(self._children[parent])
        if head < 0 or head >= self._next.shape[0]:
            self._cache[parent] = self._empty
            return self._empty

        chain: List[int] = []
        seen: set[int] = set()
        current = head
        while 0 <= current < self._next.shape[0] and current not in seen:
            chain.append(current)
            seen.add(current)
            nxt = int(self._next[current])
            if nxt < 0 or nxt == current:
                break
            current = nxt

        arr = np.asarray(chain, dtype=np.int64) if chain else self._empty
        self._cache[parent] = arr
        return arr


def _fallback_bruteforce(
    query: np.ndarray, points: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-NN via dense distances as a safety net."""

    diff = points - query[None, :]
    dists = np.linalg.norm(diff, axis=1)
    order = np.argsort(dists)[:k]
    return order.astype(np.int64), dists[order].astype(np.float64)


def _single_query_knn(
    query: np.ndarray,
    *,
    points: np.ndarray,
    si_cache: np.ndarray,
    child_cache: _ChildChainCache,
    root_indices: Sequence[int],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a cover-tree walk to collect the k nearest neighbours for one query."""

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("Cannot query an empty tree.")

    visited: set[int] = set()
    enqueued: set[int] = set()
    best_heap: List[Tuple[float, int]] = []  # max-heap of (-distance, index)
    candidate_heap: List[Tuple[float, int, int, float]] = []  # (lower, order, idx, dist)
    counter = 0
    distance_cache: dict[int, float] = {}

    def _push_candidate(idx: int) -> None:
        nonlocal counter
        if idx < 0 or idx >= num_points:
            return
        if idx in visited or idx in enqueued:
            return

        dist = distance_cache.get(idx)
        if dist is None:
            point = points[idx]
            dist = float(np.linalg.norm(query - point))
            distance_cache[idx] = dist
        radius = float(si_cache[idx]) if si_cache.size > idx else 0.0
        if math.isinf(radius):
            lower_bound = 0.0
        else:
            lower_bound = max(dist - radius, 0.0)

        current_bound = -best_heap[0][0] if len(best_heap) >= k else math.inf
        if lower_bound > current_bound:
            return

        heapq.heappush(candidate_heap, (lower_bound, counter, idx, dist))
        enqueued.add(idx)
        counter += 1

    for root_idx in root_indices:
        _push_candidate(int(root_idx))

    while candidate_heap:
        lower_bound, _, node_idx, node_dist = heapq.heappop(candidate_heap)
        enqueued.discard(node_idx)
        if node_idx in visited:
            continue

        current_bound = -best_heap[0][0] if len(best_heap) >= k else math.inf
        if len(best_heap) >= k and lower_bound > current_bound:
            break

        visited.add(node_idx)

        if len(best_heap) < k:
            heapq.heappush(best_heap, (-node_dist, node_idx))
        else:
            worst_dist, worst_idx = best_heap[0]
            worst_dist = -worst_dist
            if node_dist < worst_dist or (
                math.isclose(node_dist, worst_dist)
                and node_idx < worst_idx
            ):
                heapq.heapreplace(best_heap, (-node_dist, node_idx))

        for child in child_cache.get(node_idx):
            _push_candidate(int(child))

    if len(best_heap) < k:
        return _fallback_bruteforce(query, points, k)

    ordered = sorted((-dist, idx) for dist, idx in best_heap)
    indices = np.asarray([idx for _, idx in ordered], dtype=np.int64)
    distances = np.asarray([dist for dist, _ in ordered], dtype=np.float64)
    return indices, distances


def knn(
    tree: PCCTree,
    query_points: Any,
    *,
    k: int,
    return_distances: bool = False,
    backend: TreeBackend | None = None,
) -> Tuple[Any, Any] | Any:
    backend = backend or tree.backend
    with log_operation(LOGGER, "knn_query") as op_log:
        return _knn_impl(
            op_log,
            tree,
            query_points,
            k=k,
            return_distances=return_distances,
            backend=backend,
        )


def _knn_impl(
    op_log: Any,
    tree: PCCTree,
    query_points: Any,
    *,
    k: int,
    return_distances: bool,
    backend: TreeBackend,
) -> Tuple[Any, Any] | Any:
    if tree.is_empty():
        raise ValueError("Cannot query an empty tree.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > tree.num_points:
        raise ValueError("k cannot exceed the number of points in the tree.")

    batch = backend.asarray(query_points, dtype=backend.default_float)
    if batch.ndim == 1:
        batch = batch[None, :]

    runtime = cx_config.runtime_config()
    use_numba = runtime.enable_numba and NUMBA_QUERY_AVAILABLE

    batch_np = _to_numpy_array(backend, batch, dtype=np.float64)
    num_queries = batch_np.shape[0]

    if use_numba:
        view = materialise_tree_view_cached(tree)
        numba_indices, numba_distances = _knn_numba(
            view,
            batch_np,
            k=int(k),
            return_distances=True,
        )
        indices_arr = (
            numba_indices if numba_indices.ndim > 1 else numba_indices[None, :]
        )
        distances_arr = (
            numba_distances if numba_distances.ndim > 1 else numba_distances[None, :]
        )
    else:
        points_np = _to_numpy_array(backend, tree.points, dtype=np.float64)
        parents_np = _to_numpy_array(backend, tree.parents, dtype=np.int64)
        children_np = _to_numpy_array(backend, tree.children, dtype=np.int64)
        next_cache_np = _to_numpy_array(backend, tree.next_cache, dtype=np.int64)
        si_cache_np = _to_numpy_array(backend, tree.si_cache, dtype=np.float64)
        child_cache = _ChildChainCache(children_np, next_cache_np)

        root_candidates = np.where(parents_np < 0)[0]
        if root_candidates.size == 0:
            root_candidates = np.asarray([0], dtype=np.int64)

        results_indices: List[np.ndarray] = []
        results_distances: List[np.ndarray] = []

        for query in batch_np:
            indices, distances = _single_query_knn(
                query,
                points=points_np,
                si_cache=si_cache_np,
                child_cache=child_cache,
                root_indices=root_candidates,
                k=int(k),
            )
            results_indices.append(indices)
            results_distances.append(distances)

        indices_arr = np.stack(results_indices, axis=0)
        distances_arr = np.stack(results_distances, axis=0)

    sorted_indices = backend.asarray(indices_arr, dtype=backend.default_int)

    if op_log is not None:
        op_log.add_metadata(
            queries=num_queries,
            k=k,
            return_distances=bool(return_distances),
        )

    if not return_distances:
        if sorted_indices.shape[0] == 1:
            squeezed = sorted_indices[0]
            return squeezed if squeezed.shape[0] > 1 else squeezed[0]
        return sorted_indices

    sorted_distances = backend.asarray(distances_arr, dtype=backend.default_float)
    if sorted_indices.shape[0] == 1:
        squeezed_idx = sorted_indices[0]
        squeezed_dist = sorted_distances[0]
        if squeezed_idx.shape[0] == 1:
            return squeezed_idx[0], squeezed_dist[0]
        return squeezed_idx, squeezed_dist
    return sorted_indices, sorted_distances


def nearest_neighbor(
    tree: PCCTree,
    query_points: Any,
    *,
    return_distances: bool = False,
    backend: TreeBackend | None = None,
) -> Tuple[Any, Any] | Any:
    return knn(
        tree,
        query_points,
        k=1,
        return_distances=return_distances,
        backend=backend,
    )
