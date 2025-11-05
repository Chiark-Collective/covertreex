from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import math
import numpy as np
import weakref

try:  # pragma: no cover - optional dependency
    from numba import njit, prange  # type: ignore

    NUMBA_QUERY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numba missing
    njit = None  # type: ignore
    prange = None  # type: ignore
    NUMBA_QUERY_AVAILABLE = False

from covertreex.core.tree import PCCTree, TreeBackend


@dataclass(frozen=True)
class NumbaTreeView:
    """Lightweight NumPy materialisation of tree buffers for Numba kernels."""

    points: np.ndarray
    si_cache: np.ndarray
    top_levels: np.ndarray
    children: np.ndarray
    next_cache: np.ndarray
    children_offsets: np.ndarray
    children_list: np.ndarray
    root_indices: np.ndarray


def _to_numpy(array: Any, backend: TreeBackend, *, dtype: Any) -> np.ndarray:
    """Return a NumPy array copy of a backend buffer using the desired dtype."""

    return np.asarray(backend.to_numpy(array), dtype=dtype)


if NUMBA_QUERY_AVAILABLE:
    _EPS = 1e-12

    @njit(cache=True)
    def _sqdist_row(query: np.ndarray, point: np.ndarray) -> float:
        total = 0.0
        for d in range(query.shape[0]):
            diff = query[d] - point[d]
            total += diff * diff
        return total

    @njit(cache=True)
    def _insert_partial(indices: np.ndarray, dists: np.ndarray, count: int, idx: int, dist: float) -> None:
        pos = count
        while pos > 0:
            prev_dist = dists[pos - 1]
            prev_idx = indices[pos - 1]
            if dist < prev_dist - _EPS or (abs(dist - prev_dist) <= _EPS and idx < prev_idx):
                dists[pos] = prev_dist
                indices[pos] = prev_idx
                pos -= 1
            else:
                break
        dists[pos] = dist
        indices[pos] = idx

    @njit(cache=True)
    def _insert_full(indices: np.ndarray, dists: np.ndarray, idx: int, dist: float) -> None:
        pos = indices.shape[0] - 1
        while pos > 0:
            prev_dist = dists[pos - 1]
            prev_idx = indices[pos - 1]
            if dist < prev_dist - _EPS or (abs(dist - prev_dist) <= _EPS and idx < prev_idx):
                dists[pos] = prev_dist
                indices[pos] = prev_idx
                pos -= 1
            else:
                break
        dists[pos] = dist
        indices[pos] = idx

    @njit(cache=True)
    def _knn_single(query: np.ndarray, points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx_out = np.full(k, -1, dtype=np.int64)
        dist_out = np.full(k, math.inf, dtype=np.float64)
        count = 0

        for idx in range(points.shape[0]):
            dist_sq = _sqdist_row(query, points[idx])
            if count < k:
                _insert_partial(idx_out, dist_out, count, idx, dist_sq)
                count += 1
            else:
                worst_dist = dist_out[k - 1]
                worst_idx = idx_out[k - 1]
                if dist_sq < worst_dist - _EPS or (abs(dist_sq - worst_dist) <= _EPS and idx < worst_idx):
                    _insert_full(idx_out, dist_out, idx, dist_sq)

        for pos in range(k):
            dist_out[pos] = math.sqrt(dist_out[pos])
        return idx_out, dist_out

    @njit(cache=True)
    def _cover_tree_knn_single(
        query: np.ndarray,
        points: np.ndarray,
        si_cache: np.ndarray,
        children: np.ndarray,
        next_cache: np.ndarray,
        children_offsets: np.ndarray,
        children_list: np.ndarray,
        roots: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_points = points.shape[0]
        best_idx = np.full(k, -1, dtype=np.int64)
        best_dist = np.full(k, math.inf, dtype=np.float64)
        best_count = 0

        visited = np.zeros(num_points, dtype=np.uint8)
        enqueued = np.zeros(num_points, dtype=np.uint8)
        heap_node = np.empty(num_points, dtype=np.int64)
        heap_lower = np.empty(num_points, dtype=np.float64)
        heap_order = np.empty(num_points, dtype=np.int64)
        heap_size = 0
        order_counter = 0

        node_distances = np.full(num_points, -1.0, dtype=np.float64)

        def _heap_push(node: int, lower: float, dist: float, order: int) -> None:
            nonlocal heap_size
            pos = heap_size
            heap_node[pos] = node
            heap_lower[pos] = lower
            heap_order[pos] = order
            heap_size += 1
            while pos > 0:
                parent = (pos - 1) // 2
                parent_lower = heap_lower[parent]
                parent_order = heap_order[parent]
                if (
                    lower < parent_lower - _EPS
                    or (abs(lower - parent_lower) <= _EPS and order < parent_order)
                ):
                    heap_node[pos] = heap_node[parent]
                    heap_lower[pos] = parent_lower
                    heap_order[pos] = parent_order
                    heap_node[parent] = node
                    heap_lower[parent] = lower
                    heap_order[parent] = order
                    pos = parent
                else:
                    break

        def _heap_pop() -> Tuple[int, float, float, float]:
            nonlocal heap_size
            node = heap_node[0]
            lower = heap_lower[0]
            order = heap_order[0]
            heap_size -= 1
            if heap_size > 0:
                last_node = heap_node[heap_size]
                last_lower = heap_lower[heap_size]
                last_order = heap_order[heap_size]
                pos = 0
                heap_node[0] = last_node
                heap_lower[0] = last_lower
                heap_order[0] = last_order
                while True:
                    left = 2 * pos + 1
                    right = left + 1
                    smallest = pos
                    if left < heap_size:
                        left_lower = heap_lower[left]
                        left_order = heap_order[left]
                        cur_lower = heap_lower[smallest]
                        cur_order = heap_order[smallest]
                        if (
                            left_lower < cur_lower - _EPS
                            or (abs(left_lower - cur_lower) <= _EPS and left_order < cur_order)
                        ):
                            smallest = left
                    if right < heap_size:
                        right_lower = heap_lower[right]
                        right_order = heap_order[right]
                        cur_lower = heap_lower[smallest]
                        cur_order = heap_order[smallest]
                        if (
                            right_lower < cur_lower - _EPS
                            or (abs(right_lower - cur_lower) <= _EPS and right_order < cur_order)
                        ):
                            smallest = right
                    if smallest == pos:
                        break
                    tmp_node = heap_node[pos]
                    tmp_lower = heap_lower[pos]
                    tmp_order = heap_order[pos]
                    heap_node[pos] = heap_node[smallest]
                    heap_lower[pos] = heap_lower[smallest]
                    heap_order[pos] = heap_order[smallest]
                    heap_node[smallest] = tmp_node
                    heap_lower[smallest] = tmp_lower
                    heap_order[smallest] = tmp_order
                    pos = smallest
            return node, lower, node_distances[node], order

        for r in roots:
            idx = int(r)
            if idx < 0 or idx >= num_points:
                continue
            if visited[idx] or enqueued[idx]:
                continue
            if node_distances[idx] < 0.0:
                node_distances[idx] = math.sqrt(_sqdist_row(query, points[idx]))
            dist = node_distances[idx]
            radius = si_cache[idx] if idx < si_cache.shape[0] else 0.0
            lower_bound = 0.0 if np.isinf(radius) else dist - radius
            if lower_bound < 0.0:
                lower_bound = 0.0
            current_bound = best_dist[k - 1] if best_count == k else math.inf
            if lower_bound > current_bound:
                continue
            enqueued[idx] = 1
            _heap_push(idx, lower_bound, dist, order_counter)
            order_counter += 1

        while heap_size > 0:
            node_idx, best_lower, node_dist, order_val = _heap_pop()
            enqueued[node_idx] = 0

            if visited[node_idx]:
                continue

            current_bound = best_dist[k - 1] if best_count == k else math.inf
            if best_count == k and best_lower > current_bound:
                break

            visited[node_idx] = 1
            if best_count < k:
                _insert_partial(best_idx, best_dist, best_count, node_idx, node_dist)
                best_count += 1
            else:
                worst_dist = best_dist[k - 1]
                worst_idx = best_idx[k - 1]
                if node_dist < worst_dist - _EPS or (
                    abs(node_dist - worst_dist) <= _EPS and node_idx < worst_idx
                ):
                    _insert_full(best_idx, best_dist, node_idx, node_dist)

            if node_idx < children_offsets.shape[0] - 1:
                start = children_offsets[node_idx]
                end = children_offsets[node_idx + 1]
                for pos in range(start, end):
                    child = children_list[pos]
                    if child < 0 or child >= num_points:
                        continue
                    if visited[child] or enqueued[child]:
                        continue
                    if node_distances[child] < 0.0:
                        node_distances[child] = math.sqrt(_sqdist_row(query, points[child]))
                    dist_child = node_distances[child]
                    radius_child = si_cache[child] if child < si_cache.shape[0] else 0.0
                    lower_child = dist_child - radius_child if not np.isinf(radius_child) else 0.0
                    if lower_child < 0.0:
                        lower_child = 0.0
                    current_bound = best_dist[k - 1] if best_count == k else math.inf
                    if lower_child <= current_bound + _EPS:
                        enqueued[child] = 1
                        _heap_push(child, lower_child, dist_child, order_counter)
                        order_counter += 1

        if best_count < k:
            return _knn_single(query, points, k)

        out_idx = np.empty(k, dtype=np.int64)
        out_dist = np.empty(k, dtype=np.float64)
        for i in range(k):
            out_idx[i] = best_idx[i]
            out_dist[i] = best_dist[i]
        return out_idx, out_dist

    @njit(cache=True, parallel=True)
    def _knn_batch_cover(
        queries: np.ndarray,
        points: np.ndarray,
        si_cache: np.ndarray,
        children: np.ndarray,
        next_cache: np.ndarray,
        children_offsets: np.ndarray,
        children_list: np.ndarray,
        roots: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_queries = queries.shape[0]
        indices = np.empty((num_queries, k), dtype=np.int64)
        distances = np.empty((num_queries, k), dtype=np.float64)
        for qi in prange(num_queries):
            idx_row, dist_row = _cover_tree_knn_single(
                queries[qi],
                points,
                si_cache,
                children,
                next_cache,
                children_offsets,
                children_list,
                roots,
                k,
            )
            indices[qi, :] = idx_row
            distances[qi, :] = dist_row
        return indices, distances


def materialise_tree_view(tree: PCCTree) -> NumbaTreeView:
    """Return a host-side view of the immutable tree buffers."""

    backend = tree.backend
    points = _to_numpy(tree.points, backend, dtype=np.float64)
    si_cache = _to_numpy(tree.si_cache, backend, dtype=np.float64)
    top_levels = _to_numpy(tree.top_levels, backend, dtype=np.int64)
    children = _to_numpy(tree.children, backend, dtype=np.int64)
    next_cache = _to_numpy(tree.next_cache, backend, dtype=np.int64)
    parents = _to_numpy(tree.parents, backend, dtype=np.int64)
    root_indices = np.where(parents < 0)[0]
    if root_indices.size == 0 and tree.num_points:
        root_indices = np.asarray([0], dtype=np.int64)

    child_counts = np.zeros(children.shape[0], dtype=np.int64)
    child = children.copy()
    for parent in range(children.shape[0]):
        idx = child[parent]
        while idx >= 0:
            child_counts[parent] += 1
            if idx >= next_cache.shape[0]:
                break
            idx = next_cache[idx]

    offsets = np.zeros(children.shape[0] + 1, dtype=np.int64)
    total = 0
    for i in range(children.shape[0]):
        offsets[i] = total
        total += child_counts[i]
    offsets[children.shape[0]] = total
    children_list = np.full(total, -1, dtype=np.int64)
    for parent in range(children.shape[0]):
        start = offsets[parent]
        count = child_counts[parent]
        idx = children[parent]
        pos = start
        while idx >= 0 and pos < start + count:
            children_list[pos] = idx
            pos += 1
            if idx >= next_cache.shape[0]:
                break
            idx = next_cache[idx]

    return NumbaTreeView(
        points=points,
        si_cache=si_cache,
        top_levels=top_levels,
        children=children,
        next_cache=next_cache,
        root_indices=root_indices,
        children_offsets=offsets,
        children_list=children_list,
    )


def knn_numba(
    view: NumbaTreeView,
    queries: Iterable[np.ndarray],
    *,
    k: int,
    return_distances: bool,
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Execute a k-NN lookup using Numba-accelerated dense distances."""

    if not NUMBA_QUERY_AVAILABLE:
        raise RuntimeError("Numba is not available for k-NN queries.")

    if k <= 0:
        raise ValueError("k must be positive.")
    num_points = view.points.shape[0]
    if num_points == 0:
        raise ValueError("Cannot query an empty tree.")
    if k > num_points:
        raise ValueError("k cannot exceed the number of points in the tree.")

    queries_arr = np.asarray(queries, dtype=np.float64)
    squeeze = False
    if queries_arr.ndim == 1:
        queries_arr = queries_arr[None, :]
        squeeze = True

    if queries_arr.shape[1] != view.points.shape[1]:
        raise ValueError("Query dimensionality does not match tree points.")

    if NUMBA_QUERY_AVAILABLE:
        indices, distances = _knn_batch_cover(
            queries_arr,
            view.points,
            view.si_cache,
            view.children,
            view.next_cache,
            view.children_offsets,
            view.children_list,
            view.root_indices,
            k,
        )
    else:  # pragma: no cover - defensive fallback
        indices, distances = _knn_batch_cover(
            queries_arr,
            view.points,
            view.si_cache,
            view.children,
            view.next_cache,
            view.children_offsets,
            view.children_list,
            view.root_indices,
            k,
        )

    if squeeze:
        indices = indices[0]
        distances = distances[0]

    if return_distances:
        return indices, distances
    return indices


def materialise_tree_view_cached(tree: PCCTree) -> NumbaTreeView:
    """Return a cached host-side view for the given tree."""

    key = id(tree)
    entry = _VIEW_CACHE.get(key)
    if entry is not None:
        tree_ref, view = entry
        cached_tree = tree_ref()
        if cached_tree is tree:
            return view
        if cached_tree is None:
            _VIEW_CACHE.pop(key, None)

    view = materialise_tree_view(tree)

    def _cleanup(_: weakref.ReferenceType[PCCTree]) -> None:
        _VIEW_CACHE.pop(key, None)

    _VIEW_CACHE[key] = (weakref.ref(tree, _cleanup), view)
    return view


_VIEW_CACHE: Dict[int, Tuple[weakref.ReferenceType[PCCTree], NumbaTreeView]] = {}
