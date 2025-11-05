from __future__ import annotations

import math
from typing import List

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit  # type: ignore

    NUMBA_SPARSE_TRAVERSAL_AVAILABLE = True
except Exception:  # pragma: no cover - when numba unavailable
    njit = None  # type: ignore
    NUMBA_SPARSE_TRAVERSAL_AVAILABLE = False

from covertreex.queries._knn_numba import NumbaTreeView

_EPS = 1e-12

if NUMBA_SPARSE_TRAVERSAL_AVAILABLE:

    @njit(cache=True)
    def _sqdist_row(query: np.ndarray, point: np.ndarray) -> float:
        total = 0.0
        for d in range(query.shape[0]):
            diff = query[d] - point[d]
            total += diff * diff
        return total

    @njit(cache=True)
    def _cover_radius(index: int, top_levels: np.ndarray, si_cache: np.ndarray) -> float:
        level = int(top_levels[index])
        base = math.ldexp(1.0, level + 1)
        si_val = 0.0
        if index < si_cache.shape[0]:
            si_val = si_cache[index]
        return si_val if si_val > base else base

    @njit(cache=True)
    def _collect_scope_single(
        query: np.ndarray,
        parent: int,
        radius: float,
        points: np.ndarray,
        top_levels: np.ndarray,
        si_cache: np.ndarray,
        children_offsets: np.ndarray,
        children_list: np.ndarray,
        roots: np.ndarray,
        next_cache: np.ndarray,
    ) -> np.ndarray:
        num_nodes = points.shape[0]
        if num_nodes == 0 or parent < 0 or parent >= num_nodes:
            return np.empty(0, dtype=np.int64)

        stack = np.empty(num_nodes, dtype=np.int64)
        stack_size = 0
        visited = np.zeros(num_nodes, dtype=np.uint8)
        included = np.zeros(num_nodes, dtype=np.uint8)
        collected = np.empty(num_nodes, dtype=np.int64)
        count = 0

        for r in roots:
            idx = int(r)
            if 0 <= idx < num_nodes:
                stack[stack_size] = idx
                stack_size += 1

        while stack_size > 0:
            stack_size -= 1
            node = stack[stack_size]
            if node < 0 or node >= num_nodes:
                continue
            if visited[node]:
                continue
            visited[node] = 1

            dist = math.sqrt(_sqdist_row(query, points[node]))
            if dist <= radius + _EPS and included[node] == 0:
                collected[count] = node
                count += 1
                included[node] = 1

            cover = _cover_radius(node, top_levels, si_cache)
            lower_bound = dist - cover
            if lower_bound <= radius + _EPS:
                start = children_offsets[node]
                end = children_offsets[node + 1]
                for pos in range(start, end):
                    child = children_list[pos]
                    if child >= 0:
                        stack[stack_size] = child
                        stack_size += 1

        current = parent
        steps = 0
        while 0 <= current < num_nodes and steps < num_nodes:
            if included[current] == 0:
                collected[count] = current
                count += 1
                included[current] = 1
            nxt = -1
            if current < next_cache.shape[0]:
                nxt = next_cache[current]
            if nxt == current:
                break
            current = nxt
            steps += 1

        if count > 1:
            for i in range(1, count):
                idx = collected[i]
                level = top_levels[idx]
                j = i - 1
                while j >= 0:
                    prev = collected[j]
                    prev_level = top_levels[prev]
                    if prev_level < level or (prev_level == level and prev > idx):
                        collected[j + 1] = prev
                        j -= 1
                    else:
                        break
                collected[j + 1] = idx

        return collected[:count].copy()


def collect_sparse_scopes(
    view: NumbaTreeView,
    queries: np.ndarray,
    parents: np.ndarray,
    radii: np.ndarray,
) -> List[np.ndarray]:
    if not NUMBA_SPARSE_TRAVERSAL_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError("Sparse traversal requires numba to be installed.")

    results: List[np.ndarray] = []
    for idx in range(queries.shape[0]):
        parent = int(parents[idx])
        radius = float(radii[idx])
        if parent < 0 or queries.shape[1] != view.points.shape[1]:
            results.append(np.empty(0, dtype=np.int64))
            continue
        scope = _collect_scope_single(
            queries[idx],
            parent,
            radius,
            view.points,
            view.top_levels,
            view.si_cache,
            view.children_offsets,
            view.children_list,
            view.root_indices,
            view.next_cache,
        )
        results.append(scope)
    return results
