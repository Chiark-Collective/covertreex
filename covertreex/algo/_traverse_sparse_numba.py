from __future__ import annotations

import math
from typing import List as PyList, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit  # type: ignore
    from numba.typed import List as TypedList  # type: ignore

    NUMBA_SPARSE_TRAVERSAL_AVAILABLE = True
except Exception:  # pragma: no cover - when numba unavailable
    njit = None  # type: ignore
    NUMBA_SPARSE_TRAVERSAL_AVAILABLE = False
    TypedList = None  # type: ignore

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
    def _collect_scope_single_into(
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
        out_buf: np.ndarray,
    ) -> int:
        num_nodes = points.shape[0]
        if num_nodes == 0 or parent < 0 or parent >= num_nodes:
            return 0

        stack = np.empty(num_nodes, dtype=np.int64)
        stack_size = 0
        visited = np.zeros(num_nodes, dtype=np.uint8)
        included = np.zeros(num_nodes, dtype=np.uint8)

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
                out_buf[count] = node
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
                out_buf[count] = current
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
            min_level = top_levels[out_buf[0]]
            max_level = min_level
            for i in range(1, count):
                lvl = top_levels[out_buf[i]]
                if lvl < min_level:
                    min_level = lvl
                if lvl > max_level:
                    max_level = lvl
            span = int(max_level - min_level + 1)
            if span <= 0:
                span = 1
            levels_buf = np.empty(count, dtype=np.int64)
            counts = np.zeros(span, dtype=np.int64)
            max_lvl_int = int(max_level)
            for i in range(count):
                lvl = int(top_levels[out_buf[i]])
                levels_buf[i] = lvl
                counts[max_lvl_int - lvl] += 1
            offsets = np.empty(span + 1, dtype=np.int64)
            offsets[0] = 0
            for b in range(span):
                offsets[b + 1] = offsets[b] + counts[b]
            cursors = offsets[:-1].copy()
            ordered = np.empty(count, dtype=np.int64)
            for i in range(count):
                bucket = max_lvl_int - levels_buf[i]
                pos = cursors[bucket]
                ordered[pos] = out_buf[i]
                cursors[bucket] = pos + 1
            for b in range(span):
                start = offsets[b]
                end = offsets[b + 1]
                if end - start > 1:
                    segment = np.sort(ordered[start:end])
                    ordered[start:end] = segment
            out_buf[:count] = ordered

        return count

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
        collected = np.empty(num_nodes, dtype=np.int64)
        count = _collect_scope_single_into(
            query,
            parent,
            radius,
            points,
            top_levels,
            si_cache,
            children_offsets,
            children_list,
            roots,
            next_cache,
            collected,
        )
        return collected[:count].copy()

    @njit(cache=True)
    def _collect_scopes_csr(
        queries: np.ndarray,
        parents: np.ndarray,
        radii: np.ndarray,
        points: np.ndarray,
        top_levels: np.ndarray,
        si_cache: np.ndarray,
        children_offsets: np.ndarray,
        children_list: np.ndarray,
        roots: np.ndarray,
        next_cache: np.ndarray,
        chunk_target: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        batch_size = queries.shape[0]
        num_nodes = points.shape[0]
        counts = np.zeros(batch_size, dtype=np.int64)
        buffer = np.empty(num_nodes, dtype=np.int64)
        per_query = TypedList()
        chunk_segments = 0
        chunk_emitted = 0
        chunk_max_members = 0

        for idx in range(batch_size):
            parent = int(parents[idx])
            radius = float(radii[idx])
            if parent < 0 or num_nodes == 0:
                counts[idx] = 0
                per_query.append(np.empty(0, dtype=np.int64))
            else:
                count_val = _collect_scope_single_into(
                    queries[idx],
                    parent,
                    radius,
                    points,
                    top_levels,
                    si_cache,
                    children_offsets,
                    children_list,
                    roots,
                    next_cache,
                    buffer,
                )
                counts[idx] = count_val
                arr = np.empty(count_val, dtype=np.int64)
                if count_val:
                    arr[:count_val] = buffer[:count_val]
                    if chunk_target > 0:
                        pruned_count = 0
                        for pos in range(count_val):
                            node = arr[pos]
                            dist_sq = _sqdist_row(queries[idx], points[node])
                            if dist_sq <= radius * radius + _EPS:
                                arr[pruned_count] = node
                                pruned_count += 1
                        count_val = pruned_count
                        arr = arr[:count_val]
                per_query.append(arr)
                count_val = arr.shape[0]
                counts[idx] = count_val

            count_val = counts[idx]
            if chunk_target > 0:
                segments = 1
                if count_val > 0:
                    segments = (count_val + chunk_target - 1) // chunk_target
                    current_max = chunk_target if count_val > chunk_target else int(count_val)
                    if current_max > chunk_max_members:
                        chunk_max_members = current_max
                chunk_segments += segments
                if segments > 1:
                    chunk_emitted += segments - 1
            else:
                chunk_segments += 1
                if count_val > chunk_max_members:
                    chunk_max_members = int(count_val)

        total_scope = int(np.sum(counts))
        indptr = np.empty(batch_size + 1, dtype=np.int64)
        indptr[0] = 0
        for idx in range(batch_size):
            indptr[idx + 1] = indptr[idx] + counts[idx]

        indices = np.empty(total_scope, dtype=np.int64)
        if total_scope > 0:
            for idx in range(batch_size):
                count = int(counts[idx])
                if count == 0:
                    continue
                start = int(indptr[idx])
                query_scope = per_query[idx]
                indices[start : start + count] = query_scope

        if chunk_target > 0 and chunk_max_members == 0 and total_scope > 0:
            chunk_max_members = int(np.max(counts))

        return indptr, indices, int(chunk_segments), int(chunk_emitted), int(chunk_max_members)

else:  # pragma: no cover - executed when numba missing

    def _collect_scope_single_into(*_args, **_kwargs):
        raise RuntimeError("Sparse traversal requires numba to be installed.")

    def _collect_scope_single(*_args, **_kwargs):
        raise RuntimeError("Sparse traversal requires numba to be installed.")

    def _collect_scopes_csr(*_args, **_kwargs):
        raise RuntimeError("Sparse traversal requires numba to be installed.")


def collect_sparse_scopes(
    view: NumbaTreeView,
    queries: np.ndarray,
    parents: np.ndarray,
    radii: np.ndarray,
) -> PyList[np.ndarray]:
    if not NUMBA_SPARSE_TRAVERSAL_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError("Sparse traversal requires numba to be installed.")

    results: PyList[np.ndarray] = []
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


def collect_sparse_scopes_csr(
    view: NumbaTreeView,
    queries: np.ndarray,
    parents: np.ndarray,
    radii: np.ndarray,
    chunk_target: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    if not NUMBA_SPARSE_TRAVERSAL_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError("Sparse traversal requires numba to be installed.")

    indptr, indices, chunk_segments, chunk_emitted, chunk_max_members = _collect_scopes_csr(
        queries,
        parents,
        radii,
        view.points,
        view.top_levels,
        view.si_cache,
        view.children_offsets,
        view.children_list,
        view.root_indices,
        view.next_cache,
        int(chunk_target),
    )
    return indptr, indices, chunk_segments, chunk_emitted, chunk_max_members
