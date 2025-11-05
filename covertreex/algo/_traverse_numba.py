from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba as nb

    NUMBA_TRAVERSAL_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nb = None  # type: ignore
    NUMBA_TRAVERSAL_AVAILABLE = False

I64 = np.int64


def _require_numba() -> None:
    if not NUMBA_TRAVERSAL_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError(
            "Numba traversal helpers requested but `numba` is not available. "
            "Install the '[numba]' extra or disable the feature via "
            "COVERTREEX_ENABLE_NUMBA=0."
        )


if NUMBA_TRAVERSAL_AVAILABLE:

    @nb.njit(cache=True)
    def _mark_next_chains(mask: np.ndarray, parents: np.ndarray, next_cache: np.ndarray) -> None:
        batch = parents.shape[0]
        num_points = mask.shape[1]
        limit = min(num_points, next_cache.shape[0])
        for row in range(batch):
            parent = parents[row]
            if parent < 0 or parent >= limit:
                continue
            current = parent
            steps = 0
            while 0 <= current < limit and steps < limit:
                mask[row, current] = True
                nxt = int(next_cache[current])
                if nxt < 0 or nxt == current:
                    break
                current = nxt
                steps += 1

    @nb.njit(cache=True)
    def _sort_scope_nodes(nodes: np.ndarray, top_levels: np.ndarray) -> None:
        for i in range(1, nodes.shape[0]):
            node = nodes[i]
            level = top_levels[node]
            j = i - 1
            while j >= 0:
                prev = nodes[j]
                prev_level = top_levels[prev]
                if prev_level < level or (prev_level == level and prev > node):
                    nodes[j + 1] = prev
                    j -= 1
                else:
                    break
            nodes[j + 1] = node

    @nb.njit(cache=True)
    def _mask_to_csr(mask: np.ndarray, top_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = mask.shape[0]
        num_points = mask.shape[1]

        counts = np.zeros(batch, dtype=I64)
        total = I64(0)
        for row in range(batch):
            count = I64(0)
            for col in range(num_points):
                if mask[row, col]:
                    count += 1
            counts[row] = count
            total += count

        indptr = np.empty(batch + 1, dtype=I64)
        indptr[0] = 0
        for row in range(batch):
            indptr[row + 1] = indptr[row] + counts[row]

        indices = np.empty(total, dtype=I64)
        offset = 0
        for row in range(batch):
            count = counts[row]
            if count == 0:
                continue
            temp = np.empty(count, dtype=I64)
            pos = 0
            for col in range(num_points):
                if mask[row, col]:
                    temp[pos] = col
                    pos += 1
            _sort_scope_nodes(temp, top_levels)
            for pos in range(count):
                indices[offset + pos] = temp[pos]
            offset += count

        return indptr, indices

    @nb.njit(cache=True)
    def build_scopes_with_chains(
        mask: np.ndarray,
        parents: np.ndarray,
        next_cache: np.ndarray,
        top_levels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _mark_next_chains(mask, parents, next_cache)
        return _mask_to_csr(mask, top_levels)


def build_scopes_numba(
    mask: np.ndarray,
    parents: np.ndarray,
    next_cache: np.ndarray,
    top_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR buffers for traversal scopes using the Numba helper."""

    _require_numba()
    mask_arr = np.array(mask, dtype=np.bool_, copy=True)
    parents_arr = np.asarray(parents, dtype=I64)
    next_cache_arr = np.asarray(next_cache, dtype=I64)
    top_levels_arr = np.asarray(top_levels, dtype=I64)
    return build_scopes_with_chains(mask_arr, parents_arr, next_cache_arr, top_levels_arr)


__all__ = ["NUMBA_TRAVERSAL_AVAILABLE", "build_scopes_numba"]
