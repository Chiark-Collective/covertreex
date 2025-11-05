from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numba absent
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    DEGREE_THRESHOLD = 8

    @njit(cache=True)
    def _hash_priority(node: int, seed: int, iteration: int) -> float:
        """Return a deterministic pseudo-random priority in [0, 1)."""

        x = np.uint32(node) ^ np.uint32(seed)
        x ^= np.uint32(iteration * 0x9E3779B9)
        x ^= x << np.uint32(13)
        x ^= x >> np.uint32(17)
        x ^= x << np.uint32(5)
        return (np.float64(x) + 1.0) / np.float64(2**32)

    @njit(cache=True)
    def _max_degree(indptr: np.ndarray) -> int:
        max_deg = 0
        for node in range(indptr.size - 1):
            deg = indptr[node + 1] - indptr[node]
            if deg > max_deg:
                max_deg = deg
        return max_deg

    @njit(cache=True)
    def _greedy_low_degree(
        indptr: np.ndarray,
        indices: np.ndarray,
        num_nodes: int,
    ) -> np.ndarray:
        """Deterministic greedy for low-degree components."""

        active = np.ones(num_nodes, dtype=np.uint8)
        selected = np.zeros(num_nodes, dtype=np.uint8)

        for node in range(num_nodes):
            if not active[node]:
                continue
            selected[node] = 1
            start = indptr[node]
            end = indptr[node + 1]
            for offset in range(start, end):
                neighbor = indices[offset]
                active[neighbor] = 0
            active[node] = 0

        return selected

    @njit(cache=True)
    def _run_mis_numba_impl(
        indptr: np.ndarray,
        indices: np.ndarray,
        num_nodes: int,
        seed: int,
    ) -> Tuple[np.ndarray, int]:
        max_deg = _max_degree(indptr)
        if max_deg <= DEGREE_THRESHOLD:
            greedy = _greedy_low_degree(indptr, indices, num_nodes)
            return greedy.astype(np.int8), 1

        active = np.ones(num_nodes, dtype=np.uint8)
        selected = np.zeros(num_nodes, dtype=np.uint8)
        iterations = 0

        while np.any(active):
            iterations += 1
            priorities = np.empty(num_nodes, dtype=np.float64)
            for node in range(num_nodes):
                priorities[node] = _hash_priority(node, seed, iterations)

            winners = np.zeros(num_nodes, dtype=np.uint8)
            for node in range(num_nodes):
                if not active[node]:
                    continue
                priority = priorities[node]
                best = True
                start = indptr[node]
                end = indptr[node + 1]
                for offset in range(start, end):
                    neighbor = indices[offset]
                    if not active[neighbor]:
                        continue
                    neighbor_priority = priorities[neighbor]
                    if (
                        neighbor_priority > priority
                        or (
                            neighbor_priority == priority
                            and neighbor < node
                        )
                    ):
                        best = False
                        break
                if best:
                    winners[node] = 1

            if not np.any(winners):
                for node in range(num_nodes):
                    if active[node]:
                        winners[node] = 1
                        break

            dominated = np.zeros(num_nodes, dtype=np.uint8)
            for node in range(num_nodes):
                if not winners[node]:
                    continue
                start = indptr[node]
                end = indptr[node + 1]
                for offset in range(start, end):
                    dominated[indices[offset]] = 1

            for node in range(num_nodes):
                if winners[node]:
                    selected[node] = 1
                if winners[node] or dominated[node]:
                    active[node] = 0

        return selected.astype(np.int8), iterations


def run_mis_numba(indptr: np.ndarray, indices: np.ndarray, seed: int) -> Tuple[np.ndarray, int]:
    """Execute Luby MIS using the Numba implementation."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available.")
    return _run_mis_numba_impl(indptr, indices, indptr.shape[0] - 1, seed)
