from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit, prange  # type: ignore

    NUMBA_RESIDUAL_AVAILABLE = True
except Exception:  # pragma: no cover - when numba unavailable
    njit = None  # type: ignore
    prange = None  # type: ignore
    NUMBA_RESIDUAL_AVAILABLE = False


if NUMBA_RESIDUAL_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _distance_chunk(
        v_query: np.ndarray,
        v_chunk: np.ndarray,
        kernel_chunk: np.ndarray,
        p_i: float,
        p_chunk: np.ndarray,
        norm_query: float,
        norm_chunk: np.ndarray,
        radius: float,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        chunk_size = v_chunk.shape[0]
        rank = v_query.shape[0]
        distances = np.empty(chunk_size, dtype=np.float64)
        within = np.zeros(chunk_size, dtype=np.uint8)

        threshold = 1.0 - radius * radius
        if radius >= 1.0:
            threshold = -1.0  # effectively disable pruning

        for j in prange(chunk_size):
            denom = math.sqrt(max(p_i * p_chunk[j], eps * eps))
            partial = 0.0
            accum_q = 0.0
            accum_c = 0.0
            remaining_q = norm_query
            remaining_c = norm_chunk[j]
            pruned = False

            for d in range(rank):
                vq = v_query[d]
                vc = v_chunk[j, d]
                partial += vq * vc
                accum_q += vq * vq
                accum_c += vc * vc
                remaining_q = max(norm_query - accum_q, 0.0)
                remaining_c = max(norm_chunk[j] - accum_c, 0.0)
                rem_bound = math.sqrt(remaining_q * remaining_c)

                if denom > 0.0 and threshold > 0.0:
                    base = kernel_chunk[j] - partial
                    if rem_bound > 0.0:
                        hi = abs(base + rem_bound)
                        lo = abs(base - rem_bound)
                        max_abs = hi if hi > lo else lo
                    else:
                        max_abs = abs(base)
                    max_rho = max_abs / denom
                    if max_rho + eps < threshold:
                        distances[j] = radius + eps
                        within[j] = 0
                        pruned = True
                        break

            if pruned:
                continue

            numerator = kernel_chunk[j] - partial
            if denom > 0.0:
                rho = numerator / denom
            else:
                rho = 0.0
            if rho > 1.0:
                rho = 1.0
            elif rho < -1.0:
                rho = -1.0
            dist = math.sqrt(max(0.0, 1.0 - abs(rho)))
            distances[j] = dist
            if dist <= radius + eps:
                within[j] = 1

        return distances, within

    @njit(cache=True, fastmath=True, parallel=True)
    def _gate1_whitened_chunk(
        v_query: np.ndarray,
        v_chunk: np.ndarray,
        threshold_sq: float,
    ) -> np.ndarray:
        chunk_size = v_chunk.shape[0]
        rank = v_query.shape[0]
        keep = np.zeros(chunk_size, dtype=np.uint8)

        if not np.isfinite(threshold_sq):
            for j in range(chunk_size):
                keep[j] = 1
            return keep

        if threshold_sq <= 0.0:
            return keep

        for j in prange(chunk_size):
            dist_sq = 0.0
            for d in range(rank):
                diff = v_chunk[j, d] - v_query[d]
                dist_sq += diff * diff
                if dist_sq > threshold_sq:
                    break
            if dist_sq <= threshold_sq:
                keep[j] = 1

        return keep


def compute_distance_chunk(
    v_query: np.ndarray,
    v_chunk: np.ndarray,
    kernel_chunk: np.ndarray,
    p_i: float,
    p_chunk: np.ndarray,
    norm_query: float,
    norm_chunk: np.ndarray,
    radius: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return distances and inclusion mask for a query against a chunk.

    Falls back to pure NumPy when numba is unavailable.
    """

    if NUMBA_RESIDUAL_AVAILABLE:
        return _distance_chunk(
            v_query,
            v_chunk,
            kernel_chunk,
            p_i,
            p_chunk,
            norm_query,
            norm_chunk,
            radius,
            eps,
        )

    distances = np.empty(v_chunk.shape[0], dtype=np.float64)
    within = np.zeros(v_chunk.shape[0], dtype=np.uint8)
    threshold = 1.0 - radius * radius
    if radius >= 1.0:
        threshold = -1.0

    for j in range(v_chunk.shape[0]):
        denom = math.sqrt(max(p_i * p_chunk[j], eps * eps))
        partial = 0.0
        accum_q = 0.0
        accum_c = 0.0
        pruned = False
        for d in range(v_query.shape[0]):
            vq = v_query[d]
            vc = v_chunk[j, d]
            partial += vq * vc
            accum_q += vq * vq
            accum_c += vc * vc
            rem_bound = math.sqrt(max(norm_query - accum_q, 0.0) * max(norm_chunk[j] - accum_c, 0.0))
            if denom > 0.0 and threshold > 0.0:
                base = kernel_chunk[j] - partial
                if rem_bound > 0.0:
                    hi = abs(base + rem_bound)
                    lo = abs(base - rem_bound)
                    max_abs = hi if hi > lo else lo
                else:
                    max_abs = abs(base)
                max_rho = max_abs / denom
                if max_rho + eps < threshold:
                    distances[j] = radius + eps
                    within[j] = 0
                    pruned = True
                    break
        if pruned:
            continue
        numerator = kernel_chunk[j] - partial
        rho = numerator / denom if denom > 0.0 else 0.0
        rho = max(min(rho, 1.0), -1.0)
        dist = math.sqrt(max(0.0, 1.0 - abs(rho)))
        distances[j] = dist
        if dist <= radius + eps:
            within[j] = 1

    return distances, within


def gate1_whitened_mask(
    v_query: np.ndarray,
    v_chunk: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a uint8 mask indicating which candidates pass gate-1."""

    if not np.isfinite(threshold):
        return np.ones(v_chunk.shape[0], dtype=np.uint8)
    threshold = float(max(threshold, 0.0))
    threshold_sq = threshold * threshold

    if v_chunk.size == 0 or threshold_sq <= 0.0:
        return np.zeros(v_chunk.shape[0], dtype=np.uint8)

    if NUMBA_RESIDUAL_AVAILABLE:
        return _gate1_whitened_chunk(v_query, v_chunk, threshold_sq)

    diff = v_chunk - v_query
    dist_sq = np.sum(diff * diff, axis=1, dtype=np.float32)
    keep = dist_sq <= threshold_sq
    return keep.astype(np.uint8, copy=False)


__all__ = [
    "NUMBA_RESIDUAL_AVAILABLE",
    "compute_distance_chunk",
    "gate1_whitened_mask",
]
