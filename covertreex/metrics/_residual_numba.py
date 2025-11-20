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
    def _distance_block_no_gate(
        p_diag: np.ndarray,
        query_indices: np.ndarray,
        chunk_indices: np.ndarray,
        kernel_block: np.ndarray,
        dot_block: np.ndarray,
        radii: np.ndarray,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_queries = query_indices.shape[0]
        chunk_size = chunk_indices.shape[0]
        distances = np.empty((num_queries, chunk_size), dtype=np.float64)
        mask = np.zeros((num_queries, chunk_size), dtype=np.uint8)

        for qi in prange(num_queries):
            query_idx = int(query_indices[qi])
            p_i = float(p_diag[query_idx])
            radius = float(radii[qi])
            
            for cj in range(chunk_size):
                cand_idx = int(chunk_indices[cj])
                p_cand = float(p_diag[cand_idx])
                
                kernel_val = kernel_block[qi, cj]
                partial = dot_block[qi, cj]
                
                denom = math.sqrt(max(p_i * p_cand, eps * eps))
                numerator = kernel_val - partial
                
                if denom > 0.0:
                    rho = numerator / denom
                else:
                    rho = 0.0
                
                # Clamp correlation
                if rho > 1.0:
                    rho = 1.0
                elif rho < -1.0:
                    rho = -1.0
                    
                dist = math.sqrt(max(0.0, 1.0 - abs(rho)))
                distances[qi, cj] = dist
                if dist <= radius + eps:
                    mask[qi, cj] = 1

        return distances, mask


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


def distance_block_no_gate(
    p_diag: np.ndarray,
    query_indices: np.ndarray,
    chunk_indices: np.ndarray,
    kernel_block: np.ndarray,
    dot_block: np.ndarray,
    radii: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if NUMBA_RESIDUAL_AVAILABLE:
        return _distance_block_no_gate(
            p_diag,
            query_indices,
            chunk_indices,
            kernel_block,
            dot_block,
            radii,
            eps,
        )

    # Python fallback
    num_queries = query_indices.shape[0]
    chunk_size = chunk_indices.shape[0]
    distances = np.empty((num_queries, chunk_size), dtype=np.float64)
    mask = np.zeros((num_queries, chunk_size), dtype=np.uint8)
    
    for qi in range(num_queries):
        query_idx = int(query_indices[qi])
        p_i = float(p_diag[query_idx])
        radius = float(radii[qi])
        
        for cj in range(chunk_size):
            cand_idx = int(chunk_indices[cj])
            p_cand = float(p_diag[cand_idx])
            
            kernel_val = kernel_block[qi, cj]
            partial = dot_block[qi, cj]
            
            denom = math.sqrt(max(p_i * p_cand, eps * eps))
            numerator = kernel_val - partial
            
            if denom > 0.0:
                rho = numerator / denom
            else:
                rho = 0.0
            
            if rho > 1.0:
                rho = 1.0
            elif rho < -1.0:
                rho = -1.0
                
            dist = math.sqrt(max(0.0, 1.0 - abs(rho)))
            distances[qi, cj] = dist
            if dist <= radius + eps:
                mask[qi, cj] = 1
                
    return distances, mask


__all__ = [
    "NUMBA_RESIDUAL_AVAILABLE",
    "compute_distance_chunk",
    "distance_block_no_gate",
]
