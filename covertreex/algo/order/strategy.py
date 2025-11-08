from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BatchOrderResult:
    permutation: Optional[np.ndarray]
    metrics: Dict[str, float]


def compute_batch_order(
    points: np.ndarray,
    *,
    strategy: str,
    seed: Optional[int] = None,
) -> BatchOrderResult:
    """Return a permutation for the requested ordering strategy."""

    if points.ndim != 2:
        raise ValueError("Expected a 2-D array of points for batch ordering.")
    n_points = points.shape[0]
    if n_points <= 1 or strategy == "natural":
        return BatchOrderResult(permutation=None, metrics={})
    if strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_points).astype(np.int64, copy=False)
        return BatchOrderResult(permutation=perm, metrics={"seed": float(seed or 0)})
    if strategy == "hilbert":
        perm, metrics = _hilbert_permutation(points, seed=seed)
        return BatchOrderResult(permutation=perm, metrics=metrics)
    raise ValueError(f"Unsupported batch order strategy '{strategy}'.")


def _hilbert_permutation(points: np.ndarray, *, seed: Optional[int]) -> tuple[np.ndarray, Dict[str, float]]:
    n_points, dims = points.shape
    bits = max(1, min(16, 64 // max(1, dims)))
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scaled = (points - lo) / span
    max_val = (1 << bits) - 1
    ints = np.clip(np.round(scaled * max_val), 0, max_val).astype(np.uint64, copy=False)
    codes = np.empty(n_points, dtype=np.uint64)
    for idx in range(n_points):
        coords = ints[idx].tolist()
        transpose = _axes_to_transpose(coords, bits)
        codes[idx] = _transpose_to_hilbert(transpose, bits, dims)
    order = np.lexsort((np.arange(n_points, dtype=np.int64), codes))
    sorted_codes = codes[order]
    diffs = np.diff(sorted_codes) if sorted_codes.size > 1 else np.array([0], dtype=np.uint64)
    spread = float(np.std(diffs)) if diffs.size else 0.0
    metrics = {
        "hilbert_bits_per_dim": float(bits),
        "hilbert_code_spread": spread,
    }
    return order.astype(np.int64, copy=False), metrics


def _axes_to_transpose(coords: list[int], bits: int) -> list[int]:
    n_dims = len(coords)
    coords = coords.copy()
    M = 1 << (bits - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n_dims):
            if coords[i] & Q:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        Q >>= 1
    for i in range(1, n_dims):
        coords[i] ^= coords[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if coords[n_dims - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n_dims):
        coords[i] ^= t
    return coords


def _transpose_to_hilbert(transpose: list[int], bits: int, dims: int) -> int:
    index = 0
    for bit in range(bits - 1, -1, -1):
        word = 0
        for axis in range(dims):
            bit_val = (transpose[axis] >> bit) & 1
            word = (word << 1) | bit_val
        index = (index << dims) | word
    return index


__all__ = ["BatchOrderResult", "compute_batch_order"]
