from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike

from .core import ResidualCorrHostData

__all__ = ["build_residual_backend"]


def _rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2, dtype=np.float64)
    denom = max(lengthscale, 1e-12)
    scaled = -0.5 * sq_dist / (denom * denom)
    return float(variance) * np.exp(scaled, dtype=np.float64)


def _build_sgemm_rbf_provider(
    points: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    points_f32 = np.ascontiguousarray(points, dtype=np.float32)
    row_norms = np.sum(points_f32 * points_f32, axis=1, dtype=np.float32)
    variance32 = np.float32(variance)
    denom = max(lengthscale, 1e-12)
    gamma = np.float32(1.0 / (denom * denom))

    def provider(row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        row_idx = np.asarray(row_indices, dtype=np.int64)
        col_idx = np.asarray(col_indices, dtype=np.int64)
        if row_idx.size == 0 or col_idx.size == 0:
            return np.zeros((row_idx.size, col_idx.size), dtype=np.float32)
        rows = points_f32[row_idx]
        cols = points_f32[col_idx]
        gram = rows @ cols.T  # float32 SGEMM
        dist2 = row_norms[row_idx][:, None] + row_norms[col_idx][None, :]
        dist2 -= 2.0 * gram
        np.maximum(dist2, 0.0, out=dist2)
        # reuse dist2 buffer for scaled distances
        dist2 *= (-0.5 * gamma)
        np.exp(dist2, out=dist2)
        dist2 *= variance32
        return dist2

    return points_f32, row_norms, provider


def _point_decoder_factory(points: np.ndarray) -> Callable[[ArrayLike], np.ndarray]:
    points_contig = np.ascontiguousarray(points, dtype=np.float64)
    point_keys = [tuple(row.tolist()) for row in points_contig]
    index_map: dict[tuple[float, ...], int] = {}
    for idx, key in enumerate(point_keys):
        index_map.setdefault(key, idx)

    def decoder(values: ArrayLike) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != points_contig.shape[1]:
            raise ValueError(
                "Residual point decoder expected payload dimensionality "
                f"{points_contig.shape[1]}, received {arr.shape[1]}."
            )
        rows = np.ascontiguousarray(arr, dtype=np.float64)
        indices = np.empty(rows.shape[0], dtype=np.int64)
        for i, row in enumerate(rows):
            key = tuple(row.tolist())
            if key not in index_map:
                raise KeyError("Residual point decoder received unknown payload.")
            indices[i] = index_map[key]
        return indices

    return decoder


def build_residual_backend(
    points: np.ndarray,
    *,
    seed: int,
    inducing_count: int,
    variance: float,
    lengthscale: float,
    chunk_size: int = 512,
    rng: Generator | None = None,
) -> ResidualCorrHostData:
    """
    Build a :class:`ResidualCorrHostData` cache for the residual-correlation metric.
    """

    if points.size == 0:
        raise ValueError("Residual metric requires at least one point to configure caches.")

    points_np = np.asarray(points, dtype=np.float64)
    generator = rng or default_rng(seed)
    n_points = points.shape[0]
    inducing = min(inducing_count, n_points)
    if inducing <= 0:
        inducing = min(32, n_points)
    if inducing < n_points:
        inducing_idx = np.sort(generator.choice(n_points, size=inducing, replace=False))
    else:
        inducing_idx = np.arange(n_points)
    inducing_points = points_np[inducing_idx]

    k_mm = _rbf_kernel(inducing_points, inducing_points, variance=variance, lengthscale=lengthscale)
    jitter = 1e-6 * variance
    k_mm = k_mm + np.eye(inducing_points.shape[0], dtype=np.float64) * jitter
    l_mm = np.linalg.cholesky(k_mm)

    k_xm = _rbf_kernel(points_np, inducing_points, variance=variance, lengthscale=lengthscale)
    solve_result = np.linalg.solve(l_mm, k_xm.T)
    v_matrix = solve_result.T

    kernel_diag = np.full(n_points, variance, dtype=np.float64)
    p_diag = np.maximum(kernel_diag - np.sum(v_matrix * v_matrix, axis=1), 1e-9)

    point_decoder = _point_decoder_factory(points_np)

    kernel_points_f32, kernel_row_norms, kernel_provider = _build_sgemm_rbf_provider(
        points_np,
        variance=variance,
        lengthscale=lengthscale,
    )

    host_backend = ResidualCorrHostData(
        v_matrix=np.asarray(v_matrix, dtype=np.float32),
        p_diag=np.asarray(p_diag, dtype=np.float32),
        kernel_diag=np.asarray(kernel_diag, dtype=np.float32),
        kernel_provider=kernel_provider,
        point_decoder=point_decoder,
        chunk_size=int(chunk_size),
        kernel_points_f32=kernel_points_f32,
        kernel_row_norms_f32=kernel_row_norms,
    )
    return host_backend
