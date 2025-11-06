from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np

from ._residual_numba import compute_distance_chunk

ArrayLike = np.ndarray | Sequence[float] | Sequence[int]
_EPS = 1e-9


class KernelProvider(Protocol):
    def __call__(self, row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        ...


class PointDecoder(Protocol):
    def __call__(self, values: ArrayLike) -> np.ndarray:
        ...


def _default_point_decoder(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return np.asarray([arr], dtype=np.int64)
    if arr.ndim == 1:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.int64, copy=False)
    raise ValueError(
        "Residual-correlation metric expects integer point identifiers. "
        "Provide a custom point_decoder that can extract dataset indices "
        "from the supplied point representation."
    )


@dataclass(frozen=True)
class ResidualCorrHostData:
    """Container for host-side caches backing the residual-correlation metric.

    Parameters
    ----------
    v_matrix:
        Array with shape (n_points, rank) containing the low-rank factors
        V = L_mm^{-1} K(X, U) materialised on the host (float64 recommended).
    p_diag:
        Diagonal term p_i = max(K_xx - ||V_i||^2, eps) for each training point.
    kernel_diag:
        Raw kernel diagonal K(X_i, X_i). Used to guard normalisation when the
        kernel provider does not materialise diagonal entries.
    kernel_provider:
        Callable that returns raw kernel entries K(X_i, X_j) given arrays of
        row/column indices. Expected to return a dense matrix with shape
        (row_indices.size, col_indices.size) using float64 precision.
    point_decoder:
        Optional callable that converts point payloads passed to the metric
        into dataset indices. Defaults to treating the payload as an integer id.
    """

    v_matrix: np.ndarray
    p_diag: np.ndarray
    kernel_diag: np.ndarray
    kernel_provider: KernelProvider
    point_decoder: PointDecoder = _default_point_decoder
    chunk_size: int = 512
    v_norm_sq: np.ndarray = None  # type: ignore[misc]

    def __post_init__(self) -> None:
        if self.v_matrix.ndim != 2:
            raise ValueError("v_matrix must be two-dimensional.")
        if self.p_diag.ndim != 1:
            raise ValueError("p_diag must be one-dimensional.")
        if self.kernel_diag.ndim != 1:
            raise ValueError("kernel_diag must be one-dimensional.")
        if self.v_matrix.shape[0] != self.p_diag.shape[0]:
            raise ValueError("v_matrix and p_diag must have consistent length.")
        if self.kernel_diag.shape[0] != self.p_diag.shape[0]:
            raise ValueError("kernel_diag must align with the cached points.")
        object.__setattr__(self, "v_norm_sq", np.sum(self.v_matrix * self.v_matrix, axis=1))

    @property
    def num_points(self) -> int:
        return int(self.v_matrix.shape[0])

    @property
    def rank(self) -> int:
        return int(self.v_matrix.shape[1])


_ACTIVE_BACKEND: Optional[ResidualCorrHostData] = None


def set_residual_backend(backend: ResidualCorrHostData | None) -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def get_residual_backend() -> ResidualCorrHostData:
    if _ACTIVE_BACKEND is None:
        raise RuntimeError(
            "Residual-correlation backend has not been configured. "
            "Call covertreex.metrics.residual.configure_residual_correlation(...) "
            "after staging the host caches."
        )
    return _ACTIVE_BACKEND


def _normalise_indices(indices: np.ndarray, total: int) -> np.ndarray:
    if np.any(indices < 0) or np.any(indices >= total):
        raise IndexError(f"Residual metric received out-of-range indices (0..{total - 1}).")
    return indices


def decode_indices(host_backend: ResidualCorrHostData, payload: ArrayLike) -> np.ndarray:
    raw = host_backend.point_decoder(payload)
    arr = np.asarray(raw, dtype=np.int64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr


def _compute_distances_from_kernel_block(
    backend: ResidualCorrHostData,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    kernel_block: np.ndarray,
) -> np.ndarray:
    if kernel_block.shape != (lhs_indices.size, rhs_indices.size):
        raise ValueError(
            "Kernel block shape mismatch: "
            f"expected {(lhs_indices.size, rhs_indices.size)}, got {kernel_block.shape}."
        )

    v_lhs = backend.v_matrix[lhs_indices]
    v_rhs = backend.v_matrix[rhs_indices]
    p_lhs = np.maximum(backend.p_diag[lhs_indices], _EPS)
    p_rhs = np.maximum(backend.p_diag[rhs_indices], _EPS)

    dot_products = v_lhs @ v_rhs.T
    denom = np.sqrt(np.maximum(p_lhs[:, None] * p_rhs[None, :], _EPS * _EPS))
    residual_corr = (kernel_block - dot_products) / denom
    residual_corr = np.clip(residual_corr, -1.0, 1.0)
    return np.sqrt(np.maximum(1.0 - np.abs(residual_corr), 0.0))


def compute_residual_distances_with_kernel(
    backend: ResidualCorrHostData,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    *,
    chunk_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual distances and the raw kernel block for the given indices."""

    lhs = _normalise_indices(lhs_indices.astype(np.int64, copy=False), backend.num_points)
    rhs = _normalise_indices(rhs_indices.astype(np.int64, copy=False), backend.num_points)

    if lhs.size == 0 or rhs.size == 0:
        shape = (lhs.size, rhs.size)
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)

    chunk = int(chunk_size or backend.chunk_size or 512)
    total = rhs.size
    result = np.empty((lhs.size, total), dtype=np.float64)
    kernel_matrix = np.empty((lhs.size, total), dtype=np.float64)

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        rhs_chunk = rhs[start:stop]
        kernel_block = backend.kernel_provider(lhs, rhs_chunk)
        distances = _compute_distances_from_kernel_block(backend, lhs, rhs_chunk, kernel_block)
        kernel_matrix[:, start:stop] = kernel_block
        result[:, start:stop] = distances
    return result, kernel_matrix


def compute_residual_distances(
    backend: ResidualCorrHostData,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    *,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Return the residual-correlation distances for two index sets."""

    distances, _ = compute_residual_distances_with_kernel(
        backend,
        lhs_indices,
        rhs_indices,
        chunk_size=chunk_size,
    )
    return distances


def compute_residual_distances_from_kernel(
    backend: ResidualCorrHostData,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    kernel_block: np.ndarray,
) -> np.ndarray:
    lhs = _normalise_indices(lhs_indices.astype(np.int64, copy=False), backend.num_points)
    rhs = _normalise_indices(rhs_indices.astype(np.int64, copy=False), backend.num_points)
    return _compute_distances_from_kernel_block(backend, lhs, rhs, kernel_block)


def compute_residual_lower_bounds_from_kernel(
    backend: ResidualCorrHostData,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    kernel_block: np.ndarray,
) -> np.ndarray:
    lhs = _normalise_indices(lhs_indices.astype(np.int64, copy=False), backend.num_points)
    rhs = _normalise_indices(rhs_indices.astype(np.int64, copy=False), backend.num_points)
    if kernel_block.shape != (lhs.size, rhs.size):
        raise ValueError(
            "Kernel block shape mismatch for lower-bound computation: "
            f"expected {(lhs.size, rhs.size)}, got {kernel_block.shape}."
        )
    diag_lhs = backend.kernel_diag[lhs]
    diag_rhs = backend.kernel_diag[rhs]
    denom = np.sqrt(np.maximum(diag_lhs[:, None] * diag_rhs[None, :], _EPS * _EPS))
    ratio = np.clip(kernel_block / denom, -1.0, 1.0)
    return np.sqrt(np.maximum(1.0 - ratio, 0.0))


def compute_residual_distances_with_radius(
    backend: ResidualCorrHostData,
    query_index: int,
    chunk_indices: np.ndarray,
    kernel_row: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if chunk_indices.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        mask = np.zeros((0,), dtype=np.uint8)
        return empty, mask

    v_query = backend.v_matrix[query_index]
    v_chunk = backend.v_matrix[chunk_indices]
    p_i = float(backend.p_diag[query_index])
    p_chunk = backend.p_diag[chunk_indices]
    norm_query = float(backend.v_norm_sq[query_index])
    norm_chunk = backend.v_norm_sq[chunk_indices]
    distances, mask = compute_distance_chunk(
        v_query=v_query,
        v_chunk=v_chunk,
        kernel_chunk=kernel_row,
        p_i=p_i,
        p_chunk=p_chunk,
        norm_query=norm_query,
        norm_chunk=norm_chunk,
        radius=float(radius),
        eps=_EPS,
    )
    return distances, mask


def compute_residual_distance_single(
    backend: ResidualCorrHostData,
    lhs_index: int,
    rhs_index: int,
) -> float:
    indices = np.asarray([rhs_index], dtype=np.int64)
    result = compute_residual_distances(
        backend,
        np.asarray([lhs_index], dtype=np.int64),
        indices,
    )
    return float(result[0, 0])


def configure_residual_correlation(backend: ResidualCorrHostData) -> None:
    """Install residual-correlation kernels using the supplied backend."""

    from covertreex.core.metrics import configure_residual_metric

    set_residual_backend(backend)

    def pairwise_kernel(tree_backend, lhs: ArrayLike, rhs: ArrayLike):
        host_backend = get_residual_backend()
        lhs_indices = decode_indices(host_backend, lhs)
        rhs_indices = decode_indices(host_backend, rhs)
        distances = compute_residual_distances(
            host_backend,
            lhs_indices,
            rhs_indices,
        )
        return tree_backend.asarray(distances, dtype=tree_backend.default_float)

    def pointwise_kernel(tree_backend, lhs: ArrayLike, rhs: ArrayLike):
        host_backend = get_residual_backend()
        lhs_indices = decode_indices(host_backend, lhs)
        rhs_indices = decode_indices(host_backend, rhs)
        if lhs_indices.size != 1 or rhs_indices.size != 1:
            raise ValueError("Pointwise residual distance expects single-element inputs.")
        value = compute_residual_distance_single(
            host_backend,
            int(lhs_indices[0]),
            int(rhs_indices[0]),
        )
        return tree_backend.asarray(value, dtype=tree_backend.default_float)

    configure_residual_metric(pairwise=pairwise_kernel, pointwise=pointwise_kernel)


__all__ = [
    "ResidualCorrHostData",
    "configure_residual_correlation",
    "get_residual_backend",
    "set_residual_backend",
    "compute_residual_distances_with_kernel",
    "compute_residual_distances",
    "compute_residual_distance_single",
    "compute_residual_distances_with_radius",
    "decode_indices",
    "compute_residual_distances_from_kernel",
    "compute_residual_lower_bounds_from_kernel",
]
