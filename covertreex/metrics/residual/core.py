from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from covertreex import config as cx_config
from .._residual_numba import (
    compute_distance_chunk,
    distance_block_no_gate,
)
from .scope_caps import ResidualScopeCapRecorder, ResidualScopeCaps

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
        V = L_mm^{-1} K(X, U) materialised on the host. Stored as float32 to
        minimise residency during traversal; float64 copies are cached on demand
        for auditing.
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
    v_norm_sq: np.ndarray | None = None
    kernel_points_f32: np.ndarray | None = None
    kernel_row_norms_f32: np.ndarray | None = None
    v_matrix_f64: np.ndarray | None = None
    p_diag_f64: np.ndarray | None = None
    kernel_diag_f64: np.ndarray | None = None
    scope_caps: ResidualScopeCaps | None = None
    scope_cap_recorder: ResidualScopeCapRecorder | None = None
    scope_cap_output_path: str | None = None

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
        v_matrix_f32 = np.asarray(self.v_matrix, dtype=np.float32)
        p_diag_f32 = np.asarray(self.p_diag, dtype=np.float32)
        kernel_diag_f32 = np.asarray(self.kernel_diag, dtype=np.float32)
        object.__setattr__(self, "v_matrix", v_matrix_f32)
        object.__setattr__(self, "p_diag", p_diag_f32)
        object.__setattr__(self, "kernel_diag", kernel_diag_f32)
        if self.v_matrix_f64 is not None and self.v_matrix_f64.shape != self.v_matrix.shape:
            raise ValueError("v_matrix_f64 must match v_matrix shape when provided.")
        if self.p_diag_f64 is not None and self.p_diag_f64.shape != self.p_diag.shape:
            raise ValueError("p_diag_f64 must match p_diag shape when provided.")
        if self.kernel_diag_f64 is not None and self.kernel_diag_f64.shape != self.kernel_diag.shape:
            raise ValueError("kernel_diag_f64 must match kernel_diag shape when provided.")
        norms = np.sum(v_matrix_f32 * v_matrix_f32, axis=1, dtype=np.float64)
        object.__setattr__(self, "v_norm_sq", norms.astype(np.float32, copy=False))

    @property
    def num_points(self) -> int:
        return int(self.v_matrix.shape[0])

    @property
    def rank(self) -> int:
        return int(self.v_matrix.shape[1])

    def v_matrix_view(self, dtype: np.dtype | type[np.floating]) -> np.ndarray:
        target = np.dtype(dtype)
        if target == np.float32:
            return self.v_matrix
        if target == np.float64:
            cached = self.v_matrix_f64
            if cached is None:
                cached = self.v_matrix.astype(np.float64, copy=False)
                object.__setattr__(self, "v_matrix_f64", cached)
            return cached
        raise ValueError(f"Unsupported dtype for v_matrix_view: {dtype}")

    def p_diag_view(self, dtype: np.dtype | type[np.floating]) -> np.ndarray:
        target = np.dtype(dtype)
        if target == np.float32:
            return self.p_diag
        if target == np.float64:
            cached = self.p_diag_f64
            if cached is None:
                cached = self.p_diag.astype(np.float64, copy=False)
                object.__setattr__(self, "p_diag_f64", cached)
            return cached
        raise ValueError(f"Unsupported dtype for p_diag_view: {dtype}")

    def kernel_diag_view(self, dtype: np.dtype | type[np.floating]) -> np.ndarray:
        target = np.dtype(dtype)
        if target == np.float32:
            return self.kernel_diag
        if target == np.float64:
            cached = self.kernel_diag_f64
            if cached is None:
                cached = self.kernel_diag.astype(np.float64, copy=False)
                object.__setattr__(self, "kernel_diag_f64", cached)
            return cached
        raise ValueError(f"Unsupported dtype for kernel_diag_view: {dtype}")


@dataclass
class ResidualWorkspace:
    """Reusable buffers for computations."""

    max_queries: int = 1
    max_chunk: int = 1
    gram: np.ndarray = field(init=False, repr=False)
    dist2: np.ndarray = field(init=False, repr=False)
    mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._resize(max(1, int(self.max_queries)), max(1, int(self.max_chunk)))

    def ensure_capacity(self, rows: int, cols: int) -> None:
        rows = max(1, int(rows))
        cols = max(1, int(cols))
        if rows <= self.max_queries and cols <= self.max_chunk:
            return
        self._resize(max(rows, self.max_queries), max(cols, self.max_chunk))

    def gram_view(self, rows: int, cols: int) -> np.ndarray:
        self.ensure_capacity(rows, cols)
        return self.gram[:rows, :cols]

    def dist2_view(self, rows: int, cols: int) -> np.ndarray:
        self.ensure_capacity(rows, cols)
        return self.dist2[:rows, :cols]

    def mask_view(self, rows: int, cols: int) -> np.ndarray:
        self.ensure_capacity(rows, cols)
        return self.mask[:rows, :cols]

    def _resize(self, rows: int, cols: int) -> None:
        self.max_queries = rows
        self.max_chunk = cols
        self.gram = np.empty((rows, cols), dtype=np.float32)
        self.dist2 = np.empty((rows, cols), dtype=np.float32)
        self.mask = np.empty((rows, cols), dtype=np.uint8)


@dataclass
class ResidualDistanceTelemetry:
    """Counters describing residual metric work."""

    kernel_pairs: int = 0
    kernel_seconds: float = 0.0
    kernel_calls: int = 0

    def record_kernel(self, rows: int, cols: int, seconds: float) -> None:
        r = max(0, int(rows))
        c = max(0, int(cols))
        self.kernel_calls += 1
        self.kernel_pairs += r * c
        self.kernel_seconds += float(seconds)


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

    lhs = np.asarray(lhs_indices, dtype=np.int64)
    rhs = np.asarray(rhs_indices, dtype=np.int64)
    
    v_lhs = backend.v_matrix[lhs]
    v_rhs = backend.v_matrix[rhs]
    dot_products = v_lhs @ v_rhs.T  # float32 SGEMM

    # Use Numba kernel with a large radius to compute all distances without pruning
    radii = np.full(lhs.size, 2.0, dtype=np.float64)
    
    distances, _ = distance_block_no_gate(
        backend.p_diag,
        lhs,
        rhs,
        kernel_block.astype(np.float32, copy=False),
        dot_products,
        radii,
        _EPS,
    )
    return distances


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


def compute_residual_pairwise_matrix(
    host_backend: ResidualCorrHostData,
    batch_indices: np.ndarray,
    *,
    telemetry: ResidualDistanceTelemetry | None = None,
) -> np.ndarray:
    total = int(batch_indices.shape[0])
    if total == 0:
        return np.empty((0, 0), dtype=np.float32)
    result = np.empty((total, total), dtype=np.float32)
    chunk = int(host_backend.chunk_size or 512)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        rows = batch_indices[start:stop]
        kernel_start = time.perf_counter()
        kernel_block = host_backend.kernel_provider(rows, batch_indices)
        kernel_elapsed = time.perf_counter() - kernel_start
        if telemetry is not None:
            telemetry.record_kernel(int(rows.shape[0]), total, kernel_elapsed)
        distances = compute_residual_distances_from_kernel(
            host_backend,
            rows,
            batch_indices,
            kernel_block,
        ).astype(np.float32)
        result[start:stop, :] = distances
    return result


def compute_residual_distances_with_radius(
    backend: ResidualCorrHostData,
    query_index: int,
    chunk_indices: np.ndarray,
    kernel_row: np.ndarray | None,
    radius: float,
    *,
    workspace: ResidualWorkspace | None = None,
    telemetry: ResidualDistanceTelemetry | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute residual distances with radius check (dense fallback)."""
    candidate_idx = np.asarray(chunk_indices, dtype=np.int64)
    if candidate_idx.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        mask = np.zeros((0,), dtype=np.uint8)
        return empty, mask

    row_indices = np.asarray([query_index], dtype=np.int64)
    row_count = int(row_indices.size)
    col_total = int(candidate_idx.size)
    if kernel_row is not None:
        kernel_vals = np.asarray(kernel_row, dtype=np.float32)
    else:
        kernel_start = time.perf_counter()
        kernel_vals = backend.kernel_provider(row_indices, candidate_idx)[0]
        kernel_elapsed = time.perf_counter() - kernel_start
        if telemetry is not None:
            telemetry.record_kernel(row_count, col_total, kernel_elapsed)

    v_query = backend.v_matrix[query_index]
    v_chunk_full = backend.v_matrix[candidate_idx]
    p_i = float(backend.p_diag[query_index])
    p_chunk_full = backend.p_diag[candidate_idx]
    norm_query = float(backend.v_norm_sq[query_index])
    norm_chunk_full = backend.v_norm_sq[candidate_idx]

    distances, mask = compute_distance_chunk(
        v_query=v_query,
        v_chunk=v_chunk_full,
        kernel_chunk=kernel_vals,
        p_i=p_i,
        p_chunk=p_chunk_full,
        norm_query=norm_query,
        norm_chunk=norm_chunk_full,
        radius=float(radius),
        eps=_EPS,
    )
    return distances, mask


def compute_residual_distances_block_no_gate(
    backend: ResidualCorrHostData,
    query_indices: np.ndarray,
    chunk_indices: np.ndarray,
    kernel_block: np.ndarray,
    radii: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    query_arr = np.asarray(query_indices, dtype=np.int64)
    chunk_arr = np.asarray(chunk_indices, dtype=np.int64)
    if query_arr.size == 0 or chunk_arr.size == 0:
        shape = (int(query_arr.size), int(chunk_arr.size))
        return (
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.uint8),
        )
    kernel_block = np.asarray(kernel_block, dtype=np.float32)
    if kernel_block.shape != (query_arr.size, chunk_arr.size):
        raise ValueError(
            "Kernel block shape mismatch for residual block streaming. "
            f"Expected {(query_arr.size, chunk_arr.size)}, got {kernel_block.shape}."
        )
    radii_arr = np.asarray(radii, dtype=np.float64)
    if radii_arr.shape[0] != query_arr.shape[0]:
        raise ValueError("Radius array must align with query count for block streaming.")

    v_query = backend.v_matrix[query_arr]
    v_chunk = backend.v_matrix[chunk_arr]
    dot_block = v_query @ v_chunk.T  # float32 SGEMM via BLAS

    distances, mask = distance_block_no_gate(
        backend.p_diag,
        query_arr,
        chunk_arr,
        kernel_block,
        dot_block,
        radii_arr,
        _EPS,
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


def configure_residual_correlation(
    backend: ResidualCorrHostData,
    *,
    runtime: cx_config.RuntimeConfig | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> None:
    """Install residual-correlation kernels using the supplied backend."""

    from covertreex.core.metrics import configure_residual_metric

    resolved_runtime = runtime
    if resolved_runtime is None:
        if context is not None:
            resolved_runtime = context.config
        else:
            active = cx_config.current_runtime_context()
            if active is not None:
                resolved_runtime = active.config
            else:
                resolved_runtime = cx_config.RuntimeConfig.from_env()

    if backend.v_norm_sq is None:
        v_matrix = backend.v_matrix_view(np.float64)
        object.__setattr__(backend, "v_norm_sq", np.sum(v_matrix * v_matrix, axis=1))

    # Configure scope caps
    cap_path = resolved_runtime.residual_scope_cap_path
    cap_default = resolved_runtime.residual_scope_cap_default
    cap_output = resolved_runtime.residual_scope_cap_output
    
    recorder = None
    caps = None
    
    if cap_path or cap_default is not None:
        caps = ResidualScopeCaps.load(cap_path, default=cap_default)
        
    if cap_output:
        recorder = ResidualScopeCapRecorder(
            percentile=resolved_runtime.residual_scope_cap_percentile,
            margin=resolved_runtime.residual_scope_cap_margin,
        )
        
    object.__setattr__(backend, "scope_caps", caps)
    object.__setattr__(backend, "scope_cap_recorder", recorder)
    object.__setattr__(backend, "scope_cap_output_path", cap_output)

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
    "ResidualWorkspace",
    "ResidualDistanceTelemetry",
    "configure_residual_correlation",
    "get_residual_backend",
    "set_residual_backend",
    "compute_residual_distances_with_kernel",
    "compute_residual_distances",
    "compute_residual_distance_single",
    "compute_residual_distances_with_radius",
    "compute_residual_distances_block_no_gate",
    "decode_indices",
    "compute_residual_distances_from_kernel",
    "compute_residual_lower_bounds_from_kernel",
    "compute_residual_pairwise_matrix",
]
