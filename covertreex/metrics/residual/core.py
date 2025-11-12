from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np

from covertreex import config as cx_config
from .._residual_numba import (
    compute_distance_chunk,
    distance_block_no_gate,
)
from .policy import (
    RESIDUAL_EPS,
    ResidualGateLookup,
    ResidualGateProfile,
    ResidualGateTelemetry,
    ResidualPolicy,
    get_residual_policy,
)

ArrayLike = np.ndarray | Sequence[float] | Sequence[int]
_EPS = RESIDUAL_EPS


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
    gate1_enabled: bool | None = None
    gate1_alpha: float | None = None
    gate1_margin: float | None = None
    gate1_eps: float | None = None
    gate1_audit: bool | None = None
    gate1_radius_cap: float | None = None
    gate1_band_eps: float | None = None
    gate1_keep_pct: float | None = None
    gate1_prune_pct: float | None = None
    gate_v32: np.ndarray | None = None
    gate_norm32: np.ndarray | None = None
    gate_stats: ResidualGateTelemetry = field(default_factory=ResidualGateTelemetry)
    gate_profile_path: str | None = None
    gate_profile_bins: int | None = None
    gate_profile: "ResidualGateProfile | None" = None
    gate_lookup_path: str | None = None
    gate_lookup_margin: float | None = None
    gate_lookup: "ResidualGateLookup | None" = None
    grid_whiten_scale: float | None = None
    v_matrix_f64: np.ndarray | None = None
    p_diag_f64: np.ndarray | None = None
    kernel_diag_f64: np.ndarray | None = None

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
    """Reusable buffers for whitened block computations and gate masks."""

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
    """Counters describing whitened-block vs kernel-provider work."""

    whitened_pairs: int = 0
    whitened_seconds: float = 0.0
    whitened_calls: int = 0
    kernel_pairs: int = 0
    kernel_seconds: float = 0.0
    kernel_calls: int = 0

    def record_whitened(self, rows: int, cols: int, seconds: float) -> None:
        r = max(0, int(rows))
        c = max(0, int(cols))
        self.whitened_calls += 1
        self.whitened_pairs += r * c
        self.whitened_seconds += float(seconds)

    def record_kernel(self, rows: int, cols: int, seconds: float) -> None:
        r = max(0, int(rows))
        c = max(0, int(cols))
        self.kernel_calls += 1
        self.kernel_pairs += r * c
        self.kernel_seconds += float(seconds)


_ACTIVE_BACKEND: Optional[ResidualCorrHostData] = None


def _finalize_gate_profile(backend: ResidualCorrHostData | None) -> None:
    if backend is None:
        return
    profile = getattr(backend, "gate_profile", None)
    if isinstance(profile, ResidualGateProfile):
        profile.dump(getattr(backend, "gate_profile_path", None), force=True)


def set_residual_backend(backend: ResidualCorrHostData | None) -> None:
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is not None and _ACTIVE_BACKEND is not backend:
        _finalize_gate_profile(_ACTIVE_BACKEND)
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

    v_lhs = np.asarray(backend.v_matrix[lhs_indices], dtype=np.float32)
    v_rhs = np.asarray(backend.v_matrix[rhs_indices], dtype=np.float32)
    p_lhs = np.asarray(backend.p_diag[lhs_indices], dtype=np.float32)
    p_rhs = np.asarray(backend.p_diag[rhs_indices], dtype=np.float32)
    kernel_vals = np.asarray(kernel_block, dtype=np.float32)

    dot_products = v_lhs @ v_rhs.T  # float32
    dot_products = dot_products.astype(np.float64, copy=False)
    denom = np.sqrt(
        np.maximum(
            p_lhs.astype(np.float64, copy=False)[:, None]
            * p_rhs.astype(np.float64, copy=False)[None, :],
            float(_EPS) * float(_EPS),
        )
    )
    residual_corr = (kernel_vals.astype(np.float64, copy=False) - dot_products) / denom
    residual_corr = np.clip(residual_corr, -1.0, 1.0)
    distances = np.sqrt(np.maximum(1.0 - np.abs(residual_corr), 0.0))
    return distances


def _compute_gate1_whitened(matrix: np.ndarray, *, regularisation: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return (
            np.zeros_like(matrix, dtype=np.float32),
            np.zeros(matrix.shape[0], dtype=np.float32),
        )

    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    rank = centered.shape[1]
    if rank == 0:
        return (
            np.zeros((matrix.shape[0], 0), dtype=np.float32),
            np.zeros(matrix.shape[0], dtype=np.float32),
        )

    cov = centered.T @ centered
    denom = max(centered.shape[0] - 1, 1)
    cov = cov / float(denom)
    cov += np.eye(rank, dtype=np.float64) * regularisation
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = regularisation
        for _ in range(5):
            cov += np.eye(rank, dtype=np.float64) * jitter
            try:
                chol = np.linalg.cholesky(cov)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise
    inv_chol = np.linalg.solve(chol, np.eye(rank, dtype=np.float64))
    whitened = centered @ inv_chol.T
    v32 = whitened.astype(np.float32, copy=False)
    norms32 = np.sqrt(np.sum(v32 * v32, axis=1, dtype=np.float64)).astype(np.float32)
    return v32, norms32


def _ensure_whitened_cache(backend: ResidualCorrHostData) -> tuple[np.ndarray, np.ndarray]:
    v32 = getattr(backend, "gate_v32", None)
    n32 = getattr(backend, "gate_norm32", None)
    if v32 is None or n32 is None:
        v32, n32 = _compute_gate1_whitened(backend.v_matrix_view(np.float64))
        object.__setattr__(backend, "gate_v32", v32)
        object.__setattr__(backend, "gate_norm32", n32)
    return v32, n32


def _compute_gate1_distances(
    backend: ResidualCorrHostData,
    query_index: int,
    candidate_idx: np.ndarray,
) -> np.ndarray | None:
    if backend.gate_v32 is None or candidate_idx.size == 0:
        return None
    v_query = backend.gate_v32[query_index]
    v_chunk = backend.gate_v32[candidate_idx]
    diff = v_chunk - v_query
    dist_sq = np.sum(diff * diff, axis=1, dtype=np.float64)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq, dtype=np.float64)


def compute_whitened_block(
    backend: ResidualCorrHostData,
    query_indices: np.ndarray,
    chunk_indices: np.ndarray,
    *,
    workspace: ResidualWorkspace | None = None,
) -> np.ndarray:
    query_arr = np.asarray(query_indices, dtype=np.int64)
    chunk_arr = np.asarray(chunk_indices, dtype=np.int64)
    rows = int(query_arr.size)
    cols = int(chunk_arr.size)
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)

    gate_vectors, gate_norms = _ensure_whitened_cache(backend)
    q_vec = np.asarray(gate_vectors[query_arr], dtype=np.float32)
    c_vec = np.asarray(gate_vectors[chunk_arr], dtype=np.float32)
    q_norm_sq = np.square(np.asarray(gate_norms[query_arr], dtype=np.float32), dtype=np.float32)
    c_norm_sq = np.square(np.asarray(gate_norms[chunk_arr], dtype=np.float32), dtype=np.float32)

    if workspace is None:
        gram = np.empty((rows, cols), dtype=np.float32)
        dist2 = np.empty((rows, cols), dtype=np.float32)
    else:
        gram = workspace.gram_view(rows, cols)
        dist2 = workspace.dist2_view(rows, cols)

    np.dot(q_vec, c_vec.T, out=gram)
    dist2[:] = q_norm_sq[:, None] + c_norm_sq[None, :]
    dist2 -= 2.0 * gram
    np.maximum(dist2, 0.0, out=dist2)
    np.sqrt(dist2, out=dist2, dtype=np.float32)
    return dist2[:rows, :cols]


def _resolve_gate1_config(
    backend: ResidualCorrHostData,
    *,
    policy: ResidualPolicy | None = None,
) -> tuple[bool, float, float, float, bool, float, float, float, float]:
    runtime_policy = policy or get_residual_policy()

    enabled = runtime_policy.gate1_enabled if backend.gate1_enabled is None else backend.gate1_enabled
    alpha = backend.gate1_alpha if backend.gate1_alpha is not None else runtime_policy.gate1_alpha
    margin = backend.gate1_margin if backend.gate1_margin is not None else runtime_policy.gate1_margin
    eps = backend.gate1_eps if backend.gate1_eps is not None else runtime_policy.gate1_eps
    audit = backend.gate1_audit if backend.gate1_audit is not None else runtime_policy.gate1_audit
    cap = backend.gate1_radius_cap if backend.gate1_radius_cap is not None else runtime_policy.gate1_radius_cap
    band_eps = backend.gate1_band_eps if backend.gate1_band_eps is not None else runtime_policy.gate1_band_eps
    keep_pct = backend.gate1_keep_pct if backend.gate1_keep_pct is not None else runtime_policy.gate1_keep_pct
    prune_pct = backend.gate1_prune_pct if backend.gate1_prune_pct is not None else runtime_policy.gate1_prune_pct

    keep_pct = float(max(min(keep_pct, 100.0), 0.0))
    prune_pct = float(max(min(prune_pct, 100.0), 0.0))
    if prune_pct < keep_pct:
        prune_pct = keep_pct

    return (
        bool(enabled),
        float(alpha),
        float(margin),
        float(eps),
        bool(audit),
        float(cap),
        max(float(band_eps), 0.0),
        keep_pct,
        prune_pct,
    )


def _audit_gate1_pruned(
    *,
    backend: ResidualCorrHostData,
    query_index: int,
    candidate_idx: np.ndarray,
    kernel_vals: np.ndarray,
    keep_mask: np.ndarray,
    radius: float,
    whitened_distances: np.ndarray | None,
) -> None:
    if keep_mask.size == 0 or np.all(keep_mask):
        return
    pruned_idx = np.nonzero(~keep_mask)[0]
    if pruned_idx.size == 0:
        return
    v_query = backend.v_matrix[query_index]
    v_pruned = backend.v_matrix[candidate_idx[pruned_idx]]
    p_i = float(backend.p_diag[query_index])
    p_pruned = backend.p_diag[candidate_idx[pruned_idx]]
    norm_query = float(backend.v_norm_sq[query_index])
    norm_pruned = backend.v_norm_sq[candidate_idx[pruned_idx]]
    distances, mask = compute_distance_chunk(
        v_query=v_query,
        v_chunk=v_pruned,
        kernel_chunk=kernel_vals[pruned_idx],
        p_i=p_i,
        p_chunk=p_pruned,
        norm_query=norm_query,
        norm_chunk=norm_pruned,
        radius=float(radius),
        eps=_EPS,
    )
    profile = getattr(backend, "gate_profile", None)
    if isinstance(profile, ResidualGateProfile) and whitened_distances is not None and pruned_idx.size:
        profile.record_false_negatives(
            residual_distances=distances,
            whitened_distances=whitened_distances,
            inclusion_mask=mask,
        )
    if np.any(mask):
        raise RuntimeError(
            "Residual gate pruned a candidate that lies within the requested radius."
        )


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
    force_whitened: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_idx = np.asarray(chunk_indices, dtype=np.int64)
    if candidate_idx.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        mask = np.zeros((0,), dtype=np.uint8)
        return empty, mask

    row_indices = np.asarray([query_index], dtype=np.int64)
    row_count = int(row_indices.size)
    col_total = int(candidate_idx.size)
    if kernel_row is not None:
        kernel_vals_full = np.asarray(kernel_row, dtype=np.float32)
    else:
        kernel_vals_full = None

    def _fetch_full_kernel() -> np.ndarray:
        nonlocal kernel_vals_full
        if kernel_vals_full is None:
            kernel_start = time.perf_counter()
            kernel_vals_full = backend.kernel_provider(row_indices, candidate_idx)[0]
            kernel_elapsed = time.perf_counter() - kernel_start
            if telemetry is not None:
                telemetry.record_kernel(row_count, col_total, kernel_elapsed)
        return kernel_vals_full

    def _fetch_kernel_subset(indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if kernel_vals_full is not None:
            return kernel_vals_full[indices]
        cols = candidate_idx[indices]
        kernel_start = time.perf_counter()
        subset = backend.kernel_provider(row_indices, cols)[0]
        kernel_elapsed = time.perf_counter() - kernel_start
        if telemetry is not None:
            telemetry.record_kernel(row_count, int(indices.size), kernel_elapsed)
        return subset

    v_query = backend.v_matrix[query_index]
    v_chunk_full = backend.v_matrix[candidate_idx]
    p_i = float(backend.p_diag[query_index])
    p_chunk_full = backend.p_diag[candidate_idx]
    norm_query = float(backend.v_norm_sq[query_index])
    norm_chunk_full = backend.v_norm_sq[candidate_idx]

    profile: ResidualGateProfile | None = (
        backend.gate_profile if isinstance(getattr(backend, "gate_profile", None), ResidualGateProfile) else None
    )
    lookup: ResidualGateLookup | None = (
        backend.gate_lookup if isinstance(getattr(backend, "gate_lookup", None), ResidualGateLookup) else None
    )
    (
        gate_enabled,
        gate_alpha,
        gate_margin,
        gate_eps,
        gate_audit,
        gate_radius_cap,
        gate_band_eps,
        gate_keep_pct,
        gate_prune_pct,
    ) = _resolve_gate1_config(backend)
    radius_value = float(radius)
    radius_cap = max(float(gate_radius_cap), 0.0)
    effective_radius = min(radius_value, radius_cap) if radius_cap > 0.0 else radius_value
    effective_radius = max(effective_radius, 0.0)

    gate_keep_mask: np.ndarray | None = None
    gate_can_run = gate_enabled and radius_cap > 0.0
    need_whitened = force_whitened or gate_can_run or profile is not None or gate_audit
    whitened_distances: np.ndarray | None = None
    if need_whitened:
        whitened_start = time.perf_counter()
        whitened_block = compute_whitened_block(
            backend,
            query_indices=row_indices,
            chunk_indices=candidate_idx,
            workspace=workspace,
        )
        whitened_elapsed = time.perf_counter() - whitened_start
        whitened_distances = np.asarray(whitened_block[0], dtype=np.float64)
        if telemetry is not None:
            telemetry.record_whitened(row_count, col_total, whitened_elapsed)

    if gate_can_run and whitened_distances is not None:
        gate_start = time.perf_counter()
        if lookup is not None:
            base_keep, base_prune = lookup.thresholds(effective_radius)
            keep_threshold = max(base_keep * max(1.0 - gate_margin, 0.0), 0.0)
            prune_threshold = max(base_prune * (1.0 + gate_margin), keep_threshold)
        else:
            base = gate_alpha * effective_radius + gate_margin
            keep_threshold = 0.0
            prune_threshold = max(base, keep_threshold)
        band_eps = max(float(gate_band_eps), 0.0)
        prune_cutoff = prune_threshold + band_eps
        if prune_cutoff <= 0.0:
            gate_elapsed = time.perf_counter() - gate_start
            total = int(candidate_idx.size)
            backend.gate_stats.candidates += total
            backend.gate_stats.kept += total
            backend.gate_stats.seconds += gate_elapsed
        else:
            prune_mask = whitened_distances >= prune_cutoff
            gate_elapsed = time.perf_counter() - gate_start
            survivors_mask = ~prune_mask
            total = int(candidate_idx.size)
            kept = int(np.count_nonzero(survivors_mask))
            pruned = total - kept
            backend.gate_stats.candidates += total
            backend.gate_stats.kept += kept
            backend.gate_stats.pruned += pruned
            backend.gate_stats.seconds += gate_elapsed
            if kept == total:
                gate_keep_mask = None
            elif kept == 0:
                if gate_audit and pruned > 0:
                    audit_whitened = np.asarray(whitened_distances, dtype=np.float64)
                    audit_kernel = _fetch_full_kernel()
                    _audit_gate1_pruned(
                        backend=backend,
                        query_index=query_index,
                        candidate_idx=candidate_idx,
                        kernel_vals=audit_kernel,
                        keep_mask=np.zeros_like(prune_mask, dtype=bool),
                        radius=radius,
                        whitened_distances=audit_whitened,
                    )
                distances = np.full(total, float(radius) + gate_eps, dtype=np.float64)
                mask = np.zeros(total, dtype=np.uint8)
                if profile is not None and whitened_distances is not None:
                    profile.record_chunk(
                        residual_distances=distances,
                        whitened_distances=whitened_distances,
                        inclusion_mask=mask,
                    )
                return distances, mask
            else:
                gate_keep_mask = survivors_mask
                if gate_audit and pruned > 0:
                    audit_subset = np.asarray(whitened_distances[prune_mask], dtype=np.float64)
                    audit_kernel = _fetch_full_kernel()
                    _audit_gate1_pruned(
                        backend=backend,
                        query_index=query_index,
                        candidate_idx=candidate_idx,
                        kernel_vals=audit_kernel,
                        keep_mask=gate_keep_mask,
                        radius=radius,
                        whitened_distances=audit_subset,
                    )

    if gate_keep_mask is None:
        kernel_vals = _fetch_full_kernel()
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
        if profile is not None and whitened_distances is not None:
            profile.record_chunk(
                residual_distances=distances,
                whitened_distances=whitened_distances,
                inclusion_mask=mask,
            )
        return distances, mask

    survivors_idx = np.nonzero(gate_keep_mask)[0]
    if survivors_idx.size == 0:
        distances = np.full(candidate_idx.size, float(radius) + gate_eps, dtype=np.float64)
        mask = np.zeros(candidate_idx.size, dtype=np.uint8)
        if profile is not None and whitened_distances is not None:
            profile.record_chunk(
                residual_distances=distances,
                whitened_distances=whitened_distances,
                inclusion_mask=mask,
            )
        return distances, mask

    kernel_survivors = _fetch_kernel_subset(survivors_idx)
    distances_surv, mask_surv = compute_distance_chunk(
        v_query=v_query,
        v_chunk=v_chunk_full[survivors_idx],
        kernel_chunk=kernel_survivors,
        p_i=p_i,
        p_chunk=p_chunk_full[survivors_idx],
        norm_query=norm_query,
        norm_chunk=norm_chunk_full[survivors_idx],
        radius=float(radius),
        eps=_EPS,
    )

    distances = np.full(candidate_idx.size, float(radius) + gate_eps, dtype=np.float64)
    mask = np.zeros(candidate_idx.size, dtype=np.uint8)
    distances[survivors_idx] = distances_surv
    mask[survivors_idx] = mask_surv
    if profile is not None and whitened_distances is not None:
        profile.record_chunk(
            residual_distances=distances,
            whitened_distances=whitened_distances,
            inclusion_mask=mask,
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

    distances, mask = distance_block_no_gate(
        backend.v_matrix,
        backend.p_diag,
        backend.v_norm_sq,
        query_arr,
        chunk_arr,
        kernel_block,
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


def configure_residual_correlation(backend: ResidualCorrHostData) -> None:
    """Install residual-correlation kernels using the supplied backend."""

    from covertreex.core.metrics import configure_residual_metric

    if backend.v_norm_sq is None:
        v_matrix = backend.v_matrix_view(np.float64)
        object.__setattr__(backend, "v_norm_sq", np.sum(v_matrix * v_matrix, axis=1))

    policy = get_residual_policy()
    runtime = cx_config.runtime_config()
    (
        enabled,
        alpha,
        margin,
        eps,
        audit,
        radius_cap,
        band_eps,
        keep_pct,
        prune_pct,
    ) = _resolve_gate1_config(backend, policy=policy)
    profile_path = backend.gate_profile_path or policy.gate1_profile_path
    profile_bins = backend.gate_profile_bins or policy.gate1_profile_bins
    lookup_path = backend.gate_lookup_path or policy.gate1_lookup_path
    lookup_margin = (
        backend.gate_lookup_margin if backend.gate_lookup_margin is not None else policy.gate1_lookup_margin
    )
    grid_mode_requested = runtime.metric == "residual_correlation" and runtime.conflict_graph_impl == "grid"
    grid_whiten_scale = float(getattr(runtime, "residual_grid_whiten_scale", 1.0))
    if grid_whiten_scale <= 0.0:
        grid_whiten_scale = 1.0

    radius_max_for_profile = max(1.0, radius_cap if radius_cap > 0.0 else 1.0)
    profile_obj = None
    profile_quantiles: list[float] = []
    if keep_pct > 0.0:
        profile_quantiles.append(keep_pct)
    if prune_pct > 0.0:
        profile_quantiles.append(prune_pct)
    profile_quantiles.extend([95.0, 99.0, 99.9])
    profile_quantiles = sorted({round(val, 6) for val in profile_quantiles})
    if profile_path:
        profile_obj = ResidualGateProfile.create(
            bins=int(profile_bins),
            radius_max=float(radius_max_for_profile),
            path=profile_path,
            radius_eps=policy.radius_floor,
            quantile_percentiles=profile_quantiles,
        )

    lookup_obj = None
    if lookup_path:
        lookup_obj = ResidualGateLookup.load(
            lookup_path,
            margin=float(lookup_margin),
            keep_pct=keep_pct,
            prune_pct=prune_pct,
        )

    need_whitened = (
        enabled
        or profile_obj is not None
        or lookup_obj is not None
        or grid_mode_requested
        or grid_whiten_scale != 1.0
    )
    if need_whitened:
        v32, n32 = _compute_gate1_whitened(backend.v_matrix_view(np.float64))
    else:
        v32 = None
        n32 = None
    object.__setattr__(backend, "gate1_enabled", enabled)
    object.__setattr__(backend, "gate1_alpha", alpha)
    object.__setattr__(backend, "gate1_margin", margin)
    object.__setattr__(backend, "gate1_eps", eps)
    object.__setattr__(backend, "gate1_audit", audit)
    object.__setattr__(backend, "gate1_radius_cap", radius_cap)
    object.__setattr__(backend, "gate1_band_eps", band_eps)
    object.__setattr__(backend, "gate1_keep_pct", keep_pct)
    object.__setattr__(backend, "gate1_prune_pct", prune_pct)
    object.__setattr__(backend, "gate_v32", v32)
    object.__setattr__(backend, "gate_norm32", n32)
    object.__setattr__(backend, "gate_stats", ResidualGateTelemetry())
    object.__setattr__(backend, "gate_profile_path", profile_path)
    object.__setattr__(backend, "gate_profile_bins", profile_bins if profile_obj else None)
    object.__setattr__(backend, "gate_profile", profile_obj)
    object.__setattr__(backend, "gate_lookup_path", lookup_path)
    object.__setattr__(backend, "gate_lookup_margin", lookup_margin)
    object.__setattr__(backend, "gate_lookup", lookup_obj)
    object.__setattr__(backend, "grid_whiten_scale", grid_whiten_scale)

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
    "ResidualGateTelemetry",
    "ResidualGateProfile",
    "ResidualGateLookup",
    "configure_residual_correlation",
    "get_residual_backend",
    "set_residual_backend",
    "compute_whitened_block",
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
