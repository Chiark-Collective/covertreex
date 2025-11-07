from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np

from covertreex import config as cx_config
from ._residual_numba import compute_distance_chunk, gate1_whitened_mask

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


@dataclass
class ResidualGateTelemetry:
    candidates: int = 0
    kept: int = 0
    pruned: int = 0
    seconds: float = 0.0

    def snapshot(self) -> Tuple[int, int, int, float]:
        return (self.candidates, self.kept, self.pruned, self.seconds)

    def delta(self, snapshot: Tuple[int, int, int, float]) -> "ResidualGateTelemetry":
        cand0, kept0, pruned0, seconds0 = snapshot
        return ResidualGateTelemetry(
            candidates=self.candidates - cand0,
            kept=self.kept - kept0,
            pruned=self.pruned - pruned0,
            seconds=self.seconds - seconds0,
        )


@dataclass
class ResidualGateProfile:
    bin_edges: np.ndarray
    max_whitened: np.ndarray
    max_ratio: np.ndarray
    counts: np.ndarray
    samples_total: int = 0
    false_negative_samples: int = 0
    radius_eps: float = _EPS
    path: Path | None = None
    dirty: bool = False

    @classmethod
    def create(
        cls,
        *,
        bins: int,
        radius_max: float,
        path: str | Path | None,
        radius_eps: float = _EPS,
    ) -> "ResidualGateProfile":
        edges = np.linspace(0.0, radius_max, bins + 1, dtype=np.float64)
        size = max(edges.size - 1, 1)
        target = Path(path).expanduser() if path else None
        return cls(
            bin_edges=edges,
            max_whitened=np.zeros(size, dtype=np.float64),
            max_ratio=np.zeros(size, dtype=np.float64),
            counts=np.zeros(size, dtype=np.int64),
            radius_eps=radius_eps,
            path=target,
        )

    def _bin_indices(self, values: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self.bin_edges, values, side="right") - 1
        return np.clip(indices, 0, self.max_whitened.size - 1)

    def record_chunk(
        self,
        *,
        residual_distances: np.ndarray,
        whitened_distances: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> None:
        if whitened_distances is None or residual_distances.size == 0:
            return
        mask = inclusion_mask.astype(bool, copy=False)
        if not np.any(mask):
            return
        residual = np.asarray(residual_distances[mask], dtype=np.float64, copy=False)
        whitened = np.asarray(whitened_distances[mask], dtype=np.float64, copy=False)
        residual = np.clip(residual, self.radius_eps, None)
        idx = self._bin_indices(residual)
        np.maximum.at(self.max_whitened, idx, whitened)
        np.maximum.at(self.max_ratio, idx, np.divide(whitened, residual))
        np.add.at(self.counts, idx, 1)
        self.samples_total += whitened.size
        self.dirty = True

    def record_false_negatives(
        self,
        *,
        residual_distances: np.ndarray,
        whitened_distances: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> None:
        self.record_chunk(
            residual_distances=residual_distances,
            whitened_distances=whitened_distances,
            inclusion_mask=inclusion_mask,
        )
        self.false_negative_samples += int(np.count_nonzero(inclusion_mask))

    def _cumulative_whitened(self) -> np.ndarray:
        return np.maximum.accumulate(self.max_whitened)

    def _cumulative_ratio(self) -> np.ndarray:
        return np.maximum.accumulate(self.max_ratio)

    def to_dict(self) -> dict:
        payload = {
            "schema": 1,
            "radius_bin_edges": self.bin_edges.tolist(),
            "max_whitened": self._cumulative_whitened().tolist(),
            "max_ratio": self._cumulative_ratio().tolist(),
            "counts": self.counts.tolist(),
            "samples_total": int(self.samples_total),
            "false_negative_samples": int(self.false_negative_samples),
            "radius_eps": float(self.radius_eps),
        }
        return payload

    def dump(self, path: str | Path | None = None, *, force: bool = False) -> None:
        target = Path(path).expanduser() if path else self.path
        if target is None or (not self.dirty and not force):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["generated_at"] = float(time.time())
        payload["bins"] = int(self.max_whitened.size)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.path = target
        self.dirty = False


@dataclass
class ResidualGateLookup:
    radius_bins: np.ndarray
    thresholds: np.ndarray
    margin: float

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        *,
        margin: float,
    ) -> "ResidualGateLookup":
        edges = np.asarray(payload.get("radius_bin_edges", []), dtype=np.float64)
        maxima = np.asarray(payload.get("max_whitened", []), dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Residual gate lookup file is missing 'radius_bin_edges'.")
        bins = edges[1:]
        if maxima.size != bins.size:
            raise ValueError("Residual gate lookup file has mismatched 'max_whitened' length.")
        thresholds = np.maximum.accumulate(maxima.astype(np.float64, copy=False))
        return cls(radius_bins=bins, thresholds=thresholds, margin=float(margin))

    @classmethod
    def load(cls, path: str | Path, *, margin: float) -> "ResidualGateLookup":
        target = Path(path).expanduser()
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_payload(payload, margin=margin)

    def threshold(self, radius: float) -> float:
        if self.radius_bins.size == 0:
            return float(self.margin)
        clipped = np.clip(radius, 0.0, float(self.radius_bins[-1]))
        idx = int(np.searchsorted(self.radius_bins, clipped, side="left"))
        if idx >= self.radius_bins.size:
            idx = self.radius_bins.size - 1
        return float(self.thresholds[idx] + self.margin)


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
    gate1_enabled: bool | None = None
    gate1_alpha: float | None = None
    gate1_margin: float | None = None
    gate1_eps: float | None = None
    gate1_audit: bool | None = None
    gate1_radius_cap: float | None = None
    gate_v32: np.ndarray | None = None
    gate_norm32: np.ndarray | None = None
    gate_stats: ResidualGateTelemetry = field(default_factory=ResidualGateTelemetry)
    gate_profile_path: str | None = None
    gate_profile_bins: int | None = None
    gate_profile: "ResidualGateProfile | None" = None
    gate_lookup_path: str | None = None
    gate_lookup_margin: float | None = None
    gate_lookup: "ResidualGateLookup | None" = None

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

    v_lhs = backend.v_matrix[lhs_indices]
    v_rhs = backend.v_matrix[rhs_indices]
    p_lhs = np.maximum(backend.p_diag[lhs_indices], _EPS)
    p_rhs = np.maximum(backend.p_diag[rhs_indices], _EPS)

    dot_products = v_lhs @ v_rhs.T
    denom = np.sqrt(np.maximum(p_lhs[:, None] * p_rhs[None, :], _EPS * _EPS))
    residual_corr = (kernel_block - dot_products) / denom
    residual_corr = np.clip(residual_corr, -1.0, 1.0)
    return np.sqrt(np.maximum(1.0 - np.abs(residual_corr), 0.0))


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


def _resolve_gate1_config(
    backend: ResidualCorrHostData,
) -> tuple[bool, float, float, float, bool, float]:
    runtime = cx_config.runtime_config()

    enabled = backend.gate1_enabled
    if enabled is None:
        enabled = runtime.residual_gate1_enabled
    alpha = backend.gate1_alpha if backend.gate1_alpha is not None else runtime.residual_gate1_alpha
    margin = backend.gate1_margin if backend.gate1_margin is not None else runtime.residual_gate1_margin
    eps = backend.gate1_eps if backend.gate1_eps is not None else runtime.residual_gate1_eps
    audit = backend.gate1_audit if backend.gate1_audit is not None else runtime.residual_gate1_audit
    cap = backend.gate1_radius_cap if backend.gate1_radius_cap is not None else runtime.residual_gate1_radius_cap

    return bool(enabled), float(alpha), float(margin), float(eps), bool(audit), float(cap)


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
) -> np.ndarray:
    total = int(batch_indices.shape[0])
    if total == 0:
        return np.empty((0, 0), dtype=np.float64)
    result = np.empty((total, total), dtype=np.float64)
    chunk = int(host_backend.chunk_size or 512)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        rows = batch_indices[start:stop]
        kernel_block = host_backend.kernel_provider(rows, batch_indices)
        distances = compute_residual_distances_from_kernel(
            host_backend,
            rows,
            batch_indices,
            kernel_block,
        )
        result[start:stop, :] = distances
    return result


def compute_residual_distances_with_radius(
    backend: ResidualCorrHostData,
    query_index: int,
    chunk_indices: np.ndarray,
    kernel_row: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_idx = np.asarray(chunk_indices, dtype=np.int64)
    if candidate_idx.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        mask = np.zeros((0,), dtype=np.uint8)
        return empty, mask

    kernel_vals = np.asarray(kernel_row, dtype=np.float64)
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
    gate_enabled, gate_alpha, gate_margin, gate_eps, gate_audit, gate_radius_cap = _resolve_gate1_config(backend)
    radius_value = float(radius)
    radius_cap = max(float(gate_radius_cap), 0.0)
    effective_radius = min(radius_value, radius_cap) if radius_cap > 0.0 else radius_value
    effective_radius = max(effective_radius, 0.0)
    gate_keep_mask: np.ndarray | None = None

    gate_can_run = (
        gate_enabled
        and backend.gate_v32 is not None
        and backend.gate_v32.shape[0] == backend.num_points
        and radius_cap > 0.0
    )
    need_whitened_logging = (profile is not None or (gate_audit and gate_can_run)) and backend.gate_v32 is not None
    whitened_distances = (
        _compute_gate1_distances(backend, query_index, candidate_idx)
        if need_whitened_logging
        else None
    )

    if gate_can_run:
        gate_start = time.perf_counter()
        if lookup is not None:
            threshold = lookup.threshold(effective_radius)
        else:
            threshold = gate_alpha * effective_radius + gate_margin
        keep_uint8 = gate1_whitened_mask(
            backend.gate_v32[query_index],
            backend.gate_v32[candidate_idx],
            threshold,
        )
        gate_elapsed = time.perf_counter() - gate_start
        gate_keep_mask = keep_uint8.astype(bool, copy=False)
        total = int(candidate_idx.size)
        kept = int(np.count_nonzero(gate_keep_mask))
        pruned = total - kept
        backend.gate_stats.candidates += total
        backend.gate_stats.kept += kept
        backend.gate_stats.pruned += pruned
        backend.gate_stats.seconds += gate_elapsed
        if kept == total:
            gate_keep_mask = None
        elif kept == 0:
            if gate_audit and pruned > 0:
                audit_whitened = None
                if whitened_distances is not None:
                    audit_whitened = np.asarray(whitened_distances, dtype=np.float64, copy=False)
                _audit_gate1_pruned(
                    backend=backend,
                    query_index=query_index,
                    candidate_idx=candidate_idx,
                    kernel_vals=kernel_vals,
                    keep_mask=np.zeros_like(gate_keep_mask, dtype=bool),
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
            if gate_audit and pruned > 0:
                audit_subset = None
                if whitened_distances is not None:
                    audit_subset = np.asarray(whitened_distances[~gate_keep_mask], dtype=np.float64, copy=False)
                _audit_gate1_pruned(
                    backend=backend,
                    query_index=query_index,
                    candidate_idx=candidate_idx,
                    kernel_vals=kernel_vals,
                    keep_mask=gate_keep_mask,
                    radius=radius,
                    whitened_distances=audit_subset,
                )

    if gate_keep_mask is None:
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
    pruned_idx = np.nonzero(~gate_keep_mask)[0]
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

    distances_surv, mask_surv = compute_distance_chunk(
        v_query=v_query,
        v_chunk=v_chunk_full[survivors_idx],
        kernel_chunk=kernel_vals[survivors_idx],
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
        v_matrix = np.asarray(backend.v_matrix, dtype=np.float64)
        object.__setattr__(backend, "v_norm_sq", np.sum(v_matrix * v_matrix, axis=1))

    runtime = cx_config.runtime_config()
    enabled, alpha, margin, eps, audit, radius_cap = _resolve_gate1_config(backend)
    profile_path = backend.gate_profile_path or runtime.residual_gate1_profile_path
    profile_bins = backend.gate_profile_bins or runtime.residual_gate1_profile_bins
    lookup_path = backend.gate_lookup_path or runtime.residual_gate1_lookup_path
    lookup_margin = (
        backend.gate_lookup_margin if backend.gate_lookup_margin is not None else runtime.residual_gate1_lookup_margin
    )

    radius_max_for_profile = max(1.0, radius_cap if radius_cap > 0.0 else 1.0)
    profile_obj = None
    if profile_path:
        profile_obj = ResidualGateProfile.create(
            bins=int(profile_bins),
            radius_max=float(radius_max_for_profile),
            path=profile_path,
            radius_eps=runtime.residual_radius_floor,
        )

    lookup_obj = None
    if lookup_path:
        lookup_obj = ResidualGateLookup.load(lookup_path, margin=float(lookup_margin))

    need_whitened = enabled or profile_obj is not None or lookup_obj is not None
    if need_whitened:
        v32, n32 = _compute_gate1_whitened(np.asarray(backend.v_matrix, dtype=np.float64))
    else:
        v32 = None
        n32 = None
    object.__setattr__(backend, "gate1_enabled", enabled)
    object.__setattr__(backend, "gate1_alpha", alpha)
    object.__setattr__(backend, "gate1_margin", margin)
    object.__setattr__(backend, "gate1_eps", eps)
    object.__setattr__(backend, "gate1_audit", audit)
    object.__setattr__(backend, "gate1_radius_cap", radius_cap)
    object.__setattr__(backend, "gate_v32", v32)
    object.__setattr__(backend, "gate_norm32", n32)
    object.__setattr__(backend, "gate_stats", ResidualGateTelemetry())
    object.__setattr__(backend, "gate_profile_path", profile_path)
    object.__setattr__(backend, "gate_profile_bins", profile_bins if profile_obj else None)
    object.__setattr__(backend, "gate_profile", profile_obj)
    object.__setattr__(backend, "gate_lookup_path", lookup_path)
    object.__setattr__(backend, "gate_lookup_margin", lookup_margin)
    object.__setattr__(backend, "gate_lookup", lookup_obj)

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
    "ResidualGateTelemetry",
    "ResidualGateProfile",
    "ResidualGateLookup",
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
    "compute_residual_pairwise_matrix",
]
