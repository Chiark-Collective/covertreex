from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from covertreex import config as cx_config

ArrayLike = Any


class PairwiseKernel(Protocol):
    def __call__(self, backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
        ...


class PointwiseKernel(Protocol):
    def __call__(self, backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
        ...


@dataclass(frozen=True)
class Metric:
    """Container for metric kernels used by tree algorithms."""

    name: str
    pairwise_kernel: PairwiseKernel
    pointwise_kernel: PointwiseKernel

    def pairwise(self, backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
        return self.pairwise_kernel(backend, lhs, rhs)

    def pointwise(self, backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
        return self.pointwise_kernel(backend, lhs, rhs)


class MetricRegistry:
    """Minimal registry for runtime-selectable metrics."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}

    def register(self, metric: Metric, *, overwrite: bool = False) -> None:
        name = metric.name.lower()
        if not overwrite and name in self._metrics:
            raise ValueError(f"Metric '{metric.name}' already registered.")
        self._metrics[name] = metric

    def get(self, name: str) -> Metric:
        key = name.lower()
        if key not in self._metrics:
            raise KeyError(f"Metric '{name}' not registered.")
        return self._metrics[key]

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._metrics.keys()))


def _ensure_2d(backend: "TreeBackend", array: ArrayLike) -> ArrayLike:
    arr = backend.asarray(array, dtype=backend.default_float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _euclidean_pairwise(backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
    xp = backend.xp
    lhs_arr = _ensure_2d(backend, lhs)
    rhs_arr = _ensure_2d(backend, rhs)
    if lhs_arr.size == 0 or rhs_arr.size == 0:
        shape = (lhs_arr.shape[0], rhs_arr.shape[0])
        return backend.device_put(xp.zeros(shape, dtype=backend.default_float))
    diff = lhs_arr[:, None, :] - rhs_arr[None, :, :]
    dist_sq = xp.sum(diff * diff, axis=-1)
    return backend.device_put(xp.sqrt(dist_sq))


def _euclidean_pointwise(backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
    xp = backend.xp
    lhs_arr = backend.asarray(lhs, dtype=backend.default_float)
    rhs_arr = backend.asarray(rhs, dtype=backend.default_float)
    if lhs_arr.shape != rhs_arr.shape:
        raise ValueError("Pointwise metric operands must have identical shapes.")
    if lhs_arr.ndim == 1:
        diff = lhs_arr - rhs_arr
        return backend.device_put(xp.sqrt(xp.sum(diff * diff)))
    diff = lhs_arr - rhs_arr
    return backend.device_put(xp.sqrt(xp.sum(diff * diff, axis=-1)))


def _load_runtime_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(
        Metric(
            name="euclidean",
            pairwise_kernel=_euclidean_pairwise,
            pointwise_kernel=_euclidean_pointwise,
        )
    )
    return registry


_RESIDUAL_PAIRWISE_IMPL: Optional[PairwiseKernel] = None
_RESIDUAL_POINTWISE_IMPL: Optional[PointwiseKernel] = None


def _residual_pairwise(backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
    if _RESIDUAL_PAIRWISE_IMPL is None:
        raise RuntimeError(
            "Residual-correlation metric is not configured. "
            "Call configure_residual_metric() with pairwise/pointwise handlers."
        )
    return _RESIDUAL_PAIRWISE_IMPL(backend, lhs, rhs)


def _residual_pointwise(backend: "TreeBackend", lhs: ArrayLike, rhs: ArrayLike) -> ArrayLike:
    if _RESIDUAL_POINTWISE_IMPL is not None:
        return _RESIDUAL_POINTWISE_IMPL(backend, lhs, rhs)
    if _RESIDUAL_PAIRWISE_IMPL is None:
        raise RuntimeError(
            "Residual-correlation metric is not configured. "
            "Call configure_residual_metric() with pairwise/pointwise handlers."
        )

    lhs_arr = backend.asarray(lhs, dtype=backend.default_float)
    rhs_arr = backend.asarray(rhs, dtype=backend.default_float)
    if lhs_arr.ndim == 1:
        lhs_arr = lhs_arr[None, :]
    if rhs_arr.ndim == 1:
        rhs_arr = rhs_arr[None, :]
    pairwise_vals = _RESIDUAL_PAIRWISE_IMPL(backend, lhs_arr, rhs_arr)
    pairwise_vals = backend.asarray(pairwise_vals, dtype=backend.default_float)
    xp = backend.xp
    if pairwise_vals.ndim == 0:
        return backend.device_put(pairwise_vals)
    if pairwise_vals.ndim == 1:
        return backend.device_put(pairwise_vals)
    diag = xp.diagonal(pairwise_vals)
    return backend.device_put(diag)


_REGISTRY = _load_runtime_registry()
_REGISTRY.register(
    Metric(
        name="residual_correlation",
        pairwise_kernel=_residual_pairwise,
        pointwise_kernel=_residual_pointwise,
    )
)


def get_metric(name: str | None = None) -> Metric:
    """Return a registered metric, defaulting to the runtime-selected metric."""

    if name is None:
        name = cx_config.runtime_config().metric
    return _REGISTRY.get(name)


def configure_residual_metric(
    *,
    pairwise: PairwiseKernel,
    pointwise: PointwiseKernel | None = None,
) -> None:
    """Install handlers for the residual-correlation metric."""

    global _RESIDUAL_PAIRWISE_IMPL, _RESIDUAL_POINTWISE_IMPL
    _RESIDUAL_PAIRWISE_IMPL = pairwise
    _RESIDUAL_POINTWISE_IMPL = pointwise


def reset_residual_metric() -> None:
    """Clear any configured residual metric handlers (used in tests)."""

    global _RESIDUAL_PAIRWISE_IMPL, _RESIDUAL_POINTWISE_IMPL
    _RESIDUAL_PAIRWISE_IMPL = None
    _RESIDUAL_POINTWISE_IMPL = None
    try:  # keep optional to avoid circular import issues during interpreter teardown
        from covertreex.metrics.residual import set_residual_backend  # type: ignore

        set_residual_backend(None)
    except Exception:  # pragma: no cover - defensive
        pass


def available_metrics() -> Tuple[str, ...]:
    return _REGISTRY.names()


# Avoid circular import at module load.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from covertreex.core.tree import TreeBackend


__all__ = [
    "Metric",
    "MetricRegistry",
    "available_metrics",
    "configure_residual_metric",
    "get_metric",
    "reset_residual_metric",
]
