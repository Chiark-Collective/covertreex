"""Parallel compressed cover tree library."""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("covertreex")
except Exception:  # pragma: no cover - best effort during local development
    __version__ = "0.0.1"

from .core import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
    available_metrics,
    configure_residual_metric,
    get_metric,
    reset_residual_metric,
)
from .metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
)
from .baseline import (
    BaselineCoverTree,
    BaselineNode,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
)

__all__ = [
    "__version__",
    "PCCTree",
    "TreeBackend",
    "TreeLogStats",
    "available_metrics",
    "configure_residual_metric",
    "configure_residual_correlation",
    "get_metric",
    "reset_residual_metric",
    "ResidualCorrHostData",
    "BaselineCoverTree",
    "BaselineNode",
    "ExternalCoverTreeBaseline",
    "GPBoostCoverTreeBaseline",
    "has_external_cover_tree",
    "has_gpboost_cover_tree",
]
