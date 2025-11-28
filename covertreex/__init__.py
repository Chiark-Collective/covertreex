"""Covertreex: High-performance cover tree for k-NN queries.

Quick Start
-----------
>>> import numpy as np
>>> from covertreex import CoverTree
>>>
>>> # Basic Euclidean k-NN
>>> points = np.random.randn(10000, 3)
>>> tree = CoverTree().fit(points)
>>> neighbors = tree.knn(points[:100], k=10)

Residual Correlation (Vecchia GP)
---------------------------------
>>> from covertreex import ResidualCoverTree
>>>
>>> coords = np.random.randn(10000, 3).astype(np.float32)
>>> tree = ResidualCoverTree(coords, variance=1.0, lengthscale=1.0)
>>> neighbors = tree.knn(k=50)
>>> neighbors = tree.knn(k=50, predecessor_mode=True)  # Vecchia constraint

Classes
-------
ResidualCoverTree : Simplified API for residual correlation k-NN (Vecchia GP).
CoverTree : General-purpose cover tree for Euclidean and custom metrics.
Runtime : Configuration for backend, metric, and engine selection.
"""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("covertreex")
except Exception:  # pragma: no cover - best effort during local development
    __version__ = "0.0.1"

# Primary user-facing API
from .api import CoverTree, Runtime, Residual, PCCT
from .residual_tree import ResidualCoverTree

# Internal/advanced APIs
from .engine import CoverTree as EngineCoverTree, build_tree, get_engine
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
    MlpackCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
    has_mlpack_cover_tree,
)

__all__ = [
    # Primary API
    "__version__",
    "ResidualCoverTree",
    "CoverTree",
    "Runtime",
    "Residual",
    "PCCT",  # Deprecated alias
    # Engine-level API
    "build_tree",
    "get_engine",
    "EngineCoverTree",
    # Internal
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
    "MlpackCoverTreeBaseline",
    "has_external_cover_tree",
    "has_gpboost_cover_tree",
    "has_mlpack_cover_tree",
]
