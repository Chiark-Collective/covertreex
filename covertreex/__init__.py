"""Covertreex: High-performance cover tree for k-NN queries.

Quick Start
-----------
>>> import numpy as np
>>> from covertreex import cover_tree
>>>
>>> # Basic Euclidean k-NN
>>> tree = cover_tree(coords)
>>> neighbors = tree.knn(k=10)

Residual Correlation (Vecchia GP)
---------------------------------
>>> from covertreex import cover_tree
>>> from covertreex.kernels import Matern52
>>>
>>> # Option 1: Provide a kernel (we build V-matrix)
>>> tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0))
>>> neighbors = tree.knn(k=50)
>>>
>>> # Option 2: Provide pre-computed V-matrix (from your GP)
>>> tree = cover_tree(coords, v_matrix=V, p_diag=p_diag)
>>> neighbors = tree.knn(k=50, predecessor_mode=True)  # Vecchia constraint

Functions
---------
cover_tree : Build a cover tree (recommended entry point).

Classes
-------
CoverTree : General-purpose cover tree for Euclidean and custom metrics.
Runtime : Configuration for backend, metric, and engine selection.
Matern52, RBF : GP kernel classes for residual correlation metric.
"""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("covertreex")
except Exception:  # pragma: no cover - best effort during local development
    __version__ = "0.0.1"

# Primary user-facing API
from .api import CoverTree, Runtime, Residual, PCCT
from .api.factory import cover_tree
from .kernels import Kernel, RBF, Matern52
from .residual_tree import ResidualCoverTree  # Deprecated, use cover_tree instead

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
    "cover_tree",
    "CoverTree",
    "Runtime",
    "Residual",
    # Kernels
    "Kernel",
    "RBF",
    "Matern52",
    # Deprecated
    "ResidualCoverTree",  # Deprecated, use cover_tree instead
    "PCCT",  # Deprecated alias for CoverTree
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
