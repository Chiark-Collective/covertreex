"""Simplified high-level API for residual correlation cover trees.

This module provides a single-class interface that handles all the setup
complexity for residual correlation k-NN queries used in Vecchia GP.

Example
-------
>>> from covertreex import ResidualCoverTree
>>> import numpy as np
>>>
>>> coords = np.random.randn(10000, 3).astype(np.float32)
>>> tree = ResidualCoverTree(coords, variance=1.0, lengthscale=1.0)
>>> neighbors = tree.knn(k=50)  # k nearest for all points
>>> neighbors = tree.knn(k=50, predecessor_mode=True)  # Vecchia constraint
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class ResidualCoverTree:
    """High-performance cover tree for residual correlation k-NN queries.

    This is the recommended API for Vecchia-style Gaussian process applications.
    It handles all setup internally: V-matrix computation, backend configuration,
    and tree construction with Hilbert curve optimization.

    Parameters
    ----------
    coords : array_like
        Spatial coordinates, shape (N, D). Converted to float32 internally.
    variance : float, default=1.0
        Kernel variance parameter.
    lengthscale : float or array_like, default=1.0
        Kernel lengthscale(s). Scalar for isotropic, array for ARD.
    inducing_count : int, default=512
        Number of inducing points for low-rank V-matrix approximation.
    seed : int, default=42
        Random seed for inducing point selection.
    engine : str, default="rust-hilbert"
        Execution engine. Options: "rust-hilbert" (fastest), "rust-natural".
    kernel_type : str or int, default="rbf"
        Kernel type: "rbf" (0) or "matern52" (1).
    chunk_size : int, default=512
        Batch size for tree construction.

    Examples
    --------
    Basic usage:

    >>> tree = ResidualCoverTree(coords)
    >>> neighbors = tree.knn(k=50)

    With Vecchia predecessor constraint (neighbor j must have j < query i):

    >>> neighbors = tree.knn(k=50, predecessor_mode=True)

    Query specific points:

    >>> neighbors = tree.knn(k=50, queries=[100, 200, 300])

    Get distances too:

    >>> neighbors, distances = tree.knn(k=50, return_distances=True)
    """

    def __init__(
        self,
        coords: np.ndarray,
        *,
        variance: float = 1.0,
        lengthscale: float | np.ndarray = 1.0,
        inducing_count: int = 512,
        seed: int = 42,
        engine: str = "rust-hilbert",
        kernel_type: str | int = "rbf",
        chunk_size: int = 512,
    ):
        # Normalize kernel_type
        if isinstance(kernel_type, str):
            kernel_type = 0 if kernel_type.lower() == "rbf" else 1

        # Store params
        self._coords = np.asarray(coords, dtype=np.float32)
        self._variance = float(variance)
        self._lengthscale = lengthscale
        self._inducing_count = inducing_count
        self._seed = seed
        self._engine = engine
        self._kernel_type = kernel_type
        self._chunk_size = chunk_size
        self._n_points = self._coords.shape[0]

        # Lazy initialization
        self._tree = None
        self._handle = None
        self._backend = None

    def _ensure_built(self) -> None:
        """Build tree on first use (lazy initialization)."""
        if self._tree is not None:
            return

        from .engine import RustHilbertEngine, RustNaturalEngine
        from .metrics.residual import build_residual_backend
        from .runtime.model import RuntimeModel

        # Build V-matrix backend
        self._backend = build_residual_backend(
            self._coords,
            seed=self._seed,
            inducing_count=self._inducing_count,
            variance=self._variance,
            lengthscale=self._lengthscale,
            chunk_size=self._chunk_size,
            kernel_type=self._kernel_type,
        )

        # Select engine
        if self._engine in ("rust-hilbert", "hilbert"):
            engine_impl = RustHilbertEngine()
        else:
            engine_impl = RustNaturalEngine()

        # Build runtime config with defaults + our overrides
        runtime_model = RuntimeModel(
            metric="residual_correlation",
            engine=self._engine,
            enable_rust=True,
        )
        runtime_cfg = runtime_model.to_runtime_config()

        residual_params = {
            "variance": self._variance,
            "lengthscale": self._lengthscale,
            "kernel_type": self._kernel_type,
        }

        self._tree = engine_impl.build(
            self._coords,
            runtime=runtime_cfg,
            residual_backend=self._backend,
            residual_params=residual_params,
            compute_predecessor_bounds=True,  # Enable Vecchia optimization
        )
        self._handle = self._tree.handle
        self._runtime_cfg = runtime_cfg

    def knn(
        self,
        k: int,
        *,
        queries: np.ndarray | Sequence[int] | None = None,
        return_distances: bool = False,
        predecessor_mode: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using residual correlation distance.

        Parameters
        ----------
        k : int
            Number of neighbors to return.
        queries : array_like, optional
            Indices of query points. If None, queries all points (0..N-1).
        return_distances : bool, default=False
            If True, also return distances.
        predecessor_mode : bool, default=False
            If True, enforce Vecchia constraint: for query index i, only
            return neighbors j where j < i. Early indices will have fewer
            than k neighbors (padded with -1).

        Returns
        -------
        neighbors : ndarray
            Neighbor indices, shape (Q, k) where Q is number of queries.
            Padded with -1 when fewer than k neighbors available.
        distances : ndarray, optional
            Distances to neighbors, same shape. Only returned if
            return_distances=True.
        """
        self._ensure_built()

        # Default: query all points
        if queries is None:
            query_indices = np.arange(self._n_points, dtype=np.int64)
        else:
            query_indices = np.asarray(queries, dtype=np.int64).ravel()

        # Reshape for engine (expects 2D)
        query_2d = query_indices.reshape(-1, 1)

        # Get context
        from . import config as cx_config

        ctx = cx_config.configure_runtime(self._runtime_cfg)

        # Run query
        result = self._tree.engine.knn(
            self._tree,
            query_2d,
            k=k,
            return_distances=return_distances,
            predecessor_mode=predecessor_mode,
            context=ctx,
            runtime=self._runtime_cfg,
        )

        return result

    @property
    def n_points(self) -> int:
        """Number of points in the tree."""
        return self._n_points

    @property
    def build_time(self) -> float | None:
        """Tree build time in seconds, or None if not yet built."""
        if self._tree is None:
            return None
        return self._tree.build_seconds

    @property
    def coords(self) -> np.ndarray:
        """Spatial coordinates (read-only view)."""
        return self._coords.view()

    def __repr__(self) -> str:
        built = "built" if self._tree is not None else "not built"
        return (
            f"ResidualCoverTree(n={self._n_points}, engine={self._engine!r}, {built})"
        )


__all__ = ["ResidualCoverTree"]
