"""Factory function for building cover trees with simplified API.

This module provides a cleaner interface for common use cases, hiding the
complexity of Runtime and residual backend configuration.

Example
-------
>>> from covertreex import cover_tree
>>> from covertreex.kernels import Matern52
>>>
>>> # Euclidean distance (default)
>>> tree = cover_tree(coords)
>>>
>>> # Residual correlation with kernel (we build V-matrix)
>>> tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0))
>>>
>>> # Residual correlation with pre-computed V-matrix (user provides V)
>>> tree = cover_tree(coords, v_matrix=V, p_diag=p_diag, kernel_diag=k_diag)
>>>
>>> # Query
>>> neighbors = tree.knn(k=50)
>>> neighbors = tree.knn(k=50, predecessor_mode=True)  # Vecchia constraint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from covertreex.kernels import Kernel

from covertreex import config as cx_config
from covertreex.engine import CoverTree as EngineCoverTree


@dataclass
class BuiltCoverTree:
    """Wrapper providing a consistent k-NN interface for built cover trees.

    This class wraps the internal tree representation and provides a simple
    interface for k-NN queries.
    """

    _tree: EngineCoverTree | Any
    _n_points: int
    _context: cx_config.RuntimeContext
    _is_residual: bool = False

    def knn(
        self,
        k: int,
        *,
        queries: np.ndarray | Sequence[int] | None = None,
        return_distances: bool = False,
        predecessor_mode: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors.

        Parameters
        ----------
        k : int
            Number of neighbors to return.
        queries : array_like, optional
            Query indices (for residual) or query points (for Euclidean).
            If None, queries all points (0..N-1).
        return_distances : bool, default=False
            If True, also return distances.
        predecessor_mode : bool, default=False
            If True, enforce Vecchia constraint: neighbor j < query i.

        Returns
        -------
        neighbors : ndarray, shape (Q, k)
            Neighbor indices. Padded with -1 when fewer than k available.
        distances : ndarray, optional
            Distances to neighbors (only if return_distances=True).
        """
        # Check if this is an API CoverTree (has different interface)
        from covertreex.api.pcct import CoverTree as APICoverTree

        if isinstance(self._tree, APICoverTree):
            # API CoverTree: knn(query_points, k=...) without context
            if queries is None:
                query_data = np.arange(self._n_points, dtype=np.int64).reshape(-1, 1)
            else:
                query_data = np.asarray(queries)
                if query_data.ndim == 1:
                    query_data = query_data.reshape(-1, 1)

            return self._tree.knn(
                query_data,
                k=k,
                return_distances=return_distances,
                predecessor_mode=predecessor_mode,
            )

        # Engine CoverTree: knn(query_points, k=..., context=...)
        if queries is None:
            query_data = np.arange(self._n_points, dtype=np.int64).reshape(-1, 1)
        else:
            query_data = np.asarray(queries)
            if query_data.ndim == 1:
                query_data = query_data.reshape(-1, 1)

        return self._tree.knn(
            query_data,
            k=k,
            return_distances=return_distances,
            predecessor_mode=predecessor_mode,
            context=self._context,
        )

    @property
    def n_points(self) -> int:
        """Number of points in the tree."""
        return self._n_points

    @property
    def build_time(self) -> float | None:
        """Tree build time in seconds."""
        return self._tree.build_seconds

    def __repr__(self) -> str:
        metric = "residual" if self._is_residual else "euclidean"
        return f"BuiltCoverTree(n={self._n_points}, metric={metric!r})"


def cover_tree(
    coords: np.ndarray,
    *,
    # Residual metric options (mutually exclusive approaches)
    kernel: "Kernel | None" = None,
    v_matrix: np.ndarray | None = None,
    p_diag: np.ndarray | None = None,
    kernel_diag: np.ndarray | None = None,
    # Kernel-based V-matrix building options
    inducing_count: int = 512,
    seed: int = 42,
    chunk_size: int = 512,
    # Engine options
    engine: str = "rust-natural",
) -> EngineCoverTree:
    """Build a cover tree for k-NN queries.

    This is the recommended entry point for building cover trees. It handles
    all configuration complexity internally.

    Parameters
    ----------
    coords : ndarray, shape (N, D)
        Spatial coordinates. Converted to float32 internally.

    kernel : Kernel, optional
        GP kernel for residual correlation metric. If provided, V-matrix
        is computed internally using inducing point approximation.
        Use kernels from `covertreex.kernels` (RBF, Matern52).

    v_matrix : ndarray, shape (N, M), optional
        Pre-computed whitened inducing features. If provided along with
        p_diag, uses residual correlation metric with these matrices.
        This is the preferred option when you already have V from your GP.

    p_diag : ndarray, shape (N,), optional
        Diagonal residual variances. Required if v_matrix is provided.

    kernel_diag : ndarray, shape (N,), optional
        Diagonal of kernel matrix K(x_i, x_i). If not provided when using
        v_matrix, defaults to ones (assumes variance=1).

    inducing_count : int, default=512
        Number of inducing points when building V from kernel.

    seed : int, default=42
        Random seed for inducing point selection.

    chunk_size : int, default=512
        Batch size for tree construction.

    engine : str, default="rust-natural"
        Execution engine. Options: "rust-natural" (default, best for predecessor_mode),
        "rust-hilbert" (fastest builds), "python-numba" (reference).

    Returns
    -------
    tree : CoverTree
        Built cover tree ready for k-NN queries.

    Examples
    --------
    Euclidean k-NN:

        >>> tree = cover_tree(coords)
        >>> neighbors = tree.knn(k=10)

    Residual correlation with kernel:

        >>> from covertreex.kernels import Matern52
        >>> tree = cover_tree(coords, kernel=Matern52(lengthscale=2.0))
        >>> neighbors = tree.knn(k=50)

    Residual correlation with pre-computed V (from your GP):

        >>> tree = cover_tree(coords, v_matrix=V, p_diag=p_diag)
        >>> neighbors = tree.knn(k=50, predecessor_mode=True)

    Notes
    -----
    For Vecchia GP applications, use `predecessor_mode=True` in knn() to
    ensure query i only returns neighbors j where j < i.
    """
    from covertreex import config as cx_config
    from covertreex.engine import RustHilbertEngine, RustNaturalEngine
    from covertreex.metrics.residual import build_residual_backend, configure_residual_correlation
    from covertreex.metrics.residual.core import ResidualCorrHostData
    from covertreex.runtime.model import RuntimeModel

    coords_f32 = np.asarray(coords, dtype=np.float32)
    n_points = coords_f32.shape[0]

    # Determine metric and build backend if needed
    residual_backend = None
    residual_params = None
    use_residual = kernel is not None or v_matrix is not None

    if use_residual:
        if kernel is not None and v_matrix is not None:
            raise ValueError("Provide either 'kernel' or 'v_matrix', not both.")

        if v_matrix is not None:
            # User provides pre-computed V-matrix
            if p_diag is None:
                raise ValueError("p_diag is required when v_matrix is provided.")

            v_matrix_f32 = np.asarray(v_matrix, dtype=np.float32)
            p_diag_f32 = np.asarray(p_diag, dtype=np.float32)

            if kernel_diag is None:
                # Default to variance=1 (ones)
                kernel_diag_f32 = np.ones(n_points, dtype=np.float32)
            else:
                kernel_diag_f32 = np.asarray(kernel_diag, dtype=np.float32)

            # Build a minimal kernel provider (not used for distance computation
            # when V is provided, but needed for interface compatibility)
            def _dummy_kernel_provider(row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
                # This shouldn't be called when using pre-computed V
                return np.zeros((row_idx.size, col_idx.size), dtype=np.float32)

            residual_backend = ResidualCorrHostData(
                v_matrix=v_matrix_f32,
                p_diag=p_diag_f32,
                kernel_diag=kernel_diag_f32,
                kernel_provider=_dummy_kernel_provider,
                chunk_size=chunk_size,
                # Include original coords for Rust backend
                kernel_points_f32=coords_f32,
            )
            residual_params = {
                "variance": 1.0,
                "lengthscale": 1.0,
                "kernel_type": 0,
            }
        else:
            # Build V-matrix from kernel
            residual_backend = build_residual_backend(
                coords_f32,
                seed=seed,
                inducing_count=inducing_count,
                variance=kernel.variance,
                lengthscale=kernel.lengthscale,
                chunk_size=chunk_size,
                kernel_type=kernel.kernel_type,
            )
            residual_params = {
                "variance": kernel.variance,
                "lengthscale": kernel.lengthscale,
                "kernel_type": kernel.kernel_type,
            }

    # For Euclidean metric, use the API CoverTree which handles it properly
    if not use_residual:
        from covertreex.api.pcct import CoverTree as APICoverTree
        from covertreex.api.runtime import Runtime

        runtime = Runtime(metric="euclidean", engine=engine, enable_rust=True)
        api_tree = APICoverTree(runtime).fit(coords_f32)
        ctx = runtime.activate()
        return BuiltCoverTree(
            _tree=api_tree,
            _n_points=n_points,
            _context=ctx,
            _is_residual=False,
        )

    # For residual metric, select engine
    if engine in ("rust-hilbert", "hilbert"):
        engine_impl = RustHilbertEngine()
    elif engine in ("rust-natural", "natural"):
        engine_impl = RustNaturalEngine()
    else:
        # Fall back to python-numba for residual
        from covertreex.api.pcct import CoverTree as APICoverTree
        from covertreex.api.runtime import Runtime

        runtime = Runtime(metric="residual_correlation", engine=engine)
        ctx = runtime.activate()

        if residual_backend is not None:
            configure_residual_correlation(residual_backend, context=ctx)

        api_tree = APICoverTree(runtime).fit(coords_f32)
        return BuiltCoverTree(
            _tree=api_tree,
            _n_points=n_points,
            _context=ctx,
            _is_residual=True,
        )

    # Build runtime config
    runtime_model = RuntimeModel(
        metric="residual_correlation",
        engine=engine,
        enable_rust=True,
    )
    runtime_cfg = runtime_model.to_runtime_config()
    ctx = cx_config.configure_runtime(runtime_cfg)

    if residual_backend is not None:
        configure_residual_correlation(residual_backend, context=ctx)

    # Build tree
    tree = engine_impl.build(
        coords_f32,
        runtime=runtime_cfg,
        residual_backend=residual_backend,
        residual_params=residual_params,
        compute_predecessor_bounds=True,  # Enable Vecchia optimization
    )

    return BuiltCoverTree(
        _tree=tree,
        _n_points=n_points,
        _context=ctx,
        _is_residual=True,
    )


__all__ = ["cover_tree", "BuiltCoverTree"]
