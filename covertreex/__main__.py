#!/usr/bin/env python
"""Quick-start guide for covertreex library usage.

Run with: python -m covertreex

This module intentionally avoids importing covertreex internals to provide
a fast, clean startup for displaying help text.
"""

from __future__ import annotations

import sys

QUICKSTART = """\
================================================================================
                              COVERTREEX
      High-performance cover tree for k-NN queries (Vecchia GP optimized)
================================================================================

INSTALLATION
------------
    pip install covertreex

BASIC USAGE (Euclidean k-NN)
----------------------------
    import numpy as np
    from covertreex import CoverTree

    # Build tree from points
    points = np.random.randn(10000, 3)
    tree = CoverTree().fit(points)

    # Query k nearest neighbors
    neighbors = tree.knn(points[:100], k=10)

    # With distances
    neighbors, distances = tree.knn(points[:100], k=10, return_distances=True)

RESIDUAL CORRELATION METRIC (Vecchia GP)
----------------------------------------
For Gaussian process applications with Vecchia approximations:

    import numpy as np
    from covertreex import CoverTree, Runtime
    from covertreex.metrics.residual import (
        build_residual_backend,
        configure_residual_correlation,
    )

    # Your spatial coordinates
    coords = np.random.randn(10000, 3).astype(np.float32)

    # Build residual backend (V-matrix from inducing points)
    backend = build_residual_backend(
        coords,
        seed=42,
        inducing_count=512,
        variance=1.0,
        lengthscale=1.0,
        kernel_type=0,  # 0=RBF, 1=Matern52
    )

    # Configure and build tree
    runtime = Runtime(metric="residual_correlation", engine="rust-hilbert")
    ctx = runtime.activate()
    configure_residual_correlation(backend, context=ctx)

    # Query using point indices
    query_indices = np.arange(1000, dtype=np.int64).reshape(-1, 1)
    tree = CoverTree(runtime).fit(query_indices)
    neighbors = tree.knn(query_indices, k=50)

PREDECESSOR CONSTRAINT (Vecchia GP)
-----------------------------------
For Vecchia approximations, query i must only return neighbors j < i:

    from covertreex.engine import RustHilbertEngine

    engine = RustHilbertEngine()
    tree = engine.build(
        coords,
        runtime=runtime.to_config(),
        residual_backend=backend,
        residual_params={"variance": 1.0, "lengthscale": 1.0},
        compute_predecessor_bounds=True,  # Enables subtree pruning optimization
    )

    # Query with predecessor constraint
    neighbors = tree.knn(query_indices, k=50, predecessor_mode=True)
    # Result: neighbors[i] contains only indices j where j < query_indices[i]

    # Early indices have fewer neighbors (query 0 has none, query 1 has at most 1)
    # Padded with -1 when fewer than k valid neighbors exist

ENGINE SELECTION
----------------
    # Fastest (Rust + Hilbert curve ordering)
    runtime = Runtime(engine="rust-hilbert")

    # Rust with natural ordering
    runtime = Runtime(engine="rust-natural")

    # Python/Numba reference implementation
    runtime = Runtime(engine="python-numba")

API REFERENCE
-------------
    from covertreex import CoverTree, Runtime, Residual
    help(CoverTree)   # Main tree class
    help(Runtime)     # Configuration options
    help(Residual)    # Residual metric setup

BENCHMARKING CLI
----------------
For performance testing (separate from library usage):

    python -m cli.pcct query --dimension 3 --tree-points 8192 --k 10
    python -m cli.pcct doctor  # Check environment

LINKS
-----
    PyPI:   https://pypi.org/project/covertreex/
    GitHub: https://github.com/Chiark-Collective/covertreex
    Issues: https://github.com/Chiark-Collective/covertreex/issues

================================================================================
"""


def main() -> None:
    """Print quick-start guide."""
    print(QUICKSTART)


if __name__ == "__main__":
    main()
