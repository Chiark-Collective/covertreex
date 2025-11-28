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

EUCLIDEAN K-NN
--------------
    import numpy as np
    from covertreex import CoverTree

    points = np.random.randn(10000, 3)
    tree = CoverTree().fit(points)
    neighbors = tree.knn(points[:100], k=10)
    neighbors, distances = tree.knn(points[:100], k=10, return_distances=True)

RESIDUAL CORRELATION K-NN (Vecchia GP)
--------------------------------------
For Gaussian process applications with Vecchia approximations:

    import numpy as np
    from covertreex import ResidualCoverTree

    coords = np.random.randn(10000, 3).astype(np.float32)
    tree = ResidualCoverTree(
        coords,
        variance=1.0,
        lengthscale=1.0,
        inducing_count=512,
    )

    # Query all points
    neighbors = tree.knn(k=50)

    # Query specific indices
    neighbors = tree.knn(k=50, queries=[100, 200, 300])

    # Predecessor constraint: neighbor j must have j < query i
    neighbors = tree.knn(k=50, predecessor_mode=True)

    # With distances
    neighbors, distances = tree.knn(k=50, return_distances=True)

RESIDUALCOVERTREE PARAMETERS
----------------------------
    coords           # (N, D) spatial coordinates
    variance=1.0     # Kernel variance
    lengthscale=1.0  # Kernel lengthscale (scalar or per-dimension array)
    inducing_count=512  # Number of inducing points for V-matrix
    seed=42          # Random seed for inducing point selection
    engine="rust-hilbert"  # Execution engine (or "rust-natural")
    kernel_type="rbf"      # Kernel: "rbf" or "matern52"

API REFERENCE
-------------
    from covertreex import ResidualCoverTree, CoverTree
    help(ResidualCoverTree)  # Vecchia GP k-NN (recommended)
    help(CoverTree)          # General-purpose k-NN

BENCHMARKING CLI
----------------
    python -m cli.pcct query --dimension 3 --tree-points 8192 --k 10
    python -m cli.pcct doctor  # Check environment

LINKS
-----
    PyPI:   https://pypi.org/project/covertreex/
    GitHub: https://github.com/Chiark-Collective/covertreex

================================================================================
"""


def main() -> None:
    """Print quick-start guide."""
    print(QUICKSTART)


if __name__ == "__main__":
    main()
