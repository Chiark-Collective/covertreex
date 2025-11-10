from __future__ import annotations


class ResidualPairwiseCacheError(RuntimeError):
    """Raised when residual traversal lacks cached pairwise kernels."""

