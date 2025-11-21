from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.random import Generator, default_rng

Array = np.ndarray


def _ensure_rng(rng: Generator | None) -> Generator:
    return rng or default_rng()


def gaussian_points(
    rng: Generator | None,
    count: int,
    dimension: int,
    *,
    dtype: np.dtype | type[np.floating] = np.float64,
) -> Array:
    """Sample `count` Gaussian points with the requested dimensionality."""

    generator = _ensure_rng(rng)
    if count <= 0 or dimension <= 0:
        return np.zeros((max(count, 0), max(dimension, 0)), dtype=dtype)
    samples = generator.normal(loc=0.0, scale=1.0, size=(count, dimension))
    return np.asarray(samples, dtype=dtype)


def gaussian_dataset(
    rng: Generator | None,
    *,
    tree_points: int,
    queries: int,
    dimension: int,
    dtype: np.dtype | type[np.floating] = np.float64,
) -> Tuple[Array, Array]:
    """Return a tuple `(points, queries)` drawn from the same Gaussian."""

    generator = _ensure_rng(rng)
    points = gaussian_points(generator, tree_points, dimension, dtype=dtype)
    query_points = gaussian_points(generator, queries, dimension, dtype=dtype)
    return points, query_points


def rbf_kernel(
    x: Array,
    y: Array,
    *,
    variance: float,
    lengthscale: float,
) -> Array:
    """Compute an RBF kernel between `x` and `y` using numpy arrays."""

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("RBF kernel expects 2D arrays.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("RBF kernel received mismatched dimensionality.")
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2, dtype=np.float64)
    denom = max(lengthscale, 1e-12)
    scaled = -0.5 * sq_dist / (denom * denom)
    return float(variance) * np.exp(scaled, dtype=np.float64)
