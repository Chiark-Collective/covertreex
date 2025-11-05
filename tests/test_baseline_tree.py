import numpy as np
import pytest

from covertreex import BaselineCoverTree


def _random_points(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float64)


def _bruteforce_nearest(points: np.ndarray, query: np.ndarray) -> tuple[int, float]:
    dists = np.linalg.norm(points - query, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def _bruteforce_knn(points: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    dists = np.linalg.norm(points - query, axis=1)
    return np.argsort(dists)[:k]


def test_baseline_nearest_matches_bruteforce():
    points = _random_points(64, 3, seed=7)
    tree = BaselineCoverTree.from_points(points)

    query = _random_points(1, 3, seed=11)[0]
    idx, dist = tree.nearest(query)
    ref_idx, ref_dist = _bruteforce_nearest(points, query)

    assert idx == ref_idx
    assert pytest.approx(dist, rel=1e-12, abs=1e-12) == ref_dist


def test_baseline_knn_matches_bruteforce():
    points = _random_points(128, 4, seed=13)
    tree = BaselineCoverTree.from_points(points)
    query = _random_points(1, 4, seed=19)[0]

    indices, distances = tree.knn(query, k=5, return_distances=True)
    ref_indices = _bruteforce_knn(points, query, 5)

    assert np.array_equal(indices, ref_indices)
    ref_distances = np.linalg.norm(points[ref_indices] - query, axis=1)
    assert np.allclose(distances, ref_distances)


def test_baseline_knn_batched_queries():
    points = _random_points(96, 5, seed=23)
    tree = BaselineCoverTree.from_points(points)
    queries = _random_points(3, 5, seed=29)

    indices, distances = tree.knn(queries, k=4, return_distances=True)
    assert indices.shape == (3, 4)
    assert distances.shape == (3, 4)
    for i, query in enumerate(queries):
        ref_indices = _bruteforce_knn(points, query, 4)
        assert np.array_equal(indices[i], ref_indices)
        ref_dist = np.linalg.norm(points[ref_indices] - query, axis=1)
        assert np.allclose(distances[i], ref_dist)


def test_baseline_knn_rejects_invalid_k():
    points = _random_points(10, 2, seed=31)
    tree = BaselineCoverTree.from_points(points)
    with pytest.raises(ValueError):
        tree.knn(points[0], k=0)
    with pytest.raises(ValueError):
        tree.knn(points[0], k=points.shape[0] + 1)
