import numpy as np
import pytest

from covertreex import MlpackCoverTreeBaseline, has_mlpack_cover_tree


@pytest.mark.skipif(
    not has_mlpack_cover_tree(), reason="mlpack cover tree baseline not available"
)
def test_mlpack_baseline_knn_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(64, 3)).astype(np.float64)
    tree = MlpackCoverTreeBaseline.from_points(points)

    queries = rng.normal(size=(5, 3)).astype(np.float64)
    indices, distances = tree.knn(queries, k=6, return_distances=True)

    assert indices.shape == (5, 6)
    assert distances.shape == (5, 6)

    for i, query in enumerate(queries):
        brute = np.linalg.norm(points - query, axis=1)
        order = np.argsort(brute)[:6]
        assert np.array_equal(indices[i], order)
        assert np.allclose(distances[i], brute[order])


@pytest.mark.skipif(
    not has_mlpack_cover_tree(), reason="mlpack cover tree baseline not available"
)
def test_mlpack_baseline_nearest() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(128, 2)).astype(np.float64)
    tree = MlpackCoverTreeBaseline.from_points(points)

    query = np.array([0.15, -0.3], dtype=np.float64)
    idx, dist = tree.nearest(query)

    brute = np.linalg.norm(points - query, axis=1)
    order = np.argsort(brute)
    assert idx == int(order[0])
    assert pytest.approx(float(brute[order[0]]), rel=1e-12, abs=1e-12) == dist
