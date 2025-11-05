import numpy as np
import pytest

from covertreex import ExternalCoverTreeBaseline, has_external_cover_tree


@pytest.mark.skipif(
    not has_external_cover_tree(), reason="covertree baseline not available"
)
def test_external_baseline_knn_matches_bruteforce():
    points = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float64)
    tree = ExternalCoverTreeBaseline.from_points(points)

    queries = np.random.default_rng(1).normal(size=(3, 3)).astype(np.float64)
    indices, distances = tree.knn(queries, k=4, return_distances=True)

    assert indices.shape == (3, 4)
    assert distances.shape == (3, 4)

    for i, query in enumerate(queries):
        brute_dists = np.linalg.norm(points - query, axis=1)
        order = np.argsort(brute_dists)[:4]
        assert np.array_equal(indices[i], order)
        assert np.allclose(distances[i], brute_dists[order])


@pytest.mark.skipif(
    not has_external_cover_tree(), reason="covertree baseline not available"
)
def test_external_baseline_nearest():
    points = np.random.default_rng(2).normal(size=(32, 2)).astype(np.float64)
    tree = ExternalCoverTreeBaseline.from_points(points)

    query = np.array([0.25, -0.1], dtype=np.float64)
    idx, dist = tree.nearest(query)

    brute_dists = np.linalg.norm(points - query, axis=1)
    order = np.argsort(brute_dists)
    assert idx == int(order[0])
    assert pytest.approx(dist, rel=1e-12, abs=1e-12) == float(brute_dists[order[0]])
