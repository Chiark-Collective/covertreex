import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex import config as cx_config
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats
from covertreex.queries import knn, nearest_neighbor


def _build_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([3, 2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1, 2], dtype=backend.default_int)
    children = backend.asarray([1, 2, 3, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2, 3, 4], dtype=backend.default_int)
    si_cache = backend.asarray([4.0, 2.5, 1.5, 0.5], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, 3, -1], dtype=backend.default_int)
    return PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        stats=TreeLogStats(num_batches=1),
        backend=backend,
    )


def test_knn_matches_bruteforce_distances():
    tree = _build_tree()
    queries = jnp.asarray([[0.2, 0.1], [2.6, 2.5]])

    indices, distances = knn(tree, queries, k=2, return_distances=True)

    baseline_indices = []
    baseline_distances = []
    for query in np.asarray(queries):
        dists = np.linalg.norm(np.asarray(tree.points) - query, axis=1)
        order = np.argsort(dists)[:2]
        baseline_indices.append(order)
        baseline_distances.append(dists[order])

    assert np.array_equal(np.asarray(indices), np.asarray(baseline_indices))
    assert np.allclose(np.asarray(distances), np.asarray(baseline_distances))


def test_nearest_neighbor_handles_single_query_vector():
    tree = _build_tree()
    query = jnp.asarray([0.1, 0.2])

    idx, dist = nearest_neighbor(tree, query, return_distances=True)

    assert np.isscalar(np.asarray(idx)) or np.asarray(idx).shape == ()
    assert np.isscalar(np.asarray(dist)) or np.asarray(dist).shape == ()


def test_knn_raises_for_empty_tree():
    backend = DEFAULT_BACKEND
    empty = PCCTree.empty(dimension=2, backend=backend)

    with pytest.raises(ValueError):
        knn(empty, [[0.0, 0.0]], k=1)


def test_knn_prefers_lower_indices_on_ties():
    tree = _build_tree()
    query = jnp.asarray([1.5, 1.5])

    indices = knn(tree, query, k=2)

    assert np.asarray(indices).tolist() == [1, 2]


def test_knn_returns_all_points_when_k_equals_size():
    tree = _build_tree()
    query = jnp.asarray([1.25, 1.25])

    indices, distances = knn(tree, query, k=tree.num_points, return_distances=True)

    expected_distances = np.linalg.norm(
        np.asarray(tree.points) - np.asarray(query), axis=1
    )
    expected_order = np.argsort(expected_distances)

    assert np.asarray(indices).tolist() == expected_order.tolist()
    assert np.allclose(np.asarray(distances), expected_distances[expected_order])


def test_knn_multi_query_batch_shapes():
    tree = _build_tree()
    queries = jnp.asarray([[0.1, 0.1], [2.9, 3.1]])

    indices, distances = knn(tree, queries, k=2, return_distances=True)

    assert np.asarray(indices).shape == (2, 2)
    assert np.asarray(distances).shape == (2, 2)


def test_knn_numba_matches_python(monkeypatch):
    pytest.importorskip("numba")

    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()
    tree_numba = _build_tree()
    queries = jnp.asarray([[0.2, 0.2], [2.5, 2.4]])
    indices_numba, distances_numba = knn(tree_numba, queries, k=2, return_distances=True)
    indices_numba = np.asarray(indices_numba)
    distances_numba = np.asarray(distances_numba)

    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "0")
    cx_config.reset_runtime_config_cache()
    tree_python = _build_tree()
    indices_py, distances_py = knn(tree_python, queries, k=2, return_distances=True)
    indices_py = np.asarray(indices_py)
    distances_py = np.asarray(distances_py)

    np.testing.assert_array_equal(indices_numba, indices_py)
    np.testing.assert_allclose(distances_numba, distances_py, rtol=1e-12, atol=1e-12)

    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
