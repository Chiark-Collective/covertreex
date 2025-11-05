import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import batch_delete, batch_insert
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats
from covertreex.queries import knn


def _base_tree() -> PCCTree:
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2, 3], dtype=backend.default_int)
    si_cache = backend.asarray([4.0, 2.0, 1.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
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


def test_async_batch_insert_harness():
    tree = _base_tree()
    update_batch = jnp.asarray([[0.2, 0.1], [2.9, 2.8]])
    query = jnp.asarray([[0.15, 0.05]])
    baseline_idx = np.asarray(knn(tree, query, k=1)).tolist()

    def _perform_update() -> PCCTree:
        time.sleep(0.05)
        new_tree, _ = batch_insert(tree, update_batch, mis_seed=0)
        return new_tree

    observed: list[list[int]] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_perform_update)
        while not future.done():
            current = np.asarray(knn(tree, query, k=1)).tolist()
            observed.append(current)
            time.sleep(0.005)
        updated_tree = future.result()

    assert observed, "expected to capture at least one in-flight query"
    assert all(sample == baseline_idx for sample in observed)

    post_idx = np.asarray(knn(updated_tree, query, k=1)).tolist()
    assert post_idx != baseline_idx
    assert np.asarray(knn(tree, query, k=1)).tolist() == baseline_idx


def test_device_builder_smoke():
    backend = DEFAULT_BACKEND
    tree = PCCTree.empty(dimension=2, backend=backend)

    initial = jax.device_put(
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=backend.default_float)
    )
    tree, _ = batch_insert(tree, initial, mis_seed=0)

    more = jax.device_put(
        jnp.asarray([[2.0, 2.0], [3.0, 3.0]], dtype=backend.default_float)
    )
    tree, _ = batch_insert(tree, more, mis_seed=1)

    remove_indices = jnp.asarray([1], dtype=backend.default_int)
    tree, _ = batch_delete(tree, remove_indices)

    query = jax.device_put(
        jnp.asarray([[0.2, 0.1], [2.9, 2.95]], dtype=backend.default_float)
    )
    indices, distances = knn(tree, query, k=1, return_distances=True)

    assert isinstance(indices, jax.Array)
    assert isinstance(distances, jax.Array)

    host_points = np.asarray(tree.points)
    host_query = np.asarray(query)
    expected_indices: list[int] = []
    expected_distances: list[float] = []
    for q in host_query:
        dists = np.linalg.norm(host_points - q, axis=1)
        idx = int(np.argmin(dists))
        expected_indices.append(idx)
        expected_distances.append(float(dists[idx]))

    assert np.asarray(indices).tolist() == [[idx] for idx in expected_indices]
    assert np.allclose(np.asarray(distances).flatten(), expected_distances)
