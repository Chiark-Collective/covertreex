import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex import config as cx_config
from covertreex.algo.conflict_graph import build_conflict_graph
from covertreex.algo.mis import MISResult, batch_mis_seeds, run_mis
from covertreex.algo.traverse import traverse_collect_scopes
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _build_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=backend.default_float)
    top_levels = backend.asarray([2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 3, 3], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0, 0.0], dtype=backend.default_float)
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


def test_run_mis_produces_independent_set():
    tree = _build_tree()
    batch_points = [[2.2, 2.2], [2.4, 2.4], [4.5, 4.5]]
    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)

    result = run_mis(tree.backend, graph)

    assert isinstance(result, MISResult)
    assert result.independent_set.shape == (3,)
    independent_nodes = {
        idx for idx, flag in enumerate(result.independent_set.tolist()) if flag == 1
    }
    assert len(independent_nodes) == 2

    indptr = graph.indptr.tolist()
    indices = graph.indices.tolist()
    for node in independent_nodes:
        neighbors = set(indices[indptr[node] : indptr[node + 1]])
        assert neighbors.isdisjoint(independent_nodes)

    dominated = set(range(graph.num_nodes)) - independent_nodes
    for node in dominated:
        neighbors = set(indices[indptr[node] : indptr[node + 1]])
        assert neighbors & independent_nodes

    assert result.iterations >= 1


def test_batch_mis_seeds_deterministic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COVERTREEX_MIS_SEED", "123")
    cx_config.reset_runtime_config_cache()

    seeds_first = batch_mis_seeds(5)
    seeds_second = batch_mis_seeds(5)
    assert seeds_first == seeds_second

    explicit = batch_mis_seeds(3, seed=42)
    assert explicit != seeds_first[:3]

    monkeypatch.delenv("COVERTREEX_MIS_SEED", raising=False)
    cx_config.reset_runtime_config_cache()


def test_batch_mis_seeds_rejects_negative():
    with pytest.raises(ValueError):
        batch_mis_seeds(-1)


def test_run_mis_numba_matches_jax(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("numba")

    tree = _build_tree()
    batch_points = [[2.2, 2.2], [2.4, 2.4], [4.5, 4.5]]
    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)

    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "0")
    cx_config.reset_runtime_config_cache()
    reference = run_mis(tree.backend, graph, seed=7)

    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()
    accelerated = run_mis(tree.backend, graph, seed=7)

    ref_array = np.asarray(reference.independent_set)
    acc_array = np.asarray(accelerated.independent_set)
    assert np.array_equal(acc_array, ref_array)
    assert accelerated.iterations == reference.iterations

    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
