import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo.conflict_graph import ConflictGraph, build_conflict_graph
from covertreex.algo.traverse import traverse_collect_scopes
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats
from covertreex import config as cx_config


def _sample_tree():
    backend = DEFAULT_BACKEND
    points = backend.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=backend.default_float)
    top_levels = backend.asarray([1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0], dtype=backend.default_int)
    children = backend.asarray([1, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, -1], dtype=backend.default_int)
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


def test_conflict_graph_builds_edges_from_shared_scopes():
    cx_config.reset_runtime_config_cache()
    tree = _sample_tree()
    batch_points = [[2.1, 2.1], [2.4, 2.4], [5.0, 5.0]]

    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)

    assert isinstance(graph, ConflictGraph)
    assert graph.num_nodes == 3

    # First two points should connect (within radius); third is isolated.
    assert graph.indptr.tolist() == [0, 1, 2, 2]
    assert graph.indices.tolist() == [1, 0]
    assert graph.scope_indptr.tolist() == [0, 1, 2, 3]
    assert graph.scope_indices.tolist() == [1, 1, 1]
    assert graph.pairwise_distances.shape == (3, 3)
    assert graph.radii.shape == (3,)
    assert graph.annulus_bounds.shape == (3, 2)
    assert graph.annulus_bins.shape == (3,)
    assert graph.annulus_bin_indices.shape == (3,)
    assert graph.annulus_bin_indptr.tolist()[-1] == 3
    assert graph.timings.pairwise_seconds >= 0.0
    assert graph.timings.scope_group_seconds >= 0.0
    assert graph.timings.adjacency_seconds >= 0.0
    assert graph.timings.annulus_seconds >= 0.0
    assert graph.timings.adjacency_total_pairs == pytest.approx(6.0)
    assert graph.timings.adjacency_candidate_pairs == pytest.approx(6.0)
    assert graph.timings.scope_groups == 1
    assert graph.timings.scope_groups_unique == 1


def test_segmented_conflict_graph_matches_dense(monkeypatch: pytest.MonkeyPatch):
    cx_config.reset_runtime_config_cache()
    tree = _sample_tree()
    batch_points = [[2.1, 2.1], [2.4, 2.4], [5.0, 5.0]]

    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "dense")
    cx_config.reset_runtime_config_cache()
    traversal = traverse_collect_scopes(tree, batch_points)
    dense_graph = build_conflict_graph(tree, traversal, batch_points)

    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "segmented")
    cx_config.reset_runtime_config_cache()
    traversal_segmented = traverse_collect_scopes(tree, batch_points)
    segmented_graph = build_conflict_graph(tree, traversal_segmented, batch_points)

    assert dense_graph.indptr.tolist() == segmented_graph.indptr.tolist()
    assert dense_graph.indices.tolist() == segmented_graph.indices.tolist()
    assert dense_graph.scope_indptr.tolist() == segmented_graph.scope_indptr.tolist()
    assert dense_graph.scope_indices.tolist() == segmented_graph.scope_indices.tolist()
    assert jnp.allclose(dense_graph.radii, segmented_graph.radii)
    assert jnp.allclose(dense_graph.pairwise_distances, segmented_graph.pairwise_distances)

    monkeypatch.delenv("COVERTREEX_CONFLICT_GRAPH_IMPL", raising=False)
    cx_config.reset_runtime_config_cache()
