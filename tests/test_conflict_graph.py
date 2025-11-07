import numpy as np
import pytest

from covertreex.metrics import residual as residual_mod

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo.conflict_graph import ConflictGraph, build_conflict_graph
from covertreex.algo.traverse import traverse_collect_scopes
from covertreex.core.metrics import reset_residual_metric
from covertreex.core.tree import PCCTree, TreeLogStats, get_runtime_backend
from covertreex.algo import batch_insert
from covertreex.algo._scope_numba import NUMBA_SCOPE_AVAILABLE, _chunk_ranges_from_indptr
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
    set_residual_backend,
)
from covertreex import config as cx_config


def _sample_tree():
    backend = get_runtime_backend()
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


def test_chunk_range_builder_honours_max_segments():
    indptr = np.array([0, 5, 10, 15, 20, 25], dtype=np.int64)

    ranges_unbounded = _chunk_ranges_from_indptr(indptr, chunk_target=4, max_segments=0)
    ranges_capped = _chunk_ranges_from_indptr(indptr, chunk_target=4, max_segments=2)

    assert len(ranges_unbounded) > 2
    assert len(ranges_capped) == 2
    assert ranges_capped[0] == (0, 3)
    assert ranges_capped[1] == (3, 5)


def test_conflict_graph_builds_edges_from_shared_scopes():
    cx_config.reset_runtime_config_cache()
    backend = get_runtime_backend()
    tree = PCCTree.empty(dimension=2, backend=backend)
    base_points = backend.asarray(
        [[float(i), 0.0] for i in range(8)], dtype=backend.default_float
    )
    tree, _ = batch_insert(tree, base_points, mis_seed=0)
    batch_points = [[i + 0.25, 0.1] for i in range(4)]

    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)

    assert isinstance(graph, ConflictGraph)
    assert graph.num_nodes == 3
    assert graph.indptr.shape[0] == graph.num_nodes + 1
    assert graph.indices.ndim == 1
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
    assert graph.timings.adjacency_total_pairs >= 0.0
    assert graph.timings.adjacency_candidate_pairs >= 0.0
    assert graph.timings.scope_groups >= 0
    assert graph.timings.scope_groups_unique >= 0


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

    assert dense_graph.scope_indptr.tolist() == segmented_graph.scope_indptr.tolist()
    assert dense_graph.scope_indices.tolist() == segmented_graph.scope_indices.tolist()
    assert dense_graph.scope_indptr.tolist() == segmented_graph.scope_indptr.tolist()
    assert dense_graph.scope_indices.tolist() == segmented_graph.scope_indices.tolist()
    assert jnp.allclose(dense_graph.radii, segmented_graph.radii)
    assert jnp.allclose(dense_graph.pairwise_distances, segmented_graph.pairwise_distances)

    monkeypatch.delenv("COVERTREEX_CONFLICT_GRAPH_IMPL", raising=False)
    cx_config.reset_runtime_config_cache()


def test_chunked_conflict_graph_matches_dense(monkeypatch: pytest.MonkeyPatch):
    if not NUMBA_SCOPE_AVAILABLE:
        pytest.skip("Numba conflict builder unavailable.")

    backend = get_runtime_backend()
    tree = PCCTree.empty(dimension=2, backend=backend)
    base_points = backend.asarray(
        [[float(i), 0.0] for i in range(8)], dtype=backend.default_float
    )
    tree, _ = batch_insert(tree, base_points, mis_seed=0)
    batch_points = [[i + 0.25, 0.15] for i in range(4)]

    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")

    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "0")
    cx_config.reset_runtime_config_cache()
    traversal_dense = traverse_collect_scopes(tree, batch_points)
    dense_graph = build_conflict_graph(tree, traversal_dense, batch_points)

    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "2")
    cx_config.reset_runtime_config_cache()
    traversal_chunk = traverse_collect_scopes(tree, batch_points)
    chunk_graph = build_conflict_graph(tree, traversal_chunk, batch_points)

    assert dense_graph.indptr.tolist() == chunk_graph.indptr.tolist()
    assert dense_graph.indices.tolist() == chunk_graph.indices.tolist()
    assert chunk_graph.timings.scope_chunk_segments > 1
    assert chunk_graph.timings.scope_chunk_emitted >= 1

    monkeypatch.delenv("COVERTREEX_SCOPE_CHUNK_TARGET", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()


def test_residual_conflict_graph_matches_dense(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    points = backend.asarray([[0.0], [1.0], [2.0]], dtype=backend.default_float)
    top_levels = backend.asarray([1, 0, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 0], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 3, 3], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0, 0.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
    tree = PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        backend=backend,
    )

    v_matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.9],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.9, 1.1, 1.3], dtype=np.float64)
    kernel_diag = np.array([1.1, 1.0, 1.2], dtype=np.float64)
    kernel_full = np.array(
        [
            [1.1, 0.7, 0.5],
            [0.7, 1.0, 0.65],
            [0.5, 0.65, 1.2],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    reset_residual_metric()
    set_residual_backend(None)
    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
    )
    configure_residual_correlation(backend_state)

    batch_points = [[1.0], [1.4], [1.9]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
    traversal_dense = traverse_collect_scopes(tree, batch_points)
    dense_graph = build_conflict_graph(tree, traversal_dense, batch_points)

    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()
    traversal_stream = traverse_collect_scopes(tree, batch_points)
    stream_graph = build_conflict_graph(tree, traversal_stream, batch_points)

    assert dense_graph.indptr.tolist() == stream_graph.indptr.tolist()
    assert dense_graph.indices.tolist() == stream_graph.indices.tolist()
    assert dense_graph.scope_indptr.tolist() == stream_graph.scope_indptr.tolist()
    assert dense_graph.scope_indices.tolist() == stream_graph.scope_indices.tolist()
    assert jnp.allclose(dense_graph.pairwise_distances, stream_graph.pairwise_distances)
    assert jnp.allclose(dense_graph.radii, stream_graph.radii)

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_conflict_graph_reuses_pairwise_cache(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    points = backend.asarray([[0.0], [1.0], [2.0]], dtype=backend.default_float)
    top_levels = backend.asarray([1, 0, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 0], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 3, 3], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0, 0.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
    tree = PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        backend=backend,
    )

    v_matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.9, 1.1, 1.3], dtype=np.float64)
    kernel_diag = np.array([1.1, 1.0, 1.2], dtype=np.float64)

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        values = np.linspace(0.5, 1.1, num=rows.size * cols.size)
        return values.reshape(rows.size, cols.size)

    reset_residual_metric()
    set_residual_backend(None)
    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
    )
    configure_residual_correlation(backend_state)

    calls = {"count": 0}
    original = residual_mod.compute_residual_pairwise_matrix

    def _wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "covertreex.metrics.residual.compute_residual_pairwise_matrix",
        _wrapped,
    )
    monkeypatch.setattr(
        "covertreex.algo.traverse.compute_residual_pairwise_matrix",
        _wrapped,
    )
    monkeypatch.setattr(
        "covertreex.algo.conflict_graph.compute_residual_pairwise_matrix",
        _wrapped,
    )

    batch_points = [[1.0], [1.4], [1.9]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()

    traversal = traverse_collect_scopes(tree, batch_points)
    assert traversal.residual_cache is not None
    build_conflict_graph(tree, traversal, batch_points)

    assert calls["count"] == 1

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)
