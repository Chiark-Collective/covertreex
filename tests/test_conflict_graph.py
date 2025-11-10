import numpy as np
import pytest
from dataclasses import replace

from covertreex.metrics import residual as residual_mod

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo.conflict import ConflictGraph, build_conflict_graph
from covertreex.algo.conflict import runner as conflict_runner
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
from covertreex.exceptions import ResidualPairwiseCacheError
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

    ranges_unbounded, stats_unbounded = _chunk_ranges_from_indptr(
        indptr, chunk_target=4, max_segments=0
    )
    ranges_capped, stats_capped = _chunk_ranges_from_indptr(
        indptr, chunk_target=4, max_segments=2
    )

    assert len(ranges_unbounded) > 2
    assert len(ranges_capped) == 2
    assert ranges_capped[0] == (0, 3)
    assert ranges_capped[1] == (3, 5)
    assert stats_unbounded.pair_cap == 0
    assert stats_capped.pair_cap >= 0


def test_chunk_range_builder_clamps_when_chunk_target_disabled():
    indptr = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)

    ranges, stats = _chunk_ranges_from_indptr(indptr, chunk_target=0, max_segments=2)

    assert len(ranges) == 2
    assert ranges[0] == (0, 3)
    assert ranges[1] == (3, 5)
    assert stats.pair_cap == 0


def test_chunk_range_builder_skips_zero_volume_segments_with_keep_mask():
    indptr = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)
    keep_mask = np.array([True, True, False, False, False], dtype=bool)

    ranges_full, _ = _chunk_ranges_from_indptr(indptr, chunk_target=2, max_segments=0)
    ranges_kept, _ = _chunk_ranges_from_indptr(
        indptr,
        chunk_target=2,
        max_segments=0,
        keep_mask=keep_mask,
    )

    assert len(ranges_full) == 5
    assert ranges_kept == [(0, 1), (1, 2)]


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
    assert graph.num_nodes == len(batch_points)
    assert graph.indptr.shape[0] == graph.num_nodes + 1
    assert graph.indices.ndim == 1
    assert graph.scope_indptr.tolist() == traversal.scope_indptr.tolist()
    assert graph.scope_indices.tolist() == traversal.scope_indices.tolist()
    assert graph.pairwise_distances.shape == (graph.num_nodes, graph.num_nodes)
    assert graph.radii.shape == (graph.num_nodes,)
    assert all(r > 0.0 for r in graph.radii)
    assert graph.annulus_bounds.shape == (graph.num_nodes, 2)
    assert graph.annulus_bins.shape == (graph.num_nodes,)
    assert graph.annulus_bin_indices.shape[0] == graph.annulus_bin_indptr.tolist()[-1]
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
    dense_radii = np.asarray(dense_graph.radii, dtype=np.float64)
    stream_radii = np.asarray(stream_graph.radii, dtype=np.float64)
    assert dense_radii.shape == stream_radii.shape
    assert np.all(stream_radii > 0.0)
    assert np.all(stream_radii <= dense_radii + 1e-9)

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
        "covertreex.algo.conflict.runner.compute_residual_pairwise_matrix",
        _wrapped,
    )

    batch_points = [[1.0], [1.4], [1.9]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()

    traversal = traverse_collect_scopes(tree, batch_points)
    assert traversal.residual_cache is not None
    graph = build_conflict_graph(tree, traversal, batch_points)

    assert traversal.residual_cache.scope_radii is not None
    cached_radii = np.asarray(traversal.residual_cache.scope_radii, dtype=np.float64)
    assert cached_radii.shape[0] == len(batch_points)
    graph_radii = np.asarray(graph.radii)
    assert graph_radii.shape[0] == cached_radii.shape[0]
    assert np.all(np.isfinite(cached_radii))
    assert np.allclose(graph_radii, cached_radii)

    assert calls["count"] == 1

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_conflict_graph_requires_pairwise_cache(monkeypatch: pytest.MonkeyPatch):
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
    p_diag = np.array([0.9, 1.1, 1.2], dtype=np.float64)
    kernel_diag = np.array([1.1, 1.0, 1.2], dtype=np.float64)

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        values = np.linspace(0.4, 1.0, num=rows.size * cols.size)
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

    batch_points = [[0.1], [0.5]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()

    traversal = traverse_collect_scopes(tree, batch_points)
    traversal_missing = replace(traversal, residual_cache=None)

    with pytest.raises(ResidualPairwiseCacheError):
        build_conflict_graph(tree, traversal_missing, batch_points)

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_grid_conflict_builder_forces_leaders(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    tree = PCCTree.empty(dimension=2, backend=backend)
    batch = backend.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.05],
            [0.5, 0.1],
            [3.0, 3.0],
            [3.25, 3.1],
            [3.5, 3.2],
        ],
        dtype=backend.default_float,
    )

    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "grid")
    cx_config.reset_runtime_config_cache()

    new_tree, plan = batch_insert(tree, batch, backend=backend, mis_seed=0)

    grid_stats = plan.conflict_graph
    assert grid_stats.grid_leaders_raw >= grid_stats.grid_leaders_after > 0
    assert grid_stats.grid_cells > 0
    assert grid_stats.indices.size == 0
    assert grid_stats.forced_selected is not None
    assert plan.selected_indices.size == grid_stats.grid_leaders_after
    assert new_tree.num_points >= plan.selected_indices.size

    monkeypatch.delenv("COVERTREEX_CONFLICT_GRAPH_IMPL", raising=False)
    cx_config.reset_runtime_config_cache()


def test_residual_grid_conflict_builder_emits_leaders(monkeypatch: pytest.MonkeyPatch):
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
        values = np.linspace(0.6, 1.1, num=rows.size * cols.size)
        return values.reshape(rows.size, cols.size)

    reset_residual_metric()
    set_residual_backend(None)
    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
        gate_v32=v_matrix.astype(np.float32),
    )
    configure_residual_correlation(backend_state)

    batch_points = [[0.2], [0.9], [1.8]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "grid")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1", "1")
    cx_config.reset_runtime_config_cache()

    traversal = traverse_collect_scopes(tree, batch_points)
    graph = build_conflict_graph(tree, traversal, batch_points)

    assert graph.grid_leaders_raw > 0
    assert graph.grid_leaders_after > 0
    assert graph.forced_selected is not None
    assert graph.forced_dominated is not None

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_CONFLICT_GRAPH_IMPL", raising=False)
    monkeypatch.delenv("COVERTREEX_RESIDUAL_GATE1", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_grid_uses_whitened_scale_without_gate(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    tree = PCCTree.empty(dimension=1, backend=backend)
    base_points = backend.asarray([[0.0], [1.0], [2.0]], dtype=backend.default_float)
    tree, _ = batch_insert(tree, base_points, mis_seed=0)
    batch_points = [[0.0], [1.0], [2.0]]

    v_matrix = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.2],
            [0.15, 0.05],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([1.0, 1.1, 0.95], dtype=np.float64)
    kernel_diag = np.ones(3, dtype=np.float64)

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.full((rows.size, cols.size), 0.9, dtype=np.float64)

    def point_decoder(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return np.asarray([int(arr)], dtype=np.int64)
        if arr.ndim == 1:
            return arr.astype(np.int64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0].astype(np.int64)
        return arr.reshape(-1).astype(np.int64)

    reset_residual_metric()
    set_residual_backend(None)

    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        point_decoder=point_decoder,
        chunk_size=2,
    )

    scale = 1.75
    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "grid")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1", "0")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE", str(scale))
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "0")
    cx_config.reset_runtime_config_cache()

    configure_residual_correlation(backend_state)

    captured: dict[str, np.ndarray] = {}

    def fake_grid_select_leaders(**kwargs):
        captured["points"] = kwargs["points"].copy()
        batch_size = kwargs["points"].shape[0]
        forced_selected = np.zeros(batch_size, dtype=np.uint8)
        forced_dominated = np.ones(batch_size, dtype=np.uint8)
        stats = {"cells": batch_size, "leaders_raw": batch_size, "leaders_final": batch_size, "local_edges": 0}
        return forced_selected, forced_dominated, stats

    monkeypatch.setattr(
        "covertreex.algo.conflict.builders._grid_select_leaders",
        fake_grid_select_leaders,
    )

    traversal = traverse_collect_scopes(tree, batch_points)
    build_conflict_graph(tree, traversal, batch_points)

    assert "points" in captured
    host_backend = residual_mod.get_residual_backend()
    assert host_backend.gate_v32 is not None
    decoded = residual_mod.decode_indices(host_backend, np.asarray(batch_points, dtype=np.float64))
    expected = np.asarray(host_backend.gate_v32[decoded], dtype=np.float64) * scale
    np.testing.assert_allclose(captured["points"], expected)

    for key in [
        "COVERTREEX_METRIC",
        "COVERTREEX_CONFLICT_GRAPH_IMPL",
        "COVERTREEX_RESIDUAL_GATE1",
        "COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE",
        "COVERTREEX_ENABLE_NUMBA",
    ]:
        monkeypatch.delenv(key, raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_adaptive_chunk_target_triggers_on_sparse_scopes():
    counts = np.zeros(256, dtype=np.int64)
    counts[-1] = 512
    target = conflict_runner._adaptive_scope_chunk_target(counts)
    assert target is not None
    assert target >= 8_192


def test_adaptive_chunk_target_skips_dense_scopes():
    counts = np.array([32, 32, 32], dtype=np.int64)
    target = conflict_runner._adaptive_scope_chunk_target(counts)
    assert target is None


def test_batch_insert_clamps_infinite_si_cache():
    backend = get_runtime_backend()
    points = backend.asarray([[0.0, 0.0]], dtype=backend.default_float)
    top_levels = backend.asarray([0], dtype=backend.default_int)
    parents = backend.asarray([-1], dtype=backend.default_int)
    children = backend.asarray([-1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1], dtype=backend.default_int)
    si_cache = backend.asarray([np.inf], dtype=backend.default_float)
    next_cache = backend.asarray([-1], dtype=backend.default_int)
    tree = PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        stats=TreeLogStats(),
        backend=backend,
    )
    batch = backend.asarray([[0.1, 0.0]], dtype=backend.default_float)

    new_tree, plan = batch_insert(tree, batch, backend=backend, mis_seed=0)

    assert plan.selected_indices.size + plan.dominated_indices.size > 0
    assert new_tree.num_points == tree.num_points + int(
        plan.selected_indices.size + plan.dominated_indices.size
    )
    new_si_cache = np.asarray(new_tree.si_cache)
    assert np.isfinite(new_si_cache[-1])
