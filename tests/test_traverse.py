import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex import config as cx_config
from covertreex.algo.traverse import TraversalResult, traverse_collect_scopes
from covertreex.core.metrics import reset_residual_metric
from covertreex.core.tree import PCCTree, TreeLogStats, get_runtime_backend
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
    set_residual_backend,
)


def _sample_tree():
    backend = get_runtime_backend()
    points = backend.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=backend.default_float)
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


def _install_stub_residual_backend(chunk_size: int = 2) -> None:
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
    kernel_full = np.array(
        [
            [1.1, 0.7, 0.4],
            [0.7, 1.0, 0.6],
            [0.4, 0.6, 1.2],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=chunk_size,
    )
    configure_residual_correlation(backend_state)


def test_traversal_assigns_nearest_parent():
    tree = _sample_tree()
    batch_points = [[2.0, 2.0], [3.0, 3.0]]

    result = traverse_collect_scopes(tree, batch_points)

    assert isinstance(result, TraversalResult)
    assert result.parents.shape == (2,)
    assert result.levels.shape == (2,)
    assert len(result.conflict_scopes) == 2
    assert tuple(result.parents.tolist()) == (1, 1)
    assert tuple(result.levels.tolist()) == (0, 0)
    for scope, parent in zip(result.conflict_scopes, result.parents.tolist()):
        assert parent in scope
    indptr = result.scope_indptr.tolist()
    indices = result.scope_indices.tolist()
    reconstructed = [
        tuple(indices[indptr[i] : indptr[i + 1]]) for i in range(len(result.conflict_scopes))
    ]
    assert reconstructed == list(result.conflict_scopes)
    assert result.timings.pairwise_seconds >= 0.0
    assert result.timings.mask_seconds >= 0.0
    assert result.timings.semisort_seconds >= 0.0
    assert result.timings.chain_seconds >= 0.0
    assert result.timings.nonzero_seconds >= 0.0
    assert result.timings.sort_seconds >= 0.0
    assert result.timings.assemble_seconds >= 0.0


def test_traversal_uses_tree_backend_by_default():
    tree = _sample_tree()
    result = traverse_collect_scopes(tree, [[5.0, 5.0]])

    assert result.parents.dtype == tree.backend.default_int
    assert result.levels.dtype == tree.backend.default_int
    indptr = result.scope_indptr.tolist()
    indices = result.scope_indices.tolist()
    reconstructed = [
        tuple(indices[indptr[i] : indptr[i + 1]]) for i in range(len(result.conflict_scopes))
    ]
    assert reconstructed == list(result.conflict_scopes)
    assert result.scope_indptr.tolist() == [0, len(result.conflict_scopes[0])]
    assert result.timings.pairwise_seconds >= 0.0
    assert result.timings.chain_seconds >= 0.0


def test_traversal_handles_empty_tree():
    backend = get_runtime_backend()
    empty_tree = PCCTree.empty(dimension=2, backend=backend)

    result = traverse_collect_scopes(empty_tree, [[1.0, 1.0]])

    assert tuple(result.parents.tolist()) == (-1,)
    assert tuple(result.levels.tolist()) == (-1,)
    assert result.conflict_scopes == ((),)
    assert result.scope_indptr.tolist() == [0, 0]
    assert result.scope_indices.shape[0] == 0
    assert result.timings.pairwise_seconds == 0.0
    assert result.timings.chain_seconds == 0.0


def test_traversal_semisorts_scopes_by_level():
    backend = get_runtime_backend()
    points = backend.asarray(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([0, 2, 1], dtype=backend.default_int)
    parents = backend.asarray([-1, 1, 1], dtype=backend.default_int)
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

    result = traverse_collect_scopes(tree, [[2.1, 2.1]])

    assert result.conflict_scopes == ((1, 2, 0),)
    indptr = result.scope_indptr.tolist()
    indices = result.scope_indices.tolist()
    assert indptr == [0, 3]
    assert indices == [1, 2, 0]
    assert result.timings.pairwise_seconds >= 0.0


def test_euclidean_scope_chunk_target_limits_scopes(monkeypatch: pytest.MonkeyPatch):
    tree = _sample_tree()
    batch_points = [[2.0, 2.0]]

    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "1")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)

    scopes = result.conflict_scopes
    assert all(len(scope) <= 1 for scope in scopes)
    parents = result.parents.tolist()
    for scope, parent in zip(scopes, parents):
        if parent >= 0:
            assert parent in scope

    monkeypatch.delenv("COVERTREEX_SCOPE_CHUNK_TARGET", raising=False)
    cx_config.reset_runtime_config_cache()


def test_sparse_traversal_matches_dense(monkeypatch: pytest.MonkeyPatch):
    tree = _sample_tree()
    batch_points = [[2.0, 2.0], [3.0, 3.0]]

    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()


def test_residual_scope_limit_applies_scope_chunk_target(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    points = backend.asarray(
        [[0.0], [1.0], [2.0], [3.0]],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([1, 0, 0, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 0, 0], dtype=backend.default_int)
    children = backend.asarray([1, 2, 3, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 4, 4], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0, 0.0, 0.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, 3, -1], dtype=backend.default_int)
    tree = PCCTree(
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

    v_matrix = np.zeros((8, 2), dtype=np.float64)
    p_diag = np.ones(8, dtype=np.float64)
    kernel_diag = np.ones(8, dtype=np.float64)

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.ones((rows.size, cols.size), dtype=np.float64)

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

    batch_points = [[0.0], [1.0], [2.0]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "2")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)

    indptr = result.scope_indptr.tolist()
    indices = result.scope_indices.tolist()
    scopes = [indices[indptr[i] : indptr[i + 1]] for i in range(len(batch_points))]
    assert all(len(scope) <= 2 for scope in scopes)
    parents = result.parents.tolist()
    for scope, parent in zip(scopes, parents):
        if parent >= 0:
            assert parent in scope
    assert result.timings.scope_chunk_emitted == len(batch_points)
    assert result.timings.scope_chunk_max_members <= 2

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    monkeypatch.delenv("COVERTREEX_SCOPE_CHUNK_TARGET", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_scope_cache_hits(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    points = backend.asarray(
        [[0.0], [0.5], [1.0], [1.5]],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([1, 0, 0, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 0, 0], dtype=backend.default_int)
    children = backend.asarray([1, 2, 3, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 4, 4], dtype=backend.default_int)
    si_cache = backend.asarray([0.0, 0.0, 0.0, 0.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, 3, -1], dtype=backend.default_int)
    tree = PCCTree(
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

    _install_stub_residual_backend(chunk_size=4)

    batch_points = [[0.0], [0.1]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)
    timings = result.timings
    assert timings.scope_cache_prefetch > 0
    assert timings.scope_cache_hits > 0

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_scope_chunk_target_caps_scan_points(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    points = backend.asarray(
        [[float(i)] for i in range(6)],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([1] + [0] * 5, dtype=backend.default_int)
    parents = backend.asarray([-1] + [0] * 5, dtype=backend.default_int)
    children = backend.asarray([1, 2, 3, 4, 5, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 6, 6], dtype=backend.default_int)
    si_cache = backend.asarray([0.0] * 6, dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, 3, 4, 5, -1], dtype=backend.default_int)
    tree = PCCTree(
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

    v_matrix = np.zeros((10, 2), dtype=np.float64)
    p_diag = np.ones(10, dtype=np.float64)
    kernel_diag = np.ones(10, dtype=np.float64)

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.ones((rows.size, cols.size), dtype=np.float64)

    reset_residual_metric()
    set_residual_backend(None)
    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=1,
    )
    configure_residual_correlation(backend_state)

    batch_points = [[0.0], [1.0], [2.0]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "2")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)
    cache = result.residual_cache
    assert cache is not None
    assert isinstance(cache.pairwise, np.ndarray)
    assert cache.pairwise.dtype == np.float32
    points_scanned = np.asarray(cache.scope_chunk_points, dtype=np.int64)
    assert np.all(points_scanned <= 2)
    assert result.timings.scope_chunk_saturated == len(batch_points)

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    monkeypatch.delenv("COVERTREEX_SCOPE_CHUNK_TARGET", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_scope_budget_scheduler(monkeypatch: pytest.MonkeyPatch):
    backend = get_runtime_backend()
    points = backend.asarray([[float(i)] for i in range(3)], dtype=backend.default_float)
    top_levels = backend.asarray([1] + [0] * 2, dtype=backend.default_int)
    parents = backend.asarray([-1] + [0] * 2, dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 3, 3], dtype=backend.default_int)
    si_cache = backend.asarray([0.0] * 3, dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
    tree = PCCTree(
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

    _install_stub_residual_backend(chunk_size=1)

    batch_points = [[0.0], [0.5], [1.0]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "4")
    monkeypatch.setenv("COVERTREEX_SCOPE_BUDGET_SCHEDULE", "2,4")
    monkeypatch.setenv("COVERTREEX_SCOPE_BUDGET_UP_THRESH", "2.0")
    monkeypatch.setenv("COVERTREEX_SCOPE_BUDGET_DOWN_THRESH", "1.1")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)
    timings = result.timings
    assert timings.scope_budget_start == len(batch_points) * 2
    assert timings.scope_budget_final >= timings.scope_budget_start
    assert timings.scope_budget_final <= len(batch_points) * 4
    assert timings.scope_budget_early_terminate > 0

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    monkeypatch.delenv("COVERTREEX_SCOPE_CHUNK_TARGET", raising=False)
    monkeypatch.delenv("COVERTREEX_SCOPE_BUDGET_SCHEDULE", raising=False)
    monkeypatch.delenv("COVERTREEX_SCOPE_BUDGET_UP_THRESH", raising=False)
    monkeypatch.delenv("COVERTREEX_SCOPE_BUDGET_DOWN_THRESH", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_sparse_traversal_matches_dense(monkeypatch: pytest.MonkeyPatch):
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
    kernel_full = np.array(
        [
            [1.1, 0.7, 0.4],
            [0.7, 1.0, 0.6],
            [0.4, 0.6, 1.2],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    backend_state = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
    )
    configure_residual_correlation(backend_state)

    batch_points = [[1.0], [2.0]]

    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    cx_config.reset_runtime_config_cache()
    dense_result = traverse_collect_scopes(tree, batch_points)

    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    cx_config.reset_runtime_config_cache()
    sparse_result = traverse_collect_scopes(tree, batch_points)

    assert dense_result.engine.startswith("residual")
    assert sparse_result.engine.startswith("residual")

    assert sparse_result.parents.tolist() == dense_result.parents.tolist()
    assert sparse_result.levels.tolist() == dense_result.levels.tolist()
    assert sparse_result.conflict_scopes == dense_result.conflict_scopes
    assert sparse_result.scope_indptr.tolist() == dense_result.scope_indptr.tolist()
    assert sparse_result.scope_indices.tolist() == dense_result.scope_indices.tolist()

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)
    dense_result = traverse_collect_scopes(tree, batch_points)

    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()
    sparse_result = traverse_collect_scopes(tree, batch_points)

    assert sparse_result.parents.tolist() == dense_result.parents.tolist()
    assert sparse_result.levels.tolist() == dense_result.levels.tolist()
    assert sparse_result.conflict_scopes == dense_result.conflict_scopes
    assert sparse_result.scope_indptr.tolist() == dense_result.scope_indptr.tolist()
    assert sparse_result.scope_indices.tolist() == dense_result.scope_indices.tolist()

    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()


def test_residual_scope_caps_limit_radii(monkeypatch: pytest.MonkeyPatch):
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

    reset_residual_metric()
    set_residual_backend(None)
    _install_stub_residual_backend(chunk_size=2)

    batch_points = [[1.0], [2.0]]
    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT", "1.5")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)
    cache = result.residual_cache
    assert cache is not None
    initial = np.asarray(cache.scope_radius_initial, dtype=np.float64)
    limits = np.asarray(cache.scope_radius_limits, dtype=np.float64)
    assert np.all(limits <= 1.5 + 1e-12)
    assert np.any(initial > limits)

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)


def test_residual_scope_caps_file_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path):
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

    reset_residual_metric()
    set_residual_backend(None)
    _install_stub_residual_backend(chunk_size=2)

    payload = {"schema": 1, "levels": {"0": 0.25}}
    cap_path = tmp_path / "caps.json"
    cap_path.write_text(json.dumps(payload), encoding="utf-8")

    batch_points = [[1.0], [2.0]]
    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH", str(cap_path))
    monkeypatch.setenv("COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT", "1.0")
    cx_config.reset_runtime_config_cache()

    result = traverse_collect_scopes(tree, batch_points)
    cache = result.residual_cache
    assert cache is not None
    limits = np.asarray(cache.scope_radius_limits, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.int64)
    mask_level0 = levels == 0
    assert np.any(mask_level0)
    assert np.allclose(limits[mask_level0], 0.25, atol=1e-6)

    monkeypatch.delenv("COVERTREEX_METRIC", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH", raising=False)
    monkeypatch.delenv("COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT", raising=False)
    cx_config.reset_runtime_config_cache()
    reset_residual_metric()
    set_residual_backend(None)
