import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex import config as cx_config
from covertreex.algo.traverse import TraversalResult, traverse_collect_scopes
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _sample_tree():
    backend = DEFAULT_BACKEND
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
    backend = DEFAULT_BACKEND
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
    backend = DEFAULT_BACKEND
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


def test_sparse_traversal_matches_dense(monkeypatch: pytest.MonkeyPatch):
    tree = _sample_tree()
    batch_points = [[2.0, 2.0], [3.0, 3.0]]

    monkeypatch.delenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", raising=False)
    monkeypatch.delenv("COVERTREEX_ENABLE_NUMBA", raising=False)
    cx_config.reset_runtime_config_cache()
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
