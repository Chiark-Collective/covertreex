import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import batch_delete, batch_insert, plan_batch_delete
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _setup_tree() -> PCCTree:
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.5, 2.5],
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


def _point_set(points) -> list[tuple[float, ...]]:
    return sorted(map(tuple, np.asarray(np.array(points))))


def test_batch_delete_restores_parent_chain_after_tail_removal():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2]])

    inserted, _ = batch_insert(tree, batch, mis_seed=0)

    remove_indices = np.arange(tree.num_points, inserted.num_points)
    deleted, plan_delete = batch_delete(inserted, remove_indices)

    assert plan_delete.removed_indices.tolist() == remove_indices.tolist()
    assert plan_delete.reattach_indices.tolist() == []
    assert deleted.num_points == tree.num_points
    assert np.allclose(np.asarray(deleted.points), np.asarray(tree.points))


def test_batch_delete_preserves_previous_versions():
    tree = _setup_tree()
    batch = jnp.asarray([[2.2, 2.1]])

    inserted, _ = batch_insert(tree, batch, mis_seed=0)
    snapshot = {
        "points": np.asarray(inserted.points).copy(),
        "children": np.asarray(inserted.children).copy(),
        "next": np.asarray(inserted.next_cache).copy(),
    }

    remove_indices = np.arange(tree.num_points, inserted.num_points)
    deleted, _ = batch_delete(inserted, remove_indices)

    assert np.allclose(np.asarray(inserted.points), snapshot["points"])
    assert np.array_equal(np.asarray(inserted.children), snapshot["children"])
    assert np.array_equal(np.asarray(inserted.next_cache), snapshot["next"])
    assert deleted.num_points == tree.num_points


def test_plan_batch_delete_identifies_descendants():
    tree = _setup_tree()
    plan = plan_batch_delete(tree, [0])

    assert plan.removed_indices.tolist() == [0]
    assert sorted(plan.reattach_indices.tolist()) == [1, 2]
    assert plan.created_new_root
    levels = [summary.level for summary in plan.level_summaries]
    assert levels == sorted(levels)
    uncovered_union = set()
    for summary in plan.level_summaries:
        uncovered = set(summary.uncovered.tolist())
        promoted = set(summary.promoted.tolist())
        reattached = set(summary.reattached.tolist())
        assert uncovered == promoted.union(reattached)
        uncovered_union.update(uncovered)
    assert uncovered_union == set(plan.reattach_indices.tolist())


def test_batch_delete_handles_internal_node():
    tree = _setup_tree()

    deleted, plan = batch_delete(tree, [1])

    expected_points = [tuple(row) for row in np.asarray(np.array(tree.points))[[0, 2]]]
    assert _point_set(deleted.points) == sorted(expected_points)

    parents_np = np.asarray(np.array(deleted.parents))
    points_np = np.asarray(np.array(deleted.points))

    root_idx = int(np.where((points_np == tree.points[0]).all(axis=1))[0][0])
    child_idx = int(np.where((points_np == tree.points[2]).all(axis=1))[0][0])

    assert parents_np[root_idx] == -1
    assert parents_np[child_idx] == root_idx
    assert sorted(plan.reattach_indices.tolist()) == [2]


def test_batch_delete_handles_root_removal():
    tree = _setup_tree()

    deleted, plan = batch_delete(tree, [0])

    expected_points = [tuple(row) for row in np.asarray(np.array(tree.points))[[1, 2]]]
    assert _point_set(deleted.points) == sorted(expected_points)

    parents_np = np.asarray(np.array(deleted.parents))
    points_np = np.asarray(np.array(deleted.points))

    root_candidates = np.where(parents_np == -1)[0]
    assert root_candidates.size >= 1
    for idx, parent in enumerate(parents_np):
        if parent < 0:
            continue
        assert 0 <= parent < parents_np.size
    assert sorted(plan.reattach_indices.tolist()) == [1, 2]


def test_batch_delete_supports_multiple_leaf_indices():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2], [3.2, 3.1]])

    inserted, _ = batch_insert(tree, batch, mis_seed=0)

    remove_indices = [tree.num_points, tree.num_points + 2]
    deleted, plan = batch_delete(inserted, remove_indices)

    survivors = [idx for idx in range(inserted.num_points) if idx not in remove_indices]

    assert _point_set(deleted.points) == _point_set(np.asarray(inserted.points)[survivors])
    assert plan.reattach_indices.tolist() == []


def test_batch_delete_reassigns_descendant_subtree():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2]])

    inserted, _ = batch_insert(tree, batch, mis_seed=0)

    removed_idx = tree.num_points
    deleted, plan = batch_delete(inserted, [int(removed_idx)])

    survivors = [idx for idx in range(inserted.num_points) if idx != removed_idx]
    assert _point_set(deleted.points) == _point_set(np.asarray(inserted.points)[survivors])
    assert plan.reattach_indices.tolist() == []


def test_batch_delete_noop_for_empty_selection():
    tree = _setup_tree()
    new_tree, plan = batch_delete(tree, [])

    assert new_tree is tree
    assert plan.removed_indices.shape[0] == 0
    assert plan.level_summaries == tuple()
    assert not plan.created_new_root
