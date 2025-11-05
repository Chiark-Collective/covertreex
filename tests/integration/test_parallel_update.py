import math
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import (
    batch_insert,
    batch_insert_prefix_doubling,
    batch_mis_seeds,
    plan_batch_insert,
)
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _setup_tree():
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
    next_cache = backend.asarray([-1, -1, -1], dtype=backend.default_int)
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


def _expected_level_offsets(levels):
    levels_np = np.asarray(levels, dtype=np.int64)
    if levels_np.size == 0:
        return [0]
    counts = np.bincount(levels_np, minlength=levels_np.max() + 1)
    counts_desc = counts[::-1]
    offsets = np.concatenate([[0], np.cumsum(counts_desc)])
    return offsets.tolist()


def _mutated_prefix_indices(before, after, prefix_length):
    before_np = np.asarray(before)[:prefix_length]
    after_np = np.asarray(after)[:prefix_length]
    return {
        idx
        for idx in range(prefix_length)
        if int(before_np[idx]) != int(after_np[idx])
    }


def _counts_from_offsets(offsets):
    if offsets.size <= 1:
        return np.zeros(1, dtype=np.int64)
    diffs = np.diff(offsets.astype(np.int64))
    return diffs[::-1]


def _expected_level_count_delta(inserted_levels, target_length):
    delta = np.zeros(target_length, dtype=np.int64)
    for level in inserted_levels:
        lvl = int(level)
        if lvl < 0:
            continue
        if lvl >= delta.shape[0]:
            pad = lvl - delta.shape[0] + 1
            delta = np.pad(delta, (0, pad))
        delta[lvl] += 1
    return delta


def _assert_child_sibling_consistency(tree: PCCTree) -> None:
    backend = tree.backend
    parents_np = np.asarray(backend.to_numpy(tree.parents), dtype=np.int64)
    children_np = np.asarray(backend.to_numpy(tree.children), dtype=np.int64)
    next_np = np.asarray(backend.to_numpy(tree.next_cache), dtype=np.int64)

    num_points = parents_np.shape[0]
    for parent in range(num_points):
        seen = set()
        chain = []
        head = int(children_np[parent])
        while head != -1:
            assert 0 <= head < num_points, f"Child index {head} out of bounds"
            assert parents_np[head] == parent, (
                f"Node {head} recorded under parent {parent}, but parent array "
                f"stores {parents_np[head]}"
            )
            assert head not in seen, "Detected cycle in sibling chain"
            seen.add(head)
            chain.append(head)
            head = int(next_np[head])

        expected_children = [
            idx for idx, recorded_parent in enumerate(parents_np) if recorded_parent == parent
        ]
        assert sorted(chain) == sorted(expected_children), (
            f"Sibling chain under parent {parent} mismatch: expected "
            f"{sorted(expected_children)}, got {sorted(chain)}"
        )


def test_plan_batch_insert_runs_pipeline():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6], [0.5, 0.4], [3.2, 3.1]])

    plan = plan_batch_insert(tree, batch, mis_seed=0)

    assert plan.selected_indices.shape[0] <= batch.shape[0]
    assert plan.conflict_graph.num_nodes == batch.shape[0]
    assert plan.dominated_indices.shape[0] + plan.selected_indices.shape[0] == batch.shape[0]
    summary_levels = {summary.level for summary in plan.level_summaries}
    expected_levels = {int(max(level, 0)) for level in plan.traversal.levels.tolist()}
    assert summary_levels == expected_levels
    candidate_count = sum(int(summary.candidates.shape[0]) for summary in plan.level_summaries)
    assert candidate_count == batch.shape[0]
    selected_count = sum(int(summary.selected.shape[0]) for summary in plan.level_summaries)
    dominated_count = sum(int(summary.dominated.shape[0]) for summary in plan.level_summaries)
    assert selected_count == plan.selected_indices.shape[0]
    assert dominated_count == plan.dominated_indices.shape[0]

    indicator = plan.mis_result.independent_set.tolist()
    selected = {idx for idx, flag in enumerate(indicator) if flag == 1}

    indptr = plan.conflict_graph.indptr.tolist()
    indices = plan.conflict_graph.indices.tolist()
    for node in selected:
        neighbors = set(indices[indptr[node] : indptr[node + 1]])
        assert neighbors.isdisjoint(selected)

    dominated = set(range(batch.shape[0])) - selected
    for node in dominated:
        neighbors = set(indices[indptr[node] : indptr[node + 1]])
        assert neighbors & selected
    assert set(plan.dominated_indices.tolist()) == dominated

    new_tree, returned_plan = batch_insert(tree, batch, mis_seed=0)
    assert returned_plan.selected_indices.tolist() == plan.selected_indices.tolist()
    assert new_tree.num_points == tree.num_points + batch.shape[0]
    assert tree.num_points == 3  # unchanged original tree

    appended = new_tree.points[-batch.shape[0] :]
    expected = jnp.concatenate(
        [batch[plan.selected_indices], batch[plan.dominated_indices]], axis=0
    )
    assert jnp.allclose(appended, expected)


def test_batch_insert_updates_level_offsets_and_stats():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [3.5, 3.4], [0.2, 0.1]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    assert new_tree.level_offsets.tolist() == _expected_level_offsets(
        new_tree.top_levels.tolist()
    )
    assert tree.level_offsets.tolist() == _expected_level_offsets(
        tree.top_levels.tolist()
    )
    assert new_tree.num_points == tree.num_points + batch.shape[0]

    total_inserted = int(plan.selected_indices.size + plan.dominated_indices.size)
    assert new_tree.stats.num_batches == tree.stats.num_batches + 1
    assert new_tree.stats.num_insertions == tree.stats.num_insertions + total_inserted
    assert new_tree.stats.num_deletions == tree.stats.num_deletions


def test_batch_insert_preserves_original_tree_buffers():
    tree = _setup_tree()
    original_points = tree.points.tolist()
    original_top_levels = tree.top_levels.tolist()
    original_parents = tree.parents.tolist()
    original_children = tree.children.tolist()
    original_level_offsets = tree.level_offsets.tolist()

    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2]])
    new_tree, _ = batch_insert(tree, batch, mis_seed=0)

    assert tree.points.tolist() == original_points
    assert tree.top_levels.tolist() == original_top_levels
    assert tree.parents.tolist() == original_parents
    assert tree.children.tolist() == original_children
    assert tree.level_offsets.tolist() == original_level_offsets

    assert new_tree.points.shape[0] == tree.points.shape[0] + batch.shape[0]


def test_batch_insert_only_mutates_expected_parent_nodes():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2], [3.1, 3.0]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    parents_np = np.asarray(
        jnp.concatenate(
            [
                plan.traversal.parents[plan.selected_indices],
                plan.traversal.parents[plan.dominated_indices],
            ]
        )
    )
    touched_parents = {
        int(parent)
        for parent in parents_np.tolist()
        if 0 <= int(parent) < tree.num_points
    }

    changed_children = _mutated_prefix_indices(
        tree.children, new_tree.children, tree.num_points
    )
    changed_next = _mutated_prefix_indices(
        tree.next_cache, new_tree.next_cache, tree.num_points
    )

    assert changed_children.issubset(touched_parents)
    assert changed_next.issubset(touched_parents)


def test_batch_insert_maintains_child_sibling_chains():
    tree = _setup_tree()
    batch = jnp.asarray([[2.4, 2.4], [0.3, 0.2], [3.1, 3.0]])

    new_tree, _ = batch_insert(tree, batch, mis_seed=0)

    _assert_child_sibling_consistency(new_tree)

    offsets_before = np.asarray(tree.backend.to_numpy(tree.level_offsets), dtype=np.int64)
    offsets_after = np.asarray(new_tree.backend.to_numpy(new_tree.level_offsets), dtype=np.int64)
    counts_before = _counts_from_offsets(offsets_before)
    counts_after = _counts_from_offsets(offsets_after)
    max_len = max(counts_before.shape[0], counts_after.shape[0])
    counts_before = np.pad(counts_before, (0, max_len - counts_before.shape[0]))
    counts_after = np.pad(counts_after, (0, max_len - counts_after.shape[0]))
    delta_offsets = counts_after - counts_before

    inserted_levels = np.asarray(
        new_tree.backend.to_numpy(new_tree.top_levels), dtype=np.int64
    )[tree.num_points :]
    expected_delta = _expected_level_count_delta(inserted_levels, max_len)
    expected_delta = expected_delta[: max_len]

    assert np.array_equal(delta_offsets, expected_delta)


def test_batch_insert_on_empty_tree_sets_root_level():
    backend = DEFAULT_BACKEND
    empty = PCCTree.empty(dimension=2, backend=backend)
    batch = jnp.asarray([[1.0, 1.0], [2.0, 2.0]])

    new_tree, plan = batch_insert(empty, batch, mis_seed=0)

    assert not empty.points.shape[0]
    assert new_tree.num_points == batch.shape[0]
    assert all(level >= 0 for level in new_tree.top_levels.tolist())
    assert new_tree.level_offsets.tolist() == _expected_level_offsets(
        new_tree.top_levels.tolist()
    )
    summary_levels = {summary.level for summary in plan.level_summaries}
    assert summary_levels == {0}
    assert sum(int(summary.candidates.shape[0]) for summary in plan.level_summaries) == batch.shape[0]


def test_batch_insert_splices_child_chain_for_existing_parent():
    tree = _setup_tree()
    batch = jnp.asarray([[1.1, 1.1]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    selected_parents = plan.traversal.parents[plan.selected_indices].tolist()
    assert selected_parents
    parent = int(selected_parents[0])
    assert parent >= 0

    new_idx = tree.num_points
    old_child = tree.children.tolist()[parent]

    assert int(new_tree.children[parent]) == new_idx
    expected_next = old_child if old_child >= 0 else -1
    assert int(new_tree.next_cache[new_idx]) == expected_next
    assert int(new_tree.next_cache[parent]) == tree.next_cache.tolist()[parent]


def test_batch_insert_sets_child_chain_for_parent_without_children():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    selected_parents = plan.traversal.parents[plan.selected_indices].tolist()
    assert selected_parents
    parent = int(selected_parents[0])
    assert parent >= 0

    new_idx = tree.num_points
    assert tree.children.tolist()[parent] == -1
    assert int(new_tree.children[parent]) == new_idx
    assert int(new_tree.next_cache[new_idx]) == -1
    assert int(new_tree.next_cache[parent]) == tree.next_cache.tolist()[parent]


def test_batch_insert_redistributes_dominated_levels():
    tree = _setup_tree()
    batch = jnp.asarray([[2.6, 2.6], [2.62, 2.61], [2.64, 2.63]])

    new_tree, plan = batch_insert(tree, batch, mis_seed=0)

    dominated_count = int(plan.dominated_indices.size)
    assert dominated_count > 0
    selected_count = int(plan.selected_indices.size)
    appended_levels = new_tree.top_levels[-batch.shape[0] :]
    dominated_new = np.asarray(appended_levels[selected_count:])
    dominated_original = np.asarray(plan.traversal.levels[plan.dominated_indices])
    assert np.all(dominated_new <= dominated_original)
    positive_mask = dominated_original > 0
    assert np.all(dominated_new[positive_mask] <= dominated_original[positive_mask] - 1)

    points = np.asarray(new_tree.points)
    levels = np.asarray(new_tree.top_levels)
    parents = np.asarray(new_tree.parents)
    traversal_levels = np.asarray(plan.traversal.levels)
    LOG_EPS = 1e-12
    original_candidates = np.maximum(
        traversal_levels[np.asarray(plan.dominated_indices)].astype(int) - 1, 0
    )
    for offset in range(dominated_count):
        new_idx = tree.num_points + selected_count + offset
        lvl = int(dominated_new[offset])
        parent_idx = int(parents[new_idx])
        parent_level = int(levels[parent_idx]) if parent_idx >= 0 else 0
        dist_parent = 0.0
        if parent_idx >= 0:
            parent_point = points[parent_idx]
            dist_parent = float(np.linalg.norm(points[new_idx] - parent_point))
        candidate_level = int(original_candidates[offset])
        distance_level = 0
        if parent_idx >= 0 and dist_parent > LOG_EPS:
            log_val = math.log(dist_parent, 2) - 1e-12
            distance_level = int(math.floor(log_val))
            if distance_level < 0:
                distance_level = 0
        expected_level = min(candidate_level, parent_level - 1, distance_level)
        if expected_level < 0:
            expected_level = 0
        assert lvl == expected_level

        if lvl == 0:
            if parent_idx >= 0:
                assert dist_parent <= 1.0 + 1e-12
            continue

        anchors_mask = levels >= lvl
        anchors_mask[new_idx] = False
        anchors = points[anchors_mask]
        if anchors.size == 0:
            continue
        distances = np.linalg.norm(anchors - points[new_idx], axis=1)
        assert np.all(distances > (2.0 ** lvl))
        if parent_idx >= 0:
            assert dist_parent > (2.0 ** lvl)


def test_batch_insert_persistence_across_versions():
    tree = _setup_tree()
    original_points = np.asarray(tree.points).copy()
    batch1 = jnp.asarray([[2.4, 2.4], [0.6, 0.5]])
    tree1, plan1 = batch_insert(tree, batch1, mis_seed=0)

    expected1 = np.asarray(
        jnp.concatenate(
            [batch1[plan1.selected_indices], batch1[plan1.dominated_indices]], axis=0
        )
    )
    appended1 = np.asarray(tree1.points)[tree.num_points :]
    assert np.allclose(appended1, expected1)

    batch2 = jnp.asarray([[3.1, 3.05], [3.4, 3.35], [0.1, 0.2]])
    tree2, plan2 = batch_insert(tree1, batch2, mis_seed=1)

    expected2 = np.asarray(
        jnp.concatenate(
            [batch2[plan2.selected_indices], batch2[plan2.dominated_indices]], axis=0
        )
    )

    # Original tree untouched
    assert np.allclose(np.asarray(tree.points), original_points)

    # First version unchanged by second insert
    assert np.allclose(
        np.asarray(tree1.points)[tree.num_points :], expected1
    )

    # Second version preserves first version region and appends new batch
    assert np.allclose(np.asarray(tree2.points)[: tree1.num_points], np.asarray(tree1.points))
    assert np.allclose(
        np.asarray(tree2.points)[tree1.num_points:], expected2
    )

    # Stats monotonic
    assert tree1.stats.num_batches == tree.stats.num_batches + 1
    assert tree2.stats.num_batches == tree1.stats.num_batches + 1


def test_batch_insert_prefix_doubling_matches_manual_sequence():
    tree = _setup_tree()
    batch = jnp.asarray(
        [
            [2.1, 2.0],
            [0.4, 0.5],
            [3.0, 3.1],
            [2.2, 2.3],
            [0.8, 0.7],
            [3.5, 3.6],
        ]
    )

    mis_seed = 7
    shuffle_seed = 11

    tree_pref, result = batch_insert_prefix_doubling(
        tree, batch, mis_seed=mis_seed, shuffle_seed=shuffle_seed
    )

    assert result.permutation.shape[0] == batch.shape[0]
    assert sorted(result.permutation.tolist()) == list(range(batch.shape[0]))

    tree_manual = tree
    seeds = batch_mis_seeds(len(result.groups), seed=mis_seed)
    for idx, group in enumerate(result.groups):
        pts = batch[jnp.asarray(group.permutation_indices.tolist())]
        sub_seed = int(seeds[idx]) if seeds else None
        tree_manual, _ = batch_insert(tree_manual, pts, mis_seed=sub_seed)

    assert jnp.allclose(tree_pref.points, tree_manual.points)
    assert tree_pref.stats.num_batches == tree_manual.stats.num_batches
