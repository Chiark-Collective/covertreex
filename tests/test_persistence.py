import numpy as np

from covertreex.core.persistence import SliceUpdate, clone_array_segment, clone_tree_with_updates
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _build_tree():
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


def test_clone_array_segment_preserves_original():
    backend = DEFAULT_BACKEND
    source = backend.asarray([0, 1, 2], dtype=backend.default_int)
    updates = [SliceUpdate(index=(1,), values=99)]

    target = clone_array_segment(backend, source, updates, dtype=backend.default_int)

    assert source.tolist() == [0, 1, 2]
    assert target.tolist() == [0, 99, 2]


def test_clone_tree_with_updates_replaces_values():
    tree = _build_tree()

    updated = clone_tree_with_updates(
        tree,
        points_updates=[SliceUpdate(index=(1,), values=np.array([2.0, 2.0], dtype=tree.backend.default_float))],
        parent_updates=[SliceUpdate(index=(1,), values=0)],
    )

    assert updated is not tree
    assert updated.points.tolist() == [[0.0, 0.0], [2.0, 2.0]]
    assert updated.parents.tolist() == [-1, 0]
    assert tree.points.tolist() == [[0.0, 0.0], [1.0, 1.0]]
