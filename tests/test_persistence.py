import numpy as np
import pytest

from covertreex import config as cx_config
from covertreex.core import persistence as persistence_mod
from covertreex.core.persistence import (
    SliceUpdate,
    apply_persistence_journal,
    build_persistence_journal,
    clone_array_segment,
    clone_tree_with_updates,
)
from covertreex.core.tree import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
    compute_level_offsets,
    get_runtime_backend,
)


def _build_tree(backend: TreeBackend | None = None):
    backend = backend or get_runtime_backend()
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
    backend = get_runtime_backend()
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


def test_build_persistence_journal_tracks_head_updates():
    backend = TreeBackend.numpy()
    tree = _build_tree(backend=backend)

    inserted_points = np.asarray([[2.0, 2.0], [3.0, 3.0]], dtype=float)
    inserted_levels = np.asarray([0, 0], dtype=np.int64)
    inserted_parents = np.asarray([0, 0], dtype=np.int64)
    inserted_si = np.asarray([1.0, 1.0], dtype=float)

    journal = build_persistence_journal(
        tree,
        backend=backend,
        inserted_points=inserted_points,
        inserted_levels=inserted_levels,
        inserted_parents=inserted_parents,
        inserted_si=inserted_si,
    )

    assert journal.base_length == 2
    assert journal.inserted_count == 2
    assert journal.head_parents.tolist() == [0, 0]
    assert journal.head_values.tolist() == [2, 3]
    assert journal.next_nodes.tolist() == [2, 3]
    assert journal.next_values.tolist() == [1, 2]

    expected_offsets = compute_level_offsets(
        backend,
        backend.asarray(np.asarray([1, 0, 0, 0], dtype=np.int32), dtype=backend.default_int),
    )
    np.testing.assert_array_equal(
        journal.level_offsets,
        backend.to_numpy(expected_offsets),
    )


@pytest.mark.parametrize("enable_numba", ["0", "1"])
def test_apply_persistence_journal_numpy(monkeypatch, enable_numba):
    if enable_numba == "1" and not persistence_mod.NUMBA_PERSISTENCE_AVAILABLE:
        pytest.skip("Numba persistence helpers are unavailable")

    monkeypatch.setenv("COVERTREEX_BACKEND", "numpy")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", enable_numba)
    cx_config.reset_runtime_context()

    tracker: dict[str, bool] = {"numba_called": False}

    if enable_numba == "1":
        original_apply = persistence_mod._apply_journal_numba

        def _tracking_apply(*args, **kwargs):
            tracker["numba_called"] = True
            return original_apply(*args, **kwargs)

        monkeypatch.setattr(persistence_mod, "_apply_journal_numba", _tracking_apply)

    try:
        backend = TreeBackend.numpy()
        tree = _build_tree(backend=backend)

        inserted_points = np.asarray([[2.0, 2.0]], dtype=float)
        inserted_levels = np.asarray([0], dtype=np.int64)
        inserted_parents = np.asarray([0], dtype=np.int64)
        inserted_si = np.asarray([2.0], dtype=float)

        journal = build_persistence_journal(
            tree,
            backend=backend,
            inserted_points=inserted_points,
            inserted_levels=inserted_levels,
            inserted_parents=inserted_parents,
            inserted_si=inserted_si,
        )

        updated = apply_persistence_journal(tree, journal, backend=backend)

        assert updated is not tree
        assert tree.parents.tolist() == [-1, 0]
        assert updated.parents.tolist() == [-1, 0, 0]
        assert updated.children.tolist() == [2, -1, -1]
        assert updated.next_cache.tolist() == [1, -1, 1]
        assert updated.top_levels.tolist() == [1, 0, 0]
        np.testing.assert_array_equal(
            backend.to_numpy(updated.level_offsets),
            journal.level_offsets,
        )
        assert updated.points.shape[0] == 3
        assert tree.points.shape[0] == 2
        assert updated.points.tolist()[-1] == [2.0, 2.0]

        if enable_numba == "1":
            assert tracker["numba_called"] is True
    finally:
        cx_config.reset_runtime_context()
