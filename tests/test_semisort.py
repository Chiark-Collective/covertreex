import numpy as np
import pytest

jax = pytest.importorskip("jax")

from covertreex.algo import group_by_int, select_topk_by_level
from covertreex.core.tree import get_runtime_backend


def test_group_by_int_basic():
    backend = get_runtime_backend()
    keys = backend.asarray([2, 0, 2, 1, 1, 0], dtype=backend.default_int)
    values = backend.asarray(
        np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float64),
    )

    result = group_by_int(keys, values, backend=backend)

    assert result.keys.tolist() == [0, 1, 2]
    assert result.indptr.tolist() == [0, 2, 4, 6]

    grouped = np.asarray(result.values)
    assert grouped.shape == (6, 1)
    # Verify stability for equal keys
    assert grouped[:2, 0].tolist() == [2, 6]
    assert grouped[2:4, 0].tolist() == [4, 5]
    assert grouped[4:, 0].tolist() == [1, 3]


def test_group_by_int_empty():
    backend = get_runtime_backend()
    result = group_by_int(
        backend.asarray([], dtype=backend.default_int),
        backend.asarray([], dtype=backend.default_float),
        backend=backend,
    )

    assert result.keys.shape[0] == 0
    assert result.indptr.tolist() == [0]
    assert result.values.shape[0] == 0


def test_select_topk_by_level_applies_limit():
    indices = np.array([10, 2, 5, 4, 7])
    levels = np.array([1, 3, 2, 3, 3])

    result = select_topk_by_level(indices, levels, limit=2)

    assert result.tolist() == [2, 4]


def test_select_topk_by_level_orders_full_when_unlimited():
    indices = np.array([5, 1, 9])
    levels = np.array([0, 0, 2])

    result = select_topk_by_level(indices, levels, limit=0)

    assert result.tolist() == [9, 1, 5]
