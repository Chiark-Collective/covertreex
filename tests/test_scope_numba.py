import numpy as np
import pytest

from covertreex.algo._scope_numba import (
    NUMBA_SCOPE_AVAILABLE,
    build_scope_csr_from_pairs,
)


pytestmark = pytest.mark.skipif(
    not NUMBA_SCOPE_AVAILABLE, reason="Numba scope helpers are unavailable"
)


def test_scope_builder_limits_and_orders_with_levels():
    owners = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    members = np.array([3, 5, 4, 2, 0], dtype=np.int64)
    top_levels = np.array([0, 4, 6, 7, 1, 3], dtype=np.int64)
    parents = np.array([5, 0], dtype=np.int64)

    indptr, indices = build_scope_csr_from_pairs(
        owners,
        members,
        2,
        limit=2,
        top_levels=top_levels,
        parents=parents,
    )

    assert indptr.tolist() == [0, 2, 4]
    # row0: nodes {3(lvl7),5(lvl3),4(lvl1)} -> keep [3,5]
    # row1: nodes {2(lvl6),0(lvl0)} -> keep [2,0]
    assert indices.tolist() == [3, 5, 2, 0]


def test_scope_builder_limits_without_levels():
    owners = np.array([0, 0, 0], dtype=np.int64)
    members = np.array([5, 4, 3], dtype=np.int64)

    indptr, indices = build_scope_csr_from_pairs(owners, members, 1, limit=2)

    # Expect ascending index order with cap of 2 entries.
    assert indptr.tolist() == [0, 2]
    assert indices.tolist() == [3, 4]


def test_scope_builder_orders_full_row_when_limit_disabled():
    owners = np.array([0, 0, 0], dtype=np.int64)
    members = np.array([2, 4, 3], dtype=np.int64)
    top_levels = np.array([0, 1, 5, 3, 7], dtype=np.int64)

    indptr, indices = build_scope_csr_from_pairs(
        owners,
        members,
        1,
        limit=0,
        top_levels=top_levels,
    )

    # levels -> node4(7), node2(5), node3(3)
    assert indptr.tolist() == [0, 3]
    assert indices.tolist() == [4, 2, 3]
