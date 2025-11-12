from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from covertreex.core.tree import TreeBackend


@dataclass(frozen=True)
class GroupByResult:
    keys: Any
    indptr: Any
    values: Any


def group_by_int(
    keys: Any,
    values: Any,
    *,
    backend: TreeBackend,
    stable: bool = True,
) -> GroupByResult:
    """Group `values` by integer `keys` and return CSR-style buffers.

    Parameters
    ----------
    keys:
        1-D array of integer keys.
    values:
        Array whose first dimension matches `keys`.
    backend:
        Tree backend providing array helpers.
    stable:
        Whether to preserve the relative order of equal keys (default True).
    """

    xp = backend.xp
    keys_arr = backend.asarray(keys, dtype=backend.default_int)
    values_arr = backend.asarray(values)

    if keys_arr.ndim != 1:
        raise ValueError("group_by_int expects 1-D `keys`.")
    if values_arr.shape[0] != keys_arr.shape[0]:
        raise ValueError("`values` must align with `keys` in the first dimension.")

    size = int(keys_arr.shape[0])
    if size == 0:
        empty_keys = backend.asarray([], dtype=backend.default_int)
        empty_indptr = backend.asarray([0], dtype=backend.default_int)
        return GroupByResult(keys=empty_keys, indptr=empty_indptr, values=values_arr)

    indices = xp.arange(size, dtype=backend.default_int)
    if stable:
        order = xp.lexsort((indices, keys_arr))
    else:
        order = xp.argsort(keys_arr)
    sorted_keys = keys_arr[order]
    sorted_values = values_arr[order]

    unique_keys, counts = xp.unique(sorted_keys, return_counts=True)
    counts = counts.astype(backend.default_int)
    indptr = xp.concatenate(
        (
            xp.zeros((1,), dtype=backend.default_int),
            xp.cumsum(counts, dtype=backend.default_int),
        ),
        axis=0,
    )

    return GroupByResult(
        keys=backend.device_put(unique_keys.astype(backend.default_int)),
        indptr=backend.device_put(indptr),
        values=backend.device_put(sorted_values),
    )


def select_topk_by_level(
    indices: np.ndarray,
    levels: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Return indices ordered by (level desc, index asc), truncated to ``limit``.

    When ``limit`` <= 0 the full ordered list is returned. The helper uses
    ``np.argpartition`` to avoid sorting more than ``limit`` entries.
    """

    idx = np.asarray(indices, dtype=np.int64)
    if idx.size <= 1:
        return idx.copy()
    lvl = np.asarray(levels, dtype=np.int64)
    if idx.shape != lvl.shape:
        raise ValueError("indices and levels must share the same shape")

    max_keep = int(limit) if int(limit) > 0 else 0
    if max_keep <= 0 or max_keep >= idx.size:
        order = np.lexsort((idx, -lvl))
        return idx[order]

    kth = max_keep - 1
    if kth < 0:
        kth = 0
    if kth >= idx.size:
        kth = idx.size - 1
    partition_idx = np.argpartition(-lvl, kth)[:max_keep]
    subset_idx = idx[partition_idx]
    subset_lvl = lvl[partition_idx]
    order = np.lexsort((subset_idx, -subset_lvl))
    return subset_idx[order]
