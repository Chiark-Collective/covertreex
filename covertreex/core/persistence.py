from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np

from covertreex.core.tree import PCCTree, TreeBackend


@dataclass(frozen=True)
class SliceUpdate:
    """Descriptor for a copy-on-write update applied to a single array."""

    index: Tuple[Any, ...] | Any
    values: Any


def _ensure_array(backend: TreeBackend, values: Any, *, dtype: Any) -> Any:
    return backend.asarray(values, dtype=dtype)


def _normalise_index(index: Tuple[int, ...] | Tuple[slice, ...] | int | slice) -> Tuple[Any, ...]:
    if isinstance(index, tuple):
        return index
    return (index,)


def _required_length_along_axis(index: Tuple[Any, ...], current: int) -> int:
    if not index:
        return current
    primary = index[0]
    if isinstance(primary, slice):
        if primary.stop is None:
            return current
        return max(current, int(primary.stop))
    if isinstance(primary, int):
        if primary < 0:
            return current
        return max(current, int(primary) + 1)
    return current


def clone_array_segment(
    backend: TreeBackend, source: Any, updates: Iterable[SliceUpdate], *, dtype: Any
) -> Any:
    """Clone `source` and apply `updates` without mutating the original array."""

    target = backend.asarray(source, dtype=dtype)
    xp = backend.xp
    updates_list = list(updates)

    if updates_list:
        required_length = target.shape[0] if target.ndim >= 1 else 0
        for update in updates_list:
            index = _normalise_index(update.index)
            required_length = _required_length_along_axis(index, required_length)

        if target.ndim >= 1 and required_length > target.shape[0]:
            pad_shape = list(target.shape)
            pad_shape[0] = required_length - target.shape[0]
            pad = xp.zeros(tuple(pad_shape), dtype=dtype)
            target = xp.concatenate([target, pad], axis=0)

        for update in updates_list:
            index = _normalise_index(update.index)
            values = _ensure_array(backend, update.values, dtype=target.dtype)
            if hasattr(target, "at"):
                target = target.at[index].set(values)
            else:
                target = np.array(target, copy=True)
                target[index] = values

    return backend.device_put(target)


def clone_tree_with_updates(
    tree: PCCTree,
    *,
    points_updates: Iterable[SliceUpdate] = (),
    top_level_updates: Iterable[SliceUpdate] = (),
    parent_updates: Iterable[SliceUpdate] = (),
    child_updates: Iterable[SliceUpdate] = (),
    level_offset_updates: Iterable[SliceUpdate] = (),
    si_cache_updates: Iterable[SliceUpdate] = (),
    next_cache_updates: Iterable[SliceUpdate] = (),
) -> PCCTree:
    """Produce a new `PCCTree` with updates applied via copy-on-write semantics."""

    backend = tree.backend
    return tree.replace(
        points=clone_array_segment(
            backend, tree.points, points_updates, dtype=backend.default_float
        ),
        top_levels=clone_array_segment(
            backend, tree.top_levels, top_level_updates, dtype=backend.default_int
        ),
        parents=clone_array_segment(
            backend, tree.parents, parent_updates, dtype=backend.default_int
        ),
        children=clone_array_segment(
            backend, tree.children, child_updates, dtype=backend.default_int
        ),
        level_offsets=clone_array_segment(
            backend,
            tree.level_offsets,
            level_offset_updates,
            dtype=backend.default_int,
        ),
        si_cache=clone_array_segment(
            backend, tree.si_cache, si_cache_updates, dtype=backend.default_float
        ),
        next_cache=clone_array_segment(
            backend, tree.next_cache, next_cache_updates, dtype=backend.default_int
        ),
    )
