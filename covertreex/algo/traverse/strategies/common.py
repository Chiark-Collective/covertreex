from __future__ import annotations

import numpy as np

from covertreex.core.tree import PCCTree


def _collect_next_chain(
    tree: PCCTree,
    start: int,
    *,
    next_cache: np.ndarray | None = None,
) -> tuple[int, ...]:
    cache = next_cache if next_cache is not None else np.asarray(tree.next_cache, dtype=np.int64)
    num_points = cache.shape[0]
    if start < 0 or start >= num_points:
        return ()
    chain: list[int] = []
    visited: set[int] = set()
    current = start
    while 0 <= current < num_points and current not in visited:
        chain.append(current)
        visited.add(current)
        if cache.size == 0:
            break
        nxt = int(cache[current])
        if nxt < 0:
            break
        current = nxt
    return tuple(chain)


__all__ = ["_collect_next_chain"]
