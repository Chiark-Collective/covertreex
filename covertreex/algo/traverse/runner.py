from __future__ import annotations

from typing import Any

from covertreex import config as cx_config
from covertreex.core.tree import PCCTree, TreeBackend

from .base import (
    ResidualTraversalCache,
    TraversalResult,
    TraversalTimings,
    broadcast_batch,
    empty_result,
)
from .strategies import select_traversal_strategy

__all__ = [
    "ResidualTraversalCache",
    "TraversalResult",
    "TraversalTimings",
    "traverse_collect_scopes",
]


def traverse_collect_scopes(
    tree: PCCTree,
    batch_points: Any,
    *,
    backend: TreeBackend | None = None,
) -> TraversalResult:
    """Compute parent assignments and conflict scopes for a batch of points."""

    backend = backend or tree.backend
    batch = broadcast_batch(backend, batch_points)
    batch_size = int(batch.shape[0]) if batch.size else 0

    if batch_size == 0:
        return empty_result(backend, 0)

    if tree.is_empty():
        return empty_result(backend, batch_size)

    runtime = cx_config.runtime_config()
    strategy = select_traversal_strategy(runtime, backend)
    return strategy.collect(tree, batch, backend=backend, runtime=runtime)
