from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import numpy as np

from covertreex.core.metrics import get_metric
from covertreex.core.tree import PCCTree, TreeBackend


@dataclass(frozen=True)
class ResidualTraversalCache:
    batch_indices: np.ndarray
    pairwise: np.ndarray
    scope_radii: np.ndarray | None = None
    scope_saturated: np.ndarray | None = None
    scope_chunk_iterations: np.ndarray | None = None
    scope_chunk_points: np.ndarray | None = None
    scope_dedupe_hits: np.ndarray | None = None
    scope_radius_initial: np.ndarray | None = None
    scope_radius_limits: np.ndarray | None = None
    scope_radius_caps: np.ndarray | None = None


@dataclass(frozen=True)
class TraversalResult:
    """Structured output produced by batched traversal."""

    parents: Any
    levels: Any
    conflict_scopes: Tuple[Tuple[int, ...], ...]
    scope_indptr: Any
    scope_indices: Any
    timings: "TraversalTimings"
    residual_cache: ResidualTraversalCache | None = None
    engine: str = "unknown"
    gate_active: bool = False


@dataclass(frozen=True)
class TraversalTimings:
    pairwise_seconds: float
    mask_seconds: float
    semisort_seconds: float
    chain_seconds: float = 0.0
    nonzero_seconds: float = 0.0
    sort_seconds: float = 0.0
    assemble_seconds: float = 0.0
    tile_seconds: float = 0.0
    build_wall_seconds: float = 0.0
    scope_chunk_segments: int = 0
    scope_chunk_emitted: int = 0
    scope_chunk_max_members: int = 0
    scope_chunk_scans: int = 0
    scope_chunk_points: int = 0
    scope_chunk_dedupe: int = 0
    scope_chunk_saturated: int = 0
    scope_cache_hits: int = 0
    scope_cache_prefetch: int = 0
    scope_budget_start: int = 0
    scope_budget_final: int = 0
    scope_budget_escalations: int = 0
    scope_budget_early_terminate: int = 0
    whitened_block_pairs: int = 0
    whitened_block_seconds: float = 0.0
    whitened_block_calls: int = 0
    kernel_provider_pairs: int = 0
    kernel_provider_seconds: float = 0.0
    kernel_provider_calls: int = 0


class TraversalStrategy(Protocol):
    def collect(
        self,
        tree: PCCTree,
        batch: Any,
        *,
        backend: TreeBackend,
        runtime: Any,
    ) -> TraversalResult:
        ...


def block_until_ready(value: Any) -> None:
    """Best-effort barrier for asynchronous backends (e.g., JAX)."""

    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


def broadcast_batch(backend: TreeBackend, batch_points: Any) -> Any:
    return backend.asarray(batch_points, dtype=backend.default_float)


def collect_distances(tree: PCCTree, batch: Any, backend: TreeBackend) -> Any:
    metric = get_metric()
    return metric.pairwise(backend, batch, tree.points)


def empty_result(backend: TreeBackend, batch_size: int) -> TraversalResult:
    xp = backend.xp
    parents = backend.asarray(
        xp.full((batch_size,), -1), dtype=backend.default_int
    )
    levels = backend.asarray(
        xp.full((batch_size,), -1), dtype=backend.default_int
    )
    conflict_scopes: Tuple[Tuple[int, ...], ...] = tuple(() for _ in range(batch_size))
    scope_indptr = backend.asarray([0] * (batch_size + 1), dtype=backend.default_int)
    scope_indices = backend.asarray([], dtype=backend.default_int)
    return TraversalResult(
        parents=parents,
        levels=levels,
        conflict_scopes=conflict_scopes,
        scope_indptr=scope_indptr,
        scope_indices=scope_indices,
        timings=TraversalTimings(
            pairwise_seconds=0.0,
            mask_seconds=0.0,
            semisort_seconds=0.0,
            chain_seconds=0.0,
            nonzero_seconds=0.0,
            sort_seconds=0.0,
            assemble_seconds=0.0,
        ),
    )
