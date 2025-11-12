from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import ConflictGraphContext, ConflictGraphStrategy
from .builders import (
    AdjacencyBuild,
    build_dense_adjacency,
    build_grid_adjacency,
    build_residual_adjacency,
    build_segmented_adjacency,
)


class _DenseConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_dense_adjacency(
            backend=ctx.backend,
            batch_size=ctx.batch_size,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
            pairwise=ctx.pairwise,
            radii=ctx.radii_np,
            chunk_target_override=ctx.chunk_target_override,
        )


class _SegmentedConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_segmented_adjacency(
            backend=ctx.backend,
            scope_indices=ctx.scope_indices,
            point_ids=ctx.point_ids,
            pairwise_np=ctx.pairwise_np,
            radii_np=ctx.radii_np,
        )


class _GridConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_grid_adjacency(
            backend=ctx.backend,
            batch_points=ctx.grid_points if ctx.grid_points is not None else ctx.batch,
            batch_levels=ctx.levels,
            radii=ctx.radii,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
        )


class _ResidualGridConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_grid_adjacency(
            backend=ctx.backend,
            batch_points=ctx.grid_points if ctx.grid_points is not None else ctx.batch,
            batch_levels=ctx.levels,
            radii=ctx.radii,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
        )


class _ResidualConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_residual_adjacency(
            backend=ctx.backend,
            batch_size=ctx.batch_size,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
            pairwise=ctx.pairwise,
            radii=ctx.radii_np,
            residual_pairwise=ctx.residual_pairwise_np,
            chunk_target_override=ctx.chunk_target_override,
        )


@dataclass(frozen=True)
class _ConflictStrategySpec:
    name: str
    predicate: Callable[[Any, bool, bool], bool]
    factory: Callable[[], ConflictGraphStrategy]


_CONFLICT_REGISTRY: list[_ConflictStrategySpec] = []


def register_conflict_strategy(
    name: str,
    *,
    predicate: Callable[[Any, bool, bool], bool],
    factory: Callable[[], ConflictGraphStrategy],
) -> None:
    global _CONFLICT_REGISTRY
    _CONFLICT_REGISTRY = [spec for spec in _CONFLICT_REGISTRY if spec.name != name]
    _CONFLICT_REGISTRY.append(_ConflictStrategySpec(name=name, predicate=predicate, factory=factory))


def registered_conflict_strategies() -> tuple[str, ...]:
    return tuple(spec.name for spec in _CONFLICT_REGISTRY)


register_conflict_strategy(
    "residual_grid",
    predicate=lambda runtime, residual_mode, *_: (
        residual_mode and getattr(runtime, "conflict_graph_impl", "") == "grid"
    ),
    factory=_ResidualGridConflictStrategy,
)

register_conflict_strategy(
    "residual",
    predicate=lambda runtime, residual_mode, has_residual: residual_mode and has_residual,
    factory=_ResidualConflictStrategy,
)

register_conflict_strategy(
    "segmented",
    predicate=lambda runtime, *_: getattr(runtime, "conflict_graph_impl", "") == "segmented",
    factory=_SegmentedConflictStrategy,
)

register_conflict_strategy(
    "grid",
    predicate=lambda runtime, *_: getattr(runtime, "conflict_graph_impl", "") == "grid",
    factory=_GridConflictStrategy,
)

register_conflict_strategy(
    "dense",
    predicate=lambda *_: True,
    factory=_DenseConflictStrategy,
)


def select_conflict_strategy(
    runtime: Any,
    *,
    residual_mode: bool,
    has_residual_distances: bool,
) -> ConflictGraphStrategy:
    for spec in _CONFLICT_REGISTRY:
        if spec.predicate(runtime, residual_mode, has_residual_distances):
            return spec.factory()
    raise RuntimeError("No conflict strategy registered for the current runtime.")
