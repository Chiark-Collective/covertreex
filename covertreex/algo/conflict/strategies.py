from __future__ import annotations

from typing import Any

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
            batch_points=ctx.batch,
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
        )


def select_conflict_strategy(
    runtime: Any,
    *,
    residual_mode: bool,
    has_residual_distances: bool,
) -> ConflictGraphStrategy:
    if residual_mode and has_residual_distances:
        return _ResidualConflictStrategy()
    impl = runtime.conflict_graph_impl
    if impl == "segmented":
        return _SegmentedConflictStrategy()
    if impl == "grid":
        return _GridConflictStrategy()
    return _DenseConflictStrategy()
