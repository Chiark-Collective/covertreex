from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from covertreex.core.tree import TreeBackend

if TYPE_CHECKING:  # pragma: no cover
    from .builders import AdjacencyBuild


@dataclass(frozen=True)
class ConflictGraph:
    """Conflict graph encoded in CSR form."""

    indptr: Any
    indices: Any
    pairwise_distances: Any
    scope_indptr: Any
    scope_indices: Any
    radii: Any
    annulus_bounds: Any
    annulus_bins: Any
    annulus_bin_indptr: Any
    annulus_bin_indices: Any
    annulus_bin_ids: Any
    timings: "ConflictGraphTimings"
    forced_selected: Any | None = None
    forced_dominated: Any | None = None
    grid_cells: int = 0
    grid_leaders_raw: int = 0
    grid_leaders_after: int = 0
    grid_local_edges: int = 0

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def num_edges(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_scopes(self) -> int:
        return int(self.scope_indptr.shape[0] - 1)


@dataclass(frozen=True)
class ConflictGraphTimings:
    pairwise_seconds: float
    scope_group_seconds: float
    adjacency_seconds: float
    annulus_seconds: float
    adjacency_membership_seconds: float = 0.0
    adjacency_targets_seconds: float = 0.0
    adjacency_scatter_seconds: float = 0.0
    adjacency_filter_seconds: float = 0.0
    adjacency_sort_seconds: float = 0.0
    adjacency_dedup_seconds: float = 0.0
    adjacency_extract_seconds: float = 0.0
    adjacency_csr_seconds: float = 0.0
    adjacency_total_pairs: float = 0.0
    adjacency_candidate_pairs: float = 0.0
    adjacency_max_group_size: float = 0.0
    scope_bytes_h2d: int = 0
    scope_bytes_d2h: int = 0
    scope_groups: int = 0
    scope_groups_unique: int = 0
    scope_domination_ratio: float = 0.0
    scope_chunk_segments: int = 0
    scope_chunk_emitted: int = 0
    scope_chunk_max_members: int = 0
    scope_chunk_pair_cap: int = 0
    scope_chunk_pairs_before: int = 0
    scope_chunk_pairs_after: int = 0
    mis_seconds: float = 0.0
    grid_cells: int = 0
    grid_leaders_raw: int = 0
    grid_leaders_after: int = 0
    grid_local_edges: int = 0


@dataclass(frozen=True)
class ConflictGraphContext:
    backend: TreeBackend
    batch: Any
    batch_size: int
    scope_indptr: Any
    scope_indices: Any
    radii: Any
    radii_np: np.ndarray | None
    pairwise: Any | None
    pairwise_np: np.ndarray | None
    residual_pairwise_np: np.ndarray | None
    point_ids: Any
    levels: Any


class ConflictGraphStrategy(Protocol):
    def build(self, ctx: ConflictGraphContext) -> "AdjacencyBuild":
        ...
