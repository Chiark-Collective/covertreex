from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from covertreex.algo.conflict import ConflictGraph
from covertreex.algo.mis import MISResult
from covertreex.algo.traverse import TraversalResult


@dataclass(frozen=True)
class LevelSummary:
    level: int
    candidates: Any
    selected: Any
    dominated: Any


@dataclass(frozen=True)
class BatchInsertTimings:
    traversal_seconds: float
    conflict_graph_seconds: float
    mis_seconds: float


@dataclass(frozen=True)
class BatchInsertPlan:
    traversal: TraversalResult
    conflict_graph: ConflictGraph
    mis_result: MISResult
    selected_indices: Any
    dominated_indices: Any
    level_summaries: tuple[LevelSummary, ...]
    timings: BatchInsertTimings
    batch_permutation: Any | None = None
    batch_order_strategy: str = "natural"
    batch_order_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefixBatchGroup:
    permutation_indices: Any
    plan: BatchInsertPlan
    prefix_factor: float | None = None
    domination_ratio: float | None = None


@dataclass(frozen=True)
class PrefixBatchResult:
    permutation: Any
    groups: Tuple[PrefixBatchGroup, ...]
    order_strategy: str
    order_metrics: Dict[str, float]
