"""Core data structures and persistence primitives for the PCCT."""

from .metrics import (
    Metric,
    MetricRegistry,
    available_metrics,
    configure_residual_metric,
    get_metric,
    reset_residual_metric,
)
from .persistence import SliceUpdate, clone_array_segment, clone_tree_with_updates
from .tree import DEFAULT_BACKEND, PCCTree, TreeBackend, TreeLogStats

__all__ = [
    "DEFAULT_BACKEND",
    "PCCTree",
    "TreeBackend",
    "TreeLogStats",
    "SliceUpdate",
    "clone_array_segment",
    "clone_tree_with_updates",
    "Metric",
    "MetricRegistry",
    "available_metrics",
    "configure_residual_metric",
    "get_metric",
    "reset_residual_metric",
]
