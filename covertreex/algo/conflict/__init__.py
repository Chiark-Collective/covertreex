from __future__ import annotations

from .base import ConflictGraph, ConflictGraphTimings
from .runner import build_conflict_graph

# NOTE: Plugin loading moved to explicit call to avoid circular import.
# Plugins are loaded when covertreex.plugins.conflict is imported directly,
# or when load_conflict_plugins() is called.


def load_conflict_plugins() -> None:
    """Explicitly load conflict strategy plugins.

    This is called automatically when covertreex.plugins.conflict is imported.
    Call manually if you need plugins before that import occurs.
    """
    from covertreex.plugins import conflict as _conflict_plugins  # noqa: F401


__all__ = ["ConflictGraph", "ConflictGraphTimings", "build_conflict_graph", "load_conflict_plugins"]
