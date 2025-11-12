from __future__ import annotations

from .registry import (
    register_traversal_strategy,
    registered_traversal_strategies,
    select_traversal_strategy,
)

# Import implementations for side effects so their strategies self-register.
from . import residual as _residual  # noqa: F401
from . import euclidean as _euclidean  # noqa: F401

__all__ = [
    "register_traversal_strategy",
    "registered_traversal_strategies",
    "select_traversal_strategy",
]
