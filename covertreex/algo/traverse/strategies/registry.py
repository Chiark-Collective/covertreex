from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from covertreex.core.tree import TreeBackend
from covertreex.logging import get_logger

from ..base import TraversalStrategy

LOGGER = get_logger("algo.traverse.registry")


@dataclass(frozen=True)
class _TraversalStrategySpec:
    name: str
    predicate: Callable[[Any, TreeBackend], bool]
    factory: Callable[[], TraversalStrategy]


_TRAVERSAL_REGISTRY: list[_TraversalStrategySpec] = []


def register_traversal_strategy(
    name: str,
    *,
    predicate: Callable[[Any, TreeBackend], bool],
    factory: Callable[[], TraversalStrategy],
) -> None:
    """Register or replace a traversal strategy selection rule."""

    global _TRAVERSAL_REGISTRY
    _TRAVERSAL_REGISTRY = [spec for spec in _TRAVERSAL_REGISTRY if spec.name != name]
    _TRAVERSAL_REGISTRY.append(
        _TraversalStrategySpec(name=name, predicate=predicate, factory=factory)
    )
    LOGGER.debug("Registered traversal strategy: %%s", name)


def registered_traversal_strategies() -> Tuple[str, ...]:
    return tuple(spec.name for spec in _TRAVERSAL_REGISTRY)


def select_traversal_strategy(runtime: Any, backend: TreeBackend) -> TraversalStrategy:
    for spec in _TRAVERSAL_REGISTRY:
        try:
            if spec.predicate(runtime, backend):
                return spec.factory()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Traversal strategy '%%s' predicate failed.", spec.name)
            continue
    raise RuntimeError("No traversal strategy registered for the current runtime/backend.")


__all__ = [
    "register_traversal_strategy",
    "registered_traversal_strategies",
    "select_traversal_strategy",
]
