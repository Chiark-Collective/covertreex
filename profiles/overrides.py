from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import yaml

from covertreex.runtime.model import RuntimeModel


class OverrideError(ValueError):
    """Raised when override expressions are invalid."""


def parse_override_expression(expression: str) -> Tuple[str, Any]:
    """Parse ``path=value`` expressions using dot-paths."""

    if "=" not in expression:
        raise OverrideError(f"Override '{expression}' must contain '='.")
    key, raw_value = expression.split("=", 1)
    key = key.strip()
    if not key:
        raise OverrideError(f"Override '{expression}' is missing a field path.")
    value = yaml.safe_load(raw_value)
    return key, value


def parse_override_expressions(expressions: Sequence[str] | None) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not expressions:
        return overrides
    for expression in expressions:
        field, value = parse_override_expression(expression)
        overrides[field] = value
    return overrides


def apply_overrides_to_model(
    model: RuntimeModel, overrides: Mapping[str, Any] | None
) -> RuntimeModel:
    """Return a new model with overrides applied."""

    if not overrides:
        return model
    payload = model.model_dump()
    for path, value in overrides.items():
        segments = tuple(segment for segment in path.split(".") if segment)
        if not segments:
            raise OverrideError(f"Invalid override path '{path}'.")
        target = payload
        for segment in segments[:-1]:
            current = target.get(segment)
            if current is None or not isinstance(current, dict):
                current = {}
                target[segment] = current
            target = current
        target[segments[-1]] = value
    return RuntimeModel(**payload)
