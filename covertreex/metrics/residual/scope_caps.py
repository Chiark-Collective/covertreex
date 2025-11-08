from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List

import numpy as np

_SCOPE_CAP_CACHE: Dict[str, "ResidualScopeCapTable"] = {}
_CACHE_LOCK = Lock()


def _normalise_cap(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _coerce_level_list(level_field: int | List[int] | None) -> Iterable[int]:
    if level_field is None:
        return ()
    if isinstance(level_field, list):
        return [int(level) for level in level_field]
    return (int(level_field),)


@dataclass(frozen=True)
class ResidualScopeCapTable:
    level_caps: Dict[int, float]
    default_cap: float | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> "ResidualScopeCapTable":
        schema = int(payload.get("schema", 1))
        if schema != 1:
            raise ValueError(f"Unsupported residual scope cap schema '{schema}'.")
        default_cap = _normalise_cap(payload.get("default"))
        level_caps: Dict[int, float] = {}
        entries = payload.get("levels", {})
        if isinstance(entries, dict):
            items = []
            for key, value in entries.items():
                cap_value = value.get("cap") if isinstance(value, dict) else value
                items.append({"level": int(key), "cap": cap_value})
        elif isinstance(entries, list):
            items = entries
        else:
            raise ValueError("Residual scope cap payload must provide 'levels'.")
        for entry in items:
            cap = _normalise_cap(entry.get("cap"))
            if cap is None:
                continue
            for level in _coerce_level_list(entry.get("level")):
                level_caps[int(level)] = cap
        return cls(level_caps=level_caps, default_cap=default_cap)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualScopeCapTable":
        target = Path(path).expanduser()
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_payload(payload)

    def lookup(self, levels: np.ndarray) -> np.ndarray:
        if levels.size == 0:
            return np.empty_like(levels, dtype=np.float64)
        result = np.full(levels.shape, np.nan, dtype=np.float64)
        for level, cap in self.level_caps.items():
            mask = levels == level
            if np.any(mask):
                result[mask] = cap
        return result


def get_scope_cap_table(path: str | None) -> ResidualScopeCapTable | None:
    if not path:
        return None
    target = str(Path(path).expanduser())
    with _CACHE_LOCK:
        table = _SCOPE_CAP_CACHE.get(target)
        if table is not None:
            return table
        table = ResidualScopeCapTable.load(target)
        _SCOPE_CAP_CACHE[target] = table
        return table


def reset_scope_cap_cache() -> None:
    with _CACHE_LOCK:
        _SCOPE_CAP_CACHE.clear()


__all__ = [
    "ResidualScopeCapTable",
    "get_scope_cap_table",
    "reset_scope_cap_cache",
]
