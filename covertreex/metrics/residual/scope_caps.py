from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

import numpy as np

_SCOPE_CAP_CACHE: Dict[str, "ResidualScopeCaps"] = {}
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
class ResidualScopeCaps:
    level_caps: Dict[int, float]
    default_cap: float | None = None

    @classmethod
    def from_payload(cls, payload: dict, default: float | None = None) -> "ResidualScopeCaps":
        schema = int(payload.get("schema", 1))
        if schema != 1:
            raise ValueError(f"Unsupported residual scope cap schema '{schema}'.")
        
        payload_default = _normalise_cap(payload.get("default"))
        default_cap = payload_default if payload_default is not None else default
        
        level_caps: Dict[int, float] = {}
        entries = payload.get("levels", {})
        
        # Handle the two schemas (dict vs list of entries)
        items = []
        if isinstance(entries, dict):
            for key, value in entries.items():
                # In dict schema, keys are levels. value can be float or dict
                cap_value = value.get("cap") if isinstance(value, dict) else value
                items.append({"level": int(key), "cap": cap_value})
        elif isinstance(entries, list):
            items = entries
        
        for entry in items:
            cap = _normalise_cap(entry.get("cap"))
            if cap is None:
                continue
            # 'level' can be single int or list of ints
            levels_raw = entry.get("level")
            if levels_raw is None and "level" not in entry and "levels" not in entry:
                 # Fallback if dict schema iteration injected 'level' key above
                 # If coming from raw JSON list, expect 'level' or similar
                 pass
            
            # In dict schema we injected 'level'. In list schema 'level' might be list.
            levels_iter = _coerce_level_list(levels_raw)
            
            for level in levels_iter:
                level_caps[int(level)] = cap
                
        return cls(level_caps=level_caps, default_cap=default_cap)

    @classmethod
    def load(cls, path: str | Path | None, default: float | None = None) -> "ResidualScopeCaps | None":
        if not path and default is None:
            return None
            
        if not path:
            return cls(level_caps={}, default_cap=default)
            
        target = Path(path).expanduser()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            return cls.from_payload(payload, default=default)
        except FileNotFoundError:
            # If only default provided and file missing, valid if default exists
            if default is not None:
                return cls(level_caps={}, default_cap=default)
            raise

    def get_cap(self, level: int) -> float | None:
        return self.level_caps.get(level, self.default_cap)

    def lookup(self, levels: np.ndarray) -> np.ndarray:
        result = np.full(levels.shape, np.inf, dtype=np.float64)
        unique_levels = np.unique(levels)
        for lvl in unique_levels:
            cap = self.level_caps.get(int(lvl))
            if cap is None:
                cap = self.default_cap
            if cap is not None:
                result[levels == lvl] = cap
        return result


def get_scope_cap_table(path: str | None) -> ResidualScopeCaps | None:
    if not path:
        return None
    
    key = str(Path(path).expanduser().absolute())
    with _CACHE_LOCK:
        if key in _SCOPE_CAP_CACHE:
            return _SCOPE_CAP_CACHE[key]
            
    caps = ResidualScopeCaps.load(path)
    if caps is not None:
        with _CACHE_LOCK:
            _SCOPE_CAP_CACHE[key] = caps
    return caps


def reset_scope_cap_cache() -> None:
    with _CACHE_LOCK:
        _SCOPE_CAP_CACHE.clear()


@dataclass
class ResidualScopeCapRecorder:
    """Tracks observed radii per level to derive scope caps."""
    
    percentile: float = 0.5
    margin: float = 0.05
    samples: Dict[int, List[float]] = field(default_factory=dict)
    
    def record(self, levels: np.ndarray, radii: np.ndarray) -> None:
        # Simple aggregation - for production use a reservoir or similar
        # if memory becomes an issue. For benchmarks, keeping all is fine.
        unique_levels = np.unique(levels)
        for lvl in unique_levels:
            mask = levels == lvl
            subset = radii[mask]
            if subset.size > 0:
                if lvl not in self.samples:
                    self.samples[int(lvl)] = []
                self.samples[int(lvl)].extend(subset.tolist())

    def compute_caps(self) -> ResidualScopeCaps:
        caps = {}
        for lvl, values in self.samples.items():
            if not values:
                continue
            # Compute percentile
            val = float(np.quantile(values, self.percentile))
            # Apply margin
            caps[lvl] = val * (1.0 + self.margin)
            
        # Derive a default? For now just per-level
        return ResidualScopeCaps(level_caps=caps, default_cap=None)
        
    def dump(self, path: str | Path) -> None:
        caps = self.compute_caps()
        payload = {
            "schema": 1,
            "default": caps.default_cap,
            "percentile": self.percentile,
            "margin": self.margin,
            "levels": {
                str(k): {"cap": v} for k, v in caps.level_caps.items()
            }
        }
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "ResidualScopeCaps",
    "ResidualScopeCapRecorder",
    "get_scope_cap_table",
    "reset_scope_cap_cache",
]
