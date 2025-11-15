from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

from covertreex.runtime.model import RuntimeModel


class ProfileError(RuntimeError):
    """Base exception for profile loading issues."""


class ProfileNotFoundError(ProfileError):
    """Raised when a named profile cannot be resolved."""


class ProfileFormatError(ProfileError):
    """Raised when a profile file is malformed."""


@dataclass(frozen=True)
class ProfileMetadata:
    name: str
    description: str
    workload: str | None = None
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileDefinition:
    metadata: ProfileMetadata
    model: RuntimeModel
    path: Path


_PROFILE_ROOT = Path(__file__).resolve().parent
_PROFILE_SUFFIXES = (".yaml", ".yml")


def _normalise_name(name: str) -> str:
    slug = name.strip().lower().replace(" ", "_")
    for suffix in _PROFILE_SUFFIXES:
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug


def _resolve_profile_path(name: str) -> Path:
    slug = _normalise_name(name)
    candidate = _PROFILE_ROOT / f"{slug}.yaml"
    if candidate.exists():
        return candidate
    return candidate


def available_profiles() -> Tuple[str, ...]:
    """Return the list of available profile identifiers."""

    entries = []
    for suffix in _PROFILE_SUFFIXES:
        entries.extend(_PROFILE_ROOT.glob(f"*{suffix}"))
    names = {path.stem for path in entries}
    return tuple(sorted(names))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - depends on malformed files
        raise ProfileFormatError(f"Failed to parse profile {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ProfileFormatError(f"Profile {path} must contain a mapping at the top level.")
    return raw


def _metadata_from_payload(payload: Dict[str, Any], *, default_name: str) -> ProfileMetadata:
    tags: Iterable[str] = payload.get("tags") or ()
    if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
        raise ProfileFormatError("metadata.tags must be a list of strings.")
    normalised_tags = tuple(str(tag) for tag in tags)
    return ProfileMetadata(
        name=str(payload.get("name") or default_name),
        description=str(payload.get("description") or ""),
        workload=payload.get("workload"),
        tags=normalised_tags,
    )


@lru_cache(maxsize=None)
def load_profile_definition(name: str) -> ProfileDefinition:
    """Return the parsed profile definition with metadata."""

    path = _resolve_profile_path(name)
    if not path.exists():
        raise ProfileNotFoundError(f"Profile '{name}' not found under {path.parent}.")
    payload = _load_yaml(path)
    runtime_payload = payload.get("runtime") or {
        key: value for key, value in payload.items() if key != "metadata"
    }
    if not isinstance(runtime_payload, dict):
        raise ProfileFormatError(f"Profile {path} runtime section must be a mapping.")
    metadata_payload = payload.get("metadata") or {}
    metadata = _metadata_from_payload(metadata_payload, default_name=path.stem)
    model = RuntimeModel(**runtime_payload)
    return ProfileDefinition(metadata=metadata, model=model, path=path)


def load_profile(name: str) -> RuntimeModel:
    """Return the immutable runtime model for the given profile."""

    definition = load_profile_definition(name)
    return definition.model
