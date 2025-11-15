from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _ArenaBuffer:
    data: np.ndarray

    def ensure(self, size: int) -> np.ndarray:
        if size <= 0:
            return self.data[:0]
        if self.data.size < size:
            self.data = np.empty(size, dtype=self.data.dtype)
        return self.data[:size]

    @property
    def capacity_bytes(self) -> int:
        return int(self.data.nbytes)


@dataclass
class ConflictArena:
    """Scratch space for host-side adjacency buffers."""

    sources: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.empty(0, dtype=np.int64)))
    targets: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.empty(0, dtype=np.int64)))

    def borrow_sources(self, size: int) -> np.ndarray:
        return self.sources.ensure(size)

    def borrow_targets(self, size: int) -> np.ndarray:
        return self.targets.ensure(size)

    @property
    def total_bytes(self) -> int:
        return self.sources.capacity_bytes + self.targets.capacity_bytes


_CONFLICT_ARENA = ConflictArena()


@dataclass
class ScopeBuilderArena:
    """Scratch buffers reused by the Numba scope builder."""

    counts: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.zeros(0, dtype=np.int64)))
    degree: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.zeros(0, dtype=np.int64)))
    sources: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.empty(0, dtype=np.int32)))
    targets: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.empty(0, dtype=np.int32)))
    indices: _ArenaBuffer = field(default_factory=lambda: _ArenaBuffer(np.empty(0, dtype=np.int32)))

    def borrow_counts(self, size: int) -> np.ndarray:
        buf = self.counts.ensure(size)
        if size > 0:
            buf[:size] = 0
        return buf[:size]

    def borrow_degree_usage(self, size: int) -> np.ndarray:
        buf = self.degree.ensure(size)
        if size > 0:
            buf[:size] = 0
        return buf[:size]

    def borrow_sources(self, size: int) -> np.ndarray:
        return self.sources.ensure(size)

    def borrow_targets(self, size: int) -> np.ndarray:
        return self.targets.ensure(size)

    def borrow_indices(self, size: int) -> np.ndarray:
        return self.indices.ensure(size)

    @property
    def total_bytes(self) -> int:
        return (
            self.counts.capacity_bytes
            + self.degree.capacity_bytes
            + self.sources.capacity_bytes
            + self.targets.capacity_bytes
            + self.indices.capacity_bytes
        )


_SCOPE_BUILDER_ARENA = ScopeBuilderArena()


def get_conflict_arena() -> ConflictArena:
    return _CONFLICT_ARENA


def get_scope_builder_arena() -> ScopeBuilderArena:
    return _SCOPE_BUILDER_ARENA


__all__ = [
    "ConflictArena",
    "ScopeBuilderArena",
    "get_conflict_arena",
    "get_scope_builder_arena",
]
