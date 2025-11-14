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


def get_conflict_arena() -> ConflictArena:
    return _CONFLICT_ARENA


__all__ = ["ConflictArena", "get_conflict_arena"]
