from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable

try:  # pragma: no cover - exercised via configuration
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    jax = None  # type: ignore
    jnp = None  # type: ignore
import numpy as np

from covertreex import config as cx_config
from covertreex.logging import get_logger

ArrayLike = Any


def _device_put_default(value: ArrayLike) -> ArrayLike:
    """Default device_put shim that keeps host arrays untouched."""

    return value


@dataclass(frozen=True)
class TreeBackend:
    """Thin abstraction around the array API used by the tree.

    The initial implementation targets `jax.numpy` (JIT-friendly).  The object
    intentionally mirrors a subset of the NumPy/JAX array API we rely on so we
    can later attach alternative implementations (NumPy, Numba-accelerated, …)
    without changing higher level code.
    """

    name: str
    xp: Any
    asarray: Callable[..., ArrayLike]
    stack: Callable[..., ArrayLike]
    device_put: Callable[[ArrayLike], ArrayLike] = field(default=_device_put_default)
    default_float: Any = field(default=np.float64)
    default_int: Any = field(default=np.int32)

    @classmethod
    def jax(cls, *, precision: str = "float64") -> "TreeBackend":
        """Instantiate the canonical JAX backend."""

        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but `jax` is not available.")
        default_float = {"float32": jnp.float32, "float64": jnp.float64}.get(precision)
        if default_float is None:
            raise ValueError(f"Unsupported precision '{precision}'.")
        return cls(
            name="jax",
            xp=jnp,
            asarray=jnp.asarray,
            stack=jnp.stack,
            device_put=jax.device_put,
            default_float=default_float,
            default_int=jnp.int32,
        )

    @classmethod
    def numpy(cls, *, precision: str = "float64") -> "TreeBackend":
        """Instantiate a NumPy-backed implementation."""

        default_float = {"float32": np.float32, "float64": np.float64}.get(precision)
        if default_float is None:
            raise ValueError(f"Unsupported precision '{precision}'.")
        return cls(
            name="numpy",
            xp=np,
            asarray=np.asarray,
            stack=np.stack,
            device_put=_device_put_default,
            default_float=default_float,
            default_int=np.int32,
        )

    def array(self, value: ArrayLike, *, dtype: Any | None = None) -> ArrayLike:
        """Return a backend array placed on the intended device."""

        arr = self.asarray(value, dtype=dtype or self.default_float)
        return self.device_put(arr)

    def zeros(self, shape: Iterable[int], *, dtype: Any | None = None) -> ArrayLike:
        return self.device_put(self.xp.zeros(shape, dtype=dtype or self.default_float))

    def ones(self, shape: Iterable[int], *, dtype: Any | None = None) -> ArrayLike:
        return self.device_put(self.xp.ones(shape, dtype=dtype or self.default_float))

    def empty(self, shape: Iterable[int], *, dtype: Any | None = None) -> ArrayLike:
        return self.device_put(self.xp.empty(shape, dtype=dtype or self.default_float))

    def to_numpy(self, value: ArrayLike) -> Any:
        """Convert to a host NumPy array for debugging/testing."""

        if jax is not None:
            try:
                return np.asarray(jax.device_get(value))
            except Exception:
                pass
        return np.asarray(value)


def _init_default_backend() -> TreeBackend:
    runtime = cx_config.runtime_config()
    if runtime.backend == "jax":
        return TreeBackend.jax(precision=runtime.precision)
    if runtime.backend == "numpy":
        return TreeBackend.numpy(precision=runtime.precision)
    raise NotImplementedError(f"Backend '{runtime.backend}' is not supported yet.")


DEFAULT_BACKEND = _init_default_backend()
LOGGER = get_logger("core.tree")


def compute_level_offsets(backend: TreeBackend, top_levels: ArrayLike) -> ArrayLike:
    """Return descending level offsets compatible with the compressed layout."""

    xp = backend.xp
    levels = backend.asarray(top_levels, dtype=backend.default_int)
    if levels.size == 0:
        return backend.asarray([0], dtype=backend.default_int)
    levels = xp.where(levels < 0, 0, levels)
    max_level = int(xp.max(levels))
    counts = xp.bincount(levels, minlength=max_level + 1)
    counts_desc = counts[::-1]
    zero = xp.zeros((1,), dtype=backend.default_int)
    cumulative = xp.cumsum(counts_desc, dtype=backend.default_int)
    offsets = xp.concatenate([zero, cumulative], axis=0)
    return backend.asarray(offsets, dtype=backend.default_int)


@dataclass(frozen=True)
class TreeLogStats:
    """Execution counters and metadata maintained alongside the tree."""

    num_batches: int = 0
    num_insertions: int = 0
    num_deletions: int = 0
    num_conflicts_resolved: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "num_batches": self.num_batches,
            "num_insertions": self.num_insertions,
            "num_deletions": self.num_deletions,
            "num_conflicts_resolved": self.num_conflicts_resolved,
        }


@dataclass(frozen=True)
class PCCTree:
    """Immutable representation of a parallel compressed cover tree.

    The structure mirrors the compressed cover tree layout described in the
    Elkin–Kurlin line of work, augmented with bookkeeping needed for parallel
    batch updates (conflict scope caches, flattened child tables, etc.).  Each
    instance is immutable; updates are realised via copy-on-write helpers in the
    persistence module rather than in-place mutation.
    """

    points: ArrayLike
    top_levels: ArrayLike
    parents: ArrayLike
    children: ArrayLike
    level_offsets: ArrayLike
    si_cache: ArrayLike
    next_cache: ArrayLike
    stats: TreeLogStats = field(default_factory=TreeLogStats)
    backend: TreeBackend = field(default=DEFAULT_BACKEND)

    def __post_init__(self) -> None:
        self._validate_shapes()
        LOGGER.debug(
            "PCCTree initialised with %d points, %d levels, backend=%s",
            self.num_points,
            self.num_levels,
            self.backend.name,
        )

    def _validate_shapes(self) -> None:
        """Lightweight shape sanity checks; full invariants live in tests."""

        n_points = self.num_points
        if any(
            arr is None
            for arr in (self.points, self.top_levels, self.parents, self.children)
        ):
            raise ValueError("Tree arrays must not be None.")

        if self.points.shape[0] != n_points:
            raise ValueError("First dimension of `points` must match number of nodes.")

        expected_1d = (
            ("top_levels", self.top_levels),
            ("parents", self.parents),
            ("si_cache", self.si_cache),
            ("next_cache", self.next_cache),
        )
        for name, arr in expected_1d:
            if arr.shape[0] != n_points:
                raise ValueError(f"{name} must have length equal to number of nodes.")

        if self.level_offsets.ndim != 1:
            raise ValueError("level_offsets must be 1-D.")

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0]) if self.points.size else 0

    @property
    def num_levels(self) -> int:
        return int(self.level_offsets.shape[0])

    @property
    def dimension(self) -> int:
        if self.points.ndim < 2:
            return 0
        return int(self.points.shape[1])

    def materialise(self) -> Dict[str, Any]:
        """Return a backend-neutral dictionary snapshot."""

        xp = self.backend.xp
        return {
            "points": xp.asarray(self.points),
            "top_levels": xp.asarray(self.top_levels),
            "parents": xp.asarray(self.parents),
            "children": xp.asarray(self.children),
            "level_offsets": xp.asarray(self.level_offsets),
            "si_cache": xp.asarray(self.si_cache),
            "next_cache": xp.asarray(self.next_cache),
            "stats": self.stats.as_dict(),
            "backend": self.backend.name,
        }

    def replace(
        self,
        *,
        points: ArrayLike | None = None,
        top_levels: ArrayLike | None = None,
        parents: ArrayLike | None = None,
        children: ArrayLike | None = None,
        level_offsets: ArrayLike | None = None,
        si_cache: ArrayLike | None = None,
        next_cache: ArrayLike | None = None,
        stats: TreeLogStats | None = None,
        backend: TreeBackend | None = None,
    ) -> "PCCTree":
        """Functional update helper mirroring dataclasses.replace semantics."""

        return PCCTree(
            points=points if points is not None else self.points,
            top_levels=top_levels if top_levels is not None else self.top_levels,
            parents=parents if parents is not None else self.parents,
            children=children if children is not None else self.children,
            level_offsets=level_offsets if level_offsets is not None else self.level_offsets,
            si_cache=si_cache if si_cache is not None else self.si_cache,
            next_cache=next_cache if next_cache is not None else self.next_cache,
            stats=stats if stats is not None else self.stats,
            backend=backend if backend is not None else self.backend,
        )

    def to_backend(self, backend: TreeBackend) -> "PCCTree":
        """Materialise all buffers using `backend`."""

        return self.replace(
            points=backend.asarray(self.points, dtype=backend.default_float),
            top_levels=backend.asarray(self.top_levels, dtype=backend.default_int),
            parents=backend.asarray(self.parents, dtype=backend.default_int),
            children=backend.asarray(self.children, dtype=backend.default_int),
            level_offsets=backend.asarray(self.level_offsets, dtype=backend.default_int),
            si_cache=backend.asarray(self.si_cache, dtype=backend.default_float),
            next_cache=backend.asarray(self.next_cache, dtype=backend.default_int),
            backend=backend,
        )

    @classmethod
    def empty(cls, *, dimension: int, backend: TreeBackend | None = None) -> "PCCTree":
        """Construct an empty tree with the requested dimensionality."""

        backend = backend or DEFAULT_BACKEND
        points = backend.empty((0, dimension), dtype=backend.default_float)
        zeros_1d = backend.empty((0,), dtype=backend.default_int)
        return cls(
            points=points,
            top_levels=zeros_1d,
            parents=zeros_1d,
            children=zeros_1d,
            level_offsets=backend.zeros((1,), dtype=backend.default_int),
            si_cache=backend.empty((0,), dtype=backend.default_float),
            next_cache=zeros_1d,
            stats=TreeLogStats(),
            backend=backend,
        )

    def is_empty(self) -> bool:
        return self.num_points == 0

    # Placeholder for richer invariant checking to be implemented alongside tests.
    def validate(self) -> None:
        """Perform library-level invariant checks (stub)."""

        self._validate_shapes()
