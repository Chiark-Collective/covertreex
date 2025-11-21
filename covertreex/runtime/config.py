from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .model import RuntimeModel, SeedPack

try:  # pragma: no cover - exercised indirectly via tests
    import jax  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    jax = None  # type: ignore

_LOGGER = logging.getLogger("covertreex")
_JAX_WARNING_EMITTED = False


class _RuntimeContextStack(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list["RuntimeContext"] = []


_CONTEXT_STACK = _RuntimeContextStack()


def _stack() -> list["RuntimeContext"]:
    return _CONTEXT_STACK.stack


def _current_context_or_none() -> Optional["RuntimeContext"]:
    stack = _stack()
    return stack[-1] if stack else None


def current_runtime_context() -> Optional["RuntimeContext"]:
    """Return the active runtime context without instantiating env defaults."""

    return _current_context_or_none()


def _replace_context(context: "RuntimeContext") -> None:
    stack = _stack()
    previous = list(stack)
    stack.clear()
    stack.append(context)
    context._parent_stack = previous
    context._stack_managed = False


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    precision: str
    devices: Tuple[str, ...]
    enable_numba: bool
    enable_rust: bool
    enable_sparse_traversal: bool
    enable_diagnostics: bool
    log_level: str
    seeds: SeedPack
    conflict_graph_impl: str
    scope_segment_dedupe: bool
    scope_chunk_target: int
    scope_chunk_max_segments: int
    scope_chunk_pair_merge: bool
    scope_conflict_buffer_reuse: bool
    conflict_degree_cap: int
    scope_budget_schedule: Tuple[int, ...]
    scope_budget_up_thresh: float
    scope_budget_down_thresh: float
    metric: str
    batch_order_strategy: str
    prefix_schedule: str
    prefix_density_low: float
    prefix_density_high: float
    prefix_growth_small: float
    prefix_growth_mid: float
    prefix_growth_large: float
    residual_radius_floor: float
    residual_scope_member_limit: int | None
    residual_stream_tile: int | None
    residual_scope_bitset: bool
    residual_masked_scope_append: bool
    residual_dynamic_query_block: bool
    residual_dense_scope_streamer: bool
    residual_level_cache_batching: bool
    residual_scope_cap_path: str | None
    residual_scope_cap_default: float
    residual_scope_cap_output: str | None
    residual_scope_cap_percentile: float
    residual_scope_cap_margin: float
    residual_prefilter_enabled: bool
    residual_prefilter_lookup_path: str | None
    residual_prefilter_margin: float
    residual_prefilter_radius_cap: float
    residual_prefilter_audit: bool
    residual_grid_whiten_scale: float
    residual_use_static_euclidean_tree: bool

    @property
    def mis_seed(self) -> int | None:
        return self.seeds.mis

    @property
    def batch_order_seed(self) -> int | None:
        return self.seeds.batch_order

    @property
    def jax_enable_x64(self) -> bool:
        return self.precision == "float64"

    @property
    def primary_platform(self) -> str | None:
        if not self.devices:
            return None
        return self.devices[0].split(":", 1)[0]

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return RuntimeModel.from_env().to_runtime_config()


def _apply_jax_runtime_flags(config: RuntimeConfig) -> None:
    if config.backend != "jax":
        return

    if jax is None:
        global _JAX_WARNING_EMITTED
        if not _JAX_WARNING_EMITTED:
            _LOGGER.warning(
                "JAX backend requested but `jax` is not installed; using CPU stub devices."
            )
            _JAX_WARNING_EMITTED = True
        return

    jax.config.update("jax_enable_x64", config.jax_enable_x64)

    jax.config.update("jax_platform_name", "cpu")


def _configure_logging(level: str) -> None:
    logger = logging.getLogger("covertreex")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@dataclass
class RuntimeContext:
    """Aggregate runtime configuration and lazily-resolved backend state."""

    config: RuntimeConfig
    _backend: Any = field(default=None, init=False, repr=False)
    _activated: bool = field(default=False, init=False, repr=False)
    _parent_stack: list["RuntimeContext"] | None = field(default=None, init=False, repr=False)
    _stack_managed: bool = field(default=False, init=False, repr=False)

    def activate(self) -> None:
        """Apply side effects (logging/JAX flags) once."""

        if self._activated:
            return
        _apply_jax_runtime_flags(self.config)
        _configure_logging(self.config.log_level)
        self._activated = True

    def __enter__(self) -> "RuntimeContext":
        self.activate()
        stack = _stack()
        if self._parent_stack is not None and stack and stack[-1] is self:
            self._stack_managed = True
            return self
        snapshot = list(stack)
        stack.append(self)
        self._parent_stack = snapshot
        self._stack_managed = True
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if not self._stack_managed:
            return
        stack = _stack()
        if not stack or stack[-1] is not self:
            raise RuntimeError("Runtime context stack imbalance")
        stack.pop()
        previous = self._parent_stack
        self._parent_stack = None
        self._stack_managed = False
        if previous is not None:
            stack.clear()
            stack.extend(previous)

    def get_backend(self) -> "TreeBackend":
        """Return the active backend, instantiating it lazily."""

        if self._backend is None:
            from covertreex.core.tree import TreeBackend  # lazy import to avoid cycles

            if self.config.backend == "jax":
                backend = TreeBackend.jax(precision=self.config.precision)
            elif self.config.backend == "numpy":
                backend = TreeBackend.numpy(precision=self.config.precision)
            elif self.config.backend == "gpu":
                backend = TreeBackend.gpu(precision=self.config.precision)
            else:
                raise NotImplementedError(
                    f"Backend '{self.config.backend}' is not supported yet."
                )
            self._backend = backend
        return self._backend


def runtime_context() -> RuntimeContext:
    """Return the active runtime context, constructing it if necessary."""

    context = _current_context_or_none()
    if context is None:
        config = RuntimeConfig.from_env()
        context = RuntimeContext(config=config)
        context.activate()
        _stack().append(context)
    return context


def runtime_config() -> RuntimeConfig:
    """Backwards-compatible accessor for the active runtime configuration."""

    return runtime_context().config


def configure_runtime(config: RuntimeConfig) -> RuntimeContext:
    """Force the active runtime context to use ``config`` instead of env defaults."""

    context = RuntimeContext(config=config)
    context.activate()
    _replace_context(context)
    return context


def set_runtime_context(context: RuntimeContext) -> RuntimeContext:
    """Install ``context`` as the active runtime context after activating it."""

    context.activate()
    _replace_context(context)
    return context


def reset_runtime_config_cache() -> None:
    reset_runtime_context()


def reset_runtime_context() -> None:
    """Clear the cached runtime context (used in tests)."""

    _stack().clear()


def describe_runtime() -> Dict[str, Any]:
    """Return a serialisable view of the active runtime configuration."""

    config = runtime_config()
    return {
        "backend": config.backend,
        "precision": config.precision,
        "devices": config.devices,
        "primary_platform": config.primary_platform,
        "enable_numba": config.enable_numba,
        "enable_rust": config.enable_rust,
        "enable_sparse_traversal": config.enable_sparse_traversal,
        "enable_diagnostics": config.enable_diagnostics,
        "log_level": config.log_level,
        "mis_seed": config.mis_seed,
        "seed_pack": config.seeds.model_dump(),
        "jax_enable_x64": config.jax_enable_x64,
        "conflict_graph_impl": config.conflict_graph_impl,
        "scope_segment_dedupe": config.scope_segment_dedupe,
        "scope_chunk_target": config.scope_chunk_target,
        "scope_chunk_max_segments": config.scope_chunk_max_segments,
        "conflict_degree_cap": config.conflict_degree_cap,
        "scope_budget_schedule": config.scope_budget_schedule,
        "scope_budget_up_thresh": config.scope_budget_up_thresh,
        "scope_budget_down_thresh": config.scope_budget_down_thresh,
        "metric": config.metric,
        "batch_order_strategy": config.batch_order_strategy,
        "batch_order_seed": config.batch_order_seed,
        "prefix_schedule": config.prefix_schedule,
        "prefix_density_low": config.prefix_density_low,
        "prefix_density_high": config.prefix_density_high,
        "prefix_growth_small": config.prefix_growth_small,
        "prefix_growth_mid": config.prefix_growth_mid,
        "prefix_growth_large": config.prefix_growth_large,
        "residual_radius_floor": config.residual_radius_floor,
        "residual_scope_bitset": config.residual_scope_bitset,
        "residual_dynamic_query_block": config.residual_dynamic_query_block,
        "residual_level_cache_batching": config.residual_level_cache_batching,
        "residual_scope_member_limit": config.residual_scope_member_limit,
        "residual_scope_cap_path": config.residual_scope_cap_path,
        "residual_scope_cap_default": config.residual_scope_cap_default,
        "residual_scope_cap_output": config.residual_scope_cap_output,
        "residual_scope_cap_percentile": config.residual_scope_cap_percentile,
        "residual_scope_cap_margin": config.residual_scope_cap_margin,
        "residual_grid_whiten_scale": config.residual_grid_whiten_scale,
        "residual_use_static_euclidean_tree": config.residual_use_static_euclidean_tree,
    }


__all__ = [
    "RuntimeConfig",
    "RuntimeContext",
    "current_runtime_context",
    "runtime_context",
    "runtime_config",
    "configure_runtime",
    "set_runtime_context",
    "reset_runtime_context",
    "reset_runtime_config_cache",
    "describe_runtime",
]