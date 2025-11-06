from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

try:  # pragma: no cover - exercised indirectly via tests
    import jax  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    jax = None  # type: ignore

_FALLBACK_CPU_DEVICE = ("cpu:0",)
_LOGGER = logging.getLogger("covertreex")
_JAX_WARNING_EMITTED = False

_SUPPORTED_BACKENDS = {"jax", "numpy"}
_SUPPORTED_PRECISION = {"float32", "float64"}
_CONFLICT_GRAPH_IMPLS = {"dense", "segmented", "auto"}
_DEFAULT_SCOPE_CHUNK_TARGET = 0
_DEFAULT_NUMBA_THREADING_LAYER = "tbb"


def _ensure_env(var: str, value: str) -> None:
    if os.getenv(var) is None:
        os.environ[var] = value


def _configure_threading_defaults() -> None:
    """Set conservative threading defaults unless the user overrides them."""

    _ensure_env("OMP_NUM_THREADS", "1")
    _ensure_env("OPENBLAS_NUM_THREADS", "1")
    _ensure_env("MKL_NUM_THREADS", "1")
    _ensure_env("NUMEXPR_NUM_THREADS", "1")
    threads = os.getenv("NUMBA_NUM_THREADS")
    if threads is None:
        count = os.cpu_count() or 1
        os.environ["NUMBA_NUM_THREADS"] = str(count)
    if os.getenv("NUMBA_THREADING_LAYER") is None:
        os.environ["NUMBA_THREADING_LAYER"] = _DEFAULT_NUMBA_THREADING_LAYER


_configure_threading_defaults()


def _bool_from_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_devices(raw: str | None) -> Tuple[str, ...]:
    if not raw:
        return ()
    devices = tuple(
        spec.strip().lower()
        for spec in raw.split(",")
        if spec.strip()
    )
    return devices


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value '{raw}'") from exc


def _normalise_precision(value: str | None) -> str:
    if value is None:
        return "float64"
    value = value.strip().lower()
    if value not in _SUPPORTED_PRECISION:
        raise ValueError(f"Unsupported precision '{value}'. Expected one of {_SUPPORTED_PRECISION}.")
    return value


def _parse_conflict_graph_impl(value: str | None) -> str:
    if value is None:
        return "dense"
    impl = value.strip().lower()
    if impl not in _CONFLICT_GRAPH_IMPLS:
        raise ValueError(
            f"Unsupported conflict-graph implementation '{impl}'. Expected one of {_CONFLICT_GRAPH_IMPLS}."
        )
    return impl


def _infer_precision_from_env() -> str:
    precision = os.getenv("COVERTREEX_PRECISION")
    if precision:
        return _normalise_precision(precision)
    jax_enable_x64 = os.getenv("JAX_ENABLE_X64")
    if jax_enable_x64 is not None:
        return "float64" if _bool_from_env(jax_enable_x64, default=False) else "float32"
    return "float64"


def _infer_backend_from_env() -> str:
    backend = os.getenv("COVERTREEX_BACKEND", "numpy").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {_SUPPORTED_BACKENDS}.")
    return backend


def _device_label(device: Any) -> str:
    index = getattr(device, "id", getattr(device, "device_id", 0))
    return f"{device.platform}:{index}"


def _resolve_jax_devices(requested: Tuple[str, ...]) -> Tuple[str, ...]:
    if jax is None:
        if requested and requested:
            _LOGGER.info(
                "JAX is unavailable; forcing CPU stub device while GPU support is disabled."
            )
        return _FALLBACK_CPU_DEVICE

    available = jax.devices()
    if not available:
        return ()
    cpu_devices = [device for device in available if device.platform == "cpu"]
    if cpu_devices:
        if requested and any(not spec.startswith("cpu") for spec in requested):
            _LOGGER.info(
                "GPU execution is disabled; forcing CPU devices despite request %s.",
                requested,
            )
        return tuple(_device_label(device) for device in cpu_devices)

    _LOGGER.warning(
        "GPU execution disabled but no CPU devices reported by JAX; using fallback stub."
    )
    return _FALLBACK_CPU_DEVICE


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    precision: str
    devices: Tuple[str, ...]
    enable_numba: bool
    enable_sparse_traversal: bool
    enable_diagnostics: bool
    log_level: str
    mis_seed: int | None
    conflict_graph_impl: str
    scope_segment_dedupe: bool
    scope_chunk_target: int
    metric: str

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
        backend = _infer_backend_from_env()
        precision = _normalise_precision(_infer_precision_from_env())
        requested_devices = _parse_devices(os.getenv("COVERTREEX_DEVICE"))
        devices = _resolve_jax_devices(requested_devices) if backend == "jax" else ()
        enable_numba = _bool_from_env(os.getenv("COVERTREEX_ENABLE_NUMBA"), default=False)
        enable_sparse_traversal = _bool_from_env(
            os.getenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL"), default=False
        )
        enable_diagnostics = _bool_from_env(
            os.getenv("COVERTREEX_ENABLE_DIAGNOSTICS"), default=True
        )
        log_level = os.getenv("COVERTREEX_LOG_LEVEL", "INFO").upper()
        mis_seed = _parse_optional_int(os.getenv("COVERTREEX_MIS_SEED"))
        conflict_graph_impl = _parse_conflict_graph_impl(
            os.getenv("COVERTREEX_CONFLICT_GRAPH_IMPL")
        )
        scope_segment_dedupe = _bool_from_env(
            os.getenv("COVERTREEX_SCOPE_SEGMENT_DEDUP"), default=True
        )
        raw_chunk_target = _parse_optional_int(
            os.getenv("COVERTREEX_SCOPE_CHUNK_TARGET")
        )
        if raw_chunk_target is None:
            scope_chunk_target = _DEFAULT_SCOPE_CHUNK_TARGET
        elif raw_chunk_target <= 0:
            scope_chunk_target = 0
        else:
            scope_chunk_target = raw_chunk_target
        metric = os.getenv("COVERTREEX_METRIC", "euclidean").strip().lower() or "euclidean"
        return cls(
            backend=backend,
            precision=precision,
            devices=devices,
            enable_numba=enable_numba,
            enable_sparse_traversal=enable_sparse_traversal,
            enable_diagnostics=enable_diagnostics,
            log_level=log_level,
            mis_seed=mis_seed,
            conflict_graph_impl=conflict_graph_impl,
            scope_segment_dedupe=scope_segment_dedupe,
            scope_chunk_target=scope_chunk_target,
            metric=metric,
        )


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


@lru_cache(maxsize=None)
def runtime_config() -> RuntimeConfig:
    config = RuntimeConfig.from_env()
    _apply_jax_runtime_flags(config)
    _configure_logging(config.log_level)
    return config


def reset_runtime_config_cache() -> None:
    runtime_config.cache_clear()


def describe_runtime() -> Dict[str, Any]:
    """Return a serialisable view of the active runtime configuration."""

    config = runtime_config()
    return {
        "backend": config.backend,
        "precision": config.precision,
        "devices": config.devices,
        "primary_platform": config.primary_platform,
        "enable_numba": config.enable_numba,
        "enable_sparse_traversal": config.enable_sparse_traversal,
        "enable_diagnostics": config.enable_diagnostics,
        "log_level": config.log_level,
        "mis_seed": config.mis_seed,
        "jax_enable_x64": config.jax_enable_x64,
        "conflict_graph_impl": config.conflict_graph_impl,
        "scope_segment_dedupe": config.scope_segment_dedupe,
        "scope_chunk_target": config.scope_chunk_target,
        "metric": config.metric,
    }
