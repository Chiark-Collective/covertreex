from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - exercised indirectly via tests
    import jax  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    jax = None  # type: ignore

_FALLBACK_CPU_DEVICE = ("cpu:0",)
_LOGGER = logging.getLogger("covertreex")
_JAX_WARNING_EMITTED = False

_SUPPORTED_BACKENDS = {"jax", "numpy", "gpu"}
_SUPPORTED_PRECISION = {"float32", "float64"}
_CONFLICT_GRAPH_IMPLS = {"dense", "segmented", "auto", "grid"}
_BATCH_ORDER_STRATEGIES = {"natural", "random", "hilbert"}
_PREFIX_SCHEDULES = {"doubling", "adaptive"}
_DEFAULT_SCOPE_CHUNK_TARGET = 0
_DEFAULT_SCOPE_CHUNK_MAX_SEGMENTS = 512
_DEFAULT_SCOPE_CHUNK_PAIR_MERGE = True
_DEFAULT_SCOPE_CONFLICT_BUFFER_REUSE = True
_DEFAULT_CONFLICT_DEGREE_CAP = 0
_DEFAULT_SCOPE_BUDGET_SCHEDULE: Tuple[int, ...] = ()
_DEFAULT_RESIDUAL_SCOPE_BUDGET_SCHEDULE: Tuple[int, ...] = (32, 64, 96)
_DEFAULT_RESIDUAL_STREAM_TILE = 64
_DEFAULT_RESIDUAL_DENSE_SCOPE_STREAMER = True
_DEFAULT_SCOPE_BUDGET_UP_THRESH = 0.015
_DEFAULT_SCOPE_BUDGET_DOWN_THRESH = 0.002
_DEFAULT_BATCH_ORDER_STRATEGY = "hilbert"
_DEFAULT_PREFIX_SCHEDULE = "adaptive"
_DEFAULT_PREFIX_DENSITY_LOW = 0.15
_DEFAULT_PREFIX_DENSITY_HIGH = 0.55
_DEFAULT_PREFIX_GROWTH_SMALL = 1.25
_DEFAULT_PREFIX_GROWTH_MID = 1.75
_DEFAULT_PREFIX_GROWTH_LARGE = 2.25
_DEFAULT_RESIDUAL_GATE1_ALPHA = 4.0
_DEFAULT_RESIDUAL_GATE1_MARGIN = 0.05
_DEFAULT_RESIDUAL_GATE1_EPS = 1e-6
_DEFAULT_RESIDUAL_GATE1_RADIUS_CAP = 1.0
_DEFAULT_RESIDUAL_RADIUS_FLOOR = 1e-3
_DEFAULT_RESIDUAL_GATE1_PROFILE_BINS = 256
_DEFAULT_RESIDUAL_GATE1_LOOKUP_MARGIN = 0.02
_DEFAULT_RESIDUAL_GATE1_BAND_EPS = 0.02
_DEFAULT_RESIDUAL_GATE1_KEEP_PCT = 95.0
_DEFAULT_RESIDUAL_GATE1_PRUNE_PCT = 99.9
_DEFAULT_RESIDUAL_SCOPE_CAP_DEFAULT = 0.0
_DEFAULT_RESIDUAL_PREFILTER_MARGIN = 0.02
_DEFAULT_RESIDUAL_PREFILTER_RADIUS_CAP = 10.0
_DEFAULT_RESIDUAL_PREFILTER_LOOKUP = str(
    (Path(__file__).resolve().parents[2] / "docs" / "data" / "residual_gate_profile_32768_caps.json")
)
_DEFAULT_RESIDUAL_PREFILTER_AUDIT = False
_DEFAULT_RESIDUAL_GRID_WHITEN_SCALE = 1.0
_DEFAULT_RESIDUAL_FORCE_WHITENED = False


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


def _parse_batch_order_strategy(value: str | None) -> str:
    if value is None:
        return _DEFAULT_BATCH_ORDER_STRATEGY
    strategy = value.strip().lower()
    if strategy not in _BATCH_ORDER_STRATEGIES:
        raise ValueError(
            f"Unsupported batch order strategy '{strategy}'. Expected one of {_BATCH_ORDER_STRATEGIES}."
        )
    return strategy


def _parse_prefix_schedule(
    value: str | None,
    *,
    default: str = _DEFAULT_PREFIX_SCHEDULE,
) -> str:
    if value is None:
        return default
    schedule = value.strip().lower()
    if schedule not in _PREFIX_SCHEDULES:
        raise ValueError(
            f"Unsupported prefix schedule '{schedule}'. Expected one of {_PREFIX_SCHEDULES}."
        )
    return schedule


def _parse_optional_float(raw: str | None, *, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float value '{raw}'") from exc


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


def _parse_scope_budget_schedule(raw: str | None) -> Tuple[int, ...]:
    if raw is None:
        return _DEFAULT_SCOPE_BUDGET_SCHEDULE
    tokens = [segment.strip() for segment in raw.split(",")]
    values: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer value '{token}' in COVERTREEX_SCOPE_BUDGET_SCHEDULE"
            ) from exc
        if value <= 0:
            raise ValueError(
                "COVERTREEX_SCOPE_BUDGET_SCHEDULE entries must be positive integers"
            )
        if values and value <= values[-1]:
            raise ValueError(
                "COVERTREEX_SCOPE_BUDGET_SCHEDULE entries must be strictly increasing"
            )
        values.append(value)
    return tuple(values)


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
    scope_chunk_max_segments: int
    scope_chunk_pair_merge: bool
    scope_conflict_buffer_reuse: bool
    conflict_degree_cap: int
    scope_budget_schedule: Tuple[int, ...]
    scope_budget_up_thresh: float
    scope_budget_down_thresh: float
    metric: str
    batch_order_strategy: str
    batch_order_seed: int | None
    prefix_schedule: str
    prefix_density_low: float
    prefix_density_high: float
    prefix_growth_small: float
    prefix_growth_mid: float
    prefix_growth_large: float
    residual_gate1_enabled: bool
    residual_gate1_alpha: float
    residual_gate1_margin: float
    residual_gate1_eps: float
    residual_gate1_audit: bool
    residual_gate1_radius_cap: float
    residual_gate1_band_eps: float
    residual_gate1_keep_pct: float
    residual_gate1_prune_pct: float
    residual_radius_floor: float
    residual_gate1_profile_path: str | None
    residual_gate1_profile_bins: int
    residual_gate1_lookup_path: str | None
    residual_gate1_lookup_margin: float
    residual_force_whitened: bool
    residual_scope_member_limit: int | None
    residual_stream_tile: int | None
    residual_scope_bitset: bool
    residual_masked_scope_append: bool
    residual_dynamic_query_block: bool
    residual_dense_scope_streamer: bool
    residual_level_cache_batching: bool
    residual_scope_cap_path: str | None
    residual_scope_cap_default: float
    residual_prefilter_enabled: bool
    residual_prefilter_lookup_path: str | None
    residual_prefilter_margin: float
    residual_prefilter_radius_cap: float
    residual_prefilter_audit: bool
    residual_grid_whiten_scale: float

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
        raw_chunk_segments = _parse_optional_int(
            os.getenv("COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS")
        )
        if raw_chunk_segments is None:
            scope_chunk_max_segments = _DEFAULT_SCOPE_CHUNK_MAX_SEGMENTS
        elif raw_chunk_segments <= 0:
            scope_chunk_max_segments = 0
        else:
            scope_chunk_max_segments = raw_chunk_segments
        scope_chunk_pair_merge = _bool_from_env(
            os.getenv("COVERTREEX_SCOPE_CHUNK_PAIR_MERGE"),
            default=_DEFAULT_SCOPE_CHUNK_PAIR_MERGE,
        )
        scope_conflict_buffer_reuse = _bool_from_env(
            os.getenv("COVERTREEX_SCOPE_CONFLICT_BUFFER_REUSE"),
            default=_DEFAULT_SCOPE_CONFLICT_BUFFER_REUSE,
        )
        raw_degree_cap = _parse_optional_int(os.getenv("COVERTREEX_DEGREE_CAP"))
        if raw_degree_cap is None or raw_degree_cap <= 0:
            conflict_degree_cap = _DEFAULT_CONFLICT_DEGREE_CAP
        else:
            conflict_degree_cap = raw_degree_cap
        scope_budget_schedule = _parse_scope_budget_schedule(
            os.getenv("COVERTREEX_SCOPE_BUDGET_SCHEDULE")
        )
        scope_budget_up_thresh = _parse_optional_float(
            os.getenv("COVERTREEX_SCOPE_BUDGET_UP_THRESH"),
            default=_DEFAULT_SCOPE_BUDGET_UP_THRESH,
        )
        scope_budget_down_thresh = _parse_optional_float(
            os.getenv("COVERTREEX_SCOPE_BUDGET_DOWN_THRESH"),
            default=_DEFAULT_SCOPE_BUDGET_DOWN_THRESH,
        )
        if scope_budget_schedule and scope_budget_down_thresh >= scope_budget_up_thresh:
            raise ValueError(
                "COVERTREEX_SCOPE_BUDGET_DOWN_THRESH must be smaller than "
                "COVERTREEX_SCOPE_BUDGET_UP_THRESH"
            )
        metric = os.getenv("COVERTREEX_METRIC", "euclidean").strip().lower() or "euclidean"
        residual_metric = metric == "residual_correlation"
        if not scope_budget_schedule and residual_metric:
            scope_budget_schedule = _DEFAULT_RESIDUAL_SCOPE_BUDGET_SCHEDULE
        batch_order_strategy = _parse_batch_order_strategy(
            os.getenv("COVERTREEX_BATCH_ORDER")
        )
        batch_order_seed = _parse_optional_int(
            os.getenv("COVERTREEX_BATCH_ORDER_SEED")
        )
        prefix_schedule = _parse_prefix_schedule(
            os.getenv("COVERTREEX_PREFIX_SCHEDULE"),
            default="doubling" if residual_metric else _DEFAULT_PREFIX_SCHEDULE,
        )
        prefix_density_low = _parse_optional_float(
            os.getenv("COVERTREEX_PREFIX_DENSITY_LOW"),
            default=_DEFAULT_PREFIX_DENSITY_LOW,
        )
        prefix_density_high = _parse_optional_float(
            os.getenv("COVERTREEX_PREFIX_DENSITY_HIGH"),
            default=_DEFAULT_PREFIX_DENSITY_HIGH,
        )
        prefix_growth_small = _parse_optional_float(
            os.getenv("COVERTREEX_PREFIX_GROWTH_SMALL"),
            default=_DEFAULT_PREFIX_GROWTH_SMALL,
        )
        prefix_growth_mid = _parse_optional_float(
            os.getenv("COVERTREEX_PREFIX_GROWTH_MID"),
            default=_DEFAULT_PREFIX_GROWTH_MID,
        )
        prefix_growth_large = _parse_optional_float(
            os.getenv("COVERTREEX_PREFIX_GROWTH_LARGE"),
            default=_DEFAULT_PREFIX_GROWTH_LARGE,
        )
        residual_gate1_enabled = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_GATE1"),
            default=True if residual_metric else False,
        )
        residual_gate1_alpha = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_ALPHA"),
            default=_DEFAULT_RESIDUAL_GATE1_ALPHA,
        )
        residual_gate1_margin = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_MARGIN"),
            default=_DEFAULT_RESIDUAL_GATE1_MARGIN,
        )
        residual_gate1_eps = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_EPS"),
            default=_DEFAULT_RESIDUAL_GATE1_EPS,
        )
        residual_gate1_audit = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_AUDIT"),
            default=False,
        )
        residual_gate1_radius_cap = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP"),
            default=_DEFAULT_RESIDUAL_GATE1_RADIUS_CAP,
        )
        residual_gate1_band_eps = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_BAND_EPS"),
            default=_DEFAULT_RESIDUAL_GATE1_BAND_EPS,
        )
        if residual_gate1_band_eps < 0.0:
            residual_gate1_band_eps = _DEFAULT_RESIDUAL_GATE1_BAND_EPS
        residual_gate1_keep_pct = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_KEEP_PCT"),
            default=_DEFAULT_RESIDUAL_GATE1_KEEP_PCT,
        )
        residual_gate1_prune_pct = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_PRUNE_PCT"),
            default=_DEFAULT_RESIDUAL_GATE1_PRUNE_PCT,
        )
        residual_gate1_keep_pct = float(min(max(residual_gate1_keep_pct, 0.0), 100.0))
        residual_gate1_prune_pct = float(min(max(residual_gate1_prune_pct, 0.0), 100.0))
        if residual_gate1_prune_pct < residual_gate1_keep_pct:
            residual_gate1_prune_pct = residual_gate1_keep_pct
        residual_radius_floor = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_RADIUS_FLOOR"),
            default=_DEFAULT_RESIDUAL_RADIUS_FLOOR,
        )
        residual_gate1_profile_path = os.getenv("COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH")
        raw_profile_bins = _parse_optional_int(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_PROFILE_BINS")
        )
        if raw_profile_bins is None or raw_profile_bins <= 0:
            residual_gate1_profile_bins = _DEFAULT_RESIDUAL_GATE1_PROFILE_BINS
        else:
            residual_gate1_profile_bins = raw_profile_bins
        residual_gate1_lookup_path = os.getenv("COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH")
        residual_gate1_lookup_margin = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN"),
            default=_DEFAULT_RESIDUAL_GATE1_LOOKUP_MARGIN,
        )
        residual_force_whitened = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_FORCE_WHITENED"),
            default=_DEFAULT_RESIDUAL_FORCE_WHITENED,
        )
        residual_scope_bitset = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_SCOPE_BITSET"),
            default=residual_metric,
        )
        residual_masked_scope_append = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_MASKED_SCOPE_APPEND"),
            default=True,
        )
        residual_dynamic_query_block = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_DYNAMIC_QUERY_BLOCK"),
            default=True,
        )
        residual_dense_scope_streamer = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_DENSE_SCOPE_STREAMER"),
            default=_DEFAULT_RESIDUAL_DENSE_SCOPE_STREAMER,
        )
        residual_level_cache_batching = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_LEVEL_CACHE_BATCHING"),
            default=True,
        )
        raw_scope_member_limit = _parse_optional_int(
            os.getenv("COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT")
        )
        if raw_scope_member_limit is None:
            residual_scope_member_limit = None
        elif raw_scope_member_limit <= 0:
            residual_scope_member_limit = 0
        else:
            residual_scope_member_limit = raw_scope_member_limit
        raw_stream_tile = _parse_optional_int(os.getenv("COVERTREEX_RESIDUAL_STREAM_TILE"))
        if raw_stream_tile is None or raw_stream_tile <= 0:
            residual_stream_tile = _DEFAULT_RESIDUAL_STREAM_TILE if residual_metric else None
        else:
            residual_stream_tile = raw_stream_tile
        residual_scope_cap_path = os.getenv("COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH")
        residual_scope_cap_default = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT"),
            default=_DEFAULT_RESIDUAL_SCOPE_CAP_DEFAULT,
        )
        residual_prefilter_enabled = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_PREFILTER"),
            default=False,
        )
        residual_prefilter_lookup_path = os.getenv("COVERTREEX_RESIDUAL_PREFILTER_LOOKUP_PATH")
        residual_prefilter_margin = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_PREFILTER_MARGIN"),
            default=_DEFAULT_RESIDUAL_PREFILTER_MARGIN,
        )
        residual_prefilter_radius_cap = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_PREFILTER_RADIUS_CAP"),
            default=_DEFAULT_RESIDUAL_PREFILTER_RADIUS_CAP,
        )
        residual_prefilter_audit = _bool_from_env(
            os.getenv("COVERTREEX_RESIDUAL_PREFILTER_AUDIT"),
            default=_DEFAULT_RESIDUAL_PREFILTER_AUDIT,
        )
        residual_grid_whiten_scale = _parse_optional_float(
            os.getenv("COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE"),
            default=_DEFAULT_RESIDUAL_GRID_WHITEN_SCALE,
        )
        if residual_grid_whiten_scale <= 0.0:
            residual_grid_whiten_scale = _DEFAULT_RESIDUAL_GRID_WHITEN_SCALE
        if residual_prefilter_enabled:
            residual_gate1_enabled = True
            if residual_gate1_lookup_path is None:
                residual_gate1_lookup_path = (
                    residual_prefilter_lookup_path or _DEFAULT_RESIDUAL_PREFILTER_LOOKUP
                )
            if os.getenv("COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN") is None:
                residual_gate1_lookup_margin = residual_prefilter_margin
            if os.getenv("COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP") is None:
                residual_gate1_radius_cap = residual_prefilter_radius_cap
            if os.getenv("COVERTREEX_RESIDUAL_GATE1_AUDIT") is None:
                residual_gate1_audit = residual_prefilter_audit
        if residual_gate1_lookup_path is None and residual_metric:
            residual_gate1_lookup_path = _DEFAULT_RESIDUAL_PREFILTER_LOOKUP
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
            scope_chunk_max_segments=scope_chunk_max_segments,
            scope_chunk_pair_merge=scope_chunk_pair_merge,
            scope_conflict_buffer_reuse=scope_conflict_buffer_reuse,
            conflict_degree_cap=conflict_degree_cap,
            scope_budget_schedule=scope_budget_schedule,
            scope_budget_up_thresh=scope_budget_up_thresh,
            scope_budget_down_thresh=scope_budget_down_thresh,
            metric=metric,
            batch_order_strategy=batch_order_strategy,
            batch_order_seed=batch_order_seed,
            prefix_schedule=prefix_schedule,
            prefix_density_low=prefix_density_low,
            prefix_density_high=prefix_density_high,
            prefix_growth_small=prefix_growth_small,
            prefix_growth_mid=prefix_growth_mid,
            prefix_growth_large=prefix_growth_large,
            residual_gate1_enabled=residual_gate1_enabled,
            residual_gate1_alpha=residual_gate1_alpha,
            residual_gate1_margin=residual_gate1_margin,
            residual_gate1_eps=residual_gate1_eps,
            residual_gate1_audit=residual_gate1_audit,
            residual_gate1_radius_cap=residual_gate1_radius_cap,
            residual_gate1_band_eps=residual_gate1_band_eps,
            residual_gate1_keep_pct=residual_gate1_keep_pct,
            residual_gate1_prune_pct=residual_gate1_prune_pct,
            residual_radius_floor=residual_radius_floor,
            residual_gate1_profile_path=residual_gate1_profile_path,
            residual_gate1_profile_bins=residual_gate1_profile_bins,
            residual_gate1_lookup_path=residual_gate1_lookup_path,
            residual_gate1_lookup_margin=residual_gate1_lookup_margin,
            residual_force_whitened=residual_force_whitened,
            residual_scope_member_limit=residual_scope_member_limit,
            residual_stream_tile=residual_stream_tile,
            residual_scope_bitset=residual_scope_bitset,
            residual_masked_scope_append=residual_masked_scope_append,
            residual_dynamic_query_block=residual_dynamic_query_block,
            residual_dense_scope_streamer=residual_dense_scope_streamer,
            residual_level_cache_batching=residual_level_cache_batching,
            residual_scope_cap_path=residual_scope_cap_path,
            residual_scope_cap_default=residual_scope_cap_default,
            residual_prefilter_enabled=residual_prefilter_enabled,
            residual_prefilter_lookup_path=residual_prefilter_lookup_path,
            residual_prefilter_margin=residual_prefilter_margin,
            residual_prefilter_radius_cap=residual_prefilter_radius_cap,
            residual_prefilter_audit=residual_prefilter_audit,
            residual_grid_whiten_scale=residual_grid_whiten_scale,
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


@dataclass
class RuntimeContext:
    """Aggregate runtime configuration and lazily-resolved backend state."""

    config: RuntimeConfig
    _backend: Any = field(default=None, init=False, repr=False)
    _activated: bool = field(default=False, init=False, repr=False)

    def activate(self) -> None:
        """Apply side effects (logging/JAX flags) once."""

        if self._activated:
            return
        _apply_jax_runtime_flags(self.config)
        _configure_logging(self.config.log_level)
        self._activated = True

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


_CONTEXT_CACHE: Optional[RuntimeContext] = None


def runtime_context() -> RuntimeContext:
    """Return the cached runtime context, constructing it if necessary."""

    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is None:
        config = RuntimeConfig.from_env()
        context = RuntimeContext(config=config)
        context.activate()
        _CONTEXT_CACHE = context
    return _CONTEXT_CACHE


def runtime_config() -> RuntimeConfig:
    """Backwards-compatible accessor for the active runtime configuration."""

    return runtime_context().config


def configure_runtime(config: RuntimeConfig) -> RuntimeContext:
    """Force the active runtime context to use ``config`` instead of env defaults."""

    context = RuntimeContext(config=config)
    context.activate()
    _set_runtime_context(context)
    return context


def set_runtime_context(context: RuntimeContext) -> RuntimeContext:
    """Install ``context`` as the active runtime context after activating it."""

    context.activate()
    _set_runtime_context(context)
    return context


def _set_runtime_context(context: RuntimeContext) -> None:
    global _CONTEXT_CACHE
    _CONTEXT_CACHE = context


def reset_runtime_config_cache() -> None:
    reset_runtime_context()


def reset_runtime_context() -> None:
    """Clear the cached runtime context (used in tests)."""

    global _CONTEXT_CACHE
    _CONTEXT_CACHE = None


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
        "residual_gate1_enabled": config.residual_gate1_enabled,
        "residual_gate1_alpha": config.residual_gate1_alpha,
        "residual_gate1_margin": config.residual_gate1_margin,
        "residual_gate1_eps": config.residual_gate1_eps,
        "residual_gate1_audit": config.residual_gate1_audit,
        "residual_gate1_radius_cap": config.residual_gate1_radius_cap,
        "residual_gate1_band_eps": config.residual_gate1_band_eps,
        "residual_gate1_keep_pct": config.residual_gate1_keep_pct,
        "residual_gate1_prune_pct": config.residual_gate1_prune_pct,
        "residual_radius_floor": config.residual_radius_floor,
        "residual_gate1_profile_path": config.residual_gate1_profile_path,
        "residual_gate1_profile_bins": config.residual_gate1_profile_bins,
        "residual_gate1_lookup_path": config.residual_gate1_lookup_path,
        "residual_gate1_lookup_margin": config.residual_gate1_lookup_margin,
        "residual_force_whitened": config.residual_force_whitened,
        "residual_scope_bitset": config.residual_scope_bitset,
        "residual_dynamic_query_block": config.residual_dynamic_query_block,
        "residual_level_cache_batching": config.residual_level_cache_batching,
        "residual_scope_member_limit": config.residual_scope_member_limit,
        "residual_scope_cap_path": config.residual_scope_cap_path,
        "residual_scope_cap_default": config.residual_scope_cap_default,
        "residual_grid_whiten_scale": config.residual_grid_whiten_scale,
    }


__all__ = [
    "RuntimeConfig",
    "RuntimeContext",
    "runtime_context",
    "runtime_config",
    "configure_runtime",
    "set_runtime_context",
    "reset_runtime_context",
    "reset_runtime_config_cache",
    "describe_runtime",
]
