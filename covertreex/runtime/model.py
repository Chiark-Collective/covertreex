"""Typed runtime configuration models and env parsing helpers."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

try:  # pragma: no cover - exercised indirectly in tests
    import jax  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    jax = None  # type: ignore

try:
    import numba  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

try:
    import covertreex_backend
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

_LOGGER = logging.getLogger("covertreex")
_FALLBACK_CPU_DEVICE = ("cpu:0",)

_SUPPORTED_BACKENDS = {"jax", "numpy", "gpu"}
_SUPPORTED_PRECISION = {"float32", "float64"}
_ENGINE_CHOICES = {"python-numba", "rust-fast", "rust-hybrid", "rust-pcct", "rust-pcct2"}
_DEFAULT_ENGINE = "python-numba"
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
_DEFAULT_RESIDUAL_RADIUS_FLOOR = 1e-3
_DEFAULT_RESIDUAL_SCOPE_CAP_DEFAULT = 0.0
_DEFAULT_RESIDUAL_SCOPE_CAP_PERCENTILE = 0.5
_DEFAULT_RESIDUAL_SCOPE_CAP_MARGIN = 0.05
_DEFAULT_RESIDUAL_PREFILTER_MARGIN = 0.02
_DEFAULT_RESIDUAL_PREFILTER_RADIUS_CAP = 10.0
_DEFAULT_RESIDUAL_PREFILTER_LOOKUP = str(
    (Path(__file__).resolve().parents[2] / "docs" / "data" / "residual_gate_profile_32768_caps.json")
)
_DEFAULT_RESIDUAL_PREFILTER_AUDIT = False
_DEFAULT_RESIDUAL_GRID_WHITEN_SCALE = 1.0


def _bool_from_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
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
    except ValueError as exc:  # pragma: no cover - validation path
        raise ValueError(f"Invalid integer value '{raw}'") from exc


def _parse_optional_float(raw: str | None, *, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - validation path
        raise ValueError(f"Invalid float value '{raw}'") from exc


def _parse_scope_budget_schedule(raw: str | None) -> Tuple[int, ...]:
    if raw is None:
        return _DEFAULT_SCOPE_BUDGET_SCHEDULE
    tokens = [segment.strip() for segment in raw.split(",")]
    values: list[int] = []
    for token in tokens:
        if not token:
            continue
        value = int(token)
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


def _normalise_precision(value: str | None) -> str:
    if value is None:
        return "float64"
    precision = value.strip().lower()
    if precision not in _SUPPORTED_PRECISION:
        raise ValueError(
            f"Unsupported precision '{precision}'. Expected one of {_SUPPORTED_PRECISION}."
        )
    return precision


def _infer_precision_from_env(env: Mapping[str, str]) -> str:
    precision = env.get("COVERTREEX_PRECISION")
    if precision:
        return _normalise_precision(precision)
    jax_enable_x64 = env.get("JAX_ENABLE_X64")
    if jax_enable_x64 is not None:
        return "float64" if _bool_from_env(jax_enable_x64, default=False) else "float32"
    return "float64"


def _infer_backend(raw: str | None) -> str:
    backend = (raw or "numpy").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Expected one of {_SUPPORTED_BACKENDS}."
        )
    return backend


def _parse_engine(raw: str | None) -> str:
    if not raw:
        return _DEFAULT_ENGINE
    engine = raw.strip().lower()
    if engine not in _ENGINE_CHOICES:
        raise ValueError(
            f"Unsupported engine '{engine}'. Expected one of {_ENGINE_CHOICES}."
        )
    return engine


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


def _parse_prefix_schedule(value: str | None, *, default: str) -> str:
    if value is None:
        return default
    schedule = value.strip().lower()
    if schedule not in _PREFIX_SCHEDULES:
        raise ValueError(
            f"Unsupported prefix schedule '{schedule}'. Expected one of {_PREFIX_SCHEDULES}."
        )
    return schedule


def _device_label(device: Any) -> str:
    index = getattr(device, "id", getattr(device, "device_id", 0))
    return f"{device.platform}:{index}"


def _resolve_jax_devices(requested: Sequence[str]) -> Tuple[str, ...]:
    if jax is None:
        if requested:
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


def _derive_seed_from_global(global_seed: int, channel: str) -> int:
    payload = f"{int(global_seed)}:{channel}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


class SeedPack(BaseModel):
    """Grouped seeds that drive stochastic subsystems."""

    model_config = ConfigDict(frozen=True)

    global_seed: int | None = None
    mis: int | None = None
    batch_order: int | None = None
    residual_grid: int | None = None

    def resolved(self, channel: str, *, fallback: int = 0) -> int:
        """Return a deterministic seed for the requested channel."""

        value = getattr(self, channel, None)
        if value is not None:
            return int(value)
        if self.global_seed is not None:
            return _derive_seed_from_global(int(self.global_seed), channel)
        return int(fallback)


class DiagnosticsConfig(BaseModel):
    """Diagnostics/logging knobs."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    log_level: str = "INFO"

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - trivial normalisation
        object.__setattr__(self, "log_level", self.log_level.upper())


class ResidualConfig(BaseModel):
    """Residual metric-specific knobs mirroring ``residual_*`` legacy fields."""

    model_config = ConfigDict(frozen=True)

    radius_floor: float = _DEFAULT_RESIDUAL_RADIUS_FLOOR
    scope_member_limit: int | None = None
    stream_tile: int | None = None
    scope_bitset: bool = False
    masked_scope_append: bool = True
    dynamic_query_block: bool = True
    dense_scope_streamer: bool = _DEFAULT_RESIDUAL_DENSE_SCOPE_STREAMER
    level_cache_batching: bool = True
    scope_cap_path: str | None = None
    scope_cap_default: float = _DEFAULT_RESIDUAL_SCOPE_CAP_DEFAULT
    scope_cap_output: str | None = None
    scope_cap_percentile: float = _DEFAULT_RESIDUAL_SCOPE_CAP_PERCENTILE
    scope_cap_margin: float = _DEFAULT_RESIDUAL_SCOPE_CAP_MARGIN
    prefilter_enabled: bool = False
    prefilter_lookup_path: str | None = _DEFAULT_RESIDUAL_PREFILTER_LOOKUP
    prefilter_margin: float = _DEFAULT_RESIDUAL_PREFILTER_MARGIN
    prefilter_radius_cap: float = _DEFAULT_RESIDUAL_PREFILTER_RADIUS_CAP
    prefilter_audit: bool = _DEFAULT_RESIDUAL_PREFILTER_AUDIT
    grid_whiten_scale: float = _DEFAULT_RESIDUAL_GRID_WHITEN_SCALE
    use_static_euclidean_tree: bool = False


class RuntimeModel(BaseModel):
    """Pydantic runtime configuration with nested sections."""

    model_config = ConfigDict(frozen=True)

    backend: str = "numpy"
    precision: str = "float64"
    engine: str = _DEFAULT_ENGINE
    devices: Tuple[str, ...] = Field(default_factory=tuple)
    enable_numba: bool = False
    enable_rust: bool = False
    enable_sparse_traversal: bool = False
    conflict_graph_impl: str = "dense"
    scope_segment_dedupe: bool = True
    scope_chunk_target: int = _DEFAULT_SCOPE_CHUNK_TARGET
    scope_chunk_max_segments: int = _DEFAULT_SCOPE_CHUNK_MAX_SEGMENTS
    scope_chunk_pair_merge: bool = _DEFAULT_SCOPE_CHUNK_PAIR_MERGE
    scope_conflict_buffer_reuse: bool = _DEFAULT_SCOPE_CONFLICT_BUFFER_REUSE
    conflict_degree_cap: int = _DEFAULT_CONFLICT_DEGREE_CAP
    scope_budget_schedule: Tuple[int, ...] = Field(default_factory=tuple)
    scope_budget_up_thresh: float = _DEFAULT_SCOPE_BUDGET_UP_THRESH
    scope_budget_down_thresh: float = _DEFAULT_SCOPE_BUDGET_DOWN_THRESH
    metric: str = "euclidean"
    batch_order_strategy: str = _DEFAULT_BATCH_ORDER_STRATEGY
    prefix_schedule: str = _DEFAULT_PREFIX_SCHEDULE
    prefix_density_low: float = _DEFAULT_PREFIX_DENSITY_LOW
    prefix_density_high: float = _DEFAULT_PREFIX_DENSITY_HIGH
    prefix_growth_small: float = _DEFAULT_PREFIX_GROWTH_SMALL
    prefix_growth_mid: float = _DEFAULT_PREFIX_GROWTH_MID
    prefix_growth_large: float = _DEFAULT_PREFIX_GROWTH_LARGE
    residual: ResidualConfig = Field(default_factory=ResidualConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    seeds: SeedPack = Field(default_factory=SeedPack)

    @property
    def mis_seed(self) -> int | None:
        return self.seeds.mis

    @property
    def batch_order_seed(self) -> int | None:
        return self.seeds.batch_order

    @property
    def enable_diagnostics(self) -> bool:
        return self.diagnostics.enabled

    @property
    def log_level(self) -> str:
        return self.diagnostics.log_level

    @property
    def residual_scope_member_limit(self) -> int | None:
        return self.residual.scope_member_limit

    @property
    def residual_stream_tile(self) -> int | None:
        return self.residual.stream_tile

    @property
    def residual_scope_bitset(self) -> bool:
        return self.residual.scope_bitset

    @property
    def residual_masked_scope_append(self) -> bool:
        return self.residual.masked_scope_append

    @property
    def residual_dynamic_query_block(self) -> bool:
        return self.residual.dynamic_query_block

    @property
    def residual_dense_scope_streamer(self) -> bool:
        return self.residual.dense_scope_streamer

    @property
    def residual_level_cache_batching(self) -> bool:
        return self.residual.level_cache_batching

    @property
    def residual_use_static_euclidean_tree(self) -> bool:
        return self.residual.use_static_euclidean_tree

    def to_runtime_config(self) -> "RuntimeConfig":
        from .config import RuntimeConfig  # local import to avoid circular dependency

        residual = self.residual
        diagnostics = self.diagnostics
        seeds = self.seeds
        return RuntimeConfig(
            backend=self.backend,
            precision=self.precision,
            engine=self.engine,
            devices=self.devices,
            enable_numba=self.enable_numba,
            enable_rust=self.enable_rust,
            enable_sparse_traversal=self.enable_sparse_traversal,
            enable_diagnostics=diagnostics.enabled,
            log_level=diagnostics.log_level,
            seeds=seeds,
            conflict_graph_impl=self.conflict_graph_impl,
            scope_segment_dedupe=self.scope_segment_dedupe,
            scope_chunk_target=self.scope_chunk_target,
            scope_chunk_max_segments=self.scope_chunk_max_segments,
            scope_chunk_pair_merge=self.scope_chunk_pair_merge,
            scope_conflict_buffer_reuse=self.scope_conflict_buffer_reuse,
            conflict_degree_cap=self.conflict_degree_cap,
            scope_budget_schedule=self.scope_budget_schedule,
            scope_budget_up_thresh=self.scope_budget_up_thresh,
            scope_budget_down_thresh=self.scope_budget_down_thresh,
            metric=self.metric,
            batch_order_strategy=self.batch_order_strategy,
            prefix_schedule=self.prefix_schedule,
            prefix_density_low=self.prefix_density_low,
            prefix_density_high=self.prefix_density_high,
            prefix_growth_small=self.prefix_growth_small,
            prefix_growth_mid=self.prefix_growth_mid,
            prefix_growth_large=self.prefix_growth_large,
            residual_radius_floor=residual.radius_floor,
            residual_scope_member_limit=residual.scope_member_limit,
            residual_stream_tile=residual.stream_tile,
            residual_scope_bitset=residual.scope_bitset,
            residual_masked_scope_append=residual.masked_scope_append,
            residual_dynamic_query_block=residual.dynamic_query_block,
            residual_dense_scope_streamer=residual.dense_scope_streamer,
            residual_level_cache_batching=residual.level_cache_batching,
            residual_scope_cap_path=residual.scope_cap_path,
            residual_scope_cap_default=residual.scope_cap_default,
            residual_scope_cap_output=residual.scope_cap_output,
            residual_scope_cap_percentile=residual.scope_cap_percentile,
            residual_scope_cap_margin=residual.scope_cap_margin,
            residual_prefilter_enabled=residual.prefilter_enabled,
            residual_prefilter_lookup_path=residual.prefilter_lookup_path,
            residual_prefilter_margin=residual.prefilter_margin,
            residual_prefilter_radius_cap=residual.prefilter_radius_cap,
            residual_prefilter_audit=residual.prefilter_audit,
            residual_grid_whiten_scale=residual.grid_whiten_scale,
            residual_use_static_euclidean_tree=residual.use_static_euclidean_tree,
        )

    @classmethod
    def from_legacy_config(cls, legacy: "RuntimeConfig") -> "RuntimeModel":
        residual = ResidualConfig(
            radius_floor=legacy.residual_radius_floor,
            scope_member_limit=legacy.residual_scope_member_limit,
            stream_tile=legacy.residual_stream_tile,
            scope_bitset=legacy.residual_scope_bitset,
            masked_scope_append=legacy.residual_masked_scope_append,
            dynamic_query_block=legacy.residual_dynamic_query_block,
            dense_scope_streamer=legacy.residual_dense_scope_streamer,
            level_cache_batching=legacy.residual_level_cache_batching,
            scope_cap_path=legacy.residual_scope_cap_path,
            scope_cap_default=legacy.residual_scope_cap_default,
            scope_cap_output=legacy.residual_scope_cap_output,
            scope_cap_percentile=legacy.residual_scope_cap_percentile,
            scope_cap_margin=legacy.residual_scope_cap_margin,
            prefilter_enabled=legacy.residual_prefilter_enabled,
            prefilter_lookup_path=legacy.residual_prefilter_lookup_path,
            prefilter_margin=legacy.residual_prefilter_margin,
            prefilter_radius_cap=legacy.residual_prefilter_radius_cap,
            prefilter_audit=legacy.residual_prefilter_audit,
            grid_whiten_scale=legacy.residual_grid_whiten_scale,
            use_static_euclidean_tree=legacy.residual_use_static_euclidean_tree,
        )
        diagnostics = DiagnosticsConfig(
            enabled=legacy.enable_diagnostics,
            log_level=legacy.log_level,
        )
        legacy_seeds = getattr(legacy, "seeds", None)
        if isinstance(legacy_seeds, SeedPack):
            seeds = legacy_seeds
        else:
            seeds = SeedPack(
                mis=getattr(legacy, "mis_seed", None),
                batch_order=getattr(legacy, "batch_order_seed", None),
            )
        return cls(
            backend=legacy.backend,
            precision=legacy.precision,
            engine=getattr(legacy, "engine", _DEFAULT_ENGINE),
            devices=legacy.devices,
            enable_numba=legacy.enable_numba,
            enable_rust=legacy.enable_rust,
            enable_sparse_traversal=legacy.enable_sparse_traversal,
            conflict_graph_impl=legacy.conflict_graph_impl,
            scope_segment_dedupe=legacy.scope_segment_dedupe,
            scope_chunk_target=legacy.scope_chunk_target,
            scope_chunk_max_segments=legacy.scope_chunk_max_segments,
            scope_chunk_pair_merge=legacy.scope_chunk_pair_merge,
            scope_conflict_buffer_reuse=legacy.scope_conflict_buffer_reuse,
            conflict_degree_cap=legacy.conflict_degree_cap,
            scope_budget_schedule=legacy.scope_budget_schedule,
            scope_budget_up_thresh=legacy.scope_budget_up_thresh,
            scope_budget_down_thresh=legacy.scope_budget_down_thresh,
            metric=legacy.metric,
            batch_order_strategy=legacy.batch_order_strategy,
            prefix_schedule=legacy.prefix_schedule,
            prefix_density_low=legacy.prefix_density_low,
            prefix_density_high=legacy.prefix_density_high,
            prefix_growth_small=legacy.prefix_growth_small,
            prefix_growth_mid=legacy.prefix_growth_mid,
            prefix_growth_large=legacy.prefix_growth_large,
            residual=residual,
            diagnostics=diagnostics,
            seeds=seeds,
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeModel":
        source = os.environ if env is None else env
        backend = _infer_backend(source.get("COVERTREEX_BACKEND"))
        precision = _normalise_precision(_infer_precision_from_env(source))
        requested_devices = _parse_devices(source.get("COVERTREEX_DEVICE"))
        engine = _parse_engine(source.get("COVERTREEX_ENGINE"))
        devices = _resolve_jax_devices(requested_devices) if backend == "jax" else ()
        enable_numba = _bool_from_env(
            source.get("COVERTREEX_ENABLE_NUMBA"), default=_NUMBA_AVAILABLE
        )
        enable_rust = _bool_from_env(
            source.get("COVERTREEX_ENABLE_RUST"), default=_RUST_AVAILABLE
        )
        enable_sparse_traversal = _bool_from_env(
            source.get("COVERTREEX_ENABLE_SPARSE_TRAVERSAL"), default=False
        )
        diagnostics = DiagnosticsConfig(
            enabled=_bool_from_env(source.get("COVERTREEX_ENABLE_DIAGNOSTICS"), default=True),
            log_level=(source.get("COVERTREEX_LOG_LEVEL", "INFO") or "INFO"),
        )
        seeds = SeedPack(
            global_seed=_parse_optional_int(source.get("COVERTREEX_GLOBAL_SEED")),
            mis=_parse_optional_int(source.get("COVERTREEX_MIS_SEED")),
            batch_order=_parse_optional_int(source.get("COVERTREEX_BATCH_ORDER_SEED")),
            residual_grid=_parse_optional_int(source.get("COVERTREEX_RESIDUAL_GRID_SEED")),
        )
        conflict_graph_impl = _parse_conflict_graph_impl(
            source.get("COVERTREEX_CONFLICT_GRAPH_IMPL")
        )
        scope_segment_dedupe = _bool_from_env(
            source.get("COVERTREEX_SCOPE_SEGMENT_DEDUP"), default=True
        )
        raw_chunk_target = _parse_optional_int(source.get("COVERTREEX_SCOPE_CHUNK_TARGET"))
        if raw_chunk_target is None:
            scope_chunk_target = _DEFAULT_SCOPE_CHUNK_TARGET
        elif raw_chunk_target <= 0:
            scope_chunk_target = 0
        else:
            scope_chunk_target = raw_chunk_target
        raw_chunk_segments = _parse_optional_int(
            source.get("COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS")
        )
        if raw_chunk_segments is None:
            scope_chunk_max_segments = _DEFAULT_SCOPE_CHUNK_MAX_SEGMENTS
        elif raw_chunk_segments <= 0:
            scope_chunk_max_segments = 0
        else:
            scope_chunk_max_segments = raw_chunk_segments
        scope_chunk_pair_merge = _bool_from_env(
            source.get("COVERTREEX_SCOPE_CHUNK_PAIR_MERGE"),
            default=_DEFAULT_SCOPE_CHUNK_PAIR_MERGE,
        )
        scope_conflict_buffer_reuse = _bool_from_env(
            source.get("COVERTREEX_SCOPE_CONFLICT_BUFFER_REUSE"),
            default=_DEFAULT_SCOPE_CONFLICT_BUFFER_REUSE,
        )
        raw_degree_cap = _parse_optional_int(source.get("COVERTREEX_DEGREE_CAP"))
        if raw_degree_cap is None or raw_degree_cap <= 0:
            conflict_degree_cap = _DEFAULT_CONFLICT_DEGREE_CAP
        else:
            conflict_degree_cap = raw_degree_cap
        scope_budget_schedule = _parse_scope_budget_schedule(
            source.get("COVERTREEX_SCOPE_BUDGET_SCHEDULE")
        )
        scope_budget_up_thresh = _parse_optional_float(
            source.get("COVERTREEX_SCOPE_BUDGET_UP_THRESH"),
            default=_DEFAULT_SCOPE_BUDGET_UP_THRESH,
        )
        scope_budget_down_thresh = _parse_optional_float(
            source.get("COVERTREEX_SCOPE_BUDGET_DOWN_THRESH"),
            default=_DEFAULT_SCOPE_BUDGET_DOWN_THRESH,
        )
        if scope_budget_schedule and scope_budget_down_thresh >= scope_budget_up_thresh:
            raise ValueError(
                "COVERTREEX_SCOPE_BUDGET_DOWN_THRESH must be smaller than "
                "COVERTREEX_SCOPE_BUDGET_UP_THRESH"
            )
        metric = (source.get("COVERTREEX_METRIC", "euclidean") or "euclidean").strip().lower()
        residual_metric = metric == "residual_correlation"
        if not scope_budget_schedule and residual_metric:
            scope_budget_schedule = _DEFAULT_RESIDUAL_SCOPE_BUDGET_SCHEDULE
        batch_order_strategy = _parse_batch_order_strategy(source.get("COVERTREEX_BATCH_ORDER"))
        prefix_schedule = _parse_prefix_schedule(
            source.get("COVERTREEX_PREFIX_SCHEDULE"),
            default="doubling" if residual_metric else _DEFAULT_PREFIX_SCHEDULE,
        )
        prefix_density_low = _parse_optional_float(
            source.get("COVERTREEX_PREFIX_DENSITY_LOW"),
            default=_DEFAULT_PREFIX_DENSITY_LOW,
        )
        prefix_density_high = _parse_optional_float(
            source.get("COVERTREEX_PREFIX_DENSITY_HIGH"),
            default=_DEFAULT_PREFIX_DENSITY_HIGH,
        )
        prefix_growth_small = _parse_optional_float(
            source.get("COVERTREEX_PREFIX_GROWTH_SMALL"),
            default=_DEFAULT_PREFIX_GROWTH_SMALL,
        )
        prefix_growth_mid = _parse_optional_float(
            source.get("COVERTREEX_PREFIX_GROWTH_MID"),
            default=_DEFAULT_PREFIX_GROWTH_MID,
        )
        prefix_growth_large = _parse_optional_float(
            source.get("COVERTREEX_PREFIX_GROWTH_LARGE"),
            default=_DEFAULT_PREFIX_GROWTH_LARGE,
        )
        
        residual_radius_floor = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_RADIUS_FLOOR"),
            default=_DEFAULT_RESIDUAL_RADIUS_FLOOR,
        )
        residual_scope_bitset = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_SCOPE_BITSET"),
            default=residual_metric,
        )
        residual_masked_scope_append = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_MASKED_SCOPE_APPEND"),
            default=True,
        )
        residual_dynamic_query_block = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_DYNAMIC_QUERY_BLOCK"),
            default=True,
        )
        residual_dense_scope_streamer = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_DENSE_SCOPE_STREAMER"),
            default=_DEFAULT_RESIDUAL_DENSE_SCOPE_STREAMER,
        )
        residual_level_cache_batching = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_LEVEL_CACHE_BATCHING"),
            default=True,
        )
        raw_scope_member_limit = _parse_optional_int(
            source.get("COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT")
        )
        if raw_scope_member_limit is None:
            residual_scope_member_limit = None
        elif raw_scope_member_limit <= 0:
            residual_scope_member_limit = 0
        else:
            residual_scope_member_limit = raw_scope_member_limit
        raw_stream_tile = _parse_optional_int(source.get("COVERTREEX_RESIDUAL_STREAM_TILE"))
        if raw_stream_tile is None or raw_stream_tile <= 0:
            residual_stream_tile = _DEFAULT_RESIDUAL_STREAM_TILE if residual_metric else None
        else:
            residual_stream_tile = raw_stream_tile
        residual_scope_cap_path = source.get("COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH")
        residual_scope_cap_default = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT"),
            default=_DEFAULT_RESIDUAL_SCOPE_CAP_DEFAULT,
        )
        residual_scope_cap_output = source.get("COVERTREEX_RESIDUAL_SCOPE_CAP_OUTPUT")
        residual_scope_cap_percentile = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_SCOPE_CAP_PERCENTILE"),
            default=_DEFAULT_RESIDUAL_SCOPE_CAP_PERCENTILE,
        )
        residual_scope_cap_margin = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_SCOPE_CAP_MARGIN"),
            default=_DEFAULT_RESIDUAL_SCOPE_CAP_MARGIN,
        )
        residual_prefilter_enabled = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_PREFILTER"),
            default=False,
        )
        residual_prefilter_lookup_path = source.get("COVERTREEX_RESIDUAL_PREFILTER_LOOKUP_PATH")
        residual_prefilter_margin = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_PREFILTER_MARGIN"),
            default=_DEFAULT_RESIDUAL_PREFILTER_MARGIN,
        )
        residual_prefilter_radius_cap = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_PREFILTER_RADIUS_CAP"),
            default=_DEFAULT_RESIDUAL_PREFILTER_RADIUS_CAP,
        )
        residual_prefilter_audit = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_PREFILTER_AUDIT"),
            default=_DEFAULT_RESIDUAL_PREFILTER_AUDIT,
        )
        residual_grid_whiten_scale = _parse_optional_float(
            source.get("COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE"),
            default=_DEFAULT_RESIDUAL_GRID_WHITEN_SCALE,
        )
        if residual_grid_whiten_scale <= 0.0:
            residual_grid_whiten_scale = _DEFAULT_RESIDUAL_GRID_WHITEN_SCALE
        
        residual_use_static_euclidean_tree = _bool_from_env(
            source.get("COVERTREEX_RESIDUAL_USE_STATIC_EUCLIDEAN_TREE"),
            default=False,
        )

        residual = ResidualConfig(
            radius_floor=residual_radius_floor,
            scope_member_limit=residual_scope_member_limit,
            stream_tile=residual_stream_tile,
            scope_bitset=residual_scope_bitset,
            masked_scope_append=residual_masked_scope_append,
            dynamic_query_block=residual_dynamic_query_block,
            dense_scope_streamer=residual_dense_scope_streamer,
            level_cache_batching=residual_level_cache_batching,
            scope_cap_path=residual_scope_cap_path,
            scope_cap_default=residual_scope_cap_default,
            scope_cap_output=residual_scope_cap_output,
            scope_cap_percentile=residual_scope_cap_percentile,
            scope_cap_margin=residual_scope_cap_margin,
            prefilter_enabled=residual_prefilter_enabled,
            prefilter_lookup_path=(
                residual_prefilter_lookup_path or _DEFAULT_RESIDUAL_PREFILTER_LOOKUP
            ),
            prefilter_margin=residual_prefilter_margin,
            prefilter_radius_cap=residual_prefilter_radius_cap,
            prefilter_audit=residual_prefilter_audit,
            grid_whiten_scale=residual_grid_whiten_scale,
            use_static_euclidean_tree=residual_use_static_euclidean_tree,
        )

        return cls(
            backend=backend,
            precision=precision,
            engine=engine,
            devices=devices,
            enable_numba=enable_numba,
            enable_rust=enable_rust,
            enable_sparse_traversal=enable_sparse_traversal,
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
            prefix_schedule=prefix_schedule,
            prefix_density_low=prefix_density_low,
            prefix_density_high=prefix_density_high,
            prefix_growth_small=prefix_growth_small,
            prefix_growth_mid=prefix_growth_mid,
            prefix_growth_large=prefix_growth_large,
            residual=residual,
            diagnostics=diagnostics,
            seeds=seeds,
        )


__all__ = [
    "RuntimeModel",
    "ResidualConfig",
    "DiagnosticsConfig",
    "SeedPack",
]
