from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Tuple

from covertreex import config as cx_config
from covertreex.metrics.residual.policy import ResidualPolicy


def _maybe_tuple(value: Iterable[str] | Tuple[str, ...] | None) -> Tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    return tuple(value)


def _apply_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _policy_to_overrides(policy: ResidualPolicy) -> Dict[str, Any]:
    return {
        "residual_gate1_enabled": policy.gate1_enabled,
        "residual_gate1_alpha": policy.gate1_alpha,
        "residual_gate1_margin": policy.gate1_margin,
        "residual_gate1_eps": policy.gate1_eps,
        "residual_gate1_audit": policy.gate1_audit,
        "residual_gate1_radius_cap": policy.gate1_radius_cap,
        "residual_gate1_band_eps": policy.gate1_band_eps,
        "residual_gate1_keep_pct": policy.gate1_keep_pct,
        "residual_gate1_prune_pct": policy.gate1_prune_pct,
        "residual_radius_floor": policy.radius_floor,
        "residual_scope_cap_path": policy.scope_cap_path,
        "residual_scope_cap_default": policy.scope_cap_default,
        "residual_gate1_profile_path": policy.gate1_profile_path,
        "residual_gate1_profile_bins": policy.gate1_profile_bins,
        "residual_gate1_lookup_path": policy.gate1_lookup_path,
        "residual_gate1_lookup_margin": policy.gate1_lookup_margin,
        "residual_prefilter_enabled": policy.prefilter_enabled,
        "residual_prefilter_lookup_path": policy.prefilter_lookup_path,
        "residual_prefilter_margin": policy.prefilter_margin,
        "residual_prefilter_radius_cap": policy.prefilter_radius_cap,
        "residual_prefilter_audit": policy.prefilter_audit,
    }


@dataclass(frozen=True)
class Residual:
    """Optional residual-metric tuning knobs grouped for readability."""

    policy: ResidualPolicy | None = None
    gate1_enabled: bool | None = None
    gate1_alpha: float | None = None
    gate1_margin: float | None = None
    gate1_eps: float | None = None
    gate1_audit: bool | None = None
    gate1_radius_cap: float | None = None
    gate1_band_eps: float | None = None
    gate1_keep_pct: float | None = None
    gate1_prune_pct: float | None = None
    radius_floor: float | None = None
    scope_cap_path: str | None = None
    scope_cap_default: float | None = None
    profile_path: str | None = None
    profile_bins: int | None = None
    lookup_path: str | None = None
    lookup_margin: float | None = None
    prefilter_enabled: bool | None = None
    prefilter_lookup_path: str | None = None
    prefilter_margin: float | None = None
    prefilter_radius_cap: float | None = None
    prefilter_audit: bool | None = None

    def as_overrides(self, base_policy: ResidualPolicy | None = None) -> Dict[str, Any]:
        policy = self.to_policy(base=base_policy)
        return _policy_to_overrides(policy)

    @classmethod
    def from_config(cls, config: cx_config.RuntimeConfig) -> "Residual":
        policy = ResidualPolicy.from_runtime(config)
        return cls(
            policy=policy,
            gate1_enabled=config.residual_gate1_enabled,
            gate1_alpha=config.residual_gate1_alpha,
            gate1_margin=config.residual_gate1_margin,
            gate1_eps=config.residual_gate1_eps,
            gate1_audit=config.residual_gate1_audit,
            gate1_radius_cap=config.residual_gate1_radius_cap,
            gate1_band_eps=config.residual_gate1_band_eps,
            gate1_keep_pct=config.residual_gate1_keep_pct,
            gate1_prune_pct=config.residual_gate1_prune_pct,
            radius_floor=config.residual_radius_floor,
            scope_cap_path=config.residual_scope_cap_path,
            scope_cap_default=config.residual_scope_cap_default,
            profile_path=config.residual_gate1_profile_path,
            profile_bins=config.residual_gate1_profile_bins,
            lookup_path=config.residual_gate1_lookup_path,
            lookup_margin=config.residual_gate1_lookup_margin,
            prefilter_enabled=config.residual_prefilter_enabled,
            prefilter_lookup_path=config.residual_prefilter_lookup_path,
            prefilter_margin=config.residual_prefilter_margin,
            prefilter_radius_cap=config.residual_prefilter_radius_cap,
            prefilter_audit=config.residual_prefilter_audit,
        )

    @classmethod
    def from_policy(cls, policy: ResidualPolicy) -> "Residual":
        return cls(
            policy=policy,
            gate1_enabled=policy.gate1_enabled,
            gate1_alpha=policy.gate1_alpha,
            gate1_margin=policy.gate1_margin,
            gate1_eps=policy.gate1_eps,
            gate1_audit=policy.gate1_audit,
            gate1_radius_cap=policy.gate1_radius_cap,
            gate1_band_eps=policy.gate1_band_eps,
            gate1_keep_pct=policy.gate1_keep_pct,
            gate1_prune_pct=policy.gate1_prune_pct,
            radius_floor=policy.radius_floor,
            scope_cap_path=policy.scope_cap_path,
            scope_cap_default=policy.scope_cap_default,
            profile_path=policy.gate1_profile_path,
            profile_bins=policy.gate1_profile_bins,
            lookup_path=policy.gate1_lookup_path,
            lookup_margin=policy.gate1_lookup_margin,
            prefilter_enabled=policy.prefilter_enabled,
            prefilter_lookup_path=policy.prefilter_lookup_path,
            prefilter_margin=policy.prefilter_margin,
            prefilter_radius_cap=policy.prefilter_radius_cap,
            prefilter_audit=policy.prefilter_audit,
        )

    def to_policy(self, base: ResidualPolicy | None = None) -> ResidualPolicy:
        policy = self.policy or base
        if policy is None:
            policy = ResidualPolicy.from_runtime(cx_config.runtime_config())
        updates: Dict[str, Any] = {}
        _apply_if_present(updates, "gate1_enabled", self.gate1_enabled)
        _apply_if_present(updates, "gate1_alpha", self.gate1_alpha)
        _apply_if_present(updates, "gate1_margin", self.gate1_margin)
        _apply_if_present(updates, "gate1_eps", self.gate1_eps)
        _apply_if_present(updates, "gate1_audit", self.gate1_audit)
        _apply_if_present(updates, "gate1_radius_cap", self.gate1_radius_cap)
        _apply_if_present(updates, "gate1_band_eps", self.gate1_band_eps)
        _apply_if_present(updates, "gate1_keep_pct", self.gate1_keep_pct)
        _apply_if_present(updates, "gate1_prune_pct", self.gate1_prune_pct)
        _apply_if_present(updates, "radius_floor", self.radius_floor)
        _apply_if_present(updates, "scope_cap_path", self.scope_cap_path)
        _apply_if_present(updates, "scope_cap_default", self.scope_cap_default)
        _apply_if_present(updates, "gate1_profile_path", self.profile_path)
        _apply_if_present(updates, "gate1_profile_bins", self.profile_bins)
        _apply_if_present(updates, "gate1_lookup_path", self.lookup_path)
        _apply_if_present(updates, "gate1_lookup_margin", self.lookup_margin)
        _apply_if_present(updates, "prefilter_enabled", self.prefilter_enabled)
        _apply_if_present(updates, "prefilter_lookup_path", self.prefilter_lookup_path)
        _apply_if_present(updates, "prefilter_margin", self.prefilter_margin)
        _apply_if_present(updates, "prefilter_radius_cap", self.prefilter_radius_cap)
        _apply_if_present(updates, "prefilter_audit", self.prefilter_audit)
        if not updates:
            return policy
        return replace(policy, **updates)


@dataclass(frozen=True)
class Runtime:
    """Declarative runtime configuration that can activate a covertreex context."""

    backend: str | None = None
    precision: str | None = None
    metric: str | None = None
    devices: Tuple[str, ...] | None = None
    enable_numba: bool | None = None
    enable_sparse_traversal: bool | None = None
    diagnostics: bool | None = None
    log_level: str | None = None
    mis_seed: int | None = None
    conflict_graph: str | None = None
    scope_segment_dedupe: bool | None = None
    scope_chunk_target: int | None = None
    scope_chunk_max_segments: int | None = None
    degree_cap: int | None = None
    batch_order: str | None = None
    batch_order_seed: int | None = None
    prefix_schedule: str | None = None
    prefix_density_low: float | None = None
    prefix_density_high: float | None = None
    prefix_growth_small: float | None = None
    prefix_growth_mid: float | None = None
    prefix_growth_large: float | None = None
    residual: Residual | None = None
    residual_force_whitened: bool | None = None
    residual_scope_member_limit: int | None = None
    residual_stream_tile: int | None = None
    residual_scope_bitset: bool | None = None
    residual_masked_scope_append: bool | None = None
    residual_dynamic_query_block: bool | None = None
    residual_dense_scope_streamer: bool | None = None
    residual_level_cache_batching: bool | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_config(self, base: cx_config.RuntimeConfig | None = None) -> cx_config.RuntimeConfig:
        base_config = base or cx_config.RuntimeConfig.from_env()
        overrides: Dict[str, Any] = {}
        _apply_if_present(overrides, "backend", self.backend)
        _apply_if_present(overrides, "precision", self.precision)
        devices = _maybe_tuple(self.devices)
        if devices is not None:
            overrides["devices"] = devices
        _apply_if_present(overrides, "metric", self.metric)
        _apply_if_present(overrides, "enable_numba", self.enable_numba)
        _apply_if_present(overrides, "enable_sparse_traversal", self.enable_sparse_traversal)
        _apply_if_present(overrides, "enable_diagnostics", self.diagnostics)
        _apply_if_present(overrides, "log_level", self.log_level)
        _apply_if_present(overrides, "mis_seed", self.mis_seed)
        _apply_if_present(overrides, "conflict_graph_impl", self.conflict_graph)
        _apply_if_present(overrides, "scope_segment_dedupe", self.scope_segment_dedupe)
        _apply_if_present(overrides, "scope_chunk_target", self.scope_chunk_target)
        _apply_if_present(overrides, "scope_chunk_max_segments", self.scope_chunk_max_segments)
        _apply_if_present(overrides, "conflict_degree_cap", self.degree_cap)
        _apply_if_present(overrides, "batch_order_strategy", self.batch_order)
        _apply_if_present(overrides, "batch_order_seed", self.batch_order_seed)
        _apply_if_present(overrides, "prefix_schedule", self.prefix_schedule)
        _apply_if_present(overrides, "prefix_density_low", self.prefix_density_low)
        _apply_if_present(overrides, "prefix_density_high", self.prefix_density_high)
        _apply_if_present(overrides, "prefix_growth_small", self.prefix_growth_small)
        _apply_if_present(overrides, "prefix_growth_mid", self.prefix_growth_mid)
        _apply_if_present(overrides, "prefix_growth_large", self.prefix_growth_large)
        if self.residual is not None:
            base_policy = ResidualPolicy.from_runtime(base_config)
            overrides.update(self.residual.as_overrides(base_policy=base_policy))
        _apply_if_present(overrides, "residual_force_whitened", self.residual_force_whitened)
        _apply_if_present(
            overrides,
            "residual_scope_member_limit",
            self.residual_scope_member_limit,
        )
        _apply_if_present(
            overrides,
            "residual_stream_tile",
            self.residual_stream_tile,
        )
        _apply_if_present(
            overrides,
            "residual_scope_bitset",
            self.residual_scope_bitset,
        )
        _apply_if_present(
            overrides,
            "residual_masked_scope_append",
            self.residual_masked_scope_append,
        )
        _apply_if_present(
            overrides,
            "residual_dynamic_query_block",
            self.residual_dynamic_query_block,
        )
        _apply_if_present(
            overrides,
            "residual_dense_scope_streamer",
            self.residual_dense_scope_streamer,
        )
        _apply_if_present(
            overrides,
            "residual_level_cache_batching",
            self.residual_level_cache_batching,
        )
        overrides.update(self.extra)
        if not overrides:
            return base_config
        return replace(base_config, **overrides)

    def activate(self) -> cx_config.RuntimeContext:
        """Install this runtime as the active global context and return it."""

        config = self.to_config()
        return cx_config.configure_runtime(config)

    def describe(self) -> Dict[str, Any]:
        config = self.to_config()
        return {
            "backend": config.backend,
            "precision": config.precision,
            "metric": config.metric,
            "devices": config.devices,
            "conflict_graph": config.conflict_graph_impl,
            "conflict_degree_cap": config.conflict_degree_cap,
            "batch_order": config.batch_order_strategy,
            "prefix_schedule": config.prefix_schedule,
            "enable_numba": config.enable_numba,
            "enable_sparse_traversal": config.enable_sparse_traversal,
            "enable_diagnostics": config.enable_diagnostics,
            "residual_force_whitened": config.residual_force_whitened,
            "residual_scope_member_limit": config.residual_scope_member_limit,
            "residual_stream_tile": config.residual_stream_tile,
            "residual_dense_scope_streamer": config.residual_dense_scope_streamer,
            "residual_scope_bitset": config.residual_scope_bitset,
            "residual_masked_scope_append": config.residual_masked_scope_append,
        }

    def with_updates(self, **kwargs: Any) -> "Runtime":
        return replace(self, **kwargs)

    @classmethod
    def from_active(cls) -> "Runtime":
        return cls.from_config(cx_config.runtime_config())

    @classmethod
    def from_config(cls, config: cx_config.RuntimeConfig) -> "Runtime":
        residual = Residual.from_config(config)
        return cls(
            backend=config.backend,
            precision=config.precision,
            metric=config.metric,
            devices=config.devices,
            enable_numba=config.enable_numba,
            enable_sparse_traversal=config.enable_sparse_traversal,
            diagnostics=config.enable_diagnostics,
            log_level=config.log_level,
            mis_seed=config.mis_seed,
            conflict_graph=config.conflict_graph_impl,
            scope_segment_dedupe=config.scope_segment_dedupe,
            scope_chunk_target=config.scope_chunk_target,
            scope_chunk_max_segments=config.scope_chunk_max_segments,
            degree_cap=config.conflict_degree_cap,
            batch_order=config.batch_order_strategy,
            batch_order_seed=config.batch_order_seed,
            prefix_schedule=config.prefix_schedule,
            prefix_density_low=config.prefix_density_low,
            prefix_density_high=config.prefix_density_high,
            prefix_growth_small=config.prefix_growth_small,
            prefix_growth_mid=config.prefix_growth_mid,
            prefix_growth_large=config.prefix_growth_large,
            residual=residual,
            residual_force_whitened=config.residual_force_whitened,
            residual_scope_member_limit=config.residual_scope_member_limit,
            residual_stream_tile=config.residual_stream_tile,
            residual_scope_bitset=config.residual_scope_bitset,
            residual_masked_scope_append=config.residual_masked_scope_append,
            residual_dense_scope_streamer=config.residual_dense_scope_streamer,
        )
