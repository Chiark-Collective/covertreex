from __future__ import annotations

from typing import Any, Mapping

from covertreex.api import Runtime as ApiRuntime, Residual as ApiResidual


def _get_arg(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _residual_policy_from_args(args: Any) -> ApiResidual | None:
    metric = _get_arg(args, "metric", "euclidean")
    if metric != "residual":
        return None
    overrides: dict[str, Any] = {}
    scope_caps = _get_arg(args, "residual_scope_caps")
    if scope_caps:
        overrides["scope_cap_path"] = scope_caps
    default_cap = _get_arg(args, "residual_scope_cap_default")
    if default_cap is not None:
        overrides["scope_cap_default"] = default_cap
    gate_mode = _get_arg(args, "residual_gate")
    if gate_mode == "off":
        overrides["gate1_enabled"] = False
    elif gate_mode == "lookup":
        overrides["gate1_enabled"] = True
        overrides["lookup_path"] = _get_arg(args, "residual_gate_lookup_path")
        overrides["lookup_margin"] = _get_arg(args, "residual_gate_margin")
        overrides["gate1_audit"] = True
        cap_value = _get_arg(args, "residual_gate_cap")
        if cap_value and cap_value > 0:
            overrides["gate1_radius_cap"] = cap_value
    profile_path = _get_arg(args, "residual_gate_profile_path")
    if profile_path:
        overrides["profile_path"] = profile_path
    profile_bins = _get_arg(args, "residual_gate_profile_bins")
    if profile_bins:
        overrides["profile_bins"] = profile_bins
    simple_map = {
        "residual_gate_alpha": "gate1_alpha",
        "residual_gate_eps": "gate1_eps",
        "residual_gate_band_eps": "gate1_band_eps",
        "residual_gate_keep_pct": "gate1_keep_pct",
        "residual_gate_prune_pct": "gate1_prune_pct",
        "residual_gate_audit": "gate1_audit",
        "residual_radius_floor": "radius_floor",
        "residual_prefilter": "prefilter_enabled",
        "residual_prefilter_lookup_path": "prefilter_lookup_path",
        "residual_prefilter_margin": "prefilter_margin",
        "residual_prefilter_radius_cap": "prefilter_radius_cap",
        "residual_prefilter_audit": "prefilter_audit",
    }
    for attr_name, field_name in simple_map.items():
        value = _get_arg(args, attr_name)
        if value is not None:
            overrides[field_name] = value
    if not overrides:
        return None
    return ApiResidual(**overrides)


def runtime_from_args(
    args: Any,
    *,
    default_metric: str = "euclidean",
    extra_overrides: Mapping[str, Any] | None = None,
) -> ApiRuntime:
    metric = _get_arg(args, "metric", default_metric) or default_metric
    runtime_kwargs: dict[str, Any] = {
        "metric": "residual_correlation" if metric == "residual" else metric,
    }
    backend = _get_arg(args, "backend")
    if backend:
        runtime_kwargs["backend"] = backend
    elif metric == "residual":
        runtime_kwargs["backend"] = "numpy"
    devices = _get_arg(args, "devices")
    if devices:
        runtime_kwargs["devices"] = tuple(devices)
    precision = _get_arg(args, "precision")
    if precision:
        runtime_kwargs["precision"] = precision
    enable_numba = _get_arg(args, "enable_numba")
    if enable_numba is not None:
        runtime_kwargs["enable_numba"] = bool(enable_numba)
    elif metric == "residual":
        runtime_kwargs["enable_numba"] = True
    diagnostics = _get_arg(args, "diagnostics")
    if diagnostics is not None:
        runtime_kwargs["diagnostics"] = bool(diagnostics)
    log_level = _get_arg(args, "log_level")
    if log_level:
        runtime_kwargs["log_level"] = log_level
    enable_sparse = _get_arg(args, "enable_sparse_traversal")
    if enable_sparse is not None:
        runtime_kwargs["enable_sparse_traversal"] = bool(enable_sparse)
    elif metric == "residual":
        runtime_kwargs["enable_sparse_traversal"] = True
    conflict_graph = _get_arg(args, "conflict_graph")
    if conflict_graph:
        runtime_kwargs["conflict_graph"] = conflict_graph
    scope_segment_dedupe = _get_arg(args, "scope_segment_dedupe")
    if scope_segment_dedupe is not None:
        runtime_kwargs["scope_segment_dedupe"] = bool(scope_segment_dedupe)
    scope_chunk_target = _get_arg(args, "scope_chunk_target")
    if scope_chunk_target is not None:
        runtime_kwargs["scope_chunk_target"] = int(scope_chunk_target)
    scope_chunk_max_segments = _get_arg(args, "scope_chunk_max_segments")
    if scope_chunk_max_segments is not None:
        runtime_kwargs["scope_chunk_max_segments"] = int(scope_chunk_max_segments)
    batch_order = _get_arg(args, "batch_order")
    if batch_order:
        runtime_kwargs["batch_order"] = batch_order
    batch_seed = _get_arg(args, "batch_order_seed")
    if batch_seed is not None:
        runtime_kwargs["batch_order_seed"] = batch_seed
    prefix_schedule = _get_arg(args, "prefix_schedule")
    if prefix_schedule:
        runtime_kwargs["prefix_schedule"] = prefix_schedule
    prefix_density_low = _get_arg(args, "prefix_density_low")
    if prefix_density_low is not None:
        runtime_kwargs["prefix_density_low"] = float(prefix_density_low)
    prefix_density_high = _get_arg(args, "prefix_density_high")
    if prefix_density_high is not None:
        runtime_kwargs["prefix_density_high"] = float(prefix_density_high)
    prefix_growth_small = _get_arg(args, "prefix_growth_small")
    if prefix_growth_small is not None:
        runtime_kwargs["prefix_growth_small"] = float(prefix_growth_small)
    prefix_growth_mid = _get_arg(args, "prefix_growth_mid")
    if prefix_growth_mid is not None:
        runtime_kwargs["prefix_growth_mid"] = float(prefix_growth_mid)
    prefix_growth_large = _get_arg(args, "prefix_growth_large")
    if prefix_growth_large is not None:
        runtime_kwargs["prefix_growth_large"] = float(prefix_growth_large)
    residual_stream_tile = _get_arg(args, "residual_stream_tile")
    if residual_stream_tile is not None:
        runtime_kwargs["residual_stream_tile"] = residual_stream_tile
    mis_seed = _get_arg(args, "mis_seed")
    if mis_seed is not None:
        runtime_kwargs["mis_seed"] = mis_seed
    residual_force_whitened = _get_arg(args, "residual_force_whitened")
    if residual_force_whitened is not None:
        runtime_kwargs["residual_force_whitened"] = bool(residual_force_whitened)
    residual_scope_member_limit = _get_arg(args, "residual_scope_member_limit")
    if residual_scope_member_limit is not None:
        runtime_kwargs["residual_scope_member_limit"] = int(residual_scope_member_limit)
    residual_policy = _residual_policy_from_args(args)
    if residual_policy is not None:
        runtime_kwargs["residual"] = residual_policy
    if extra_overrides:
        runtime_kwargs.update(extra_overrides)
    return ApiRuntime(**runtime_kwargs)


__all__ = ["runtime_from_args"]
