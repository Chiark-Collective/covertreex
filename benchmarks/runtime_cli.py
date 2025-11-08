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
    precision = _get_arg(args, "precision")
    if precision:
        runtime_kwargs["precision"] = precision
    enable_numba = _get_arg(args, "enable_numba")
    if enable_numba is not None:
        runtime_kwargs["enable_numba"] = bool(enable_numba)
    diagnostics = _get_arg(args, "diagnostics")
    if diagnostics is not None:
        runtime_kwargs["diagnostics"] = bool(diagnostics)
    enable_sparse = _get_arg(args, "enable_sparse_traversal")
    if enable_sparse is not None:
        runtime_kwargs["enable_sparse_traversal"] = bool(enable_sparse)
    batch_order = _get_arg(args, "batch_order")
    if batch_order:
        runtime_kwargs["batch_order"] = batch_order
    batch_seed = _get_arg(args, "batch_order_seed")
    if batch_seed is not None:
        runtime_kwargs["batch_order_seed"] = batch_seed
    prefix_schedule = _get_arg(args, "prefix_schedule")
    if prefix_schedule:
        runtime_kwargs["prefix_schedule"] = prefix_schedule
    mis_seed = _get_arg(args, "mis_seed")
    if mis_seed is not None:
        runtime_kwargs["mis_seed"] = mis_seed
    residual_policy = _residual_policy_from_args(args)
    if residual_policy is not None:
        runtime_kwargs["residual"] = residual_policy
        if _get_arg(args, "residual_gate") == "lookup" and "enable_sparse_traversal" not in runtime_kwargs:
            runtime_kwargs["enable_sparse_traversal"] = True
    if extra_overrides:
        runtime_kwargs.update(extra_overrides)
    return ApiRuntime(**runtime_kwargs)
