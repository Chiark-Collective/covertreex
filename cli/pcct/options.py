from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

from covertreex.api import Runtime as ApiRuntime
from profiles.loader import ProfileError
from profiles.overrides import OverrideError


@dataclass
class QueryCLIOptions:
    dimension: int = 8
    tree_points: int = 16_384
    batch_size: int = 512
    queries: int = 1_024
    k: int = 8
    seed: int = 0
    run_id: str | None = None
    profile: str | None = None
    set_override: list[str] | None = None
    metric: str = "euclidean"
    engine: str | None = None
    backend: str | None = None
    precision: str | None = None
    devices: Tuple[str, ...] | None = None
    enable_numba: bool | None = None
    enable_rust: bool | None = None
    enable_sparse_traversal: bool | None = None
    diagnostics: bool | None = None
    log_level: str | None = None
    global_seed: int | None = None
    mis_seed: int | None = None
    conflict_graph: str | None = None
    scope_segment_dedupe: bool | None = None
    scope_chunk_target: int | None = None
    scope_chunk_max_segments: int | None = None
    scope_chunk_pair_merge: bool | None = None
    scope_conflict_buffer_reuse: bool | None = None
    degree_cap: int | None = None
    batch_order: str | None = None
    batch_order_seed: int | None = None
    residual_grid_seed: int | None = None
    prefix_schedule: str | None = None
    prefix_density_low: float | None = None
    prefix_density_high: float | None = None
    prefix_growth_small: float | None = None
    prefix_growth_mid: float | None = None
    prefix_growth_large: float | None = None
    residual_lengthscale: float = 1.0
    residual_variance: float = 1.0
    residual_kernel_type: int | None = 0
    residual_inducing: int = 512
    residual_chunk_size: int = 512
    residual_stream_tile: int | None = 64
    residual_scope_member_limit: int | None = None
    residual_scope_bitset: bool | None = None
    residual_dynamic_query_block: bool | None = None
    residual_dense_scope_streamer: bool | None = None
    residual_masked_scope_append: bool | None = None
    residual_level_cache_batching: bool | None = True
    residual_scope_caps: str | None = None
    residual_scope_cap_default: float | None = None
    residual_scope_cap_output: str | None = None
    residual_scope_cap_percentile: float = 0.5
    residual_scope_cap_margin: float = 0.05
    residual_radius_floor: float | None = None
    residual_prefilter: bool | None = None
    residual_prefilter_lookup_path: str | None = None
    residual_prefilter_margin: float | None = None
    residual_prefilter_radius_cap: float | None = None
    residual_prefilter_audit: bool | None = None
    baseline: str = "none"
    log_file: str | None = None
    no_log_file: bool = False
    build_mode: str = "batch"
    predecessor_mode: bool | None = None

    @classmethod
    def from_namespace(cls, namespace: Any) -> "QueryCLIOptions":
        values = {}
        for field in cls.__dataclass_fields__:
            if hasattr(namespace, field):
                value = getattr(namespace, field)
                if field == "devices" and value:
                    value = tuple(value)
                values[field] = value
        return cls(**values)


def resolve_metric_flag(
    metric: Literal["auto", "euclidean", "residual", "residual-lite"],
    *,
    profile: Optional[str],
    overrides: Optional[List[str]],
) -> Literal["euclidean", "residual", "residual-lite"]:
    """Return the effective metric derived from CLI inputs."""

    if metric in ("euclidean", "residual", "residual-lite"):
        return metric
    if not profile:
        return "euclidean"
    try:
        runtime = ApiRuntime.from_profile(profile, overrides=overrides)
    except (ProfileError, OverrideError, ValueError) as exc:
        raise ValueError(f"Invalid profile or overrides: {exc}") from exc
    described_metric = (runtime.describe().get("metric") or "euclidean").lower()
    if "residual_correlation_lite" in described_metric or "residual-lite" in described_metric:
        return "residual-lite"
    return "residual" if "residual" in described_metric else "euclidean"


__all__ = ["resolve_metric_flag"]
