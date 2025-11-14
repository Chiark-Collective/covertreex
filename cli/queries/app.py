from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.random import default_rng
import typer
from typing_extensions import Annotated

from covertreex import config as cx_config, reset_residual_metric
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.telemetry import generate_run_id, timestamped_artifact

from cli.runtime import runtime_from_args
from tests.utils.datasets import gaussian_points

from .baselines import run_baseline_comparisons
from .benchmark import benchmark_knn_latency
from .gate import _append_gate_profile_log
from .runtime import (
    _emit_engine_banner,
    _ensure_thread_env_defaults,
    _gate_active_for_backend,
    _resolve_artifact_arg,
    _thread_env_snapshot,
)
from .telemetry import (
    CLITelemetryHandles,
    ResidualTraversalTelemetry,
    initialise_cli_telemetry,
)


@dataclass
class QueryCLIOptions:
    dimension: int = 8
    tree_points: int = 16_384
    batch_size: int = 512
    queries: int = 1_024
    k: int = 8
    seed: int = 0
    run_id: str | None = None
    metric: str = "euclidean"
    backend: str | None = None
    precision: str | None = None
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
    batch_order: str | None = None
    batch_order_seed: int | None = None
    prefix_schedule: str | None = None
    prefix_density_low: float | None = None
    prefix_density_high: float | None = None
    prefix_growth_small: float | None = None
    prefix_growth_mid: float | None = None
    prefix_growth_large: float | None = None
    residual_lengthscale: float = 1.0
    residual_variance: float = 1.0
    residual_inducing: int = 512
    residual_chunk_size: int = 512
    residual_stream_tile: int | None = 64
    residual_force_whitened: bool | None = None
    residual_scope_member_limit: int | None = None
    residual_scope_bitset: bool | None = None
    residual_dynamic_query_block: bool | None = None
    residual_dense_scope_streamer: bool | None = None
    residual_masked_scope_append: bool | None = None
    residual_gate: str | None = "off"
    residual_gate_lookup_path: str = "docs/data/residual_gate_profile_diag0.json"
    residual_gate_margin: float = 0.02
    residual_gate_cap: float = 0.0
    residual_gate_alpha: float | None = None
    residual_gate_eps: float | None = None
    residual_gate_band_eps: float | None = None
    residual_gate_keep_pct: float | None = None
    residual_gate_prune_pct: float | None = None
    residual_gate_audit: bool | None = None
    residual_gate_profile_path: str | None = None
    residual_gate_profile_bins: int = 512
    residual_gate_profile_log: str | None = None
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

    @classmethod
    def from_namespace(cls, namespace: Any) -> "QueryCLIOptions":
        values = {}
        for field in cls.__dataclass_fields__:
            if hasattr(namespace, field):
                values[field] = getattr(namespace, field)
        return cls(**values)


app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Benchmark the parallel compressed cover tree (PCCT) implementation.",
)

_SHAPE_PANEL = "Benchmark shape"
_RUNTIME_PANEL = "Runtime controls"
_RESIDUAL_PANEL = "Residual metric"
_GATE_PANEL = "Gate & prefilter"
_TELEMETRY_PANEL = "Telemetry & baselines"


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    dimension: Annotated[
        int,
        typer.Option(
            "--dimension",
            help="Dimensionality of tree/query points.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = 8,
    tree_points: Annotated[
        int,
        typer.Option(
            "--tree-points",
            help="Number of points inserted before querying.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = 16_384,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Insertion batch size.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = 512,
    queries: Annotated[
        int,
        typer.Option(
            "--queries",
            help="Number of query points per run.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = 1_024,
    k: Annotated[
        int,
        typer.Option(
            "--k",
            help="Number of neighbours requested per query.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = 8,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Base random seed for point/query generation.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = 0,
    run_id: Annotated[
        Optional[str],
        typer.Option(
            "--run-id",
            help="Optional run identifier propagated to telemetry artifacts.",
            rich_help_panel=_TELEMETRY_PANEL,
        ),
    ] = None,
    metric: Annotated[
        Literal["euclidean", "residual"],
        typer.Option(
            "--metric",
            case_sensitive=False,
            help="Distance metric to benchmark.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = "euclidean",
    backend: Annotated[
        Optional[str],
        typer.Option(
            "--backend",
            help="Runtime backend (default inferred from metric).",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    precision: Annotated[
        Optional[str],
        typer.Option(
            "--precision",
            help="Backend precision override (float32, float64, ...).",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    devices: Annotated[
        Optional[List[str]],
        typer.Option(
            "--device",
            "-d",
            help="Restrict execution to specific logical devices.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    enable_numba: Annotated[
        Optional[bool],
        typer.Option(
            "--enable-numba/--disable-numba",
            help="Force-enable or disable Numba kernels.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    enable_sparse_traversal: Annotated[
        Optional[bool],
        typer.Option(
            "--enable-sparse-traversal/--disable-sparse-traversal",
            help="Toggle sparse traversal engines when supported.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    diagnostics: Annotated[
        Optional[bool],
        typer.Option(
            "--enable-diagnostics/--disable-diagnostics",
            help="Control resource polling + diagnostic logging.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            "--log-level",
            help="Override runtime log level.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    mis_seed: Annotated[
        Optional[int],
        typer.Option(
            "--mis-seed",
            help="Sticky MIS seed when reproducing deterministic traversals.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    conflict_graph: Annotated[
        Optional[str],
        typer.Option(
            "--conflict-graph",
            help="Conflict graph implementation (dense, grid, auto, ...).",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    scope_segment_dedupe: Annotated[
        Optional[bool],
        typer.Option(
            "--scope-segment-dedupe/--no-scope-segment-dedupe",
            help="Enable dedupe for scope chunk emission.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    scope_chunk_target: Annotated[
        Optional[int],
        typer.Option(
            "--scope-chunk-target",
            help="Override scope chunk target (guards scanning depth).",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    scope_chunk_max_segments: Annotated[
        Optional[int],
        typer.Option(
            "--scope-chunk-max-segments",
            help="Upper bound on concurrent scope chunk segments.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    batch_order: Annotated[
        Optional[Literal["natural", "random", "hilbert"]],
        typer.Option(
            "--batch-order",
            help="Override insertion order (default: runtime config).",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    batch_order_seed: Annotated[
        Optional[int],
        typer.Option(
            "--batch-order-seed",
            help="Seed used when --batch-order=random.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    prefix_schedule: Annotated[
        Optional[Literal["doubling", "adaptive"]],
        typer.Option(
            "--prefix-schedule",
            help="Prefix-doubling schedule override.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    prefix_density_low: Annotated[
        Optional[float],
        typer.Option(
            "--prefix-density-low",
            help="Lower density bound for adaptive prefix.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    prefix_density_high: Annotated[
        Optional[float],
        typer.Option(
            "--prefix-density-high",
            help="Upper density bound for adaptive prefix.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    prefix_growth_small: Annotated[
        Optional[float],
        typer.Option(
            "--prefix-growth-small",
            help="Small-cluster growth factor.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    prefix_growth_mid: Annotated[
        Optional[float],
        typer.Option(
            "--prefix-growth-mid",
            help="Mid-cluster growth factor.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    prefix_growth_large: Annotated[
        Optional[float],
        typer.Option(
            "--prefix-growth-large",
            help="Large-cluster growth factor.",
            rich_help_panel=_RUNTIME_PANEL,
        ),
    ] = None,
    residual_lengthscale: Annotated[
        float,
        typer.Option(
            "--residual-lengthscale",
            help="Synthetic residual RBF lengthscale.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 1.0,
    residual_variance: Annotated[
        float,
        typer.Option(
            "--residual-variance",
            help="Synthetic residual RBF variance.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 1.0,
    residual_inducing: Annotated[
        int,
        typer.Option(
            "--residual-inducing",
            help="Number of inducing points in synthetic backend.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 512,
    residual_chunk_size: Annotated[
        int,
        typer.Option(
            "--residual-chunk-size",
            help="Residual kernel chunk size.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 512,
    residual_stream_tile: Annotated[
        Optional[int],
        typer.Option(
            "--residual-stream-tile",
            help="Tile size for dense scope streaming (default clamps to 64).",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 64,
    residual_force_whitened: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-force-whitened/--no-residual-force-whitened",
            help="Force SGEMM whitening even when the gate is off.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_scope_member_limit: Annotated[
        Optional[int],
        typer.Option(
            "--residual-scope-member-limit",
            help="Override residual scope membership cap (0 disables dense fallback).",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_scope_bitset: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-scope-bitset/--no-residual-scope-bitset",
            help="Bitset dedupe for dense residual scopes (default on).",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_dynamic_query_block: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-dynamic-query-block/--no-residual-dynamic-query-block",
            help="Prototype dynamic query-block sizing for residual traversal.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_dense_scope_streamer: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-dense-scope-streamer/--no-residual-dense-scope-streamer",
            help="Force dense scope streaming to scan each chunk once per batch.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_masked_scope_append: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-masked-scope-append/--no-residual-masked-scope-append",
            help="Use the Numba masked append path for dense scope streaming.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_scope_caps: Annotated[
        Optional[str],
        typer.Option(
            "--residual-scope-caps",
            help="JSON file describing per-level radius caps.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_scope_cap_default: Annotated[
        Optional[float],
        typer.Option(
            "--residual-scope-cap-default",
            help="Fallback radius cap when no per-level cap matches.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_scope_cap_output: Annotated[
        Optional[str],
        typer.Option(
            "--residual-scope-cap-output",
            help="Write derived per-level scope caps to this JSON file.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_scope_cap_percentile: Annotated[
        float,
        typer.Option(
            "--residual-scope-cap-percentile",
            help="Quantile (0-1) used when deriving scope caps.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 0.5,
    residual_scope_cap_margin: Annotated[
        float,
        typer.Option(
            "--residual-scope-cap-margin",
            help="Safety margin added to derived caps.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = 0.05,
    residual_radius_floor: Annotated[
        Optional[float],
        typer.Option(
            "--residual-radius-floor",
            help="Lower bound for residual scope radii.",
            rich_help_panel=_RESIDUAL_PANEL,
        ),
    ] = None,
    residual_gate: Annotated[
        Optional[Literal["off", "lookup"]],
        typer.Option(
            "--residual-gate",
            help="Residual gate preset (off keeps dense path).",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = "off",
    residual_gate_lookup_path: Annotated[
        str,
        typer.Option(
            "--residual-gate-lookup-path",
            help="Lookup JSON when --residual-gate=lookup.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = "docs/data/residual_gate_profile_diag0.json",
    residual_gate_margin: Annotated[
        float,
        typer.Option(
            "--residual-gate-margin",
            help="Lookup safety margin.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = 0.02,
    residual_gate_cap: Annotated[
        float,
        typer.Option(
            "--residual-gate-cap",
            help="Optional radius cap when using lookup presets.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = 0.0,
    residual_gate_alpha: Annotated[
        Optional[float],
        typer.Option(
            "--residual-gate-alpha",
            help="Manual override for Gate-1 alpha.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_eps: Annotated[
        Optional[float],
        typer.Option(
            "--residual-gate-eps",
            help="Gate-1 epsilon override.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_band_eps: Annotated[
        Optional[float],
        typer.Option(
            "--residual-gate-band-eps",
            help="Gate-1 band epsilon override.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_keep_pct: Annotated[
        Optional[float],
        typer.Option(
            "--residual-gate-keep-pct",
            help="Gate-1 keep percentage.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_prune_pct: Annotated[
        Optional[float],
        typer.Option(
            "--residual-gate-prune-pct",
            help="Gate-1 prune percentage.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_audit: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-gate-audit/--no-residual-gate-audit",
            help="Emit gate audit payloads.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_profile_path: Annotated[
        Optional[str],
        typer.Option(
            "--residual-gate-profile-path",
            help="Write gate profile samples to this JSON file.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_gate_profile_bins: Annotated[
        int,
        typer.Option(
            "--residual-gate-profile-bins",
            help="Histogram bins when recording gate profiles.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = 512,
    residual_gate_profile_log: Annotated[
        Optional[str],
        typer.Option(
            "--residual-gate-profile-log",
            help="Append profile metadata to this JSONL log.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_prefilter: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-prefilter/--no-residual-prefilter",
            help="Enable lookup-driven residual prefilter before traversal.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_prefilter_lookup_path: Annotated[
        Optional[str],
        typer.Option(
            "--residual-prefilter-lookup-path",
            help="Lookup JSON file for the residual prefilter.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_prefilter_margin: Annotated[
        Optional[float],
        typer.Option(
            "--residual-prefilter-margin",
            help="Safety margin for the residual prefilter.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_prefilter_radius_cap: Annotated[
        Optional[float],
        typer.Option(
            "--residual-prefilter-radius-cap",
            help="Radius cap when the prefilter is enabled.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    residual_prefilter_audit: Annotated[
        Optional[bool],
        typer.Option(
            "--residual-prefilter-audit/--no-residual-prefilter-audit",
            help="Emit prefilter audit payloads.",
            rich_help_panel=_GATE_PANEL,
        ),
    ] = None,
    baseline: Annotated[
        Literal["none", "sequential", "gpboost", "external", "both", "all"],
        typer.Option(
            "--baseline",
            help="Run optional baseline comparisons.",
            rich_help_panel=_TELEMETRY_PANEL,
        ),
    ] = "none",
    log_file: Annotated[
        Optional[str],
        typer.Option(
            "--log-file",
            help="Write per-batch telemetry JSONL to this path.",
            rich_help_panel=_TELEMETRY_PANEL,
        ),
    ] = None,
    no_log_file: Annotated[
        bool,
        typer.Option(
            "--no-log-file",
            help="Disable JSONL telemetry emission (not recommended).",
            rich_help_panel=_TELEMETRY_PANEL,
        ),
    ] = False,
    build_mode: Annotated[
        Literal["batch", "prefix"],
        typer.Option(
            "--build-mode",
            help="Choose between batch or prefix-doubling construction.",
            rich_help_panel=_SHAPE_PANEL,
        ),
    ] = "batch",
) -> None:
    options = QueryCLIOptions(
        dimension=dimension,
        tree_points=tree_points,
        batch_size=batch_size,
        queries=queries,
        k=k,
        seed=seed,
        run_id=run_id,
        metric=metric,
        backend=backend,
        precision=precision,
        devices=tuple(devices) if devices else None,
        enable_numba=enable_numba,
        enable_sparse_traversal=enable_sparse_traversal,
        diagnostics=diagnostics,
        log_level=log_level,
        mis_seed=mis_seed,
        conflict_graph=conflict_graph,
        scope_segment_dedupe=scope_segment_dedupe,
        scope_chunk_target=scope_chunk_target,
        scope_chunk_max_segments=scope_chunk_max_segments,
        batch_order=batch_order,
        batch_order_seed=batch_order_seed,
        prefix_schedule=prefix_schedule,
        prefix_density_low=prefix_density_low,
        prefix_density_high=prefix_density_high,
        prefix_growth_small=prefix_growth_small,
        prefix_growth_mid=prefix_growth_mid,
        prefix_growth_large=prefix_growth_large,
        residual_lengthscale=residual_lengthscale,
        residual_variance=residual_variance,
        residual_inducing=residual_inducing,
        residual_chunk_size=residual_chunk_size,
        residual_stream_tile=residual_stream_tile,
        residual_force_whitened=residual_force_whitened,
        residual_scope_member_limit=residual_scope_member_limit,
        residual_scope_bitset=residual_scope_bitset,
        residual_dynamic_query_block=residual_dynamic_query_block,
        residual_dense_scope_streamer=residual_dense_scope_streamer,
        residual_scope_caps=residual_scope_caps,
        residual_scope_cap_default=residual_scope_cap_default,
        residual_scope_cap_output=residual_scope_cap_output,
        residual_scope_cap_percentile=residual_scope_cap_percentile,
        residual_scope_cap_margin=residual_scope_cap_margin,
        residual_radius_floor=residual_radius_floor,
        residual_gate=residual_gate,
        residual_gate_lookup_path=residual_gate_lookup_path,
        residual_gate_margin=residual_gate_margin,
        residual_gate_cap=residual_gate_cap,
        residual_gate_alpha=residual_gate_alpha,
        residual_gate_eps=residual_gate_eps,
        residual_gate_band_eps=residual_gate_band_eps,
        residual_gate_keep_pct=residual_gate_keep_pct,
        residual_gate_prune_pct=residual_gate_prune_pct,
        residual_gate_audit=residual_gate_audit,
        residual_gate_profile_path=residual_gate_profile_path,
        residual_gate_profile_bins=residual_gate_profile_bins,
        residual_gate_profile_log=residual_gate_profile_log,
        residual_prefilter=residual_prefilter,
        residual_prefilter_lookup_path=residual_prefilter_lookup_path,
        residual_prefilter_margin=residual_prefilter_margin,
        residual_prefilter_radius_cap=residual_prefilter_radius_cap,
        residual_prefilter_audit=residual_prefilter_audit,
        baseline=baseline,
        log_file=log_file,
        no_log_file=no_log_file,
        build_mode=build_mode,
    )
    ctx.obj = options
    if ctx.invoked_subcommand is None:
        run_queries(options)


def _validate_residual_runtime(snapshot: Mapping[str, Any]) -> None:
    errors = []
    backend = snapshot.get("backend")
    if backend != "numpy":
        errors.append(f"expected backend 'numpy' but runtime selected {backend!r}")
    if not snapshot.get("enable_numba"):
        errors.append(
            "Numba acceleration is required for residual traversal; enable it by leaving "
            "COVERTREEX_ENABLE_NUMBA unset or passing --enable-numba=1."
        )
    if errors:
        joined = "\n - ".join(errors)
        raise RuntimeError(
            "Residual metric requires the residual traversal engine; unable to satisfy the"
            f" prerequisites:\n - {joined}"
        )


def run_queries(options: QueryCLIOptions) -> None:
    args = options
    run_id = args.run_id or generate_run_id()
    if args.residual_gate and args.metric != "residual":
        raise ValueError("--residual-gate presets are only supported when --metric residual is selected.")

    if args.metric == "residual":
        profile_log_path = _resolve_artifact_arg(args.residual_gate_profile_log, category="profiles")
        if args.residual_gate_profile_path:
            profile_json_path = _resolve_artifact_arg(
                args.residual_gate_profile_path,
                category="profiles",
            )
        elif profile_log_path:
            profile_json_path = str(
                timestamped_artifact(
                    category="profiles",
                    prefix=f"residual_gate_profile_{run_id}",
                    suffix=".json",
                )
            )
        else:
            profile_json_path = None
        args.residual_gate_profile_path = profile_json_path
        args.residual_gate_profile_log = profile_log_path
    else:
        args.residual_gate_profile_path = None
        args.residual_gate_profile_log = None

    _ensure_thread_env_defaults()
    cli_runtime = runtime_from_args(args)
    runtime_snapshot = dict(cli_runtime.describe())
    if args.metric == "residual":
        _validate_residual_runtime(runtime_snapshot)
    cli_runtime.activate()
    thread_snapshot = _thread_env_snapshot()
    runtime_snapshot["runtime_blas_threads"] = thread_snapshot["blas_threads"]
    runtime_snapshot["runtime_numba_threads"] = thread_snapshot["numba_threads"]
    engine_label = "euclidean_dense"
    gate_flag = False
    if args.metric != "residual":
        runtime_snapshot["runtime_traversal_engine"] = engine_label
        runtime_snapshot["runtime_gate_active"] = gate_flag
        _emit_engine_banner(engine_label, gate_flag, thread_snapshot)
    log_metadata = {
        "benchmark": "cli.queries",
        "dimension": args.dimension,
        "tree_points": args.tree_points,
        "batch_size": args.batch_size,
        "queries": args.queries,
        "k": args.k,
        "metric": args.metric,
        "build_mode": args.build_mode,
        "baseline": args.baseline,
    }
    telemetry_handles: CLITelemetryHandles | None = None
    log_writer = None
    log_path: str | None = None
    scope_cap_recorder = None
    telemetry_view: ResidualTraversalTelemetry | None = None

    try:
        telemetry_handles = initialise_cli_telemetry(
            args=args,
            run_id=run_id,
            runtime_snapshot=runtime_snapshot,
            log_metadata=log_metadata,
        )
        log_writer = telemetry_handles.log_writer
        log_path = telemetry_handles.log_path
        scope_cap_recorder = telemetry_handles.scope_cap_recorder
        telemetry_view = telemetry_handles.traversal_view

        point_rng = default_rng(args.seed)
        points_np = gaussian_points(point_rng, args.tree_points, args.dimension, dtype=np.float64)
        query_rng = default_rng(args.seed + 1)
        queries_np = gaussian_points(query_rng, args.queries, args.dimension, dtype=np.float64)

        if args.metric == "residual":
            residual_backend = build_residual_backend(
                points_np,
                seed=args.seed,
                inducing_count=args.residual_inducing,
                variance=args.residual_variance,
                lengthscale=args.residual_lengthscale,
                chunk_size=args.residual_chunk_size,
            )
            configure_residual_correlation(residual_backend)
            gate_flag = _gate_active_for_backend(residual_backend)
            engine_label = "residual_serial" if gate_flag else "residual_parallel"
            runtime_snapshot["runtime_traversal_engine"] = engine_label
            runtime_snapshot["runtime_gate_active"] = gate_flag
            _emit_engine_banner(engine_label, gate_flag, thread_snapshot)
        else:
            runtime_snapshot.setdefault("runtime_traversal_engine", engine_label)
            runtime_snapshot.setdefault("runtime_gate_active", gate_flag)

        tree, result = benchmark_knn_latency(
            dimension=args.dimension,
            tree_points=args.tree_points,
            query_count=args.queries,
            k=args.k,
            batch_size=args.batch_size,
            seed=args.seed,
            prebuilt_points=points_np,
            prebuilt_queries=queries_np,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            build_mode=args.build_mode,
            plan_callback=telemetry_view.observe_plan if telemetry_view is not None else None,
        )

        print(
            f"pcct | build={result.build_seconds:.4f}s "
            f"queries={result.queries} k={result.k} "
            f"time={result.elapsed_seconds:.4f}s "
            f"latency={result.latency_ms:.4f}ms "
            f"throughput={result.queries_per_second:,.1f} q/s"
        )

        if args.baseline != "none":
            baseline_results = run_baseline_comparisons(
                points_np,
                queries_np,
                k=args.k,
                mode=args.baseline,
            )
            for baseline in baseline_results:
                slowdown = (
                    baseline.latency_ms / result.latency_ms if result.latency_ms else float("inf")
                )

                print(
                    f"baseline[{baseline.name}] | build={baseline.build_seconds:.4f}s "
                    f"time={baseline.elapsed_seconds:.4f}s "
                    f"latency={baseline.latency_ms:.4f}ms "
                    f"throughput={baseline.queries_per_second:,.1f} q/s "
                    f"slowdown={slowdown:.3f}x"
                )
        if telemetry_view is not None and telemetry_view.has_data:
            for line in telemetry_view.render_summary():
                print(line)
    finally:
        reset_residual_metric()
        if args.metric == "residual":
            _append_gate_profile_log(
                profile_json_path=args.residual_gate_profile_path,
                profile_log_path=args.residual_gate_profile_log,
                run_id=run_id,
                log_metadata=log_metadata,
                runtime_snapshot=runtime_snapshot,
                batch_log_path=log_path,
            )
        cx_config.reset_runtime_context()
        if telemetry_handles is not None:
            telemetry_handles.close()


def main() -> None:
    app()


__all__ = ["main"]
