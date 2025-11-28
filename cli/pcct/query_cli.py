from __future__ import annotations

from types import SimpleNamespace

import typer

from . import option_defs as opts
from .execution import benchmark_run
from .options import QueryCLIOptions, resolve_metric_flag
from .query import execute_query_benchmark

_QUERY_HELP = """Run k-NN benchmark with tree construction and query throughput measurement.

[bold cyan]Examples[/bold cyan]

  [dim]#[/dim] Basic Euclidean benchmark (default metric)
  python -m cli.pcct query --dimension 3 --tree-points 8192 --k 10

  [dim]#[/dim] High-performance Rust backend (recommended for production)
  python -m cli.pcct query --engine rust-hilbert --tree-points 32768 --k 50

  [dim]#[/dim] Residual correlation metric for GP workloads
  python -m cli.pcct query --metric residual --engine rust-hilbert \\
      --tree-points 32768 --dimension 3 --k 50

  [dim]#[/dim] Load a profile preset with overrides
  python -m cli.pcct query --profile residual-gold --set seeds.global_seed=42

  [dim]#[/dim] Compare against baseline implementations
  python -m cli.pcct query --baseline scipy,sklearn --tree-points 4096

[bold cyan]Key Options[/bold cyan]

  [bold]--engine[/bold]      Execution backend: python-numba, rust-natural, rust-hilbert
  [bold]--metric[/bold]      Distance metric: euclidean, residual, residual-lite
  [bold]--profile[/bold]     Load YAML preset (see: profile list)
  [bold]--tree-points[/bold] Number of points to insert into tree
  [bold]--k[/bold]           Number of nearest neighbors per query

[bold cyan]Output[/bold cyan]

  Reports tree construction time (sec), query throughput (queries/sec),
  and median query latency. Telemetry written to logs/ by default."""

query_app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
    help=_QUERY_HELP,
)

@query_app.callback()
def query(
    ctx: typer.Context,
    dimension: opts.DimensionOption = 8,
    tree_points: opts.TreePointsOption = 16_384,
    batch_size: opts.BatchSizeOption = 512,
    queries: opts.QueriesOption = 1_024,
    k: opts.KOption = 8,
    seed: opts.SeedOption = 0,
    run_id: opts.RunIdOption = None,
    profile: opts.ProfileOption = None,
    set_override: opts.SetOverrideOption = None,
    metric: opts.MetricOption = "auto",
    engine: opts.EngineOption = None,
    backend: opts.BackendOption = None,
    precision: opts.PrecisionOption = None,
    devices: opts.DevicesOption = None,
    enable_numba: opts.EnableNumbaOption = None,
    enable_sparse_traversal: opts.EnableSparseTraversalOption = None,
    diagnostics: opts.DiagnosticsOption = None,
    log_level: opts.LogLevelOption = None,
    global_seed: opts.GlobalSeedOption = None,
    mis_seed: opts.MisSeedOption = None,
    conflict_graph: opts.ConflictGraphOption = None,
    scope_segment_dedupe: opts.ScopeSegmentDedupeOption = None,
    scope_chunk_target: opts.ScopeChunkTargetOption = None,
    scope_chunk_max_segments: opts.ScopeChunkMaxSegmentsOption = None,
    scope_chunk_pair_merge: opts.ScopeChunkPairMergeOption = None,
    scope_conflict_buffer_reuse: opts.ScopeConflictBufferReuseOption = None,
    degree_cap: opts.DegreeCapOption = None,
    batch_order: opts.BatchOrderOption = None,
    batch_order_seed: opts.BatchOrderSeedOption = None,
    residual_grid_seed: opts.ResidualGridSeedOption = None,
    prefix_schedule: opts.PrefixScheduleOption = None,
    prefix_density_low: opts.PrefixDensityLowOption = None,
    prefix_density_high: opts.PrefixDensityHighOption = None,
    prefix_growth_small: opts.PrefixGrowthSmallOption = None,
    prefix_growth_mid: opts.PrefixGrowthMidOption = None,
    prefix_growth_large: opts.PrefixGrowthLargeOption = None,
    residual_lengthscale: opts.ResidualLengthscaleOption = 1.0,
    residual_variance: opts.ResidualVarianceOption = 1.0,
    residual_kernel_type: opts.ResidualKernelTypeOption = 0,
    residual_inducing: opts.ResidualInducingOption = 512,
    residual_chunk_size: opts.ResidualChunkSizeOption = 512,
    residual_stream_tile: opts.ResidualStreamTileOption = 64,
    residual_force_whitened: opts.ResidualForceWhitenedOption = None,
    residual_scope_member_limit: opts.ResidualScopeMemberLimitOption = None,
    residual_scope_bitset: opts.ResidualScopeBitsetOption = None,
    residual_dynamic_query_block: opts.ResidualDynamicQueryBlockOption = None,
    residual_dense_scope_streamer: opts.ResidualDenseScopeStreamerOption = None,
    residual_masked_scope_append: opts.ResidualMaskedScopeAppendOption = None,
    residual_level_cache_batching: opts.ResidualLevelCacheBatchingOption = None,
    residual_scope_caps: opts.ResidualScopeCapsOption = None,
    residual_scope_cap_default: opts.ResidualScopeCapDefaultOption = None,
    residual_scope_cap_output: opts.ResidualScopeCapOutputOption = None,
    residual_scope_cap_percentile: opts.ResidualScopeCapPercentileOption = 0.5,
    residual_scope_cap_margin: opts.ResidualScopeCapMarginOption = 0.05,
    residual_radius_floor: opts.ResidualRadiusFloorOption = None,
    residual_prefilter: opts.ResidualPrefilterOption = None,
    residual_prefilter_lookup_path: opts.ResidualPrefilterLookupPathOption = None,
    residual_prefilter_margin: opts.ResidualPrefilterMarginOption = None,
    residual_prefilter_radius_cap: opts.ResidualPrefilterRadiusCapOption = None,
    residual_prefilter_audit: opts.ResidualPrefilterAuditOption = None,
    baseline: opts.BaselineOption = "none",
    log_file: opts.LogFileOption = None,
    no_log_file: opts.NoLogFileOption = False,
    build_mode: opts.BuildModeOption = "batch",
    predecessor_mode: opts.PredecessorModeOption = None,
) -> None:
    if set_override and not profile:
        raise typer.BadParameter("--set overrides require --profile.")
    if profile:
        override_list = list(set_override or [])
        if global_seed is not None:
            override_list.append(f"seeds.global_seed={global_seed}")
        if residual_grid_seed is not None:
            override_list.append(f"seeds.residual_grid={residual_grid_seed}")
        overrides = override_list or None
    else:
        overrides = set_override or None
    try:
        resolved_metric = resolve_metric_flag(metric, profile=profile, overrides=overrides)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    ctx.params["metric"] = resolved_metric
    ctx.params["set_override"] = overrides
    ctx.params["profile"] = profile
    namespace = SimpleNamespace(**ctx.params)
    options = QueryCLIOptions.from_namespace(namespace)

    log_metadata = {
        "benchmark": "pcct.query",
        "profile": options.profile,
        "dimension": options.dimension,
        "tree_points": options.tree_points,
        "batch_size": options.batch_size,
        "queries": options.queries,
        "k": options.k,
        "build_mode": options.build_mode,
        "baseline": options.baseline,
        "metric": options.metric,
    }
    with benchmark_run(options, benchmark="pcct.query", metadata=log_metadata) as run:
        execute_query_benchmark(options, run)


__all__ = ["query_app"]
