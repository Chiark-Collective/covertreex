from __future__ import annotations

import statistics
from dataclasses import replace
from types import SimpleNamespace
from typing import List, Optional

import typer

from . import option_defs as opts
from .execution import benchmark_run
from .options import QueryCLIOptions, resolve_metric_flag
from .query import execute_query_benchmark

benchmark_app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    invoke_without_command=True,
    help="Repeat PCCT queries to gather aggregate latency statistics.",
)


@benchmark_app.callback()
def benchmark(
    ctx: typer.Context,
    dimension: opts.DimensionOption = 8,
    tree_points: opts.TreePointsOption = 16_384,
    batch_size: opts.BatchSizeOption = 512,
    queries: opts.QueriesOption = 1_024,
    k: opts.KOption = 8,
    seed: opts.SeedOption = 0,
    repeat: opts.RepeatOption = 3,
    seed_step: opts.SeedStepOption = 1,
    run_id: opts.RunIdOption = None,
    profile: opts.ProfileOption = None,
    set_override: opts.SetOverrideOption = None,
    metric: opts.MetricOption = "auto",
    baseline: opts.BaselineOption = "none",
    build_mode: opts.BuildModeOption = "batch",
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
    log_file: opts.LogFileOption = None,
    no_log_file: opts.NoLogFileOption = False,
) -> None:
    if repeat <= 0:
        raise typer.BadParameter("--repeat must be positive.")
    if seed_step <= 0:
        raise typer.BadParameter("--seed-step must be positive.")
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
    ctx.params["run_id"] = run_id
    namespace = SimpleNamespace(**ctx.params)
    base_options = QueryCLIOptions.from_namespace(namespace)

    latencies_ms: List[float] = []
    build_seconds: List[float] = []
    log_paths: List[str] = []

    for iteration in range(repeat):
        iteration_seed = seed + iteration * seed_step
        iteration_run_id = f"{run_id}-i{iteration+1}" if run_id else None
        options = replace(base_options, seed=iteration_seed, run_id=iteration_run_id)
        log_metadata = {
            "benchmark": "pcct.benchmark",
            "profile": profile,
            "dimension": dimension,
            "tree_points": tree_points,
            "batch_size": batch_size,
            "queries": queries,
            "k": k,
            "iteration": iteration,
            "build_mode": build_mode,
            "baseline": baseline,
            "metric": options.metric,
        }
        with benchmark_run(options, benchmark="pcct.benchmark", metadata=log_metadata) as run:
            result = execute_query_benchmark(options, run)
            latencies_ms.append(result.latency_ms)
            if result.build_seconds is not None:
                build_seconds.append(result.build_seconds)
            if run.log_path:
                log_paths.append(run.log_path)

    if not latencies_ms:
        return

    mean_latency = statistics.mean(latencies_ms)
    median_latency = statistics.median(latencies_ms)
    min_latency = min(latencies_ms)
    summary = (
        f"pcct benchmark | runs={repeat} mean_latency={mean_latency:.4f}ms "
        f"median={median_latency:.4f}ms best={min_latency:.4f}ms"
    )
    if build_seconds:
        summary += f" mean_build={statistics.mean(build_seconds):.4f}s"
    typer.echo(summary)
    if log_paths:
        typer.echo(f"[artifacts] batch telemetry files: {', '.join(log_paths)}")


__all__ = ["benchmark_app"]
