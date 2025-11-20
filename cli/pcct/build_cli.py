from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import typer

from .support.benchmark_utils import build_tree as _build_tree
from .support.runtime_utils import resolve_artifact_arg as _resolve_artifact_arg
from covertreex.core.tree import PCCTree

from . import option_defs as opts
from .execution import benchmark_run
from .options import QueryCLIOptions, resolve_metric_flag

build_app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    invoke_without_command=True,
    help="Construct PCCTrees with profile-driven runtime settings.",
)


def _export_tree_npz(tree: PCCTree, path: str) -> None:
    backend = tree.backend
    payload = {
        "points": np.asarray(backend.to_numpy(tree.points), dtype=np.float64),
        "top_levels": np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int32),
        "parents": np.asarray(backend.to_numpy(tree.parents), dtype=np.int32),
        "children": np.asarray(backend.to_numpy(tree.children), dtype=np.int32),
        "level_offsets": np.asarray(backend.to_numpy(tree.level_offsets), dtype=np.int64),
        "si_cache": np.asarray(backend.to_numpy(tree.si_cache), dtype=np.float64),
        "next_cache": np.asarray(backend.to_numpy(tree.next_cache), dtype=np.int32),
    }
    np.savez(path, **payload)


@build_app.callback()
def build(
    ctx: typer.Context,
    dimension: opts.DimensionOption = 8,
    tree_points: opts.TreePointsOption = 16_384,
    batch_size: opts.BatchSizeOption = 512,
    seed: opts.SeedOption = 0,
    run_id: opts.RunIdOption = None,
    profile: opts.ProfileOption = None,
    set_override: opts.SetOverrideOption = None,
    metric: opts.MetricOption = "auto",
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
    export_tree: opts.ExportTreeOption = None,
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
    ctx.params["run_id"] = run_id
    ctx.params["queries"] = 0
    ctx.params["k"] = 1
    ctx.params["baseline"] = "none"
    namespace = SimpleNamespace(**ctx.params)
    options = QueryCLIOptions.from_namespace(namespace)
    log_metadata = {
        "benchmark": "pcct.build",
        "profile": profile,
        "dimension": dimension,
        "tree_points": tree_points,
        "batch_size": batch_size,
        "build_mode": build_mode,
        "metric": options.metric,
    }
    with benchmark_run(options, benchmark="pcct.build", metadata=log_metadata) as run:
        tree, _, build_seconds = _build_tree(
            dimension=dimension,
            tree_points=tree_points,
            batch_size=batch_size,
            seed=seed,
            log_writer=run.log_writer,
            scope_cap_recorder=run.scope_cap_recorder,
            build_mode=build_mode,
            plan_callback=run.telemetry_view.observe_plan if run.telemetry_view is not None else None,
            context=run.context,
        )
        typer.echo(
            f"pcct build | points={tree_points} dimension={dimension} "
            f"build={build_seconds:.4f}s mode={build_mode}"
        )
        if export_tree:
            export_path = _resolve_artifact_arg(export_tree, category="trees")
            _export_tree_npz(tree, export_path)
            typer.echo(f"[artifacts] tree exported to {export_path}")


__all__ = ["build_app"]