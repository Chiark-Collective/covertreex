from __future__ import annotations

from typing import List, Literal, Optional

import typer
from typing_extensions import Annotated

from .help_panels import GATE_PANEL, RESIDUAL_PANEL, RUNTIME_PANEL, SHAPE_PANEL, TELEMETRY_PANEL


# Benchmark shape
DimensionOption = Annotated[
    int,
    typer.Option("--dimension", help="Dimensionality of tree/query points.", rich_help_panel=SHAPE_PANEL),
]
TreePointsOption = Annotated[
    int,
    typer.Option("--tree-points", help="Number of tree points to insert.", rich_help_panel=SHAPE_PANEL),
]
BatchSizeOption = Annotated[
    int,
    typer.Option("--batch-size", help="Batch size for insertions.", rich_help_panel=SHAPE_PANEL),
]
QueriesOption = Annotated[
    int,
    typer.Option("--queries", help="Number of queries per run.", rich_help_panel=SHAPE_PANEL),
]
KOption = Annotated[
    int,
    typer.Option("--k", help="Nearest neighbours requested per query.", rich_help_panel=SHAPE_PANEL),
]
SeedOption = Annotated[
    int,
    typer.Option("--seed", help="Base random seed for dataset generation.", rich_help_panel=SHAPE_PANEL),
]
BuildModeOption = Annotated[
    Literal["batch", "prefix"],
    typer.Option("--build-mode", help="Choose between batch or prefix-doubling construction.", rich_help_panel=SHAPE_PANEL),
]
PredecessorModeOption = Annotated[
    Optional[bool],
    typer.Option(
        "--predecessor-mode/--no-predecessor-mode",
        help="Enable Vecchia constraint: neighbor j < query i.",
        rich_help_panel=SHAPE_PANEL,
    ),
]

# Telemetry
RunIdOption = Annotated[
    Optional[str],
    typer.Option("--run-id", help="Optional run identifier propagated to telemetry artifacts.", rich_help_panel=TELEMETRY_PANEL),
]
BaselineOption = Annotated[
    str,
    typer.Option("--baseline", help="Optional baseline comparison to run alongside PCCT (comma-separated).", rich_help_panel=TELEMETRY_PANEL),
]
LogFileOption = Annotated[
    Optional[str],
    typer.Option("--log-file", help="Write per-batch telemetry to the given path.", rich_help_panel=TELEMETRY_PANEL),
]
NoLogFileOption = Annotated[
    bool,
    typer.Option("--no-log-file", help="Disable telemetry emission (not recommended).", rich_help_panel=TELEMETRY_PANEL),
]
ExportTreeOption = Annotated[
    Optional[str],
    typer.Option(
        "--export-tree",
        help="Persist the constructed tree as a .npz artifact (path resolved under artifacts/trees).",
        rich_help_panel=TELEMETRY_PANEL,
    ),
]
RepeatOption = Annotated[
    int,
    typer.Option("--repeat", help="Number of benchmark iterations to execute.", rich_help_panel=TELEMETRY_PANEL),
]
SeedStepOption = Annotated[
    int,
    typer.Option("--seed-step", help="Seed increment applied between iterations.", rich_help_panel=TELEMETRY_PANEL),
]

# Runtime controls
ProfileOption = Annotated[
    Optional[str],
    typer.Option(
        "--profile",
        help="YAML preset: default, residual-gold, residual-fast, residual-audit, cpu-debug. See: profile list",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
SetOverrideOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--set",
        metavar="PATH=VALUE",
        help="Apply dot-path overrides (repeatable) when loading the profile.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
MetricOption = Annotated[
    Literal["auto", "euclidean", "residual", "residual-lite"],
    typer.Option(
        "--metric",
        help="Distance metric: euclidean (standard), residual (GP correlation), residual-lite (faster residual), auto (infer from profile).",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
EngineOption = Annotated[
    Optional[
        Literal[
            "python-numba",
            "rust-natural",
            "rust-hybrid",
            "rust-hilbert",
            "rust-fast",
        ]
    ],
    typer.Option(
        "--engine",
        help="Execution engine: python-numba (reference), rust-natural (fast), rust-hilbert (fastest, recommended).",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
BackendOption = Annotated[
    Optional[str],
    typer.Option("--backend", help="Runtime backend override (numpy, jax, ...).", rich_help_panel=RUNTIME_PANEL),
]
PrecisionOption = Annotated[
    Optional[str],
    typer.Option("--precision", help="Backend precision override (float32, float64, ...).", rich_help_panel=RUNTIME_PANEL),
]
DevicesOption = Annotated[
    Optional[List[str]],
    typer.Option("--device", "-d", help="Restrict execution to specific logical devices.", rich_help_panel=RUNTIME_PANEL),
]
EnableNumbaOption = Annotated[
    Optional[bool],
    typer.Option("--enable-numba/--disable-numba", help="Force-enable or disable Numba kernels.", rich_help_panel=RUNTIME_PANEL),
]
EnableSparseTraversalOption = Annotated[
    Optional[bool],
    typer.Option(
        "--enable-sparse-traversal/--disable-sparse-traversal",
        help="Toggle sparse traversal engines when supported.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
DiagnosticsOption = Annotated[
    Optional[bool],
    typer.Option(
        "--enable-diagnostics/--disable-diagnostics",
        help="Control resource polling + diagnostic logging.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
LogLevelOption = Annotated[
    Optional[str],
    typer.Option("--log-level", help="Override runtime log level.", rich_help_panel=RUNTIME_PANEL),
]
MisSeedOption = Annotated[
    Optional[int],
    typer.Option("--mis-seed", help="Sticky MIS seed when reproducing deterministic traversals.", rich_help_panel=RUNTIME_PANEL),
]
GlobalSeedOption = Annotated[
    Optional[int],
    typer.Option(
        "--global-seed",
        help="Base SeedPack seed applied when per-channel seeds are omitted.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
ConflictGraphOption = Annotated[
    Optional[str],
    typer.Option("--conflict-graph", help="Conflict graph implementation (dense, grid, auto, ...).", rich_help_panel=RUNTIME_PANEL),
]
ScopeSegmentDedupeOption = Annotated[
    Optional[bool],
    typer.Option(
        "--scope-segment-dedupe/--no-scope-segment-dedupe",
        help="Enable dedupe for scope chunk emission.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
ScopeChunkTargetOption = Annotated[
    Optional[int],
    typer.Option("--scope-chunk-target", help="Override scope chunk target (guards scanning depth).", rich_help_panel=RUNTIME_PANEL),
]
ScopeChunkMaxSegmentsOption = Annotated[
    Optional[int],
    typer.Option(
        "--scope-chunk-max-segments",
        help="Upper bound on concurrent scope chunk segments.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
ScopeChunkPairMergeOption = Annotated[
    Optional[bool],
    typer.Option(
        "--scope-chunk-pair-merge/--no-scope-chunk-pair-merge",
        help="Merge scope chunks based on pair-count heuristics.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
ScopeConflictBufferReuseOption = Annotated[
    Optional[bool],
    typer.Option(
        "--scope-conflict-buffer-reuse/--no-scope-conflict-buffer-reuse",
        help="Reuse conflict-builder buffers when using Numba scopes.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
DegreeCapOption = Annotated[
    Optional[int],
    typer.Option("--degree-cap", help="Limit conflict-graph degree per node (0 disables).", rich_help_panel=RUNTIME_PANEL),
]
BatchOrderOption = Annotated[
    Optional[Literal["natural", "random", "hilbert"]],
    typer.Option("--batch-order", help="Override insertion order.", rich_help_panel=RUNTIME_PANEL),
]
BatchOrderSeedOption = Annotated[
    Optional[int],
    typer.Option("--batch-order-seed", help="Seed used when --batch-order=random.", rich_help_panel=RUNTIME_PANEL),
]
ResidualGridSeedOption = Annotated[
    Optional[int],
    typer.Option(
        "--residual-grid-seed",
        help="Seed used by residual grid leader selection.",
        rich_help_panel=RUNTIME_PANEL,
    ),
]
PrefixScheduleOption = Annotated[
    Optional[Literal["doubling", "adaptive"]],
    typer.Option("--prefix-schedule", help="Prefix-doubling schedule override.", rich_help_panel=RUNTIME_PANEL),
]
PrefixDensityLowOption = Annotated[
    Optional[float],
    typer.Option("--prefix-density-low", help="Lower density bound for adaptive prefix.", rich_help_panel=RUNTIME_PANEL),
]
PrefixDensityHighOption = Annotated[
    Optional[float],
    typer.Option("--prefix-density-high", help="Upper density bound for adaptive prefix.", rich_help_panel=RUNTIME_PANEL),
]
PrefixGrowthSmallOption = Annotated[
    Optional[float],
    typer.Option("--prefix-growth-small", help="Small-cluster growth factor.", rich_help_panel=RUNTIME_PANEL),
]
PrefixGrowthMidOption = Annotated[
    Optional[float],
    typer.Option("--prefix-growth-mid", help="Mid-cluster growth factor.", rich_help_panel=RUNTIME_PANEL),
]
PrefixGrowthLargeOption = Annotated[
    Optional[float],
    typer.Option("--prefix-growth-large", help="Large-cluster growth factor.", rich_help_panel=RUNTIME_PANEL),
]

# Residual metric
ResidualLengthscaleOption = Annotated[
    float,
    typer.Option("--residual-lengthscale", help="Synthetic residual RBF lengthscale.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualVarianceOption = Annotated[
    float,
    typer.Option("--residual-variance", help="Synthetic residual RBF variance.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualKernelTypeOption = Annotated[
    Optional[int],
    typer.Option("--residual-kernel-type", help="GP kernel: 0=RBF (squared exponential), 1=Matérn 5/2.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualInducingOption = Annotated[
    int,
    typer.Option("--residual-inducing", help="Number of inducing points in the synthetic residual backend.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualChunkSizeOption = Annotated[
    int,
    typer.Option("--residual-chunk-size", help="Residual kernel chunk size.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualStreamTileOption = Annotated[
    Optional[int],
    typer.Option(
        "--residual-stream-tile",
        help="Tile size for dense scope streaming (None lets the runtime decide).",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualForceWhitenedOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-force-whitened/--no-residual-force-whitened",
        help="Force SGEMM whitening even when the gate is off.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualScopeMemberLimitOption = Annotated[
    Optional[int],
    typer.Option(
        "--residual-scope-member-limit",
        help="Override residual scope membership cap (0 disables dense fallback).",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualScopeBitsetOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-scope-bitset/--no-residual-scope-bitset",
        help="Bitset dedupe for dense residual scopes.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualDynamicQueryBlockOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-dynamic-query-block/--no-residual-dynamic-query-block",
        help="Prototype dynamic query-block sizing for residual traversal.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualDenseScopeStreamerOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-dense-scope-streamer/--no-residual-dense-scope-streamer",
        help="Force dense scope streaming to scan each chunk once per batch.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualMaskedScopeAppendOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-masked-scope-append/--no-residual-masked-scope-append",
        help="Use the Numba masked append path for dense scope streaming.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualLevelCacheBatchingOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-level-cache-batching/--no-residual-level-cache-batching",
        help="Batch level-scope cache prefetching + parent-chain insertion.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualScopeCapsOption = Annotated[
    Optional[str],
    typer.Option("--residual-scope-caps", help="JSON file describing per-level radius caps.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualScopeCapDefaultOption = Annotated[
    Optional[float],
    typer.Option(
        "--residual-scope-cap-default",
        help="Fallback radius cap when no per-level cap matches.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualScopeCapOutputOption = Annotated[
    Optional[str],
    typer.Option(
        "--residual-scope-cap-output",
        help="Write derived per-level scope caps to this JSON file.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualScopeCapPercentileOption = Annotated[
    float,
    typer.Option(
        "--residual-scope-cap-percentile",
        help="Quantile (0-1) used when deriving scope caps.",
        rich_help_panel=RESIDUAL_PANEL,
    ),
]
ResidualScopeCapMarginOption = Annotated[
    float,
    typer.Option("--residual-scope-cap-margin", help="Margin added to derived scope caps.", rich_help_panel=RESIDUAL_PANEL),
]
ResidualRadiusFloorOption = Annotated[
    Optional[float],
    typer.Option("--residual-radius-floor", help="Lower bound for residual scope radii.", rich_help_panel=RESIDUAL_PANEL),
]

# Gate & prefilter
ResidualPrefilterOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-prefilter/--no-residual-prefilter",
        help="Enable the residual prefilter.",
        rich_help_panel=GATE_PANEL,
    ),
]
ResidualPrefilterLookupPathOption = Annotated[
    Optional[str],
    typer.Option(
        "--residual-prefilter-lookup-path",
        help="Prefilter lookup JSON when the prefilter is enabled.",
        rich_help_panel=GATE_PANEL,
    ),
]
ResidualPrefilterMarginOption = Annotated[
    Optional[float],
    typer.Option("--residual-prefilter-margin", help="Safety margin for the residual prefilter.", rich_help_panel=GATE_PANEL),
]
ResidualPrefilterRadiusCapOption = Annotated[
    Optional[float],
    typer.Option(
        "--residual-prefilter-radius-cap",
        help="Radius cap when the prefilter is enabled.",
        rich_help_panel=GATE_PANEL,
    ),
]
ResidualPrefilterAuditOption = Annotated[
    Optional[bool],
    typer.Option(
        "--residual-prefilter-audit/--no-residual-prefilter-audit",
        help="Emit prefilter audit payloads.",
        rich_help_panel=GATE_PANEL,
    ),
]

__all__ = [
    "DimensionOption",
    "TreePointsOption",
    "BatchSizeOption",
    "QueriesOption",
    "KOption",
    "SeedOption",
    "BuildModeOption",
    "PredecessorModeOption",
    "RunIdOption",
    "BaselineOption",
    "LogFileOption",
    "NoLogFileOption",
    "ExportTreeOption",
    "RepeatOption",
    "SeedStepOption",
    "ProfileOption",
    "SetOverrideOption",
    "MetricOption",
    "EngineOption",
    "BackendOption",
    "PrecisionOption",
    "DevicesOption",
    "EnableNumbaOption",
    "EnableSparseTraversalOption",
    "DiagnosticsOption",
    "LogLevelOption",
    "MisSeedOption",
    "ConflictGraphOption",
    "ScopeSegmentDedupeOption",
    "ScopeChunkTargetOption",
    "ScopeChunkMaxSegmentsOption",
    "ScopeChunkPairMergeOption",
    "ScopeConflictBufferReuseOption",
    "DegreeCapOption",
    "BatchOrderOption",
    "BatchOrderSeedOption",
    "PrefixScheduleOption",
    "PrefixDensityLowOption",
    "PrefixDensityHighOption",
    "PrefixGrowthSmallOption",
    "PrefixGrowthMidOption",
    "PrefixGrowthLargeOption",
    "ResidualLengthscaleOption",
    "ResidualVarianceOption",
    "ResidualKernelTypeOption",
    "ResidualInducingOption",
    "ResidualChunkSizeOption",
    "ResidualStreamTileOption",
    "ResidualForceWhitenedOption",
    "ResidualScopeMemberLimitOption",
    "ResidualScopeBitsetOption",
    "ResidualDynamicQueryBlockOption",
    "ResidualDenseScopeStreamerOption",
    "ResidualMaskedScopeAppendOption",
    "ResidualLevelCacheBatchingOption",
    "ResidualScopeCapsOption",
    "ResidualScopeCapDefaultOption",
    "ResidualScopeCapOutputOption",
    "ResidualScopeCapPercentileOption",
    "ResidualScopeCapMarginOption",
    "ResidualRadiusFloorOption",
    "ResidualPrefilterOption",
    "ResidualPrefilterLookupPathOption",
    "ResidualPrefilterMarginOption",
    "ResidualPrefilterRadiusCapOption",
    "ResidualPrefilterAuditOption",
]
