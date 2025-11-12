from __future__ import annotations

from typing import Dict, Tuple

BENCHMARK_BATCH_SCHEMA_ID = "covertreex.benchmark_batch.v1"
BENCHMARK_BATCH_SCHEMA: Dict[str, object] = {
    "id": BENCHMARK_BATCH_SCHEMA_ID,
    "description": "Per-batch PCCT telemetry emitted during benchmark runs.",
    "required": (
        "schema_id",
        "run_id",
        "timestamp",
        "batch_index",
        "batch_size",
        "traversal_ms",
        "conflict_graph_ms",
        "mis_ms",
    ),
}

RESIDUAL_SCOPE_CAP_SCHEMA_VERSION = 1
RESIDUAL_SCOPE_CAP_SCHEMA_ID = "covertreex.residual_scope_cap_summary.v1"
RESIDUAL_SCOPE_CAP_SCHEMA: Dict[str, object] = {
    "id": RESIDUAL_SCOPE_CAP_SCHEMA_ID,
    "version": RESIDUAL_SCOPE_CAP_SCHEMA_VERSION,
    "description": "Summary tables derived from traversal residual scope radii.",
    "required": ("schema", "schema_id", "levels", "metadata"),
}

RESIDUAL_GATE_PROFILE_SCHEMA_VERSION = 2
RESIDUAL_GATE_PROFILE_SCHEMA_ID = "covertreex.residual_gate_profile.v2"
RESIDUAL_GATE_PROFILE_SCHEMA: Dict[str, object] = {
    "id": RESIDUAL_GATE_PROFILE_SCHEMA_ID,
    "version": RESIDUAL_GATE_PROFILE_SCHEMA_VERSION,
    "description": "Aggregated Gate-1 profile payload derived from residual benchmark telemetry.",
    "required": (
        "schema_id",
        "run_id",
        "radius_bin_edges",
        "max_whitened",
        "max_ratio",
        "counts",
        "samples_total",
    ),
}

RUNTIME_BREAKDOWN_SCHEMA_ID = "covertreex.runtime_breakdown.v1"
RUNTIME_BREAKDOWN_SCHEMA: Dict[str, object] = {
    "id": RUNTIME_BREAKDOWN_SCHEMA_ID,
    "description": "Warm-up vs steady-state runtime metrics for PCCT and baselines.",
    "fields": (
        "schema_id",
        "run",
        "label",
        "build_warmup_seconds",
        "build_steady_seconds",
        "build_total_seconds",
        "query_warmup_seconds",
        "query_steady_seconds",
        "build_cpu_seconds",
        "build_cpu_utilisation",
        "build_rss_delta_bytes",
        "build_max_rss_bytes",
        "query_cpu_seconds",
        "query_cpu_utilisation",
        "query_rss_delta_bytes",
        "query_max_rss_bytes",
    ),
}

RUNTIME_BREAKDOWN_CHUNK_FIELDS: Tuple[str, ...] = (
    "traversal_chunk_segments_warmup",
    "traversal_chunk_segments_steady",
    "traversal_chunk_emitted_warmup",
    "traversal_chunk_emitted_steady",
    "traversal_chunk_max_members_warmup",
    "traversal_chunk_max_members_steady",
    "conflict_chunk_segments_warmup",
    "conflict_chunk_segments_steady",
    "conflict_chunk_emitted_warmup",
    "conflict_chunk_emitted_steady",
    "conflict_chunk_max_members_warmup",
    "conflict_chunk_max_members_steady",
)

RUNTIME_BREAKDOWN_FIELDNAMES: Tuple[str, ...] = (
    "schema_id",
    "run_id",
    "run",
    "label",
    "build_warmup_seconds",
    "build_steady_seconds",
    "build_total_seconds",
    "query_warmup_seconds",
    "query_steady_seconds",
    "build_cpu_seconds",
    "build_cpu_utilisation",
    "build_rss_delta_bytes",
    "build_max_rss_bytes",
    "query_cpu_seconds",
    "query_cpu_utilisation",
    "query_rss_delta_bytes",
    "query_max_rss_bytes",
) + RUNTIME_BREAKDOWN_CHUNK_FIELDS


def runtime_breakdown_fieldnames(*, include_run: bool = True) -> Tuple[str, ...]:
    """Return stable CSV headers for runtime breakdown telemetry."""

    fields = list(RUNTIME_BREAKDOWN_FIELDNAMES)
    if not include_run and "run" in fields:
        fields.remove("run")
    return tuple(fields)


BATCH_OPS_RESULT_SCHEMA_ID = "covertreex.batch_ops_summary.v1"
BATCH_OPS_RESULT_SCHEMA: Dict[str, object] = {
    "id": BATCH_OPS_RESULT_SCHEMA_ID,
    "description": "Throughput benchmark summary emitted by benchmarks.batch_ops.",
    "required": (
        "schema_id",
        "run_id",
        "timestamp",
        "mode",
        "batches",
        "batch_size",
        "points_processed",
        "elapsed_seconds",
    ),
}

__all__ = [
    "BENCHMARK_BATCH_SCHEMA",
    "BENCHMARK_BATCH_SCHEMA_ID",
    "RESIDUAL_SCOPE_CAP_SCHEMA",
    "RESIDUAL_SCOPE_CAP_SCHEMA_ID",
    "RESIDUAL_SCOPE_CAP_SCHEMA_VERSION",
    "RESIDUAL_GATE_PROFILE_SCHEMA",
    "RESIDUAL_GATE_PROFILE_SCHEMA_ID",
    "RESIDUAL_GATE_PROFILE_SCHEMA_VERSION",
    "RUNTIME_BREAKDOWN_SCHEMA",
    "RUNTIME_BREAKDOWN_SCHEMA_ID",
    "RUNTIME_BREAKDOWN_CHUNK_FIELDS",
    "RUNTIME_BREAKDOWN_FIELDNAMES",
    "runtime_breakdown_fieldnames",
    "BATCH_OPS_RESULT_SCHEMA",
    "BATCH_OPS_RESULT_SCHEMA_ID",
]
