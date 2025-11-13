from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import sys

from covertreex import config as cx_config
from covertreex.telemetry import (
    BenchmarkLogWriter,
    ResidualScopeCapRecorder,
    timestamped_artifact,
)

from .runtime import _resolve_artifact_arg


@dataclass
class CLITelemetryHandles:
    log_path: Optional[str]
    log_writer: Optional[BenchmarkLogWriter]
    scope_cap_recorder: Optional[ResidualScopeCapRecorder]
    traversal_view: Optional["ResidualTraversalTelemetry"]

    def close(self) -> None:
        if self.scope_cap_recorder is not None:
            self.scope_cap_recorder.dump()
        if self.log_writer is not None:
            self.log_writer.close()


def initialise_cli_telemetry(
    *,
    args: Any,
    run_id: str,
    runtime_snapshot: Mapping[str, Any],
    log_metadata: Mapping[str, Any],
) -> CLITelemetryHandles:
    if getattr(args, "no_log_file", False):
        log_path = None
    elif getattr(args, "log_file", None):
        log_path = str(_resolve_artifact_arg(args.log_file))
    else:
        log_path = str(
            timestamped_artifact(
                category="benchmarks",
                prefix=f"queries_{run_id}",
                suffix=".jsonl",
            )
        )
    log_writer = None
    if log_path:
        print(f"[queries] writing batch telemetry to {log_path}")
        log_writer = BenchmarkLogWriter(
            log_path,
            run_id=run_id,
            runtime=runtime_snapshot,
            metadata=log_metadata,
        )
    scope_cap_recorder: Optional[ResidualScopeCapRecorder] = None
    scope_cap_output = getattr(args, "residual_scope_cap_output", None)
    if getattr(args, "metric", None) == "residual" and scope_cap_output:
        runtime_config = cx_config.runtime_config()
        resolved_output = str(_resolve_artifact_arg(scope_cap_output))
        scope_cap_recorder = ResidualScopeCapRecorder(
            output=resolved_output,
            percentile=getattr(args, "residual_scope_cap_percentile", 0.5),
            margin=getattr(args, "residual_scope_cap_margin", 0.05),
            radius_floor=runtime_config.residual_radius_floor,
        )
        scope_cap_recorder.annotate(
            run_id=run_id,
            log_file=log_path,
            tree_points=getattr(args, "tree_points", 0),
            batch_size=getattr(args, "batch_size", 0),
            scope_chunk_target=runtime_config.scope_chunk_target,
            scope_chunk_max_segments=runtime_config.scope_chunk_max_segments,
            residual_scope_cap_default=getattr(args, "residual_scope_cap_default", None),
            seed=getattr(args, "seed", 0),
            build_mode=getattr(args, "build_mode", "batch"),
        )
    traversal_view: Optional[ResidualTraversalTelemetry]
    if getattr(args, "metric", None) == "residual":
        traversal_view = ResidualTraversalTelemetry()
    else:
        traversal_view = None
    return CLITelemetryHandles(
        log_path=log_path,
        log_writer=log_writer,
        scope_cap_recorder=scope_cap_recorder,
        traversal_view=traversal_view,
    )

@dataclass
class _ResidualBatchTelemetry:
    batch_index: int
    batch_size: int
    whitened_pairs: float
    whitened_ms: float
    whitened_calls: int
    kernel_pairs: float
    kernel_ms: float
    kernel_calls: int
    pairwise_reused: bool


class ResidualTraversalTelemetry:
    """Collects per-batch residual traversal counters for CLI summaries."""

    def __init__(self) -> None:
        self._records: List[_ResidualBatchTelemetry] = []

    def observe_plan(self, plan: Any, batch_index: int, batch_size: int) -> None:
        traversal = getattr(plan, "traversal", None)
        if traversal is None:
            return
        timings = getattr(traversal, "timings", None)
        if timings is None:
            return
        conflict_graph = getattr(plan, "conflict_graph", None)
        conflict_timings = getattr(conflict_graph, "timings", None) if conflict_graph else None
        pairwise_flag = int(getattr(conflict_timings, "pairwise_reused", 1)) if conflict_timings else 1
        if pairwise_flag != 1:
            self._handle_pairwise_reuse_failure(batch_index, batch_size, pairwise_flag)
        self._records.append(
            _ResidualBatchTelemetry(
                batch_index=batch_index,
                batch_size=batch_size,
                whitened_pairs=float(getattr(timings, "whitened_block_pairs", 0)),
                whitened_ms=float(getattr(timings, "whitened_block_seconds", 0.0)),
                whitened_calls=int(getattr(timings, "whitened_block_calls", 0)),
                kernel_pairs=float(getattr(timings, "kernel_provider_pairs", 0)),
                kernel_ms=float(getattr(timings, "kernel_provider_seconds", 0.0)),
                kernel_calls=int(getattr(timings, "kernel_provider_calls", 0)),
                pairwise_reused=bool(pairwise_flag),
            )
        )

    @property
    def has_data(self) -> bool:
        return bool(self._records)

    def render_summary(self) -> Sequence[str]:
        if not self._records:
            return ()
        wh_pairs = np.asarray([rec.whitened_pairs for rec in self._records], dtype=np.float64)
        wh_ms = np.asarray([rec.whitened_ms for rec in self._records], dtype=np.float64)
        wh_calls = np.asarray([rec.whitened_calls for rec in self._records], dtype=np.float64)
        kernel_pairs = np.asarray([rec.kernel_pairs for rec in self._records], dtype=np.float64)
        kernel_ms = np.asarray([rec.kernel_ms for rec in self._records], dtype=np.float64)
        kernel_calls = np.asarray([rec.kernel_calls for rec in self._records], dtype=np.float64)

        wh_pairs_sum = float(np.sum(wh_pairs))
        wh_ms_sum = float(np.sum(wh_ms))
        kernel_pairs_sum = float(np.sum(kernel_pairs))
        kernel_ms_sum = float(np.sum(kernel_ms))
        total_pairs = wh_pairs_sum + kernel_pairs_sum
        total_ms = wh_ms_sum + kernel_ms_sum
        if total_pairs > 0:
            share_pairs = wh_pairs_sum / total_pairs
            kernel_pair_share = 1.0 - share_pairs
        else:
            share_pairs = 0.0
            kernel_pair_share = 0.0
        if total_ms > 0:
            share_ms = wh_ms_sum / total_ms
            kernel_ms_share = 1.0 - share_ms
        else:
            share_ms = 0.0
            kernel_ms_share = 0.0

        ratio_series = []
        for rec in self._records:
            if rec.kernel_pairs > 0:
                ratio_series.append(rec.whitened_pairs / rec.kernel_pairs)
            elif rec.whitened_pairs > 0:
                ratio_series.append(float("inf"))
        ratio_series_np = np.asarray(ratio_series, dtype=np.float64) if ratio_series else np.empty(0)
        aggregate_ratio = (
            (wh_pairs_sum / kernel_pairs_sum)
            if kernel_pairs_sum > 0
            else (float("inf") if wh_pairs_sum > 0 else 0.0)
        )

        lines: List[str] = []
        lines.append(f"[queries] residual traversal telemetry (batches={len(self._records)})")
        lines.append(
            "  whitened pairs: sum={} (median/batch={}, p90={}), calls={}, time={} (~{} per call)".format(
                _format_pairs(wh_pairs_sum),
                _format_pairs(_median(wh_pairs)),
                _format_pairs(_percentile(wh_pairs, 90.0)),
                _format_int(np.sum(wh_calls)),
                _format_time(wh_ms_sum),
                _format_time(_per_unit(wh_ms_sum, np.sum(wh_calls))),
            )
        )
        lines.append(
            "  kernel pairs:   sum={} (median/batch={}, p90={}), calls={}, time={} (~{} per call)".format(
                _format_pairs(kernel_pairs_sum),
                _format_pairs(_median(kernel_pairs)),
                _format_pairs(_percentile(kernel_pairs, 90.0)),
                _format_int(np.sum(kernel_calls)),
                _format_time(kernel_ms_sum),
                _format_time(_per_unit(kernel_ms_sum, np.sum(kernel_calls))),
            )
        )
        if ratio_series_np.size:
            lines.append(
                "  coverage ratio (whitened/kernel): aggregate={} median={} min={} max={}".format(
                    _format_ratio(aggregate_ratio),
                    _format_ratio(_median(ratio_series_np)),
                    _format_ratio(np.min(ratio_series_np)),
                    _format_ratio(np.max(ratio_series_np)),
                )
            )
        else:
            lines.append(
                "  coverage ratio (whitened/kernel): aggregate={} (kernel tiles absent)".format(
                    _format_ratio(aggregate_ratio)
                )
            )
        lines.append(
            "  pair/time mix: whitened pairs={} / kernel pairs={}, whitened time={} / kernel time={}".format(
                f"{share_pairs * 100.0:.1f}%",
                f"{kernel_pair_share * 100.0:.1f}%",
                f"{share_ms * 100.0:.1f}%",
                f"{kernel_ms_share * 100.0:.1f}%",
            )
        )
        reuse_ok = sum(1 for rec in self._records if rec.pairwise_reused)
        ratio = (reuse_ok / len(self._records)) * 100.0
        lines.append(
            "  conflict pairwise reuse: {}/{} batches ({:.1f}%)".format(
                reuse_ok,
                len(self._records),
                ratio,
            )
        )
        return tuple(lines)

    @staticmethod
    def _handle_pairwise_reuse_failure(batch_index: int, batch_size: int, flag: int) -> None:
        message = (
            "[queries] residual conflict graph missing cached pairwise distances "
            f"(batch={batch_index}, batch_size={batch_size}, conflict_pairwise_reused={flag}); "
            "ensure residual traversal caching stays enabled."
        )
        print(message, file=sys.stderr, flush=True)
        raise RuntimeError(
            "Residual conflict graph reported conflict_pairwise_reused=0; enable residual traversal "
            "caching to satisfy the audit invariant."
        )


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _percentile(values: np.ndarray, pct: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, pct))


def _per_unit(numerator: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numerator / denom


def _format_pairs(value: float) -> str:
    return _format_scaled(value, "pairs")


def _format_int(value: float) -> str:
    return f"{int(value):,}"


def _format_time(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    return f"{seconds * 1e3:.1f}ms"


__all__ = [
    "CLITelemetryHandles",
    "ResidualTraversalTelemetry",
    "initialise_cli_telemetry",
]


def _format_ratio(value: float) -> str:
    if np.isinf(value):
        return "∞"
    if np.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _format_scaled(value: float, suffix: str) -> str:
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}B {suffix}"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M {suffix}"
    if abs_value >= 1e3:
        return f"{value / 1e3:.2f}K {suffix}"
    return f"{value:.0f} {suffix}"


__all__ = ["ResidualTraversalTelemetry"]
