from __future__ import annotations

import csv
import enum
import io
import json
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np
import typer

from covertreex.telemetry import BenchmarkBatchRecord


class OutputFormat(str, enum.Enum):
    json = "json"
    md = "md"
    csv = "csv"


class ShowSection(str, enum.Enum):
    fields = "fields"


telemetry_app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Inspect and summarise telemetry artifacts emitted by `pcct` commands.",
)


@telemetry_app.command("render")
def render_telemetry(
    log: Path = typer.Argument(..., exists=True, readable=True, help="Path to a benchmark JSONL log."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format for the summary (json, md, csv).",
    ),
    show: List[ShowSection] = typer.Option(None, "--show", help="Include supplementary sections."),
) -> None:
    """Render a benchmark JSONL log as a concise summary."""

    records = _load_records(log)
    if not records:
        typer.echo("[telemetry] No telemetry records found.", err=True)
        raise typer.Exit(code=1)
    summary = _build_summary(records)
    if output_format is OutputFormat.json:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    elif output_format is OutputFormat.md:
        typer.echo(_render_markdown(summary))
    else:
        typer.echo(_render_csv(summary))

    if show and ShowSection.fields in show:
        typer.echo("\n# Measurement fields")
        for field in summary["fields"]:
            typer.echo(field)


def _load_records(path: Path) -> List[BenchmarkBatchRecord]:
    records: List[BenchmarkBatchRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            data = json.loads(payload)
            records.append(BenchmarkBatchRecord.from_payload(data))
    return records


def _numeric_series(records: Iterable[BenchmarkBatchRecord], key: str) -> np.ndarray:
    values: List[float] = []
    for record in records:
        value = record.measurements.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _summarise(values: np.ndarray) -> Mapping[str, float] | Mapping[str, float | int]:
    if values.size == 0:
        return {}
    summary: dict[str, float | int] = {
        "samples": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }
    return summary


def _build_summary(records: List[BenchmarkBatchRecord]) -> dict:
    first = records[0]
    measurement_fields = sorted({key for record in records for key in record.measurements})
    summary: dict[str, object] = {
        "schema_id": first.schema_id,
        "schema_version": first.schema_version,
        "run_id": first.run_id,
        "run_hash": first.run_hash,
        "runtime_digest": first.runtime_digest,
        "batches": len(records),
        "total_points": int(sum(record.batch_size for record in records)),
        "runtime": dict(first.runtime),
        "metadata": dict(first.metadata),
        "seed_pack": dict(first.seed_pack),
        "fields": measurement_fields,
        "measurements": {},
    }

    measurements = summary["measurements"]  # type: ignore[assignment]
    for metric in ("traversal_ms", "conflict_graph_ms", "mis_ms", "rss_delta_bytes"):
        stats = _summarise(_numeric_series(records, metric))
        if stats:
            measurements[metric] = stats  # type: ignore[index]

    batch_sizes = np.asarray([record.batch_size for record in records], dtype=np.float64)
    summary["batch_size"] = _summarise(batch_sizes)

    residual = _residual_summary(records)
    if residual:
        summary["residual"] = residual

    pairwise = _pairwise_reuse(records)
    if pairwise is not None:
        summary["conflict_pairwise_reuse_pct"] = pairwise
    return summary


def _residual_summary(records: List[BenchmarkBatchRecord]) -> dict:
    whitened_pairs = float(_numeric_series(records, "traversal_whitened_block_pairs").sum())
    kernel_pairs = float(_numeric_series(records, "traversal_kernel_provider_pairs").sum())
    whitened_ms = float(_numeric_series(records, "traversal_whitened_block_ms").sum())
    kernel_ms = float(_numeric_series(records, "traversal_kernel_provider_ms").sum())
    if not (whitened_pairs or kernel_pairs or whitened_ms or kernel_ms):
        return {}
    pair_total = whitened_pairs + kernel_pairs
    time_total = whitened_ms + kernel_ms
    summary = {
        "whitened_pairs": whitened_pairs,
        "kernel_pairs": kernel_pairs,
        "whitened_ms": whitened_ms,
        "kernel_ms": kernel_ms,
    }
    if pair_total > 0:
        summary["whitened_pair_share"] = whitened_pairs / pair_total
    if time_total > 0:
        summary["whitened_time_share"] = whitened_ms / time_total
    return summary


def _pairwise_reuse(records: List[BenchmarkBatchRecord]) -> float | None:
    flags = [
        1.0 if record.measurements.get("conflict_pairwise_reused") else 0.0
        for record in records
        if "conflict_pairwise_reused" in record.measurements
    ]
    if not flags:
        return None
    values = np.asarray(flags, dtype=np.float64)
    return float(np.mean(values) * 100.0)


def _render_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "## Run Summary",
        f"- schema: {summary['schema_id']} (v{summary['schema_version']})",
        f"- run: {summary['run_id']} (hash={summary['run_hash']})",
        f"- batches: {summary['batches']} (points={summary['total_points']})",
    ]
    digest = summary.get("runtime_digest")
    if digest:
        lines.append(f"- digest: {digest}")

    runtime = summary.get("runtime") or {}
    if runtime:
        lines.append("- runtime: " + ", ".join(f"{key}={value}" for key, value in runtime.items()))
    seed_pack = summary.get("seed_pack") or {}
    if seed_pack:
        parts = ", ".join(f"{key}={value}" for key, value in sorted(seed_pack.items()))
        lines.append(f"- seeds: {parts}")

    measurements = summary.get("measurements") or {}
    lines.append("\n### Measurements")
    if measurements:
        lines.append("| metric | mean | p50 | p90 | p95 | max | samples |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for metric, stats in measurements.items():
            lines.append(
                "| {metric} | {mean:.3f} | {p50:.3f} | {p90:.3f} | {p95:.3f} | {max:.3f} | {samples} |".format(
                    metric=metric,
                    mean=stats.get("mean", 0.0),
                    p50=stats.get("p50", 0.0),
                    p90=stats.get("p90", 0.0),
                    p95=stats.get("p95", 0.0),
                    max=stats.get("max", 0.0),
                    samples=int(stats.get("samples", 0)),
                )
            )
    else:
        lines.append("No numeric measurements present.")

    residual = summary.get("residual") or {}
    if residual:
        lines.append("\n### Residual coverage")
        lines.append(
            "- whitened pairs: {pairs:.0f} ({share:.1f}% of pairs)".format(
                pairs=residual.get("whitened_pairs", 0.0),
                share=100.0 * residual.get("whitened_pair_share", 0.0),
            )
        )
        lines.append(
            "- whitened time: {time:.2f} ms ({share:.1f}% of time)".format(
                time=residual.get("whitened_ms", 0.0),
                share=100.0 * residual.get("whitened_time_share", 0.0),
            )
        )

    if summary.get("conflict_pairwise_reuse_pct") is not None:
        lines.append(
            "\n- conflict pairwise reuse: {:.1f}% of batches".format(summary["conflict_pairwise_reuse_pct"])
        )

    return "\n".join(lines)


def _render_csv(summary: Mapping[str, object]) -> str:
    rows: List[tuple[str, object]] = [
        ("schema_id", summary["schema_id"]),
        ("schema_version", summary["schema_version"]),
        ("run_id", summary["run_id"]),
        ("run_hash", summary["run_hash"]),
        ("batches", summary["batches"]),
        ("total_points", summary["total_points"]),
    ]

    runtime = summary.get("runtime") or {}
    for key, value in runtime.items():
        rows.append((f"runtime.{key}", value))

    measurements = summary.get("measurements") or {}
    for metric, stats in measurements.items():
        for field, value in stats.items():
            rows.append((f"measurements.{metric}.{field}", value))

    residual = summary.get("residual") or {}
    for key, value in residual.items():
        rows.append((f"residual.{key}", value))

    if summary.get("conflict_pairwise_reuse_pct") is not None:
        rows.append(("conflict_pairwise_reuse_pct", summary["conflict_pairwise_reuse_pct"]))

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["key", "value"])
    for key, value in rows:
        writer.writerow([key, value])
    return buffer.getvalue().strip()


__all__ = ["telemetry_app"]
