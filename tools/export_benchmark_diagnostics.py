from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON line in {path}: {exc}") from exc
    if not records:
        raise ValueError(f"No telemetry records found in {path}")
    return records


def _detect_metric(records: Iterable[Dict[str, Any]]) -> str | None:
    for record in records:
        metric = record.get("metric") or record.get("runtime_metric")
        if metric:
            return str(metric)
    return None


def _aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    batches = len(records)
    metric_label = _detect_metric(records)
    residual_mode = metric_label in {"residual", "residual_correlation"}
    traversal_ms = [rec.get("traversal_ms", 0.0) for rec in records]
    traversal_build_ms = [rec.get("traversal_build_wall_ms", 0.0) for rec in records]
    conflict_ms = [rec.get("conflict_graph_ms", 0.0) for rec in records]
    budget_start = sum(rec.get("traversal_scope_budget_start", 0) for rec in records)
    budget_final = sum(rec.get("traversal_scope_budget_final", 0) for rec in records)
    budget_escalations = sum(rec.get("traversal_scope_budget_escalations", 0) for rec in records)
    budget_early = sum(rec.get("traversal_scope_budget_early_terminate", 0) for rec in records)
    scope_saturated = sum(rec.get("traversal_scope_chunk_saturated", 0) for rec in records)
    scope_points = sum(rec.get("traversal_scope_chunk_points", 0) for rec in records)
    scope_scans = sum(rec.get("traversal_scope_chunk_scans", 0) for rec in records)
    whitened_pairs = sum(rec.get("traversal_whitened_block_pairs", 0) for rec in records)
    whitened_ms = sum(rec.get("traversal_whitened_block_ms", 0.0) for rec in records)
    kernel_pairs = sum(rec.get("traversal_kernel_provider_pairs", 0) for rec in records)
    kernel_ms = sum(rec.get("traversal_kernel_provider_ms", 0.0) for rec in records)
    whitened_calls = sum(rec.get("traversal_whitened_block_calls", 0) for rec in records)
    kernel_calls = sum(rec.get("traversal_kernel_provider_calls", 0) for rec in records)
    pairwise_reused = sum(1 for rec in records if rec.get("conflict_pairwise_reused", 0))
    pairwise_ratio = pairwise_reused / batches if batches else 0.0
    def _per(value: float, denom: float) -> float:
        return float(value) / denom if denom else 0.0

    if residual_mode and batches and pairwise_reused != batches:
        missing = batches - pairwise_reused
        raise ValueError(
            f"residual conflict batches missing cached pairwise distances: {missing}/{batches} reported conflict_pairwise_reused=0"
        )

    return {
        "metric": metric_label or "unknown",
        "batches": batches,
        "traversal_ms_sum": sum(traversal_ms),
        "traversal_ms_median": median(traversal_ms),
        "traversal_build_ms_sum": sum(traversal_build_ms),
        "traversal_build_ms_median": median(traversal_build_ms),
        "conflict_ms_sum": sum(conflict_ms),
        "conflict_ms_median": median(conflict_ms),
        "whitened_block_pairs_sum": whitened_pairs,
        "whitened_block_ms_sum": whitened_ms,
        "whitened_block_calls_sum": whitened_calls,
        "whitened_block_pairs_per_batch": _per(whitened_pairs, batches),
        "whitened_block_ms_per_call": _per(whitened_ms, whitened_calls),
        "whitened_block_ms_per_pair": _per(whitened_ms, whitened_pairs),
        "kernel_provider_pairs_sum": kernel_pairs,
        "kernel_provider_ms_sum": kernel_ms,
        "kernel_provider_calls_sum": kernel_calls,
        "kernel_provider_pairs_per_batch": _per(kernel_pairs, batches),
        "kernel_provider_ms_per_call": _per(kernel_ms, kernel_calls),
        "kernel_provider_ms_per_pair": _per(kernel_ms, kernel_pairs),
        "whitened_to_kernel_pair_ratio": _per(whitened_pairs, kernel_pairs),
        "scope_chunk_points_sum": scope_points,
        "scope_chunk_scans_sum": scope_scans,
        "scope_chunk_saturated": scope_saturated,
        "scope_budget_start_sum": budget_start,
        "scope_budget_final_sum": budget_final,
        "scope_budget_escalations_sum": budget_escalations,
        "scope_budget_early_sum": budget_early,
        "scope_budget_final_over_start": (budget_final / budget_start) if budget_start else 0.0,
        "pairwise_reused_batches": pairwise_reused,
        "pairwise_reused_ratio": pairwise_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate cli.queries telemetry JSONL files into a CSV summary."
    )
    parser.add_argument("--output", required=True, help="Destination CSV path")
    parser.add_argument("paths", nargs="+", help="Telemetry JSONL files to aggregate")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {path}")
        records = _load_records(path)
        try:
            aggregates = _aggregate(records)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        aggregates["log_path"] = str(path)
        rows.append(aggregates)

    fieldnames = [
        "log_path",
        "metric",
        "batches",
        "traversal_ms_sum",
        "traversal_ms_median",
        "traversal_build_ms_sum",
        "traversal_build_ms_median",
        "conflict_ms_sum",
        "conflict_ms_median",
        "whitened_block_pairs_sum",
        "whitened_block_ms_sum",
        "whitened_block_calls_sum",
        "whitened_block_pairs_per_batch",
        "whitened_block_ms_per_call",
        "whitened_block_ms_per_pair",
        "kernel_provider_pairs_sum",
        "kernel_provider_ms_sum",
        "kernel_provider_calls_sum",
        "kernel_provider_pairs_per_batch",
        "kernel_provider_ms_per_call",
        "kernel_provider_ms_per_pair",
        "whitened_to_kernel_pair_ratio",
        "scope_chunk_points_sum",
        "scope_chunk_scans_sum",
        "scope_chunk_saturated",
        "scope_budget_start_sum",
        "scope_budget_final_sum",
        "scope_budget_escalations_sum",
        "scope_budget_early_sum",
        "scope_budget_final_over_start",
        "pairwise_reused_batches",
        "pairwise_reused_ratio",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} aggregated rows to {output_path}")


if __name__ == "__main__":
    main()
