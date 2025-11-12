#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

from covertreex.telemetry.schemas import RESIDUAL_GATE_PROFILE_SCHEMA_ID


@dataclass
class ProfileRecord:
    edges: np.ndarray
    max_whitened: np.ndarray
    max_ratio: np.ndarray
    counts: np.ndarray
    samples_total: int
    false_negative_samples: int
    metadata: Dict[str, Any]
    source: str
    quantiles: Dict[str, np.ndarray]
    quantile_percentiles: np.ndarray
    quantile_capacity: int
    quantile_counts: np.ndarray
    quantile_totals: np.ndarray


def _coerce_array(values: Iterable[Any], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {arr.shape}.")
    return arr


def _coerce_counts(values: Iterable[Any], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {arr.shape}.")
    return arr


def _iter_payloads(path: Path) -> Iterator[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} contains invalid JSON line: {exc}") from exc
        return
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
            else:
                raise ValueError(f"{path} contains a non-dict entry inside JSON list")
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def _extract_profile(record: Dict[str, Any], *, source: Path) -> ProfileRecord | None:
    schema_id = record.get("schema_id")
    legacy_ids = {"covertreex.residual_gate_profile.v1", RESIDUAL_GATE_PROFILE_SCHEMA_ID}
    if schema_id and schema_id not in legacy_ids:
        return None
    payload = record.get("profile", record)
    if "radius_bin_edges" not in payload:
        return None
    edges = _coerce_array(payload["radius_bin_edges"], "radius_bin_edges")
    max_whitened = _coerce_array(payload.get("max_whitened", []), "max_whitened")
    max_ratio = _coerce_array(payload.get("max_ratio", []), "max_ratio")
    counts = _coerce_counts(payload.get("counts", []), "counts")
    if max_whitened.size != edges.size - 1:
        raise ValueError(f"{source} max_whitened length mismatch")
    if max_ratio.size != edges.size - 1:
        raise ValueError(f"{source} max_ratio length mismatch")
    if counts.size != edges.size - 1:
        raise ValueError(f"{source} counts length mismatch")
    quantiles_field = payload.get("quantiles", {}) or {}
    quantiles: Dict[str, np.ndarray] = {}
    for key, values in quantiles_field.items():
        arr = _coerce_array(values, f"quantiles[{key}]")
        if arr.size != edges.size - 1:
            raise ValueError(f"{source} quantile '{key}' length mismatch")
        quantiles[str(key)] = arr
    quantile_percentiles = _coerce_array(payload.get("quantile_percentiles", []), "quantile_percentiles")
    quantile_capacity = int(payload.get("quantile_capacity", 0))
    quantile_counts = _coerce_counts(payload.get("quantile_counts", [0] * (edges.size - 1)), "quantile_counts")
    quantile_totals = _coerce_counts(payload.get("quantile_totals", [0] * (edges.size - 1)), "quantile_totals")
    samples_total = int(payload.get("samples_total", int(counts.sum())))
    false_negatives = int(payload.get("false_negative_samples", 0))
    metadata = dict(payload.get("metadata", {}))
    if run_id := record.get("run_id"):
        metadata.setdefault("run_id", run_id)
    metadata.setdefault("source_path", str(source))
    return ProfileRecord(
        edges=edges,
        max_whitened=max_whitened,
        max_ratio=max_ratio,
        counts=counts,
        samples_total=samples_total,
        false_negative_samples=false_negatives,
        metadata=metadata,
        source=str(source),
        quantiles=quantiles,
        quantile_percentiles=quantile_percentiles,
        quantile_capacity=quantile_capacity,
        quantile_counts=quantile_counts,
        quantile_totals=quantile_totals,
    )


def load_profile_records(paths: List[str]) -> List[ProfileRecord]:
    records: List[ProfileRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Profile telemetry file not found: {raw_path}")
        for payload in _iter_payloads(path):
            entry = _extract_profile(payload, source=path)
            if entry is not None:
                records.append(entry)
    return records


def merge_profile_records(
    records: List[ProfileRecord],
    *,
    metadata_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("No gate profile telemetry records were provided.")
    edges = records[0].edges
    bin_count = edges.size - 1
    max_whitened = np.zeros(bin_count, dtype=np.float64)
    max_ratio = np.zeros(bin_count, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int64)
    samples_total = 0
    false_negatives = 0
    merged_metadata: Dict[str, Any] = {
        "sources": [],
        "run_ids": [],
    }
    merged_quantiles: Dict[str, np.ndarray] = {}
    quantile_percentiles: np.ndarray | None = None
    quantile_capacity = 0
    quantile_counts = np.zeros(bin_count, dtype=np.int64)
    quantile_totals = np.zeros(bin_count, dtype=np.int64)
    for record in records:
        if record.edges.size != edges.size or not np.allclose(record.edges, edges):
            raise ValueError("All records must share identical radius_bin_edges.")
        max_whitened = np.maximum(max_whitened, record.max_whitened)
        max_ratio = np.maximum(max_ratio, record.max_ratio)
        counts += record.counts
        samples_total += record.samples_total
        false_negatives += record.false_negative_samples
        merged_metadata["sources"].append(record.source)
        run_id = record.metadata.get("run_id")
        if run_id:
            merged_metadata["run_ids"].append(run_id)
        for key, value in record.metadata.items():
            merged_metadata.setdefault(key, value)
        for label, values in record.quantiles.items():
            existing = merged_quantiles.get(label)
            if existing is None:
                merged_quantiles[label] = np.array(values, dtype=np.float64, copy=True)
            else:
                merged_quantiles[label] = np.maximum(existing, values)
        if quantile_percentiles is None and record.quantile_percentiles.size:
            quantile_percentiles = record.quantile_percentiles
        quantile_capacity = max(quantile_capacity, record.quantile_capacity)
        if record.quantile_counts.size == bin_count:
            quantile_counts += record.quantile_counts
        if record.quantile_totals.size == bin_count:
            quantile_totals += record.quantile_totals
    if metadata_overrides:
        merged_metadata.update(metadata_overrides)
    schema_version = 2 if merged_quantiles else 1
    payload = {
        "schema": schema_version,
        "schema_id": RESIDUAL_GATE_PROFILE_SCHEMA_ID,
        "generated_at": time.time(),
        "radius_bin_edges": edges.tolist(),
        "max_whitened": np.maximum.accumulate(max_whitened).tolist(),
        "max_ratio": np.maximum.accumulate(max_ratio).tolist(),
        "counts": counts.tolist(),
        "samples_total": int(samples_total),
        "false_negative_samples": int(false_negatives),
        "metadata": merged_metadata,
    }
    if merged_quantiles:
        quantile_payload = {
            label: np.maximum.accumulate(values).tolist()
            for label, values in merged_quantiles.items()
        }
        payload["quantiles"] = quantile_payload
        payload["quantile_capacity"] = int(quantile_capacity)
        payload["quantile_counts"] = quantile_counts.astype(int).tolist()
        payload["quantile_totals"] = quantile_totals.astype(int).tolist()
        if quantile_percentiles is not None:
            payload["quantile_percentiles"] = quantile_percentiles.tolist()
    payload["bins"] = int(bin_count)
    payload.setdefault("radius_eps", 1e-9)
    return payload


def _parse_metadata(pairs: List[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Metadata override '{item}' must use key=value format.")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate residual Gate-1 profile telemetry (JSON/JSONL) into a lookup JSON file.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSON/JSONL telemetry files (e.g. cli.queries --residual-gate-profile-log outputs).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the merged profile JSON.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Optional key=value metadata overrides to stamp onto the output profile.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        records = load_profile_records(args.inputs)
    except Exception as exc:  # pragma: no cover - argparse already exercises this path
        parser.error(str(exc))
    overrides: Dict[str, Any] = {}
    if args.metadata:
        try:
            overrides = _parse_metadata(args.metadata)
        except ValueError as exc:  # pragma: no cover - validated through unit tests
            parser.error(str(exc))
    payload = merge_profile_records(records, metadata_overrides=overrides)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[ingest] wrote {output_path} | sources={len(payload['metadata'].get('sources', []))} "
        f"samples={payload['samples_total']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI entry
    sys.exit(main())
