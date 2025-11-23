from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:  # pragma: no cover - platform specific fallback
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore

import numpy as np

from .schemas import (
    BENCHMARK_BATCH_SCHEMA_ID,
    BENCHMARK_BATCH_SCHEMA_VERSION,
    RESIDUAL_SCOPE_CAP_SCHEMA_ID,
    RESIDUAL_SCOPE_CAP_SCHEMA_VERSION,
)

__all__ = ["BenchmarkLogWriter", "ResidualScopeCapRecorder"]


def _read_rss_bytes() -> int | None:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            contents = handle.readline().strip().split()
        if len(contents) >= 2:
            rss_pages = int(contents[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(rss_pages * page_size)
    except (OSError, ValueError, AttributeError):
        pass
    if resource is None:  # pragma: no cover - Windows fallback
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if getattr(usage, "ru_maxrss", 0):
        return int(usage.ru_maxrss * 1024)
    return None


def _ms(value: float) -> float:
    return float(value) * 1e3


def _summarise_metric(record: Dict[str, Any], prefix: str, values: np.ndarray) -> None:
    if values.size == 0:
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    record[f"{prefix}_samples"] = int(finite.size)
    record[f"{prefix}_min"] = float(np.min(finite))
    record[f"{prefix}_max"] = float(np.max(finite))
    record[f"{prefix}_mean"] = float(np.mean(finite))
    for pct in (50, 90, 95, 99):
        record[f"{prefix}_p{pct}"] = float(np.percentile(finite, pct))


def _metric_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    summary: Dict[str, float] = {
        "samples": float(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }
    for pct in (50, 90, 95, 99):
        summary[f"p{pct}"] = float(np.percentile(finite, pct))
    return summary


def _augment_residual_scope_metrics(record: Dict[str, Any], residual_cache: Any) -> None:
    scope_radii = getattr(residual_cache, "scope_radii", None)
    if scope_radii is not None:
        _summarise_metric(
            record,
            "traversal_scope_radius_obs",
            np.asarray(scope_radii, dtype=np.float64),
        )
    initial = getattr(residual_cache, "scope_radius_initial", None)
    if initial is not None:
        _summarise_metric(
            record,
            "traversal_scope_radius_initial",
            np.asarray(initial, dtype=np.float64),
        )
    limits = getattr(residual_cache, "scope_radius_limits", None)
    if limits is not None:
        _summarise_metric(
            record,
            "traversal_scope_radius_limit",
            np.asarray(limits, dtype=np.float64),
        )
    caps = getattr(residual_cache, "scope_radius_caps", None)
    if caps is not None:
        _summarise_metric(
            record,
            "traversal_scope_radius_cap_values",
            np.asarray(caps, dtype=np.float64),
        )
    if initial is not None and limits is not None:
        init_np = np.asarray(initial, dtype=np.float64)
        limit_np = np.asarray(limits, dtype=np.float64)
        clamp_mask = np.isfinite(init_np) & np.isfinite(limit_np) & (init_np > limit_np + 1e-12)
        if clamp_mask.size:
            record["traversal_scope_radius_cap_hits"] = int(np.count_nonzero(clamp_mask))
            if np.any(clamp_mask):
                delta = init_np[clamp_mask] - limit_np[clamp_mask]
                record["traversal_scope_radius_cap_delta_mean"] = float(np.mean(delta))
                record["traversal_scope_radius_cap_delta_max"] = float(np.max(delta))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    normalised = json.dumps(payload, sort_keys=True, separators=( "," , ":" )).encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()


class BenchmarkLogWriter:
    def __init__(
        self,
        path: str,
        *,
        run_id: str | None = None,
        runtime: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        run_hash: str | None = None,
    ):
        self._path = Path(path).expanduser()
        if self._path.parent:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._previous_rss = _read_rss_bytes()
        self._schema_id = BENCHMARK_BATCH_SCHEMA_ID
        self._schema_version = BENCHMARK_BATCH_SCHEMA_VERSION
        self._run_id = run_id or "run-0"
        self._runtime_snapshot: Dict[str, Any] = dict(runtime or {})
        self._seed_pack: Dict[str, Any] = dict(self._runtime_snapshot.get("seeds") or {})
        runtime_items = self._runtime_snapshot.items()
        self._runtime_fields: Dict[str, Any] = {f"runtime_{key}": value for key, value in runtime_items}
        self._metadata: Dict[str, Any] = dict(metadata or {})
        self._runtime_digest = _hash_payload({"runtime": self._runtime_snapshot})
        hash_payload = {"runtime_digest": self._runtime_digest, "seeds": self._seed_pack}
        self._run_hash = run_hash or _hash_payload(hash_payload)
        self._batch_event_index = 0

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.close()

    def record_batch(
        self,
        *,
        batch_index: int,
        batch_size: int,
        plan: Any,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        traversal_timings = plan.traversal.timings
        conflict_timings = plan.conflict_graph.timings
        rss_now = _read_rss_bytes()
        rss_delta = None
        if rss_now is not None and self._previous_rss is not None:
            rss_delta = rss_now - self._previous_rss
        self._previous_rss = rss_now

        record = {
            "schema_id": self._schema_id,
            "schema_version": self._schema_version,
            "run_id": self._run_id,
            "run_hash": self._run_hash,
            "runtime_digest": self._runtime_digest,
            "seed_pack": self._seed_pack,
            "batch_event_index": int(self._batch_event_index),
            "timestamp": time.time(),
            "batch_index": int(batch_index),
            "batch_size": int(batch_size),
            "runtime": self._runtime_snapshot,
            "metadata": self._metadata,
            "candidates": int(plan.traversal.parents.shape[0]),
            "selected": int(plan.selected_indices.size),
            "dominated": int(plan.dominated_indices.size),
            "mis_iterations": int(getattr(plan.mis_result, "iterations", 0)),
            "traversal_ms": _ms(plan.timings.traversal_seconds),
            "conflict_graph_ms": _ms(plan.timings.conflict_graph_seconds),
            "mis_ms": _ms(plan.timings.mis_seconds),
            "traversal_pairwise_ms": _ms(traversal_timings.pairwise_seconds),
            "traversal_mask_ms": _ms(traversal_timings.mask_seconds),
            "traversal_semisort_ms": _ms(traversal_timings.semisort_seconds),
            "traversal_tile_ms": _ms(traversal_timings.tile_seconds),
            "traversal_build_wall_ms": _ms(traversal_timings.build_wall_seconds),
            "conflict_pairwise_ms": _ms(conflict_timings.pairwise_seconds),
            "conflict_scope_group_ms": _ms(conflict_timings.scope_group_seconds),
            "conflict_adjacency_ms": _ms(conflict_timings.adjacency_seconds),
            "conflict_annulus_ms": _ms(conflict_timings.annulus_seconds),
            "conflict_adj_scatter_ms": _ms(conflict_timings.adjacency_scatter_seconds),
            "conflict_adj_filter_ms": _ms(conflict_timings.adjacency_filter_seconds),
            "conflict_adj_pairs": int(conflict_timings.adjacency_total_pairs),
            "conflict_adj_candidates": int(conflict_timings.adjacency_candidate_pairs),
            "traversal_scope_chunk_segments": int(traversal_timings.scope_chunk_segments),
            "traversal_scope_chunk_emitted": int(traversal_timings.scope_chunk_emitted),
            "traversal_scope_chunk_max_members": int(traversal_timings.scope_chunk_max_members),
            "traversal_scope_chunk_scans": int(traversal_timings.scope_chunk_scans),
            "traversal_scope_chunk_points": int(traversal_timings.scope_chunk_points),
            "traversal_scope_chunk_dedupe": int(traversal_timings.scope_chunk_dedupe),
            "traversal_scope_chunk_saturated": int(traversal_timings.scope_chunk_saturated),
            "traversal_scope_budget_start": int(traversal_timings.scope_budget_start),
            "traversal_scope_budget_final": int(traversal_timings.scope_budget_final),
            "traversal_scope_budget_escalations": int(traversal_timings.scope_budget_escalations),
            "traversal_scope_budget_early_terminate": int(traversal_timings.scope_budget_early_terminate),
            "conflict_scope_chunk_segments": int(conflict_timings.scope_chunk_segments),
            "conflict_scope_chunk_emitted": int(conflict_timings.scope_chunk_emitted),
            "conflict_scope_chunk_max_members": int(conflict_timings.scope_chunk_max_members),
            "conflict_scope_chunk_pair_cap": int(conflict_timings.scope_chunk_pair_cap),
            "conflict_scope_chunk_pairs_before": int(conflict_timings.scope_chunk_pairs_before),
            "conflict_scope_chunk_pairs_after": int(conflict_timings.scope_chunk_pairs_after),
            "conflict_scope_chunk_pair_merges": int(conflict_timings.scope_chunk_pair_merges),
            "conflict_scope_domination_ratio": float(conflict_timings.scope_domination_ratio),
            "conflict_pairwise_reused": int(conflict_timings.pairwise_reused),
            "conflict_arena_bytes": int(conflict_timings.arena_bytes),
            "conflict_degree_cap": int(conflict_timings.degree_cap),
            "conflict_degree_pruned_pairs": int(conflict_timings.degree_pruned_pairs),
            "traversal_whitened_block_pairs": int(traversal_timings.whitened_block_pairs),
            "traversal_whitened_block_ms": _ms(traversal_timings.whitened_block_seconds),
            "traversal_whitened_block_calls": int(traversal_timings.whitened_block_calls),
            "traversal_kernel_provider_pairs": int(traversal_timings.kernel_provider_pairs),
            "traversal_kernel_provider_ms": _ms(traversal_timings.kernel_provider_seconds),
            "traversal_kernel_provider_calls": int(traversal_timings.kernel_provider_calls),
            "batch_order_strategy": plan.batch_order_strategy,
            "conflict_grid_cells": int(plan.conflict_graph.grid_cells),
            "conflict_grid_leaders_raw": int(plan.conflict_graph.grid_leaders_raw),
            "conflict_grid_leaders_after": int(plan.conflict_graph.grid_leaders_after),
            "conflict_grid_local_edges": int(plan.conflict_graph.grid_local_edges),
        }
        record["traversal_engine"] = getattr(plan.traversal, "engine", "unknown")
        record["traversal_gate_active"] = int(
            getattr(plan.traversal, "gate_active", False)
        )
        if plan.batch_permutation is not None:
            record["batch_order_permutation_size"] = int(len(plan.batch_permutation))
        for key, value in plan.batch_order_metrics.items():
            record[f"batch_order_{key}"] = float(value)
        residual_cache = getattr(plan.traversal, "residual_cache", None)
        if residual_cache is not None:
            _augment_residual_scope_metrics(record, residual_cache)
        if extra:
            for key, value in extra.items():
                if isinstance(value, (int, float)):
                    record[key] = value
                elif value is not None:
                    record[key] = value
        if rss_now is not None:
            record["rss_bytes"] = int(rss_now)
        if rss_delta is not None:
            record["rss_delta_bytes"] = int(rss_delta)

        if self._runtime_fields:
            record.update(self._runtime_fields)
        if self._metadata:
            record.update(self._metadata)

        self._handle.write(json.dumps(record, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()
        self._batch_event_index += 1

    def has_records(self) -> bool:
        return self._batch_event_index > 0

    def record_event(self, *, event: str, extra: Mapping[str, Any] | None = None) -> None:
        record: Dict[str, Any] = {
            "schema_id": self._schema_id,
            "schema_version": self._schema_version,
            "run_id": self._run_id,
            "run_hash": self._run_hash,
            "runtime_digest": self._runtime_digest,
            "seed_pack": self._seed_pack,
            "batch_event_index": int(self._batch_event_index),
            "timestamp": time.time(),
            "event": str(event),
        }
        record["runtime"] = self._runtime_snapshot
        if self._runtime_fields:
            record.update(self._runtime_fields)
        if self._metadata:
            record.update(self._metadata)
        # Default to a “healthy” conflict reuse flag so residual aggregators do not fail.
        record.setdefault("conflict_pairwise_reused", 1)
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                record[key] = value
        self._handle.write(json.dumps(record, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()
        self._batch_event_index += 1

    def __enter__(self) -> "BenchmarkLogWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ResidualScopeCapRecorder:
    def __init__(self, *, output: str, percentile: float, margin: float, radius_floor: float):
        self._path = Path(output).expanduser()
        self._percentile = max(0.0, min(1.0, float(percentile)))
        self._margin = float(margin)
        self._radius_floor = float(radius_floor)
        self._levels: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: {"obs": [], "limit": [], "cap": []}
        )
        self._metadata: Dict[str, Any] = {}

    def annotate(self, **metadata: Any) -> None:
        for key, value in metadata.items():
            if value is None:
                continue
            self._metadata[str(key)] = value

    def capture(self, plan: Any) -> None:
        cache = getattr(plan.traversal, "residual_cache", None)
        if cache is None or cache.scope_radii is None:
            return
        obs = np.asarray(cache.scope_radii, dtype=np.float64)
        limits = (
            np.asarray(cache.scope_radius_limits, dtype=np.float64)
            if cache.scope_radius_limits is not None
            else None
        )
        caps = (
            np.asarray(cache.scope_radius_caps, dtype=np.float64)
            if cache.scope_radius_caps is not None
            else None
        )
        for summary in getattr(plan, "level_summaries", ()):  # defensive for legacy plans
            candidate_idx = np.asarray(summary.candidates, dtype=np.int64)
            if candidate_idx.size == 0:
                continue
            level_entry = self._levels[int(summary.level)]
            level_entry["obs"].append(obs[candidate_idx])
            if limits is not None:
                level_entry["limit"].append(limits[candidate_idx])
            if caps is not None:
                level_entry["cap"].append(caps[candidate_idx])

    def dump(self) -> None:
        if not self._levels:
            return
        payload: Dict[str, Any] = {
            "schema": RESIDUAL_SCOPE_CAP_SCHEMA_VERSION,
            "schema_id": RESIDUAL_SCOPE_CAP_SCHEMA_ID,
            "generated_at": float(time.time()),
            "percentile": self._percentile,
            "margin": self._margin,
            "radius_floor": self._radius_floor,
            "metadata": self._metadata,
            "levels": {},
        }
        combined_samples: List[np.ndarray] = []
        percentile_pct = self._percentile * 100.0
        for level, data in sorted(self._levels.items()):
            obs_chunks = data.get("obs", [])
            if not obs_chunks:
                continue
            obs_values = np.concatenate(obs_chunks)
            obs_summary = _metric_summary(obs_values)
            if not obs_summary:
                continue
            combined_samples.append(obs_values)
            percentile_value = float(np.percentile(obs_values, percentile_pct))
            suggested_cap = max(percentile_value + self._margin, self._radius_floor)
            level_payload: Dict[str, Any] = {
                "cap": suggested_cap,
                "obs": obs_summary,
            }
            limit_chunks = data.get("limit", [])
            if limit_chunks:
                level_payload["limit"] = _metric_summary(np.concatenate(limit_chunks))
            cap_chunks = data.get("cap", [])
            if cap_chunks:
                level_payload["applied_caps"] = _metric_summary(np.concatenate(cap_chunks))
            payload["levels"][str(level)] = level_payload
        if combined_samples:
            combined = np.concatenate(combined_samples)
            payload["overview"] = _metric_summary(combined)
            payload["default"] = max(
                float(np.percentile(combined, percentile_pct)) + self._margin,
                self._radius_floor,
            )
        else:
            payload["overview"] = {}
            payload["default"] = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
