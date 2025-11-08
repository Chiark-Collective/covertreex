from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - platform specific fallback
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng

from covertreex import reset_residual_metric
from covertreex import config as cx_config
from covertreex.algo import batch_insert, batch_insert_prefix_doubling
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics import ResidualCorrHostData, configure_residual_correlation
from covertreex.queries.knn import knn
from covertreex.baseline import (
    BaselineCoverTree,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
)


_RESIDUAL_RUNTIME_ENV_VARS = (
    "COVERTREEX_ENABLE_SPARSE_TRAVERSAL",
    "COVERTREEX_RESIDUAL_GATE1",
    "COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH",
    "COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN",
    "COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP",
    "COVERTREEX_RESIDUAL_GATE1_AUDIT",
    "COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH",
    "COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT",
)


def _apply_residual_gate_preset(args: argparse.Namespace) -> None:
    if args.residual_gate is None:
        return
    if args.residual_gate == "off":
        os.environ["COVERTREEX_RESIDUAL_GATE1"] = "0"
        return
    if args.residual_gate != "lookup":
        return
    os.environ["COVERTREEX_ENABLE_SPARSE_TRAVERSAL"] = "1"
    os.environ["COVERTREEX_RESIDUAL_GATE1"] = "1"
    if args.residual_gate_lookup_path:
        os.environ["COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH"] = args.residual_gate_lookup_path
    margin = args.residual_gate_margin
    if margin is not None:
        os.environ["COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN"] = str(margin)
    cap = args.residual_gate_cap
    if cap and cap > 0:
        os.environ["COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP"] = str(cap)
    os.environ.setdefault("COVERTREEX_RESIDUAL_GATE1_AUDIT", "1")


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


class BenchmarkLogWriter:
    def __init__(self, path: str):
        self._path = Path(path).expanduser()
        if self._path.parent:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._previous_rss = _read_rss_bytes()

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
            "timestamp": time.time(),
            "batch_index": int(batch_index),
            "batch_size": int(batch_size),
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
            "conflict_scope_chunk_segments": int(conflict_timings.scope_chunk_segments),
            "conflict_scope_chunk_emitted": int(conflict_timings.scope_chunk_emitted),
            "conflict_scope_chunk_max_members": int(conflict_timings.scope_chunk_max_members),
            "conflict_scope_chunk_pair_cap": int(conflict_timings.scope_chunk_pair_cap),
            "conflict_scope_chunk_pairs_before": int(conflict_timings.scope_chunk_pairs_before),
            "conflict_scope_chunk_pairs_after": int(conflict_timings.scope_chunk_pairs_after),
            "conflict_scope_domination_ratio": float(conflict_timings.scope_domination_ratio),
            "traversal_gate1_candidates": int(traversal_timings.gate1_candidates),
            "traversal_gate1_kept": int(traversal_timings.gate1_kept),
            "traversal_gate1_pruned": int(traversal_timings.gate1_pruned),
            "traversal_gate1_ms": _ms(traversal_timings.gate1_seconds),
            "batch_order_strategy": plan.batch_order_strategy,
            "conflict_grid_cells": int(plan.conflict_graph.grid_cells),
            "conflict_grid_leaders_raw": int(plan.conflict_graph.grid_leaders_raw),
            "conflict_grid_leaders_after": int(plan.conflict_graph.grid_leaders_after),
            "conflict_grid_local_edges": int(plan.conflict_graph.grid_local_edges),
        }
        if plan.batch_permutation is not None:
            record["batch_order_permutation_size"] = int(len(plan.batch_permutation))
        for key, value in plan.batch_order_metrics.items():
            record[f"batch_order_{key}"] = float(value)
        if traversal_timings.gate1_candidates:
            ratio = (
                traversal_timings.gate1_pruned
                / traversal_timings.gate1_candidates
            )
            record["traversal_gate1_pruned_ratio"] = float(ratio)
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

        self._handle.write(json.dumps(record, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()

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
        self._levels: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: {"obs": [], "limit": [], "cap": []})
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
            "schema": 1,
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


@dataclass(frozen=True)
class QueryBenchmarkResult:
    elapsed_seconds: float
    queries: int
    k: int
    latency_ms: float
    queries_per_second: float
    build_seconds: float | None = None


@dataclass(frozen=True)
class BaselineComparison:
    name: str
    build_seconds: float
    elapsed_seconds: float
    latency_ms: float
    queries_per_second: float


def _resolve_backend() -> TreeBackend:
    runtime = cx_config.runtime_config()
    if runtime.backend == "jax":
        return TreeBackend.jax(precision=runtime.precision)
    if runtime.backend == "numpy":
        return TreeBackend.numpy(precision=runtime.precision)
    raise NotImplementedError(f"Backend '{runtime.backend}' is not supported yet.")


def _generate_points_backend(rng: Generator, count: int, dimension: int) -> np.ndarray:
    samples = rng.normal(loc=0.0, scale=1.0, size=(count, dimension))
    backend = _resolve_backend()
    return backend.asarray(samples, dtype=backend.default_float)


def _generate_points_numpy(rng: Generator, count: int, dimension: int) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(count, dimension)).astype(np.float64, copy=False)


def _rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    return variance * np.exp(-0.5 * sq_dist / (lengthscale * lengthscale))


def _build_residual_backend(
    points: np.ndarray,
    *,
    seed: int,
    inducing_count: int,
    variance: float,
    lengthscale: float,
    chunk_size: int,
) -> ResidualCorrHostData:
    if points.size == 0:
        raise ValueError("Residual metric requires at least one point to configure caches.")

    rng = default_rng(seed)
    n_points = points.shape[0]
    inducing = min(inducing_count, n_points)
    if inducing <= 0:
        inducing = min(32, n_points)
    if inducing < n_points:
        inducing_idx = np.sort(rng.choice(n_points, size=inducing, replace=False))
    else:
        inducing_idx = np.arange(n_points)
    inducing_points = points[inducing_idx]

    k_mm = _rbf_kernel(inducing_points, inducing_points, variance=variance, lengthscale=lengthscale)
    jitter = 1e-6 * variance
    k_mm += np.eye(inducing_points.shape[0], dtype=np.float64) * jitter
    l_mm = np.linalg.cholesky(k_mm)

    k_xm = _rbf_kernel(points, inducing_points, variance=variance, lengthscale=lengthscale)
    solve_result = np.linalg.solve(l_mm, k_xm.T)
    v_matrix = solve_result.T

    kernel_diag = np.full(n_points, variance, dtype=np.float64)
    p_diag = np.maximum(kernel_diag - np.sum(v_matrix * v_matrix, axis=1), 1e-9)

    points_contig = np.ascontiguousarray(points, dtype=np.float64)
    point_keys = [tuple(row.tolist()) for row in points_contig]
    index_map: dict[tuple[float, ...], int] = {}
    for idx, key in enumerate(point_keys):
        index_map.setdefault(key, idx)

    def point_decoder(values: ArrayLike) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != points_contig.shape[1]:
            raise ValueError(
                "Residual point decoder expected payload dimensionality "
                f"{points_contig.shape[1]}, received {arr.shape[1]}."
            )
        rows = np.ascontiguousarray(arr, dtype=np.float64)
        indices = np.empty(rows.shape[0], dtype=np.int64)
        for i, row in enumerate(rows):
            key = tuple(row.tolist())
            if key not in index_map:
                raise KeyError("Residual point decoder received unknown payload.")
            indices[i] = index_map[key]
        return indices

    def kernel_provider(row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        rows = points[np.asarray(row_indices, dtype=np.int64, copy=False)]
        cols = points[np.asarray(col_indices, dtype=np.int64, copy=False)]
        return _rbf_kernel(rows, cols, variance=variance, lengthscale=lengthscale)

    host_backend = ResidualCorrHostData(
        v_matrix=np.asarray(v_matrix, dtype=np.float64, copy=False),
        p_diag=np.asarray(p_diag, dtype=np.float64, copy=False),
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        point_decoder=point_decoder,
        chunk_size=int(chunk_size),
    )

    configure_residual_correlation(host_backend)
    return host_backend


def _build_tree(
    *,
    dimension: int,
    tree_points: int,
    batch_size: int,
    seed: int,
    prebuilt_points: np.ndarray | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
) -> Tuple[PCCTree, np.ndarray, float]:
    backend = _resolve_backend()
    tree = PCCTree.empty(dimension=dimension, backend=backend)

    if build_mode == "prefix":
        if prebuilt_points is not None:
            points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        else:
            rng = default_rng(seed)
            points_np = _generate_points_numpy(rng, tree_points, dimension)
        batch = backend.asarray(points_np, dtype=backend.default_float)
        start = time.perf_counter()
        tree, prefix_result = batch_insert_prefix_doubling(
            tree,
            batch,
            backend=backend,
            mis_seed=seed,
            shuffle_seed=seed,
        )
        build_seconds = time.perf_counter() - start
        if log_writer is not None:
            runtime = cx_config.runtime_config()
            schedule = runtime.prefix_schedule
            for group_index, group in enumerate(prefix_result.groups):
                plan = group.plan
                if hasattr(plan.traversal, "parents"):
                    group_size = int(plan.traversal.parents.shape[0])
                else:
                    group_size = int(plan.traversal.levels.shape[0])
                extra = {
                    "prefix_group_index": group_index,
                    "prefix_factor": float(group.prefix_factor or 0.0),
                    "prefix_domination_ratio": float(group.domination_ratio or 0.0),
                    "prefix_schedule": schedule,
                }
                log_writer.record_batch(
                    batch_index=group_index,
                    batch_size=group_size,
                    plan=plan,
                    extra=extra,
                )
                if scope_cap_recorder is not None:
                    scope_cap_recorder.capture(plan)
        else:
            if scope_cap_recorder is not None:
                for group in prefix_result.groups:
                    scope_cap_recorder.capture(group.plan)
        return tree, points_np, build_seconds

    start = time.perf_counter()
    buffers: List[np.ndarray] = []
    idx = 0

    if prebuilt_points is not None:
        points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        total = points_np.shape[0]
        while idx * batch_size < total:
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_np = points_np[start_idx:end_idx]
            batch = backend.asarray(batch_np, dtype=backend.default_float)
            tree, plan = batch_insert(tree, batch, mis_seed=seed + idx)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=int(batch_np.shape[0]),
                    plan=plan,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            buffers.append(np.asarray(batch))
            idx += 1
    else:
        rng = default_rng(seed)
        remaining = tree_points
        while remaining > 0:
            current = min(batch_size, remaining)
            batch = _generate_points_backend(rng, current, dimension)
            tree, plan = batch_insert(tree, batch, mis_seed=seed + idx)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=current,
                    plan=plan,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            buffers.append(np.asarray(batch))
            remaining -= current
            idx += 1

    build_seconds = time.perf_counter() - start
    if buffers:
        points_np = np.concatenate(buffers, axis=0)
    else:
        points_np = np.empty((0, dimension), dtype=np.float64)
    return tree, points_np, build_seconds


def benchmark_knn_latency(
    *,
    dimension: int,
    tree_points: int,
    query_count: int,
    k: int,
    batch_size: int,
    seed: int,
    prebuilt_points: np.ndarray | None = None,
    prebuilt_tree: PCCTree | None = None,
    prebuilt_queries: np.ndarray | None = None,
    build_seconds: float | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
) -> Tuple[PCCTree, QueryBenchmarkResult]:
    tree_build_seconds: float | None = None
    if prebuilt_tree is None:
        tree, _, tree_build_seconds = _build_tree(
            dimension=dimension,
            tree_points=tree_points,
            batch_size=batch_size,
            seed=seed,
            prebuilt_points=prebuilt_points,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            build_mode=build_mode,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds

    if scope_cap_recorder is not None and tree_build_seconds is not None:
        scope_cap_recorder.annotate(tree_build_seconds=tree_build_seconds)

    if prebuilt_queries is None:
        query_rng = default_rng(seed + 1)
        queries = _generate_points_backend(query_rng, query_count, dimension)
    else:
        backend = _resolve_backend()
        queries = backend.asarray(
            prebuilt_queries, dtype=backend.default_float
        )
    start = time.perf_counter()
    knn(tree, queries, k=k)
    elapsed = time.perf_counter() - start
    qps = query_count / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / query_count) * 1e3 if query_count else 0.0
    return tree, QueryBenchmarkResult(
        elapsed_seconds=elapsed,
        queries=query_count,
        k=k,
        latency_ms=latency,
        queries_per_second=qps,
        build_seconds=tree_build_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched k-NN query latency for the PCCT implementation."
    )
    parser.add_argument("--dimension", type=int, default=8, help="Dimensionality of points.")
    parser.add_argument(
        "--tree-points",
        type=int,
        default=16_384,
        help="Number of points to populate the tree with before querying.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used while constructing the tree.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1024,
        help="Number of query points to evaluate.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of neighbours to request.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--metric",
        choices=("euclidean", "residual"),
        default="euclidean",
        help="Distance metric to benchmark (configures residual caches when 'residual').",
    )
    parser.add_argument(
        "--residual-lengthscale",
        type=float,
        default=1.0,
        help="RBF kernel lengthscale for synthetic residual caches.",
    )
    parser.add_argument(
        "--residual-variance",
        type=float,
        default=1.0,
        help="RBF kernel variance for synthetic residual caches.",
    )
    parser.add_argument(
        "--residual-inducing",
        type=int,
        default=512,
        help="Number of inducing points to use when building residual caches.",
    )
    parser.add_argument(
        "--residual-chunk-size",
        type=int,
        default=512,
        help="Chunk size for residual kernel streaming.",
    )
    parser.add_argument(
        "--baseline",
        choices=("none", "sequential", "gpboost", "external", "both", "all"),
        default="none",
        help=(
            "Include baseline comparisons. Install '.[baseline]' for the external library and "
            "'numba' extra for the GPBoost baseline. Options: 'sequential', 'gpboost', "
            "'external', 'both' (sequential + external), 'all' (sequential + gpboost + external)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write per-batch telemetry as JSON lines to the specified path.",
    )
    parser.add_argument(
        "--batch-order",
        choices=("natural", "random", "hilbert"),
        default=None,
        help="Override COVERTREEX_BATCH_ORDER for this run.",
    )
    parser.add_argument(
        "--batch-order-seed",
        type=int,
        default=None,
        help="Override COVERTREEX_BATCH_ORDER_SEED for this run.",
    )
    parser.add_argument(
        "--prefix-schedule",
        choices=("doubling", "adaptive"),
        default=None,
        help="Override COVERTREEX_PREFIX_SCHEDULE for this run.",
    )
    parser.add_argument(
        "--build-mode",
        choices=("batch", "prefix"),
        default="batch",
        help="Choose the tree construction strategy (standard batch inserts or prefix doubling).",
    )
    parser.add_argument(
        "--residual-gate",
        choices=("off", "lookup"),
        default=None,
        help="Residual-only: automatically configure Gate-1 (e.g. 'lookup' wires sparse traversal + lookup table).",
    )
    parser.add_argument(
        "--residual-gate-lookup-path",
        type=str,
        default="docs/data/residual_gate_profile_diag0.json",
        help="Lookup JSON used when --residual-gate=lookup (default: diag0 profile).",
    )
    parser.add_argument(
        "--residual-gate-margin",
        type=float,
        default=0.02,
        help="Safety margin added to lookup thresholds when --residual-gate=lookup.",
    )
    parser.add_argument(
        "--residual-gate-cap",
        type=float,
        default=0.0,
        help="Optional radius cap passed to the lookup preset (0 keeps existing env/default).",
    )
    parser.add_argument(
        "--residual-scope-caps",
        type=str,
        default=None,
        help="Residual-only: JSON file describing per-level scope radius caps.",
    )
    parser.add_argument(
        "--residual-scope-cap-default",
        type=float,
        default=None,
        help="Residual-only: fallback radius cap applied when no per-level cap matches.",
    )
    parser.add_argument(
        "--residual-scope-cap-output",
        type=str,
        default=None,
        help="Residual-only: write derived per-level scope caps to this JSON file.",
    )
    parser.add_argument(
        "--residual-scope-cap-percentile",
        type=float,
        default=0.5,
        help="Quantile (0-1) used when deriving new scope caps (default: median).",
    )
    parser.add_argument(
        "--residual-scope-cap-margin",
        type=float,
        default=0.05,
        help="Safety margin added to the sampled percentile when deriving scope caps.",
    )
    return parser.parse_args()


def _run_sequential_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    start_build = time.perf_counter()
    tree = BaselineCoverTree.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="sequential",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def _run_external_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_external_cover_tree():
        raise RuntimeError("External cover tree baseline requested but `covertree` is not available.")
    start_build = time.perf_counter()
    tree = ExternalCoverTreeBaseline.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="external",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def _run_gpboost_baseline(points: np.ndarray, queries: np.ndarray, *, k: int) -> BaselineComparison:
    if not has_gpboost_cover_tree():
        raise RuntimeError(
            "GPBoost cover tree baseline requested but 'numba' extra is not installed."
        )
    start_build = time.perf_counter()
    tree = GPBoostCoverTreeBaseline.from_points(points)
    build_seconds = time.perf_counter() - start_build
    start = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    elapsed = time.perf_counter() - start
    qps = queries.shape[0] / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / queries.shape[0]) * 1e3 if queries.shape[0] else 0.0
    return BaselineComparison(
        name="gpboost",
        build_seconds=build_seconds,
        elapsed_seconds=elapsed,
        latency_ms=latency,
        queries_per_second=qps,
    )


def run_baseline_comparisons(
    points: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    mode: str,
) -> List[BaselineComparison]:
    queries = np.asarray(queries, dtype=float)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    results: List[BaselineComparison] = []
    if mode in ("sequential", "both", "all"):
        results.append(_run_sequential_baseline(points, queries, k=k))
    if mode in ("gpboost", "all"):
        results.append(_run_gpboost_baseline(points, queries, k=k))
    if mode in ("external", "both", "all"):
        results.append(_run_external_baseline(points, queries, k=k))
    return results


def main() -> None:
    args = _parse_args()
    if args.residual_gate and args.metric != "residual":
        raise ValueError("--residual-gate presets are only supported when --metric residual is selected.")
    previous_metric = os.environ.get("COVERTREEX_METRIC")
    previous_backend = os.environ.get("COVERTREEX_BACKEND")
    previous_batch_order = os.environ.get("COVERTREEX_BATCH_ORDER")
    previous_batch_order_seed = os.environ.get("COVERTREEX_BATCH_ORDER_SEED")
    previous_prefix_schedule = os.environ.get("COVERTREEX_PREFIX_SCHEDULE")
    previous_gate_env = {var: os.environ.get(var) for var in _RESIDUAL_RUNTIME_ENV_VARS}
    log_writer: BenchmarkLogWriter | None = None
    scope_cap_recorder: ResidualScopeCapRecorder | None = None

    try:
        if args.log_file:
            log_writer = BenchmarkLogWriter(args.log_file)
        if args.metric == "residual":
            os.environ["COVERTREEX_METRIC"] = "residual_correlation"
            os.environ["COVERTREEX_BACKEND"] = "numpy"
        else:
            os.environ["COVERTREEX_METRIC"] = "euclidean"
        if args.batch_order:
            os.environ["COVERTREEX_BATCH_ORDER"] = args.batch_order
        if args.batch_order_seed is not None:
            os.environ["COVERTREEX_BATCH_ORDER_SEED"] = str(args.batch_order_seed)
        if args.prefix_schedule:
            os.environ["COVERTREEX_PREFIX_SCHEDULE"] = args.prefix_schedule
        if args.metric == "residual":
            _apply_residual_gate_preset(args)
            if args.residual_scope_caps:
                os.environ["COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH"] = args.residual_scope_caps
            if args.residual_scope_cap_default is not None:
                os.environ["COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT"] = str(args.residual_scope_cap_default)
        cx_config.reset_runtime_config_cache()

        if args.metric == "residual" and args.residual_scope_cap_output:
            runtime = cx_config.runtime_config()
            scope_cap_recorder = ResidualScopeCapRecorder(
                output=args.residual_scope_cap_output,
                percentile=args.residual_scope_cap_percentile,
                margin=args.residual_scope_cap_margin,
                radius_floor=runtime.residual_radius_floor,
            )
            scope_cap_recorder.annotate(
                tree_points=args.tree_points,
                batch_size=args.batch_size,
                scope_chunk_target=runtime.scope_chunk_target,
                scope_chunk_max_segments=runtime.scope_chunk_max_segments,
                residual_scope_cap_default=args.residual_scope_cap_default,
                seed=args.seed,
                build_mode=args.build_mode,
            )

        point_rng = default_rng(args.seed)
        points_np = _generate_points_numpy(point_rng, args.tree_points, args.dimension)
        query_rng = default_rng(args.seed + 1)
        queries_np = _generate_points_numpy(query_rng, args.queries, args.dimension)

        if args.metric == "residual":
            _build_residual_backend(
                points_np,
                seed=args.seed,
                inducing_count=args.residual_inducing,
                variance=args.residual_variance,
                lengthscale=args.residual_lengthscale,
                chunk_size=args.residual_chunk_size,
            )

        tree, result = benchmark_knn_latency(
            dimension=args.dimension,
            tree_points=args.tree_points,
            query_count=args.queries,
            k=args.k,
            batch_size=args.batch_size,
            seed=args.seed,
            prebuilt_points=points_np,
            prebuilt_queries=queries_np,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            build_mode=args.build_mode,
        )

        print(
            f"pcct | build={result.build_seconds:.4f}s "
            f"queries={result.queries} k={result.k} "
            f"time={result.elapsed_seconds:.4f}s "
            f"latency={result.latency_ms:.4f}ms "
            f"throughput={result.queries_per_second:,.1f} q/s"
        )

        if args.baseline != "none":
            baseline_results = run_baseline_comparisons(
                points_np,
                queries_np,
                k=args.k,
                mode=args.baseline,
            )
            for baseline in baseline_results:
                slowdown = (
                    baseline.latency_ms / result.latency_ms if result.latency_ms else float("inf")
                )

                print(
                    f"baseline[{baseline.name}] | build={baseline.build_seconds:.4f}s "
                    f"time={baseline.elapsed_seconds:.4f}s "
                    f"latency={baseline.latency_ms:.4f}ms "
                    f"throughput={baseline.queries_per_second:,.1f} q/s "
                    f"slowdown={slowdown:.3f}x"
                )
    finally:
        reset_residual_metric()
        if previous_metric is not None:
            os.environ["COVERTREEX_METRIC"] = previous_metric
        else:
            os.environ.pop("COVERTREEX_METRIC", None)
        if previous_backend is not None:
            os.environ["COVERTREEX_BACKEND"] = previous_backend
        else:
            os.environ.pop("COVERTREEX_BACKEND", None)
        if previous_batch_order is not None:
            os.environ["COVERTREEX_BATCH_ORDER"] = previous_batch_order
        else:
            os.environ.pop("COVERTREEX_BATCH_ORDER", None)
        if previous_batch_order_seed is not None:
            os.environ["COVERTREEX_BATCH_ORDER_SEED"] = previous_batch_order_seed
        else:
            os.environ.pop("COVERTREEX_BATCH_ORDER_SEED", None)
        if previous_prefix_schedule is not None:
            os.environ["COVERTREEX_PREFIX_SCHEDULE"] = previous_prefix_schedule
        else:
            os.environ.pop("COVERTREEX_PREFIX_SCHEDULE", None)
        for var, value in previous_gate_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        cx_config.reset_runtime_config_cache()
        if scope_cap_recorder is not None:
            scope_cap_recorder.dump()
        if log_writer is not None:
            log_writer.close()


if __name__ == "__main__":
    main()
