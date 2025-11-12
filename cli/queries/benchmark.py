from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from covertreex import config as cx_config
from covertreex.algo import batch_insert, batch_insert_prefix_doubling
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.queries.knn import knn
from covertreex.telemetry import BenchmarkLogWriter, ResidualScopeCapRecorder
from tests.utils.datasets import gaussian_points

from .runtime import _resolve_backend


@dataclass(frozen=True)
class QueryBenchmarkResult:
    elapsed_seconds: float
    queries: int
    k: int
    latency_ms: float
    queries_per_second: float
    build_seconds: float | None = None


def _generate_backend_points(
    rng: Generator,
    count: int,
    dimension: int,
    *,
    backend: TreeBackend,
) -> np.ndarray:
    samples = gaussian_points(rng, count, dimension, dtype=np.float64)
    return backend.asarray(samples, dtype=backend.default_float)


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
    plan_callback: Callable[[Any, int, int], None] | None = None,
) -> Tuple[PCCTree, np.ndarray, float]:
    backend = _resolve_backend()
    tree = PCCTree.empty(dimension=dimension, backend=backend)

    if build_mode == "prefix":
        if prebuilt_points is not None:
            points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
        else:
            rng = default_rng(seed)
            points_np = gaussian_points(rng, tree_points, dimension, dtype=np.float64)
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
        runtime = cx_config.runtime_config()
        schedule = runtime.prefix_schedule
        for group_index, group in enumerate(prefix_result.groups):
            plan = group.plan
            if hasattr(plan.traversal, "parents"):
                group_size = int(plan.traversal.parents.shape[0])
            else:
                group_size = int(plan.traversal.levels.shape[0])
            if log_writer is not None:
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
            if plan_callback is not None:
                plan_callback(plan, group_index, group_size)
        return tree, points_np, build_seconds

    start = time.perf_counter()
    buffers: list[np.ndarray] = []
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
            if plan_callback is not None:
                plan_callback(plan, idx, int(batch_np.shape[0]))
            buffers.append(np.asarray(batch))
            idx += 1
    else:
        rng = default_rng(seed)
        remaining = tree_points
        while remaining > 0:
            current = min(batch_size, remaining)
            batch = _generate_backend_points(
                rng,
                current,
                dimension,
                backend=backend,
            )
            tree, plan = batch_insert(tree, batch, mis_seed=seed + idx)
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=current,
                    plan=plan,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            if plan_callback is not None:
                plan_callback(plan, idx, current)
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
    plan_callback: Callable[[Any, int, int], None] | None = None,
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
            plan_callback=plan_callback,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds

    if scope_cap_recorder is not None and tree_build_seconds is not None:
        scope_cap_recorder.annotate(tree_build_seconds=tree_build_seconds)

    backend = tree.backend
    if prebuilt_queries is None:
        query_rng = default_rng(seed + 1)
        queries = _generate_backend_points(
            query_rng,
            query_count,
            dimension,
            backend=backend,
        )
    else:
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


__all__ = [
    "QueryBenchmarkResult",
    "_build_tree",
    "benchmark_knn_latency",
]
