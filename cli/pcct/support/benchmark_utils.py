from __future__ import annotations

import time
from dataclasses import dataclass
import os
from typing import Any, Callable, Mapping, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from covertreex import config as cx_config
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.engine import CoverTree, build_tree as build_cover_tree
from covertreex.telemetry import BenchmarkLogWriter, ResidualScopeCapRecorder

from .runtime_utils import resolve_backend, measure_resources


def _ensure_context(context: cx_config.RuntimeContext | None) -> cx_config.RuntimeContext:
    existing = context or cx_config.current_runtime_context()
    if existing is not None:
        return existing
    return cx_config.runtime_context()


@dataclass(frozen=True)
class QueryBenchmarkResult:
    elapsed_seconds: float
    queries: int
    k: int
    latency_ms: float
    queries_per_second: float
    build_seconds: float | None = None
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    rss_delta_bytes: int = 0


def _generate_backend_points(
    rng: Generator,
    count: int,
    dimension: int,
    *,
    backend: TreeBackend,
) -> np.ndarray:
    from tests.utils.datasets import gaussian_points
    samples = gaussian_points(rng, count, dimension, dtype=np.float64)
    return backend.asarray(samples, dtype=backend.default_float)


def _ensure_residual_backend(
    points: np.ndarray,
    context: cx_config.RuntimeContext,
) -> None:
    from covertreex.metrics.residual import (
        ResidualCorrHostData,
        configure_residual_correlation,
        get_residual_backend,
    )
    try:
        get_residual_backend()
        return
    except RuntimeError:
        pass
    
    # Auto-configure dummy backend
    N, D = points.shape
    # Rank 16 default
    rank = 16
    rng = default_rng(42)
    V = rng.normal(size=(N, rank)).astype(np.float32)
    p_diag = rng.uniform(0.1, 1.0, size=N).astype(np.float32)
    kernel_diag = np.ones(N, dtype=np.float32)
    
    def dummy_provider(r, c):
        return np.zeros((r.size, c.size), dtype=np.float64)
        
    host_data = ResidualCorrHostData(
        v_matrix=V,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=dummy_provider,
        chunk_size=512
    )
    # Inject points for potential Numba usage or debugging
    object.__setattr__(host_data, "kernel_points_f32", points.astype(np.float32))
    
    configure_residual_correlation(host_data, context=context)


def _collect_rust_debug_stats() -> Mapping[str, int] | None:
    if os.environ.get("COVERTREEX_RUST_DEBUG_STATS") != "1":
        return None
    try:
        import covertreex_backend  # type: ignore
    except ImportError:
        return None
    getter = getattr(covertreex_backend, "get_rust_debug_stats", None)
    if getter is None:
        return None
    dist, pushes = getter()  # type: ignore[arg-type]
    return {
        "rust_distance_evals": int(dist),
        "rust_heap_pushes": int(pushes),
    }


def build_tree(
    *,
    dimension: int,
    tree_points: int,
    batch_size: int,
    seed: int,
    prebuilt_points: np.ndarray | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
    plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
    context: cx_config.RuntimeContext | None = None,
    residual_backend: Any | None = None,
    residual_params: Mapping[str, Any] | None = None,
) -> Tuple[CoverTree | PCCTree, np.ndarray, float]:
    from tests.utils.datasets import gaussian_points
    resolved_context = _ensure_context(context)
    runtime = resolved_context.config
    engine_name = getattr(runtime, "engine", None) or "python-numba"
    is_residual = runtime.metric.startswith("residual_correlation")

    if prebuilt_points is not None:
        points_np = np.asarray(prebuilt_points, dtype=np.float64, copy=False)
    else:
        rng = default_rng(seed)
        points_np = gaussian_points(rng, tree_points, dimension, dtype=np.float64)

    if is_residual and engine_name == "python-numba" and residual_backend is None:
        _ensure_residual_backend(points_np, resolved_context)

    start = time.perf_counter()
    tree = build_cover_tree(
        points_np,
        runtime=runtime,
        engine=engine_name,
        context=resolved_context,
        batch_size=batch_size,
        seed=seed,
        build_mode=build_mode,
        log_writer=log_writer,
        scope_cap_recorder=scope_cap_recorder,
        plan_callback=plan_callback,
        residual_backend=residual_backend,
        residual_params=residual_params,
    )
    build_seconds = tree.build_seconds if isinstance(tree, CoverTree) else None
    if build_seconds is None:
        build_seconds = time.perf_counter() - start

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
    prebuilt_tree: PCCTree | CoverTree | None = None,
    prebuilt_queries: np.ndarray | None = None,
    build_seconds: float | None = None,
    log_writer: BenchmarkLogWriter | None = None,
    scope_cap_recorder: "ResidualScopeCapRecorder | None" = None,
    build_mode: str = "batch",
    plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
    context: cx_config.RuntimeContext | None = None,
    residual_backend: Any | None = None,
    residual_params: Mapping[str, Any] | None = None,
    predecessor_mode: bool = False,
) -> Tuple[CoverTree | PCCTree, QueryBenchmarkResult]:
    from covertreex.queries.knn import knn
    
    resolved_context = _ensure_context(context)
    runtime = resolved_context.config
    is_residual = (
        runtime.metric == "residual_correlation"
        or runtime.residual_use_static_euclidean_tree
    )
    tree_build_seconds: float | None = None
    if prebuilt_tree is None:
        tree, _, tree_build_seconds = build_tree(
            dimension=dimension,
            tree_points=tree_points,
            batch_size=batch_size,
            seed=seed,
        prebuilt_points=prebuilt_points,
        log_writer=log_writer,
        scope_cap_recorder=scope_cap_recorder,
        build_mode=build_mode,
        plan_callback=plan_callback,
        context=resolved_context,
        residual_backend=residual_backend,
        residual_params=residual_params,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds
    if tree_build_seconds is None and isinstance(tree, CoverTree):
        tree_build_seconds = tree.build_seconds

    if scope_cap_recorder is not None and tree_build_seconds is not None:
        scope_cap_recorder.annotate(tree_build_seconds=tree_build_seconds)

    backend = getattr(tree, "backend", None)
    if backend is None:
        backend = resolve_backend(context=resolved_context)
    dataset_size = tree_points
    if isinstance(tree, CoverTree):
        dataset_size = int(tree.meta.get("dataset_size", dataset_size))
        handle = getattr(tree, "handle", None)
        if handle is not None and hasattr(handle, "num_points"):
            dataset_size = int(getattr(handle, "num_points"))
    elif hasattr(tree, "num_points"):
        dataset_size = int(tree.num_points)
    if prebuilt_queries is None:
        query_rng = default_rng(seed + 1)
        if is_residual:
            indices = query_rng.integers(
                0,
                dataset_size,
                size=query_count,
                endpoint=False,
                dtype=np.int64,
            ).reshape(-1, 1)
            queries = backend.asarray(indices, dtype=backend.default_int)
        else:
            queries = _generate_backend_points(
                query_rng,
                query_count,
                dimension,
                backend=backend,
            )
    else:
        qp = np.asarray(prebuilt_queries)
        if is_residual and qp.dtype.kind in {"i", "u"}:
            queries = backend.asarray(qp, dtype=backend.default_int)
        else:
            queries = backend.asarray(prebuilt_queries, dtype=backend.default_float)
    
    with measure_resources() as query_stats:
        knn(tree, queries, k=k, context=resolved_context, predecessor_mode=predecessor_mode)
    
    elapsed = query_stats['wall']
    qps = query_count / elapsed if elapsed > 0 else float("inf")
    latency = (elapsed / query_count) * 1e3 if query_count else 0.0
    bench_result = QueryBenchmarkResult(
        elapsed_seconds=elapsed,
        queries=query_count,
        k=k,
        latency_ms=latency,
        queries_per_second=qps,
        build_seconds=tree_build_seconds,
        cpu_user_seconds=query_stats['user'],
        cpu_system_seconds=query_stats['system'],
        rss_delta_bytes=query_stats['rss_delta'],
    )
    engine_name = getattr(runtime, "engine", None) or "python-numba"
    if log_writer is not None and not log_writer.has_records():
        extra_payload: dict[str, Any] = {
            "metric": runtime.metric,
            "engine": engine_name,
            "queries": query_count,
            "k": k,
            "query_seconds": elapsed,
            "query_qps": qps,
            "latency_ms": latency,
            "build_seconds": tree_build_seconds or 0.0,
            "dataset_size": dataset_size,
            # Keep residual aggregators happy even without conflict telemetry.
            "conflict_pairwise_reused": 1,
        }
        if runtime.metric == "residual_correlation" and engine_name in {"rust-fast", "rust-hybrid", "rust-pcct", "rust-pcct2"}:
            rust_stats = _collect_rust_debug_stats()
            if rust_stats:
                extra_payload.update(rust_stats)
        log_writer.record_event(event="query_summary", extra=extra_payload)
    return tree, bench_result


__all__ = [
    "QueryBenchmarkResult",
    "build_tree",
    "benchmark_knn_latency",
]
