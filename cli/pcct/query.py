from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import default_rng

from covertreex.metrics import build_residual_backend, configure_residual_correlation

from .support.baseline_utils import run_baseline_comparisons
from .support.benchmark_utils import QueryBenchmarkResult, benchmark_knn_latency
from .support.runtime_utils import emit_engine_banner as _emit_engine_banner
from tests.utils.datasets import gaussian_points

from .execution import BenchmarkRun

if TYPE_CHECKING:  # pragma: no cover
    from .options import QueryCLIOptions


def _generate_datasets(options: "QueryCLIOptions") -> tuple[np.ndarray, np.ndarray]:
    point_rng = default_rng(options.seed)
    points = gaussian_points(point_rng, options.tree_points, options.dimension, dtype=np.float64)
    if options.metric == "residual":
        query_rng = default_rng(options.seed + 1)
        query_indices = query_rng.integers(
            0,
            options.tree_points,
            size=options.queries,
            endpoint=False,
            dtype=np.int64,
        )
        queries = query_indices.reshape(-1, 1)
    else:
        query_rng = default_rng(options.seed + 1)
        queries = gaussian_points(query_rng, options.queries, options.dimension, dtype=np.float64)
    return points, queries


def _run_residual_backend(options: "QueryCLIOptions", points: np.ndarray, *, context) -> tuple[str, bool, Any]:
    runtime_cfg = context.config
    seed_pack = runtime_cfg.seeds
    residual_seed = seed_pack.resolved("residual_grid", fallback=seed_pack.resolved("mis"))
    residual_backend = build_residual_backend(
        points,
        seed=residual_seed,
        inducing_count=options.residual_inducing,
        variance=options.residual_variance,
        lengthscale=options.residual_lengthscale,
        chunk_size=options.residual_chunk_size,
        kernel_type=options.residual_kernel_type,
    )
    configure_residual_correlation(residual_backend, context=context)
    gate_active = False
    engine_label = "residual_parallel"
    return engine_label, gate_active, residual_backend


def _print_baseline_results(
    options: "QueryCLIOptions",
    points: np.ndarray,
    queries: np.ndarray,
    benchmark_result,
) -> None:
    if options.metric == "residual" and queries.ndim == 2 and queries.shape[1] == 1:
        idx = np.asarray(queries, dtype=np.int64).reshape(-1)
        queries = points[idx % points.shape[0]]
    baseline_results = run_baseline_comparisons(
        points,
        queries,
        k=options.k,
        mode=options.baseline,
    )
    for baseline in baseline_results:
        slowdown = (
            baseline.latency_ms / benchmark_result.latency_ms if benchmark_result.latency_ms else float("inf")
        )
        cpu_time = (baseline.cpu_user_seconds or 0.0) + (baseline.cpu_system_seconds or 0.0)
        print(
            f"baseline[{baseline.name}] | build={baseline.build_seconds:.4f}s "
            f"time={baseline.elapsed_seconds:.4f}s "
            f"latency={baseline.latency_ms:.4f}ms "
            f"throughput={baseline.queries_per_second:,.1f} q/s "
            f"cpu={cpu_time:.4f}s "
            f"rss_delta={baseline.rss_delta_bytes / 1024 / 1024:.2f}MB "
            f"slowdown={slowdown:.3f}x"
        )


def execute_query_benchmark(options: "QueryCLIOptions", run: BenchmarkRun) -> QueryBenchmarkResult:
    """Run the k-NN benchmark with telemetry/baseline output."""

    runtime_snapshot = run.runtime_snapshot
    thread_snapshot = run.thread_snapshot
    telemetry_view = run.telemetry_view
    log_writer = run.log_writer
    scope_cap_recorder = run.scope_cap_recorder
    context = run.context

    engine_label = context.config.engine or "python-numba"
    gate_flag = False
    residual_backend: Any | None = None
    if options.metric == "residual-lite":
        runtime_snapshot["runtime_traversal_engine"] = f"{engine_label}-residual-lite"
        runtime_snapshot["runtime_gate_active"] = gate_flag
        _emit_engine_banner(runtime_snapshot["runtime_traversal_engine"], thread_snapshot)
    elif options.metric != "residual":
        runtime_snapshot["runtime_traversal_engine"] = engine_label
        runtime_snapshot["runtime_gate_active"] = gate_flag
        _emit_engine_banner(engine_label, thread_snapshot)

    points_np, queries_np = _generate_datasets(options)

    if options.metric == "residual":
        if engine_label == "python-numba":
            backend_label, gate_flag, residual_backend = _run_residual_backend(options, points_np, context=context)
            engine_label = f"{engine_label}:{backend_label}"
        runtime_snapshot["runtime_traversal_engine"] = engine_label
        runtime_snapshot["runtime_gate_active"] = gate_flag
        _emit_engine_banner(engine_label, thread_snapshot)
    elif options.metric != "residual-lite":
        runtime_snapshot.setdefault("runtime_traversal_engine", engine_label)
        runtime_snapshot.setdefault("runtime_gate_active", gate_flag)

    if (
        options.metric == "residual"
        and residual_backend is None
        and np.issubdtype(points_np.dtype, np.integer)
        and points_np.ndim == 2
        and points_np.shape[1] == 1
    ):
        raise RuntimeError(
            "Residual benchmark received integer-valued 1D points without a configured residual backend; "
            "this would degrade to Euclidean-on-indices. Provide float coordinates or configure the residual backend."
        )

    tree, result = benchmark_knn_latency(
        dimension=options.dimension,
        tree_points=options.tree_points,
        query_count=options.queries,
        k=options.k,
        batch_size=options.batch_size,
        seed=options.seed,
        prebuilt_points=points_np,
        prebuilt_queries=queries_np,
        log_writer=log_writer,
        scope_cap_recorder=scope_cap_recorder,
        build_mode=options.build_mode,
        plan_callback=telemetry_view.observe_plan if telemetry_view is not None else None,
        context=context,
        residual_backend=residual_backend,
        residual_params={
            "variance": options.residual_variance,
            "lengthscale": options.residual_lengthscale,
            "inducing_count": options.residual_inducing,
            "chunk_size": options.residual_chunk_size,
            "kernel_type": options.residual_kernel_type,
        },
        predecessor_mode=options.predecessor_mode or False,
    )

    cpu_time = (result.cpu_user_seconds or 0.0) + (result.cpu_system_seconds or 0.0)
    print(
        f"pcct | build={result.build_seconds:.4f}s "
        f"queries={result.queries} k={result.k} "
        f"time={result.elapsed_seconds:.4f}s "
        f"latency={result.latency_ms:.4f}ms "
        f"throughput={result.queries_per_second:,.1f} q/s "
        f"cpu={cpu_time:.4f}s "
        f"rss_delta={result.rss_delta_bytes / 1024 / 1024:.2f}MB"
    )

    if options.baseline != "none":
        _print_baseline_results(options, points_np, queries_np, result)

    if telemetry_view is not None and telemetry_view.has_data:
        for line in telemetry_view.render_summary():
            print(line)

    return result


__all__ = ["execute_query_benchmark"]
