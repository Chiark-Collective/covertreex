from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from typing import Any, Mapping

from covertreex import config as cx_config, reset_residual_metric
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.telemetry import (
    BenchmarkLogWriter,
    ResidualScopeCapRecorder,
    generate_run_id,
    timestamped_artifact,
)

from cli.runtime import runtime_from_args
from tests.utils.datasets import gaussian_points

from .args import _parse_args
from .baselines import run_baseline_comparisons
from .benchmark import benchmark_knn_latency
from .gate import _append_gate_profile_log
from .runtime import (
    _emit_engine_banner,
    _ensure_thread_env_defaults,
    _gate_active_for_backend,
    _resolve_artifact_arg,
    _thread_env_snapshot,
)
from .telemetry import ResidualTraversalTelemetry


def _validate_residual_runtime(snapshot: Mapping[str, Any]) -> None:
    errors = []
    backend = snapshot.get("backend")
    if backend != "numpy":
        errors.append(f"expected backend 'numpy' but runtime selected {backend!r}")
    if not snapshot.get("enable_numba"):
        errors.append(
            "Numba acceleration is required for residual traversal; enable it by leaving "
            "COVERTREEX_ENABLE_NUMBA unset or passing --enable-numba=1."
        )
    if errors:
        joined = "\n - ".join(errors)
        raise RuntimeError(
            "Residual metric requires the residual traversal engine; unable to satisfy the"
            f" prerequisites:\n - {joined}"
        )


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or generate_run_id()
    if args.residual_gate and args.metric != "residual":
        raise ValueError("--residual-gate presets are only supported when --metric residual is selected.")

    if args.metric == "residual":
        profile_log_path = _resolve_artifact_arg(args.residual_gate_profile_log, category="profiles")
        if args.residual_gate_profile_path:
            profile_json_path = _resolve_artifact_arg(
                args.residual_gate_profile_path,
                category="profiles",
            )
        elif profile_log_path:
            profile_json_path = str(
                timestamped_artifact(
                    category="profiles",
                    prefix=f"residual_gate_profile_{run_id}",
                    suffix=".json",
                )
            )
        else:
            profile_json_path = None
        args.residual_gate_profile_path = profile_json_path
        args.residual_gate_profile_log = profile_log_path
    else:
        args.residual_gate_profile_path = None
        args.residual_gate_profile_log = None

    _ensure_thread_env_defaults()
    cli_runtime = runtime_from_args(args)
    runtime_snapshot = dict(cli_runtime.describe())
    if args.metric == "residual":
        _validate_residual_runtime(runtime_snapshot)
    cli_runtime.activate()
    thread_snapshot = _thread_env_snapshot()
    runtime_snapshot["runtime_blas_threads"] = thread_snapshot["blas_threads"]
    runtime_snapshot["runtime_numba_threads"] = thread_snapshot["numba_threads"]
    engine_label = "euclidean_dense"
    gate_flag = False
    if args.metric != "residual":
        runtime_snapshot["runtime_traversal_engine"] = engine_label
        runtime_snapshot["runtime_gate_active"] = gate_flag
        _emit_engine_banner(engine_label, gate_flag, thread_snapshot)
    log_metadata = {
        "benchmark": "cli.queries",
        "dimension": args.dimension,
        "tree_points": args.tree_points,
        "batch_size": args.batch_size,
        "queries": args.queries,
        "k": args.k,
        "metric": args.metric,
        "build_mode": args.build_mode,
        "baseline": args.baseline,
    }
    log_writer: BenchmarkLogWriter | None = None
    log_path: str | None = None
    scope_cap_recorder: ResidualScopeCapRecorder | None = None

    telemetry_view = ResidualTraversalTelemetry() if args.metric == "residual" else None

    try:
        if args.no_log_file:
            log_path = None
        elif args.log_file:
            log_path = _resolve_artifact_arg(args.log_file)
        else:
            log_path = str(
                timestamped_artifact(
                    category="benchmarks",
                    prefix=f"queries_{run_id}",
                    suffix=".jsonl",
                )
            )
        scope_cap_output = _resolve_artifact_arg(args.residual_scope_cap_output)
        if args.metric == "residual" and scope_cap_output:
            runtime_config = cx_config.runtime_config()
            scope_cap_recorder = ResidualScopeCapRecorder(
                output=scope_cap_output,
                percentile=args.residual_scope_cap_percentile,
                margin=args.residual_scope_cap_margin,
                radius_floor=runtime_config.residual_radius_floor,
            )
            scope_cap_recorder.annotate(
                run_id=run_id,
                log_file=log_path,
                tree_points=args.tree_points,
                batch_size=args.batch_size,
                scope_chunk_target=runtime_config.scope_chunk_target,
                scope_chunk_max_segments=runtime_config.scope_chunk_max_segments,
                residual_scope_cap_default=args.residual_scope_cap_default,
                seed=args.seed,
                build_mode=args.build_mode,
            )

        point_rng = default_rng(args.seed)
        points_np = gaussian_points(point_rng, args.tree_points, args.dimension, dtype=np.float64)
        query_rng = default_rng(args.seed + 1)
        queries_np = gaussian_points(query_rng, args.queries, args.dimension, dtype=np.float64)

        if args.metric == "residual":
            residual_backend = build_residual_backend(
                points_np,
                seed=args.seed,
                inducing_count=args.residual_inducing,
                variance=args.residual_variance,
                lengthscale=args.residual_lengthscale,
                chunk_size=args.residual_chunk_size,
            )
            configure_residual_correlation(residual_backend)
            gate_flag = _gate_active_for_backend(residual_backend)
            engine_label = "residual_serial" if gate_flag else "residual_parallel"
            runtime_snapshot["runtime_traversal_engine"] = engine_label
            runtime_snapshot["runtime_gate_active"] = gate_flag
            _emit_engine_banner(engine_label, gate_flag, thread_snapshot)
        else:
            runtime_snapshot.setdefault("runtime_traversal_engine", engine_label)
            runtime_snapshot.setdefault("runtime_gate_active", gate_flag)

        if log_path and log_writer is None:
            print(f"[queries] writing batch telemetry to {log_path}")
            log_writer = BenchmarkLogWriter(
                log_path,
                run_id=run_id,
                runtime=runtime_snapshot,
                metadata=log_metadata,
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
            plan_callback=telemetry_view.observe_plan if telemetry_view is not None else None,
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
        if telemetry_view is not None and telemetry_view.has_data:
            for line in telemetry_view.render_summary():
                print(line)
    finally:
        reset_residual_metric()
        if args.metric == "residual":
            _append_gate_profile_log(
                profile_json_path=args.residual_gate_profile_path,
                profile_log_path=args.residual_gate_profile_log,
                run_id=run_id,
                log_metadata=log_metadata,
                runtime_snapshot=runtime_snapshot,
                batch_log_path=log_path,
            )
        cx_config.reset_runtime_context()
        if scope_cap_recorder is not None:
            scope_cap_recorder.dump()
        if log_writer is not None:
            log_writer.close()


__all__ = ["main"]
