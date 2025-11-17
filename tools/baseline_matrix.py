#!/usr/bin/env python3
"""Profile-driven PCCT vs cover-tree baseline sweeps with resource telemetry."""
from __future__ import annotations

import argparse
import json
import resource
import time
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import psutil

from covertreex import config as cx_config
from covertreex.api import Runtime
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from cli.queries.baselines import run_baseline_comparisons
from cli.queries.benchmark import benchmark_knn_latency
from tests.utils.datasets import gaussian_points

logging.getLogger("covertreex").setLevel(logging.WARNING)
logging.getLogger("covertreex.algo").setLevel(logging.WARNING)
logging.getLogger("covertreex.queries").setLevel(logging.WARNING)


def _default_list(values: Iterable[int] | None, fallback: Sequence[int]) -> List[int]:
    return list(values) if values else list(fallback)


def _artifact_path(path: str | None) -> Path:
    resolved = Path(path).expanduser() if path else Path("artifacts/benchmarks/baseline_matrix.jsonl")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _generate_datasets(dimension: int, tree_points: int, query_count: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    tree_rng = np.random.default_rng(seed)
    query_rng = np.random.default_rng(seed + 1)
    points = gaussian_points(tree_rng, tree_points, dimension, dtype=np.float64)
    queries = gaussian_points(query_rng, query_count, dimension, dtype=np.float64)
    return points, queries


def _measure(func):
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    result = func()
    wall_seconds = time.perf_counter() - start
    cpu_after = proc.cpu_times()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    cpu_util = cpu_seconds / wall_seconds if wall_seconds > 0 else 0.0
    rss_peak_bytes = max(0, (usage_after.ru_maxrss - usage_before.ru_maxrss) * 1024)
    return result, {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization": cpu_util,
        "rss_peak_bytes": float(rss_peak_bytes),
    }


def _activate_runtime(profile: str, metric: str) -> cx_config.RuntimeContext:
    runtime = Runtime.from_profile(profile)
    runtime = runtime.with_updates(metric=metric)
    return runtime.activate()


def _run_pcct(
    *,
    dimension: int,
    tree_points: int,
    query_count: int,
    k: int,
    batch_size: int,
    seed: int,
    profile: str,
    metric: str,
    residual_inducing: int,
    residual_variance: float,
    residual_lengthscale: float,
    residual_chunk_size: int,
    baseline_mode: str,
) -> Dict[str, object]:
    context = _activate_runtime(profile, metric)
    points, queries = _generate_datasets(dimension, tree_points, query_count, seed)
    try:
        if metric == "residual":
            runtime_cfg = context.config
            seed_pack = runtime_cfg.seeds
            residual_seed = seed_pack.resolved("residual_grid", fallback=seed_pack.resolved("mis"))
            residual_backend = build_residual_backend(
                points,
                seed=residual_seed,
                inducing_count=residual_inducing,
                variance=residual_variance,
                lengthscale=residual_lengthscale,
                chunk_size=residual_chunk_size,
            )
            configure_residual_correlation(residual_backend, context=context)

        def _pcct_call():
            return benchmark_knn_latency(
                dimension=dimension,
                tree_points=tree_points,
                query_count=query_count,
                k=k,
                batch_size=batch_size,
                seed=seed,
                prebuilt_points=points,
                prebuilt_queries=queries,
                log_writer=None,
                scope_cap_recorder=None,
                context=context,
            )

        (tree, bench_result), resources = _measure(_pcct_call)
    finally:
        cx_config.reset_runtime_context()

    pcct_summary = {
        "build_seconds": bench_result.build_seconds,
        "elapsed_seconds": bench_result.elapsed_seconds,
        "latency_ms": bench_result.latency_ms,
        "queries_per_second": bench_result.queries_per_second,
    }
    baselines: Dict[str, Dict[str, float]] = {}
    if metric == "euclidean" and baseline_mode != "none":
        for baseline in run_baseline_comparisons(points, queries, k=k, mode=baseline_mode):
            baselines[baseline.name] = {
                "build_seconds": baseline.build_seconds,
                "elapsed_seconds": baseline.elapsed_seconds,
                "latency_ms": baseline.latency_ms,
                "queries_per_second": baseline.queries_per_second,
            }
    return {"pcct": pcct_summary, "baselines": baselines, "resources": resources}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="default", help="PCCT profile slug to load.")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=("euclidean", "residual"),
        help="Metric(s) to benchmark (repeatable).",
    )
    parser.add_argument(
        "--dimension",
        dest="dimensions",
        action="append",
        type=int,
        help="Dimension values to sweep (repeatable).",
    )
    parser.add_argument(
        "--tree-points",
        dest="tree_points",
        action="append",
        type=int,
        help="Tree sizes to sweep (repeatable).",
    )
    parser.add_argument(
        "--k",
        dest="k_values",
        action="append",
        type=int,
        help="k values to sweep (repeatable).",
    )
    parser.add_argument("--queries", type=int, default=512, help="Number of query points.")
    parser.add_argument("--batch-size", type=int, default=512, help="Tree build batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset base seed.")
    parser.add_argument(
        "--baseline-mode",
        choices=("cover", "external", "mlpack", "sequential", "gpboost", "none"),
        default="cover",
        help="Baseline selector passed through to the runner.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats per shape.")
    parser.add_argument("--output", help="Destination JSONL file.")
    parser.add_argument("--residual-inducing", type=int, default=512, help="Residual inducing points.")
    parser.add_argument("--residual-variance", type=float, default=1.0, help="Residual RBF variance.")
    parser.add_argument("--residual-lengthscale", type=float, default=1.0, help="Residual RBF lengthscale.")
    parser.add_argument("--residual-chunk-size", type=int, default=512, help="Residual kernel chunk size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = _default_list(args.dimensions, (8,))
    tree_points_list = _default_list(args.tree_points, (2048,))
    k_values = _default_list(args.k_values, (8,))
    metrics = args.metrics or ["euclidean", "residual"]
    output_path = _artifact_path(args.output)
    scenarios = [
        (metric, dim, n_points, k_val, repeat_id)
        for metric in metrics
        for dim in dimensions
        for n_points in tree_points_list
        for k_val in k_values
        for repeat_id in range(args.repeat)
    ]
    print(f"[baseline-matrix] running {len(scenarios)} scenarios -> {output_path}")

    with output_path.open("a", encoding="utf-8") as handle:
        for metric, dimension, tree_points, k_value, repeat_id in scenarios:
            baseline_mode = args.baseline_mode if metric == "euclidean" else "none"
            print(
                f"[baseline-matrix] metric={metric} dim={dimension} points={tree_points} "
                f"k={k_value} repeat={repeat_id} baseline={baseline_mode}"
            )
            summary = _run_pcct(
                dimension=dimension,
                tree_points=tree_points,
                query_count=args.queries,
                k=k_value,
                batch_size=args.batch_size,
                seed=args.seed + repeat_id,
                profile=args.profile,
                metric=metric,
                residual_inducing=args.residual_inducing,
                residual_variance=args.residual_variance,
                residual_lengthscale=args.residual_lengthscale,
                residual_chunk_size=args.residual_chunk_size,
                baseline_mode=baseline_mode,
            )
            payload = {
                "profile": args.profile,
                "metric": metric,
                "dimension": dimension,
                "tree_points": tree_points,
                "queries": args.queries,
                "batch_size": args.batch_size,
                "k": k_value,
                "repeat_index": repeat_id,
                "baseline_mode": baseline_mode,
                **summary,
            }
            handle.write(json.dumps(payload) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
