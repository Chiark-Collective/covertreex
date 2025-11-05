from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from covertreex.algo import batch_insert
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree
from covertreex.queries.knn import knn
from covertreex.baseline import (
    BaselineCoverTree,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
)


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


def _generate_points(rng: Generator, count: int, dimension: int) -> np.ndarray:
    samples = rng.normal(loc=0.0, scale=1.0, size=(count, dimension))
    return DEFAULT_BACKEND.asarray(samples, dtype=DEFAULT_BACKEND.default_float)


def _build_tree(
    *,
    dimension: int,
    tree_points: int,
    batch_size: int,
    seed: int,
) -> Tuple[PCCTree, np.ndarray, float]:
    tree = PCCTree.empty(dimension=dimension, backend=DEFAULT_BACKEND)
    rng = default_rng(seed)
    remaining = tree_points
    idx = 0
    start = time.perf_counter()
    buffers: List[np.ndarray] = []
    while remaining > 0:
        current = min(batch_size, remaining)
        batch = _generate_points(rng, current, dimension)
        tree, _ = batch_insert(tree, batch, mis_seed=seed + idx)
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
    prebuilt_tree: PCCTree | None = None,
    prebuilt_queries: np.ndarray | None = None,
    build_seconds: float | None = None,
) -> Tuple[PCCTree, QueryBenchmarkResult]:
    tree_build_seconds: float | None = None
    if prebuilt_tree is None:
        tree, _, tree_build_seconds = _build_tree(
            dimension=dimension,
            tree_points=tree_points,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds

    if prebuilt_queries is None:
        query_rng = default_rng(seed + 1)
        queries = _generate_points(query_rng, query_count, dimension)
    else:
        queries = DEFAULT_BACKEND.asarray(
            prebuilt_queries, dtype=DEFAULT_BACKEND.default_float
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
        "--baseline",
        choices=("none", "sequential", "gpboost", "external", "both", "all"),
        default="none",
        help=(
            "Include baseline comparisons. Install '.[baseline]' for the external library and "
            "'numba' extra for the GPBoost baseline. Options: 'sequential', 'gpboost', "
            "'external', 'both' (sequential + external), 'all' (sequential + gpboost + external)."
        ),
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
    tree, points_np, build_seconds = _build_tree(
        dimension=args.dimension,
        tree_points=args.tree_points,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    query_rng = default_rng(args.seed + 1)
    queries = _generate_points(query_rng, args.queries, args.dimension)
    _, result = benchmark_knn_latency(
        dimension=args.dimension,
        tree_points=args.tree_points,
        query_count=args.queries,
        k=args.k,
        batch_size=args.batch_size,
        seed=args.seed,
        prebuilt_tree=tree,
        prebuilt_queries=queries,
        build_seconds=build_seconds,
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
            np.asarray(queries),
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


if __name__ == "__main__":
    main()
