from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from covertreex import reset_residual_metric
from covertreex import config as cx_config
from covertreex.algo import batch_insert
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
) -> Tuple[PCCTree, np.ndarray, float]:
    backend = _resolve_backend()
    tree = PCCTree.empty(dimension=dimension, backend=backend)
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
            tree, _ = batch_insert(tree, batch, mis_seed=seed + idx)
            buffers.append(np.asarray(batch))
            idx += 1
    else:
        rng = default_rng(seed)
        remaining = tree_points
        while remaining > 0:
            current = min(batch_size, remaining)
            batch = _generate_points_backend(rng, current, dimension)
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
    prebuilt_points: np.ndarray | None = None,
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
            prebuilt_points=prebuilt_points,
        )
    else:
        tree = prebuilt_tree
        tree_build_seconds = build_seconds

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
    previous_metric = os.environ.get("COVERTREEX_METRIC")
    previous_backend = os.environ.get("COVERTREEX_BACKEND")

    try:
        if args.metric == "residual":
            os.environ["COVERTREEX_METRIC"] = "residual_correlation"
            os.environ["COVERTREEX_BACKEND"] = "numpy"
        else:
            os.environ["COVERTREEX_METRIC"] = "euclidean"
        cx_config.reset_runtime_config_cache()

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
        cx_config.reset_runtime_config_cache()


if __name__ == "__main__":
    main()
