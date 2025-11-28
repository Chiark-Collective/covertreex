#!/usr/bin/env python
"""Benchmark predecessor_mode performance vs standard k-NN.

This script compares query throughput with and without predecessor_mode
to assess the performance impact of the predecessor constraint filtering.
"""
import time
import numpy as np

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import CoverTree, Runtime
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.metrics.residual import set_residual_backend


def run_benchmark(
    n_points: int = 32768,
    dimension: int = 3,
    n_queries: int = 1024,
    k: int = 50,
    engine: str = "rust-hilbert",
    runs: int = 3,
):
    """Run benchmark comparing predecessor_mode=True vs False."""
    print(f"\n{'='*60}")
    print(f"Predecessor Mode Benchmark")
    print(f"  N={n_points}, D={dimension}, queries={n_queries}, k={k}")
    print(f"  engine={engine}, runs={runs}")
    print(f"{'='*60}")

    # Reset state
    cx_config.reset_runtime_context()
    reset_residual_metric()
    set_residual_backend(None)

    # Generate data
    np.random.seed(42)
    points = np.random.randn(n_points, dimension).astype(np.float32)

    # Build residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=256,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=512,
    )
    configure_residual_correlation(backend_state)

    # Build tree
    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine=engine,
        residual_use_static_euclidean_tree=True,
    )

    print("\nBuilding tree...")
    t0 = time.perf_counter()
    tree = CoverTree(runtime).fit(points, mis_seed=7)
    build_time = time.perf_counter() - t0
    print(f"  Build time: {build_time:.3f}s")

    # Query indices - use all points for full coverage
    # For predecessor_mode, we want indices that span the full range
    query_indices = np.arange(n_points - n_queries, n_points, dtype=np.int64).reshape(-1, 1)

    results = {"standard": [], "predecessor": []}

    for run_idx in range(runs):
        print(f"\nRun {run_idx + 1}/{runs}")

        # Standard k-NN (no predecessor constraint)
        t0 = time.perf_counter()
        neighbors_std = tree.knn(query_indices, k=k, predecessor_mode=False)
        std_time = time.perf_counter() - t0
        std_qps = n_queries / std_time
        results["standard"].append(std_qps)
        print(f"  Standard:    {std_time:.4f}s  ({std_qps:,.0f} q/s)")

        # Predecessor-constrained k-NN
        t0 = time.perf_counter()
        neighbors_pred = tree.knn(query_indices, k=k, predecessor_mode=True)
        pred_time = time.perf_counter() - t0
        pred_qps = n_queries / pred_time
        results["predecessor"].append(pred_qps)
        print(f"  Predecessor: {pred_time:.4f}s  ({pred_qps:,.0f} q/s)")

        # Verify predecessor constraint
        neighbors_pred_arr = np.asarray(neighbors_pred)
        for i, q_idx in enumerate(range(n_points - n_queries, n_points)):
            valid = neighbors_pred_arr[i][neighbors_pred_arr[i] >= 0]
            violations = sum(1 for j in valid if j >= q_idx)
            if violations > 0:
                print(f"  WARNING: Query {q_idx} has {violations} constraint violations!")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (median of {runs} runs):")
    std_median = np.median(results["standard"])
    pred_median = np.median(results["predecessor"])
    speedup = pred_median / std_median

    print(f"  Standard k-NN:    {std_median:,.0f} q/s")
    print(f"  Predecessor mode: {pred_median:,.0f} q/s")
    print(f"  Ratio: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark predecessor_mode performance")
    parser.add_argument("--n-points", type=int, default=32768, help="Number of points")
    parser.add_argument("--dimension", type=int, default=3, help="Dimensionality")
    parser.add_argument("--queries", type=int, default=1024, help="Number of queries")
    parser.add_argument("--k", type=int, default=50, help="Number of neighbors")
    parser.add_argument("--engine", type=str, default="rust-hilbert", help="Engine to use")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")

    args = parser.parse_args()

    run_benchmark(
        n_points=args.n_points,
        dimension=args.dimension,
        n_queries=args.queries,
        k=args.k,
        engine=args.engine,
        runs=args.runs,
    )
