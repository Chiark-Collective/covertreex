#!/usr/bin/env python
"""Benchmark subtree bounds optimization for predecessor mode queries.

Compares rust-natural vs rust-hilbert engines with and without
compute_predecessor_bounds to measure pruning effectiveness.
"""
import time
import numpy as np

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import Runtime
from covertreex.engine import RustHilbertEngine, RustNaturalEngine
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.metrics.residual import set_residual_backend


def run_benchmark(
    n_points: int = 32768,
    dimension: int = 3,
    n_queries: int = 1024,
    k: int = 50,
    runs: int = 3,
):
    """Run benchmark comparing subtree bounds across engines."""
    print(f"\n{'='*70}")
    print(f"Subtree Bounds Optimization Benchmark")
    print(f"  N={n_points}, D={dimension}, queries={n_queries}, k={k}")
    print(f"{'='*70}")

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

    # Query indices - use the upper half to have valid predecessors
    query_indices = np.arange(n_points // 2, n_points, dtype=np.int64).reshape(-1, 1)[:n_queries]

    residual_params = {"variance": 1.0, "lengthscale": 1.0}

    # Test configurations
    configs = [
        ("rust-hilbert", RustHilbertEngine(), False),
        ("rust-hilbert + bounds", RustHilbertEngine(), True),
        ("rust-natural", RustNaturalEngine(), False),
        ("rust-natural + bounds", RustNaturalEngine(), True),
    ]

    results = {}

    for name, engine, use_bounds in configs:
        print(f"\n--- {name} ---")

        runtime = Runtime(
            backend="numpy",
            precision="float32",
            metric="residual_correlation",
            enable_rust=True,
            engine="rust-hilbert" if isinstance(engine, RustHilbertEngine) else "rust-natural",
            residual_use_static_euclidean_tree=True,
        )

        # Build tree
        t0 = time.perf_counter()
        tree = engine.build(
            points,
            runtime=runtime.to_config(),
            residual_backend=backend_state,
            residual_params=residual_params,
            compute_predecessor_bounds=use_bounds,
        )
        build_time = time.perf_counter() - t0
        print(f"  Build: {build_time:.3f}s")

        if use_bounds and tree.handle.subtree_min_bounds is not None:
            bounds = tree.handle.subtree_min_bounds
            print(f"  Bounds computed: {len(bounds)} nodes")
            # Show bounds distribution
            print(f"    Min bound range: [{bounds.min()}, {bounds.max()}]")

        qps_list = []
        ctx = cx_config.runtime_context()

        for run_idx in range(runs):
            t0 = time.perf_counter()
            neighbors = engine.knn(
                tree,
                query_indices,
                k=k,
                return_distances=False,
                predecessor_mode=True,
                context=ctx,
                runtime=runtime.to_config(),
            )
            elapsed = time.perf_counter() - t0
            qps = len(query_indices) / elapsed
            qps_list.append(qps)

            if run_idx == 0:
                # Verify constraint
                neighbors_arr = np.asarray(neighbors)
                violations = 0
                for i, q_idx in enumerate(query_indices.flatten()):
                    valid = neighbors_arr[i][neighbors_arr[i] >= 0]
                    violations += sum(1 for j in valid if j >= q_idx)
                if violations > 0:
                    print(f"  WARNING: {violations} constraint violations!")
                else:
                    print(f"  Constraint verified OK")

        median_qps = np.median(qps_list)
        results[name] = median_qps
        print(f"  Median: {median_qps:,.0f} q/s")

    # Summary comparison
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    baseline = results.get("rust-hilbert", 1)
    for name, qps in results.items():
        ratio = qps / baseline
        print(f"  {name:30s}: {qps:>8,.0f} q/s  ({ratio:.2f}x)")

    # Specific comparisons
    print(f"\nSubtree bounds impact:")
    if "rust-hilbert" in results and "rust-hilbert + bounds" in results:
        h_delta = results["rust-hilbert + bounds"] / results["rust-hilbert"]
        print(f"  rust-hilbert: {h_delta:.2f}x")
    if "rust-natural" in results and "rust-natural + bounds" in results:
        n_delta = results["rust-natural + bounds"] / results["rust-natural"]
        print(f"  rust-natural: {n_delta:.2f}x")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark subtree bounds optimization")
    parser.add_argument("--n-points", type=int, default=32768, help="Number of points")
    parser.add_argument("--dimension", type=int, default=3, help="Dimensionality")
    parser.add_argument("--queries", type=int, default=1024, help="Number of queries")
    parser.add_argument("--k", type=int, default=50, help="Number of neighbors")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")

    args = parser.parse_args()

    run_benchmark(
        n_points=args.n_points,
        dimension=args.dimension,
        n_queries=args.queries,
        k=args.k,
        runs=args.runs,
    )
