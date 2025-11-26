import time
import numpy as np
import sys
import os
from dataclasses import replace

sys.path.append(os.getcwd())

from covertreex.engine import RustNaturalEngine, RustHilbertEngine
from covertreex.runtime.config import RuntimeConfig

def benchmark_natural_vs_hilbert():
    print("--- Residual Natural vs Hilbert (Gold Standard) Benchmark ---")
    
    # Gold Standard Parameters
    n_points = 32768
    n_dims = 3
    n_queries = 1024
    k = 50
    batch_size = 512
    seed = 42
    
    # Inducing count = 512 (standard default) to match "Gold Standard" config
    inducing_count = 512
    print(f"Configuration: N={n_points}, D={n_dims}, Q={n_queries}, K={k}, Batch={batch_size}, Inducing={inducing_count}")
    
    np.random.seed(seed)
    data = np.random.rand(n_points, n_dims).astype(np.float32)
    query_indices = np.random.randint(0, n_points, size=n_queries).astype(np.int64).reshape(-1, 1)
    
    residual_params = {
        "variance": 1.0,
        "lengthscale": 1.0,
        "inducing_count": inducing_count,
        "chunk_size": 512
    }
    
    base_runtime = RuntimeConfig.from_env()
    
    # 1. Benchmark RustNaturalEngine (Natural Order)
    print("\n[1/2] Benchmarking RustNaturalEngine (Natural Order)...")
    sys.stdout.flush()
    
    runtime_natural = replace(
        base_runtime,
        metric="residual_correlation",
        batch_order_strategy="natural",
        residual_grid_whiten_scale=1.0,
        scope_chunk_target=0,
        conflict_degree_cap=0,
        scope_budget_schedule=(),
        residual_masked_scope_append=False,
        backend="numpy",
    )
    
    natural_engine = RustNaturalEngine()
    start_build = time.time()
    try:
        tree_natural = natural_engine.build(data, runtime=runtime_natural, residual_params=residual_params, batch_size=batch_size)
        build_time_natural = time.time() - start_build
        print(f"  Build Time: {build_time_natural:.4f}s")
        
        start_query = time.time()
        natural_engine.knn(tree_natural, query_indices, k=k, return_distances=False, context=None, runtime=runtime_natural)
        query_time_natural = time.time() - start_query
        qps_natural = n_queries / query_time_natural
        print(f"  Query Time: {query_time_natural:.4f}s ({qps_natural:.2f} q/s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        build_time_natural = float('nan')
        qps_natural = float('nan')

    # 2. Benchmark RustHilbertEngine (Gold / Hilbert Order)
    print("\n[2/2] Benchmarking RustHilbertEngine (Hilbert Order / Gold)...")
    sys.stdout.flush()
    
    runtime_hilbert = replace(
        base_runtime,
        metric="residual_correlation",
        batch_order_strategy="hilbert",
        residual_grid_whiten_scale=1.0,
        scope_chunk_target=0,
        conflict_degree_cap=0,
        scope_budget_schedule=(),
        residual_masked_scope_append=False,
        backend="numpy",
    )
    
    hilbert_engine = RustHilbertEngine()
    start_build = time.time()
    try:
        tree_hilbert = hilbert_engine.build(data, runtime=runtime_hilbert, residual_params=residual_params, batch_size=batch_size)
        build_time_hilbert = time.time() - start_build
        print(f"  Build Time: {build_time_hilbert:.4f}s")
        
        start_query = time.time()
        hilbert_engine.knn(tree_hilbert, query_indices, k=k, return_distances=False, context=None, runtime=runtime_hilbert)
        query_time_hilbert = time.time() - start_query
        qps_hilbert = n_queries / query_time_hilbert
        print(f"  Query Time: {query_time_hilbert:.4f}s ({qps_hilbert:.2f} q/s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        build_time_hilbert = float('nan')
        qps_hilbert = float('nan')

    # Summary
    print("\n--- Summary ---")
    print(f"{ 'Engine':<20} | { 'Build (s)':<10} | { 'QPS':<10} | { 'Query (s)':<10}")
    print("-" * 60)
    print(f"{ 'Natural':<20} | {build_time_natural:<10.4f} | {qps_natural:<10.2f} | {query_time_natural:<10.4f}")
    print(f"{ 'Hilbert (Gold)':<20} | {build_time_hilbert:<10.4f} | {qps_hilbert:<10.2f} | {query_time_hilbert:<10.4f}")

if __name__ == "__main__":
    benchmark_natural_vs_hilbert()
