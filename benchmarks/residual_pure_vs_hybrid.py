import time
import numpy as np
import sys
import os
from dataclasses import replace

sys.path.append(os.getcwd())

from covertreex.engine import RustHybridResidualEngine, RustNaturalEngine
from covertreex.runtime.config import RuntimeConfig

def benchmark_pure_vs_hybrid():
    print("Generating data...")
    n_points = 2000
    n_dims = 64
    k = 10
    n_queries = 50
    
    data = np.random.rand(n_points, n_dims).astype(np.float32)
    query_indices = np.arange(n_queries, dtype=np.int64).reshape(-1, 1)
    
    # inducing_count = n_points ensures full rank
    residual_params = {
        "variance": 1.0,
        "lengthscale": 1.0,
        "inducing_count": n_points,
        "chunk_size": 100
    }
    
    base_runtime = RuntimeConfig.from_env()
    runtime = replace(
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
    
    results = []
    
    # --- Pure Residual ---
    print("\n--- Benchmarking Pure Residual (RustNaturalEngine) ---")
    sys.stdout.flush()
    pure_engine = RustNaturalEngine()
    start_build = time.time()
    try:
        pure_tree = pure_engine.build(data, runtime=runtime, residual_params=residual_params)
        build_time_pure = time.time() - start_build
        print(f"Pure Build Time: {build_time_pure:.4f}s")
        
        start_query = time.time()
        pure_engine.knn(pure_tree, query_indices, k=k, return_distances=False, context=None, runtime=runtime)
        query_time_pure = time.time() - start_query
        print(f"Pure Query Time: {query_time_pure:.4f}s ({(n_queries/query_time_pure):.2f} q/s)")
    except Exception as e:
        print(f"Pure Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Hybrid Residual ---
    print("\n--- Benchmarking Hybrid Residual (RustHybridResidualEngine) ---")
    sys.stdout.flush()
    hybrid_engine = RustHybridResidualEngine()
    start_build = time.time()
    try:
        hybrid_tree = hybrid_engine.build(data, runtime=runtime, residual_params=residual_params)
        build_time_hybrid = time.time() - start_build
        print(f"Hybrid Build Time: {build_time_hybrid:.4f}s")
        
        start_query = time.time()
        hybrid_engine.knn(hybrid_tree, query_indices, k=k, return_distances=False, context=None, runtime=runtime)
        query_time_hybrid = time.time() - start_query
        print(f"Hybrid Query Time: {query_time_hybrid:.4f}s ({(n_queries/query_time_hybrid):.2f} q/s)")
    except Exception as e:
        print(f"Hybrid Failed: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    benchmark_pure_vs_hybrid()
