import numpy as np
import pytest
import sys
import os

# Add current directory to path so we can import covertreex
sys.path.append(os.getcwd())

from covertreex.engine import RustHybridResidualEngine, RustNaturalEngine
from covertreex.metrics.residual.core import configure_residual_correlation

def test_residual_pure_vs_hybrid_parity():
    """
    Verifies that the 'Pure' Residual Tree (built with residual metric)
    produces identical k-NN results to the 'Hybrid' Tree (built with Euclidean, queried with Residual).
    """
    # 1. Setup Data
    n_points = 500
    n_dims = 10
    k = 5
    
    # Create random data
    data = np.random.rand(n_points, n_dims).astype(np.float32)
    
    query_indices = np.arange(5, dtype=np.int64).reshape(-1, 1) # Query first 5 points
    
    # 2. Configure Residual Metric
    # We use inducing_count = n_points to ensure full rank / unique distances
    residual_params = {
        "variance": 1.0,
        "lengthscale": 1.0,
        "inducing_count": n_points, 
        "chunk_size": 100
    }
    
    # Create a real RuntimeConfig using defaults
    from covertreex.runtime.config import RuntimeConfig
    from dataclasses import replace
    
    base_runtime = RuntimeConfig.from_env()
    runtime = replace(
        base_runtime,
        metric="residual_correlation",
        batch_order_strategy="natural", # Natural order for consistency
        residual_grid_whiten_scale=1.0,
        scope_chunk_target=0,
        conflict_degree_cap=0,
        scope_budget_schedule=(),
        residual_masked_scope_append=False,
        backend="numpy",
    )
    
    # 3. Initialize Engines
    
    # Pure Residual Engine (RustNaturalEngine)
    print("Building Pure Residual Tree (RustNaturalEngine)...")
    pure_engine_impl = RustNaturalEngine()
    pure_tree = pure_engine_impl.build(data, runtime=runtime, residual_params=residual_params)
    
    # Hybrid Engine (RustHybridResidualEngine)
    print("Building Hybrid Residual Tree (RustHybridResidualEngine)...")
    hybrid_engine_impl = RustHybridResidualEngine()
    hybrid_tree = hybrid_engine_impl.build(data, runtime=runtime, residual_params=residual_params)
    
    # 4. Run Queries
    print("Querying...")
    
    # Query Pure
    pure_results = pure_engine_impl.knn(pure_tree, query_indices, k=k, return_distances=True, context=None, runtime=runtime)
    pure_indices_res = pure_results[0]
    pure_distances_res = pure_results[1]
    
    # Query Hybrid
    hybrid_results = hybrid_engine_impl.knn(hybrid_tree, query_indices, k=k, return_distances=True, context=None, runtime=runtime)
    hybrid_indices_res = hybrid_results[0]
    hybrid_distances_res = hybrid_results[1]
    
    # 5. Compare
    print(f"\nPure Indices:\n{pure_indices_res}")
    print(f"Hybrid Indices:\n{hybrid_indices_res}")
    print(f"Pure Dists:\n{pure_distances_res}")
    print(f"Hybrid Dists:\n{hybrid_distances_res}")
    
    np.testing.assert_allclose(pure_distances_res, hybrid_distances_res, rtol=1e-4, err_msg="Distances mismatch between Pure and Hybrid")
    
    # Simple set equality for indices
    # For each query
    for i in range(len(query_indices)):
        p_idx_set = set(pure_indices_res[i])
        h_idx_set = set(hybrid_indices_res[i])
        assert p_idx_set == h_idx_set, f"Nearest neighbor indices do not match for query {i}!"

if __name__ == "__main__":
    try:
        test_residual_pure_vs_hybrid_parity()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()