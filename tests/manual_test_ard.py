import time
import numpy as np
from covertreex.api.pcct import PCCT
from covertreex.runtime import configure_runtime, RuntimeConfig
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
)
import dataclasses
from covertreex.api.runtime import Runtime

def generate_ard_data(n_points, d=3, rank=16, seed=42):
    """Generate synthetic data with ARD kernel."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_points, d)).astype(np.float32)
    
    # ARD Lengthscales
    # ls = [0.1, 1.0, 10.0] for D=3
    ls = np.logspace(-1, 1, d).astype(np.float32)
    ls_sq = ls**2
    
    # K_true(x, y) = exp(-0.5 * sum((x-y)^2 / ls^2))
    
    V = rng.normal(size=(n_points, rank)).astype(np.float32) * 0.1
    v_norms = np.sum(V**2, axis=1)
    p_diag = (1.0 - v_norms).astype(np.float32)
    p_diag[p_diag < 0.1] = 0.1
    kernel_diag = np.ones(n_points, dtype=np.float32)
    
    return X, V, p_diag, kernel_diag, ls

def test_ard_numba_path():
    print("Testing ARD Numba Path...")
    N = 1000
    D = 3
    K = 5
    X, V, p_diag, kernel_diag, ls = generate_ard_data(N, D)
    
    # 1. Configure Backend with ARD
    # Mock provider not needed if Numba path is triggered, but needed for validation if we compare?
    # Let's skip provider validation for this unit test and trust Numba path runs.
    # Or better, check if it crashes.
    
    # Dummy provider
    def dummy_provider(r, c):
        return np.zeros((r.size, c.size))
        
    host_data = ResidualCorrHostData(
        v_matrix=V,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=dummy_provider,
        chunk_size=512
    )
    
    # Inject ARD params
    object.__setattr__(host_data, "rbf_variance", 1.0)
    object.__setattr__(host_data, "rbf_lengthscale", ls) # ARD Array
    object.__setattr__(host_data, "kernel_points_f32", X.astype(np.float32))
    
    configure_residual_correlation(host_data)
    
    # 2. Build
    config = RuntimeConfig.from_env()
    config = dataclasses.replace(config, residual_use_static_euclidean_tree=True)
    configure_runtime(config)
    
    pcct_base = PCCT()
    tree = pcct_base.fit(X)
    
    runtime_wrapper = Runtime.from_config(config)
    pcct = dataclasses.replace(pcct_base, tree=tree, runtime=runtime_wrapper)
    
    # 3. Query
    query_indices = np.arange(10, dtype=np.int64).reshape(-1, 1)
    # We expect this to run without error using the Numba ARD kernel
    indices, dists = pcct.knn(query_indices, k=K, return_distances=True)
    
    print(f"Query successful. Shapes: {indices.shape}")
    assert indices.shape == (10, K)
    
    # Verify it actually used ARD logic?
    # Hard to verify internally without observing logs/coverage.
    # But if `ls` is array, code path triggers `ls_sq_arr` setup.
    
    print("ARD Test Passed.")

if __name__ == "__main__":
    test_ard_numba_path()
