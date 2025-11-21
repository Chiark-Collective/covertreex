import time
import numpy as np
import covertreex_backend
from tests.utils.datasets import gaussian_points

def run_benchmark():
    print("Benchmarking: Rust Residual Query")
    
    N = 10_000
    D = 8
    K = 10
    n_queries = 1000
    
    print(f"Data: N={N}, D={D}, Queries={n_queries}, K={K}")
    
    rng = np.random.default_rng(42)
    X = gaussian_points(rng, N, D, dtype=np.float64).astype(np.float32)
    Q = gaussian_points(rng, n_queries, D, dtype=np.float64).astype(np.float32)
    
    # Residual Data
    rank = 16
    V = rng.normal(size=(N, rank)).astype(np.float32)
    p_diag = rng.uniform(0.1, 1.0, size=N).astype(np.float32)
    rbf_var = 1.0
    rbf_ls = np.ones(D, dtype=np.float32) # ARD Lengthscales (ones)
    
    # Query needs to be INDICES into the dataset for Residual Metric
    # So we append Q to X?
    # Or we assume Q is a subset?
    # For this benchmark, let's pick random indices from X as queries.
    query_indices = rng.choice(N, size=n_queries, replace=False).astype(np.int64)
    
    # Tree Structure (Linear for consistent baseline)
    parents = np.arange(-1, N-1, dtype=np.int64)
    children = np.arange(1, N+1, dtype=np.int64)
    children[-1] = -1
    next_node = np.full(N, -1, dtype=np.int64)
    levels = np.zeros(N, dtype=np.int32)
    
    # Node to Dataset Map (Identity)
    node_to_dataset = np.arange(N, dtype=np.int64)
    # Rust expects Vec<i64> for node_to_dataset?
    # PyO3 handles numpy array -> Vec conversion if we pass it right?
    # Wrapper signature: `node_to_dataset: Vec<i64>`.
    # Python passes list or array. Array is faster if PyO3 supports it directly?
    # PyO3 converts Sequence/List to Vec. Passing numpy array iterates.
    # It works.
    
    tree = covertreex_backend.CoverTreeWrapper(
        X, parents, children, next_node, levels, -10, 10
    )
    
    print("Running Rust Residual Queries...")
    t0 = time.perf_counter()
    
    # Pass node_to_dataset as list for Vec conversion
    indices, dists = tree.knn_query_residual(
        query_indices,
        node_to_dataset.tolist(),
        V,
        p_diag,
        X,
        rbf_var,
        rbf_ls,
        K
    )
    t_rust = time.perf_counter() - t0
    qps = n_queries / t_rust
    print(f"Rust Residual: {t_rust:.4f}s ({qps:.1f} q/s)")
    
    # Sanity check shape
    assert indices.shape == (n_queries, K)
    print("Success.")

if __name__ == "__main__":
    run_benchmark()
