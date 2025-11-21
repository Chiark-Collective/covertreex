import time
import numpy as np
import covertreex_backend
from tests.utils.datasets import gaussian_points

def run_benchmark():
    print("Benchmarking: Rust Full Residual (Build + Query)")
    
    N = 10_000
    D = 8
    K = 10
    batch_size = 5_000
    n_queries = 1000
    
    print(f"Data: N={N}, D={D}, BatchSize={batch_size}")
    
    rng = np.random.default_rng(42)
    X = gaussian_points(rng, N, D, dtype=np.float64).astype(np.float32)
    
    # Residual Data
    rank = 16
    V = rng.normal(size=(N, rank)).astype(np.float32)
    p_diag = rng.uniform(0.1, 1.0, size=N).astype(np.float32)
    rbf_var = 1.0
    rbf_ls = np.ones(D, dtype=np.float32)
    
    # Initial Empty Tree
    # Note: dimension passed to empty tree should be 1 (indices)?
    # Or we pass D?
    # `CoverTreeData` treats points as chunks of `dimension`.
    # If we pass indices, dimension is 1.
    # So `dummy_points` for `new` should be (0, 1).
    
    dummy_points = np.empty((0, 1), dtype=np.float32)
    dummy_parents = np.empty(0, dtype=np.int64)
    dummy_children = np.empty(0, dtype=np.int64)
    dummy_next = np.empty(0, dtype=np.int64)
    dummy_levels = np.empty(0, dtype=np.int32)
    
    print("Initializing Rust Tree (Residual Mode, Dim=1)...")
    tree = covertreex_backend.CoverTreeWrapper(
        dummy_points, dummy_parents, dummy_children, dummy_next, dummy_levels, -20, 20
    )
    
    print("Building Tree with Residual Metric...")
    t0 = time.perf_counter()
    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        # Indices batch
        indices_batch = np.arange(i, end, dtype=np.float32).reshape(-1, 1)
        
        tree.insert_residual(
            indices_batch,
            V,
            p_diag,
            X,
            rbf_var,
            rbf_ls
        )
        print(f"  Inserted {end}/{N}")
        
    t_build = time.perf_counter() - t0
    print(f"Build Time: {t_build:.4f}s ({N/t_build:.1f} pts/s)")
    
    print("Querying Tree (Residual)...")
    Q_idx = rng.choice(N, size=n_queries, replace=False).astype(np.int64)
    # Node to Dataset is Identity
    node_to_dataset = np.arange(N, dtype=np.int64).tolist()
    
    t0 = time.perf_counter()
    indices, dists = tree.knn_query_residual(
        Q_idx,
        node_to_dataset,
        V,
        p_diag,
        X,
        rbf_var,
        rbf_ls,
        K
    )
    t_query = time.perf_counter() - t0
    print(f"Query Time: {t_query:.4f}s ({n_queries/t_query:.1f} q/s)")
    
    assert indices.shape == (n_queries, K)
    # Check self-match? distance to self should be 0?
    # Yes, if Q is subset of X.
    
    print("Success.")

if __name__ == "__main__":
    run_benchmark()
