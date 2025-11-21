import time
import numpy as np
import covertreex_backend
from covertreex.queries._residual_knn_numba import residual_knn_search_numba
from tests.utils.datasets import gaussian_points

def run_benchmark():
    print("Benchmarking: Rust vs Numba (Static Tree / Euclidean)")
    
    # Setup
    N = 10_000
    D = 8
    K = 10
    n_queries = 1000
    
    print(f"Data: N={N}, D={D}, Queries={n_queries}, K={K}")
    
    rng = np.random.default_rng(42)
    # Points
    X = gaussian_points(rng, N, D, dtype=np.float64).astype(np.float32)
    # Queries
    Q = gaussian_points(rng, n_queries, D, dtype=np.float64).astype(np.float32)
    
    # Build a Manual Linear Tree to isolate Algo Overhead (exclude tree complexity)
    # 0 -> 1 -> 2 ...
    parents = np.arange(-1, N-1, dtype=np.int64)
    children = np.arange(1, N+1, dtype=np.int64)
    children[-1] = -1
    next_node = np.full(N, -1, dtype=np.int64)
    
    # Rust Setup
    print("Initializing Rust Backend...")
    t0 = time.perf_counter()
    rust_tree = covertreex_backend.CoverTreeWrapper(X, parents, children, next_node)
    print(f"Rust Init: {(time.perf_counter() - t0)*1000:.2f} ms")
    
    # Numba Setup
    # Mocking residual backend arrays to run Euclidean-like search
    # Numba path is designed for Residual, but if we set v_matrix=0, p=1, var=1, ls=inf?
    # No, Numba path is HARDCODED for Residual math.
    # We can't easily benchmark "Numba Euclidean" because we don't have one exposed directly?
    # `knn_numba` in `covertreex` usually refers to residual.
    # But we want to compare "Static Tree Query".
    # Let's use the `residual_knn_search_numba` with parameters that emulate Euclidean?
    # K(x,y) = exp(-0.5 d^2). V=0. P=1.
    # rho = K / 1 = exp(-0.5 d^2).
    # dist_res = sqrt(1 - rho) = sqrt(1 - exp(-0.5 d^2)).
    # This is monotonic with Euclidean distance d.
    # So we can verify throughput.
    
    # Setup Mock Residual Data for Numba
    v_matrix = np.zeros((N, 1), dtype=np.float32)
    p_diag = np.ones(N, dtype=np.float32)
    v_norm_sq = np.zeros(N, dtype=np.float32)
    coords = X # float32
    var = 1.0
    ls_sq = np.full(D, 1.0, dtype=np.float64) # Isotropic 1.0
    
    # Numba Arrays
    # Need children, next, parents, node_to_dataset
    node_to_dataset = np.arange(N, dtype=np.int64)
    
    # Pre-compile Numba
    print("Warming up Numba...")
    heap_keys = np.empty(10000, dtype=np.float64)
    heap_vals = np.empty(10000, dtype=np.int64)
    heap_extras = np.empty(10000, dtype=np.int64)
    knn_keys = np.empty(K, dtype=np.float64)
    knn_indices = np.full(K, -1, dtype=np.int64)
    visited = np.zeros((N+63)//64, dtype=np.int64)
    roots = np.array([0], dtype=np.int64)
    
    residual_knn_search_numba(
        children, next_node, parents,
        node_to_dataset, v_matrix, p_diag, v_norm_sq,
        coords, var, ls_sq,
        0, K, roots,
        heap_keys, heap_vals, heap_extras,
        knn_keys, knn_indices, visited
    )
    
    # Benchmark Rust
    print("Running Rust Queries...")
    t0 = time.perf_counter()
    r_idx, r_dist = rust_tree.knn_query(Q, K)
    t_rust = time.perf_counter() - t0
    qps_rust = n_queries / t_rust
    print(f"Rust: {t_rust:.4f}s ({qps_rust:.1f} q/s)")
    
    # Benchmark Numba (Serial Loop)
    print("Running Numba Queries (Serial Loop)...")
    t0 = time.perf_counter()
    for i in range(n_queries):
        residual_knn_search_numba(
            children, next_node, parents,
            node_to_dataset, v_matrix, p_diag, v_norm_sq,
            coords, var, ls_sq,
            i, K, roots,
            heap_keys, heap_vals, heap_extras,
            knn_keys, knn_indices, visited
        )
    t_numba = time.perf_counter() - t0
    qps_numba = n_queries / t_numba
    print(f"Numba: {t_numba:.4f}s ({qps_numba:.1f} q/s)")
    
    print(f"Speedup: {qps_rust / qps_numba:.2f}x")

if __name__ == "__main__":
    run_benchmark()
