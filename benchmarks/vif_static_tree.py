import time
import numpy as np
from covertreex.api.pcct import PCCT
from covertreex.runtime import configure_runtime, RuntimeConfig
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
    compute_residual_pairwise_matrix
)

def generate_synthetic_data(n_points, d=3, rank=16, seed=42):
    """Generate synthetic data consistent with RBF prior."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_points, d)).astype(np.float32)
    
    # K_true(x, y) = 1.0 * exp(-dist^2 / (2 * 0.5)) = exp(-dist^2)
    # Let's use ls=sqrt(0.5) -> 2*ls^2 = 1.
    # K_true = exp(-|x-y|^2).
    
    # V matrix (Inducing point approximation)
    # V = random
    V = rng.normal(size=(n_points, rank)).astype(np.float32) * 0.1
    
    # P diag = diag(K_true - V V').
    # We need K_true diagonal. K(x,x) = 1.0.
    # p_i = 1.0 - ||v_i||^2.
    # Ensure positive.
    v_norms = np.sum(V**2, axis=1)
    p_diag = (1.0 - v_norms).astype(np.float32)
    p_diag[p_diag < 0.1] = 0.1 # clip
    
    # Kernel Diagonal
    kernel_diag = np.ones(n_points, dtype=np.float32)
    
    return X, V, p_diag, kernel_diag

def mock_kernel_provider(row_idx, col_idx):
    """RBF Kernel Provider (Python Reference)."""
    global _GLOBAL_X
    x_rows = _GLOBAL_X[row_idx]
    x_cols = _GLOBAL_X[col_idx]
    
    if x_rows.ndim == 1: x_rows = x_rows[None, :]
    if x_cols.ndim == 1: x_cols = x_cols[None, :]
    
    diff = x_rows[:, None, :] - x_cols[None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    # K = 1.0 * exp(-dist_sq / 1.0) (ls^2 = 0.5? No, previous was 0.5 * exp(-d^2))
    # Let's stick to simple: K = exp(-dist_sq).
    # Var=1.0, ls=1/sqrt(2) approx 0.707.
    
    k_val = np.exp(-dist_sq)
    return k_val.astype(np.float64)

def evaluate_correlation(host_data, N, ls, num_samples=1000):
    """Evaluate correlation between Euclidean and Residual distances."""
    print("Evaluating Metric Correlation...")
    rng = np.random.default_rng(42)
    idx1 = rng.integers(0, int(N), size=num_samples)
    idx2 = rng.integers(0, int(N), size=num_samples)
    
    # Compute Euclidean
    # Need coords. host_data.kernel_points_f32
    X = host_data.kernel_points_f32
    diff = X[idx1] - X[idx2]
    d_euc = np.linalg.norm(diff, axis=1)
    
    # Compute Residual
    from covertreex.metrics.residual import compute_residual_distances
    # compute_residual_distances returns matrix (lhs, rhs)
    d_res_matrix = compute_residual_distances(host_data, idx1, idx2)
    d_res = np.diagonal(d_res_matrix)
    
    from scipy.stats import pearsonr, spearmanr
    p_r, _ = pearsonr(d_euc, d_res)
    s_r, _ = spearmanr(d_euc, d_res)
    
    print(f"Correlation (Euclidean vs Residual): Pearson={p_r:.4f}, Spearman={s_r:.4f}")
    return p_r

def generate_ard_data(n_points, d=3, rank=16, seed=42):
    """Generate synthetic data with ARD kernel."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_points, d)).astype(np.float32)
    
    # ARD Lengthscales
    # ls = [0.1, 1.0, 10.0] for D=3
    ls = np.logspace(-1, 1, d).astype(np.float32)
    
    V = rng.normal(size=(n_points, rank)).astype(np.float32) * 0.1
    v_norms = np.sum(V**2, axis=1)
    p_diag = (1.0 - v_norms).astype(np.float32)
    p_diag[p_diag < 0.1] = 0.1
    kernel_diag = np.ones(n_points, dtype=np.float32)
    
    return X, V, p_diag, kernel_diag, ls

def ard_kernel_provider(row_idx, col_idx):
    """ARD RBF Kernel Provider (Python Reference)."""
    global _GLOBAL_X, _GLOBAL_LS_SQ
    x_rows = _GLOBAL_X[row_idx]
    x_cols = _GLOBAL_X[col_idx]
    
    if x_rows.ndim == 1: x_rows = x_rows[None, :]
    if x_cols.ndim == 1: x_cols = x_cols[None, :]
    
    diff = x_rows[:, None, :] - x_cols[None, :, :]
    dist_sq = np.sum((diff**2) / _GLOBAL_LS_SQ, axis=-1)
    k_val = np.exp(-0.5 * dist_sq)
    return k_val.astype(np.float64)

def run_benchmark():
    N = 10_000 # Small enough for brute force comparison
    D = 3
    K = 10
    
    print(f"Generating {N} points with ARD kernel...")
    X, V, p_diag, kernel_diag, ls = generate_ard_data(N, D)
    
    # Setup Global state for mock kernel
    global _GLOBAL_V, _GLOBAL_P, _GLOBAL_X, _GLOBAL_LS_SQ
    _GLOBAL_V = V
    _GLOBAL_P = p_diag
    _GLOBAL_X = X
    _GLOBAL_LS_SQ = ls**2
    
    # 1. Configure Backend
    host_data = ResidualCorrHostData(
        v_matrix=V,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=ard_kernel_provider,
        chunk_size=512
    )
    
    # Inject ARD params for Numba Fast Path
    object.__setattr__(host_data, "rbf_variance", 1.0)
    object.__setattr__(host_data, "rbf_lengthscale", ls) # ARD Array
    object.__setattr__(host_data, "kernel_points_f32", X.astype(np.float32))
    
    configure_residual_correlation(host_data)
    
    # Evaluate Correlation
    evaluate_correlation(host_data, N, ls)
    
    # 2. Build PCCT (Euclidean)
    print("Building PCCT (Euclidean)...")
    t0 = time.perf_counter()
    pcct_base = PCCT()
    # Disable batch order to ensure Identity Mapping (Node i == Dataset i) matches our fallback
    tree = pcct_base.fit(X, apply_batch_order=False)
    t_build = time.perf_counter() - t0
    print(f"Build time: {t_build:.4f}s")
    
    # 3. Prepare Queries
    n_queries = 100
    query_indices = np.linspace(0, N-1, n_queries, dtype=np.int64).reshape(-1, 1)
    
    # 4. Enable Static Tree Mode

    print("Configuring Static Tree Mode...")
    import dataclasses
    from covertreex.api.runtime import Runtime
    
    config = RuntimeConfig.from_env()
    config = dataclasses.replace(config, residual_use_static_euclidean_tree=True)
    configure_runtime(config)
    
    # Update PCCT wrapper with the built tree AND the new runtime config
    runtime_wrapper = Runtime.from_config(config)
    pcct = dataclasses.replace(pcct_base, tree=tree, runtime=runtime_wrapper)
    
    # 5. Run Query (Static Tree)
    print(f"Running {n_queries} queries (k={K})...")
    t0 = time.perf_counter()
    indices, dists = pcct.knn(query_indices, k=K, return_distances=True)
    t_query = time.perf_counter() - t0
    print(f"Query time: {t_query:.4f}s ({n_queries/t_query:.2f} q/s)")
    
    # 6. Gold Standard (Brute Force)
    print("Running Brute Force Verification...")
    # We can use `compute_residual_pairwise_matrix` for a block?
    # Or manual scan.
    
    recall_accum = 0
    
    for i, q_idx in enumerate(query_indices):
        # Compute all distances
        # shape (1, N)
        q_arr = np.array([q_idx.item()], dtype=np.int64)
        all_indices = np.arange(N, dtype=np.int64)
        
        # Using internal helper to get true distances
        from covertreex.metrics.residual import compute_residual_distances
        true_dists = compute_residual_distances(host_data, q_arr, all_indices).flatten()
        
        # Sort
        top_k_idx = np.argsort(true_dists)[:K]
        top_k_dists = true_dists[top_k_idx]
        
        # Compare with PCCT result
        pcct_idx = indices[i]
        pcct_dists = dists[i]
        
        if i == 0:
            print(f"DEBUG Query {q_idx.item()}:")
            print(f"  True Top 10 Idx: {top_k_idx}")
            print(f"  True Top 10 Dst: {top_k_dists}")
            print(f"  PCCT Top 10 Idx: {pcct_idx}")
            print(f"  PCCT Top 10 Dst: {pcct_dists}")
        
        intersection = np.intersect1d(top_k_idx, pcct_idx)
        recall = len(intersection) / K
        recall_accum += recall
        
    avg_recall = recall_accum / n_queries
    print(f"Average Recall@{K}: {avg_recall:.4f}")
    
    if avg_recall < 0.5:
        print("WARNING: Recall is low! Static Tree pruning/ordering might be ineffective.")
    else:
        print("SUCCESS: Recall is acceptable.")

if __name__ == "__main__":
    run_benchmark()
