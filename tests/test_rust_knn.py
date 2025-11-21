import numpy as np
import pytest
import heapq

try:
    import covertreex_backend
except ImportError:
    covertreex_backend = None

@pytest.mark.skipif(covertreex_backend is None, reason="Rust backend not built")
def test_rust_knn():
    print("Testing Rust KNN...")
    # Create simple data: Points on X axis
    # 0, 1, 2, 3, 4 ... 99
    N = 100
    dim = 2
    points = np.zeros((N, dim), dtype=np.float32)
    points[:, 0] = np.arange(N, dtype=np.float32)
    
    # Build Simple Tree (Linear Chain)
    # 0 -> 1 -> 2 ...
    # parents: [-1, 0, 1, 2 ...]
    # children: [1, 2, 3, ..., -1]
    # next_node: [-1, -1, -1 ...] (no siblings)
    
    parents = np.arange(-1, N-1, dtype=np.int64)
    children = np.arange(1, N+1, dtype=np.int64)
    children[-1] = -1 # Last one has no child
    next_node = np.full(N, -1, dtype=np.int64)
    levels = np.arange(N, dtype=np.int32) # Dummy levels
    
    # Initialize Rust Wrapper
    wrapper = covertreex_backend.CoverTreeWrapper(
        points, parents, children, next_node, levels, 0, 100
    )
    
    # Query: Point at 0.5 (Should match 0 and 1)
    queries = np.array([[0.5, 0.0]], dtype=np.float32)
    k = 2
    
    indices, dists = wrapper.knn_query(queries, k)
    
    print(f"Indices: {indices}")
    print(f"Dists: {dists}")
    
    assert indices.shape == (1, k)
    
    # Check results
    # Should be 0 and 1.
    # Dist to 0: 0.5. Dist to 1: 0.5.
    
    # Sort results to verify set (order might vary if dists equal)
    res_idx = np.sort(indices[0])
    assert np.array_equal(res_idx, [0, 1])
    assert np.allclose(dists[0], [0.5, 0.5])
    
    print("Rust KNN Test Passed!")

if __name__ == "__main__":
    test_rust_knn()
