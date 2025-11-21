import numpy as np
import pytest

try:
    import covertreex_backend
except ImportError:
    covertreex_backend = None

@pytest.mark.skipif(covertreex_backend is None, reason="Rust backend not built")
def test_rust_core_wrapper():
    print("Testing Rust Core...")
    # Create dummy data
    data = np.random.rand(100, 5).astype(np.float32)
    N = 100
    parents = np.full(N, -1, dtype=np.int64)
    children = np.full(N, -1, dtype=np.int64)
    next_node = np.full(N, -1, dtype=np.int64)
    levels = np.zeros(N, dtype=np.int32)
    
    # Initialize Rust Object
    core = covertreex_backend.CoverTreeWrapper(
        data, parents, children, next_node, levels, -10, 10
    )
    
    # Verify
    assert core.point_count() == 100
    
    # Check point retrieval (wrapper doesn't expose get_point directly, but we can test via KNN?)
    # The wrapper only exposes `knn_query` and `insert`.
    # Let's just check point_count.
    
    print("Rust Core Test Passed!")

if __name__ == "__main__":
    test_rust_core_wrapper()
