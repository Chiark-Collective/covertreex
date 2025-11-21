import numpy as np
import pytest
import covertreex_backend

@pytest.mark.skipif(not hasattr(covertreex_backend, "CoverTreeWrapper"), reason="Rust backend not available")
def test_rust_batch_insert():
    print("Testing Rust Batch Insert...")
    
    # Initial Tree
    N = 10
    D = 3
    points = np.zeros((N, D), dtype=np.float32)
    parents = np.full(N, -1, dtype=np.int64)
    children = np.full(N, -1, dtype=np.int64)
    next_node = np.full(N, -1, dtype=np.int64)
    levels = np.zeros(N, dtype=np.int32)
    
    wrapper = covertreex_backend.CoverTreeWrapper(
        points, parents, children, next_node, levels, -10, 10
    )
    assert wrapper.point_count() == 10
    
    # Batch Insert
    batch = np.random.rand(5, D).astype(np.float32)
    wrapper.insert(batch)
    
    assert wrapper.point_count() == 15
    
    print("Rust Batch Insert Test Passed!")

if __name__ == "__main__":
    test_rust_batch_insert()
