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
    
    # Initialize Rust Object
    core = covertreex_backend.CoverTreeCore(data)
    
    # Verify
    assert core.point_count() == 100
    
    # Check point retrieval
    p0 = core.get_point(0)
    assert np.allclose(p0, data[0])
    
    print("Rust Core Test Passed!")

if __name__ == "__main__":
    test_rust_core_wrapper()
