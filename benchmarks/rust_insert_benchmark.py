import time
import numpy as np
import covertreex_backend
from tests.utils.datasets import gaussian_points

def run_benchmark():
    print("Benchmarking: Rust Batch Insert")
    
    N = 100_000
    D = 8
    batch_size = 10_000
    
    print(f"Data: N={N}, D={D}, BatchSize={batch_size}")
    
    rng = np.random.default_rng(42)
    points = gaussian_points(rng, N, D, dtype=np.float64).astype(np.float32)
    
    # Initial empty tree structure
    # We need to pass dummy arrays to initialize the wrapper
    # But we want to populate it via batch insert.
    # Wrapper takes initial points.
    # Let's initialize with 0 points?
    # PyO3 wrapper expects arrays.
    # Passing empty arrays:
    dummy_points = np.empty((0, D), dtype=np.float32)
    dummy_parents = np.empty(0, dtype=np.int64)
    dummy_children = np.empty(0, dtype=np.int64)
    dummy_next = np.empty(0, dtype=np.int64)
    dummy_levels = np.empty(0, dtype=np.int32)
    
    print("Initializing Rust Tree (Empty)...")
    # Note: Passing -100, 100 range.
    tree = covertreex_backend.CoverTreeWrapper(
        dummy_points, dummy_parents, dummy_children, dummy_next, dummy_levels, -20, 20
    )
    
    start_time = time.perf_counter()
    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch = points[i:end]
        # print(f"Inserting batch {i}..{end}")
        tree.insert(batch)
        
    elapsed = time.perf_counter() - start_time
    print(f"Total Time: {elapsed:.4f}s")
    print(f"Throughput: {N/elapsed:.1f} points/s")
    print(f"Final Size: {tree.point_count()}")

if __name__ == "__main__":
    run_benchmark()
