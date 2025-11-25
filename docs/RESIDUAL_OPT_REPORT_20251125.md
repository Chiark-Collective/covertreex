# Residual Optimization Report - 2025-11-25

## Summary of Experiments

### 1. Best-First Search (BFS) Attempt
*   **Implementation:** Implemented a BinaryHeap-based BFS traversal in Rust (similar to Numba's strategy) to replace the Level-Synchronous scan.
*   **Result:** Throughput dropped significantly to **~1,800 q/s** (vs ~7,200 q/s baseline).
*   **Analysis:** The overhead of heap operations and individual node processing in Rust (without JIT fusion of the metric kernel) outweighed the benefits of visiting fewer nodes. Numba's 40k q/s BFS likely benefits from aggressive inlining and kernel fusion that eliminates function call overheads for the RBF calculation.

### 2. Level-Synchronous Optimization (Selected Path)
*   **Strategy:** Reverted to the Level-Synchronous traversal which batches children effectively.
*   **Optimization:** Removed per-query `O(N)` allocations for `cached_lb` (level cache) and `seen_mask`.
    *   *Rationale:* The tree structure guarantees unique parent-child paths, making these deduplication structures largely redundant for correctness while costing significant allocation bandwidth (37k queries * 256KB allocs = ~9GB/s).
*   **Result:** Restored baseline performance to **~7,100 q/s** with reduced memory pressure.

### 3. Cleanup
*   Removed the legacy `single_residual_knn_query_heap` function and unused `batch_residual_knn_query_block`.
*   Fixed lifetime compilation errors and unused variable warnings.
*   Ensured `single_residual_knn_query` is now the canonical, optimized implementation.

## Recommendations for Future Optimization

1.  **Kernel Fusion:** To match Numba's 40k q/s, Rust likely needs to inline the `ResidualMetric` calculations directly into the traversal loop to avoid function call overhead per batch/node.
2.  **Adaptive Tiling:** Implementing Numba's dynamic block sizing (adapting `stream_tile` based on queue size) might help.
3.  **Backporting to Numba:** The `BitSet` optimization used in the BFS experiment is worth backporting to Numba if it currently uses a standard boolean array/set, as it reduces cache pressure.
