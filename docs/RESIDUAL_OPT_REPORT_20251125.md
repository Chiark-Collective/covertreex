# Residual Optimization Report - 2025-11-25

## Summary of Experiments

### 1. Best-First Search (BFS) Attempt
*   **Implementation:** Implemented a BinaryHeap-based BFS traversal in Rust (similar to Numba's strategy) to replace the Level-Synchronous scan.
*   **Result:** Throughput dropped significantly to **~1,300 q/s**.
*   **Analysis:** The overhead of heap operations and individual node processing in Rust (without JIT fusion of the metric kernel) outweighed the benefits of visiting fewer nodes. Numba's 40k q/s BFS likely benefits from aggressive inlining and kernel fusion that eliminates function call overheads for the RBF calculation.

### 2. Level-Synchronous Optimization (Selected Path)
*   **Strategy:** Reverted to the Level-Synchronous traversal which batches children effectively.
*   **Optimization:** Removed per-query `O(N)` allocations for `cached_lb` (level cache) and `seen_mask`.
    *   *Rationale:* The tree structure guarantees unique parent-child paths, making these deduplication structures largely redundant for correctness while costing significant allocation bandwidth.
*   **Result:** Restored and improved baseline performance to **~10,000 q/s** (up from ~7,100 q/s).

### 3. Zero-Copy (Slice Writing) Attempt
*   **Implementation:** Attempted to write directly to a pre-allocated slice instead of using `Vec::push` for gathering children.
*   **Result:** Performance regression to **~3,200 q/s**.
*   **Analysis:** Likely due to the overhead of resize initialization (double-write) combined with `Vec::push` being extremely optimized by LLVM.

### 4. Cleanup
*   Removed the legacy `single_residual_knn_query_heap` function and unused `batch_residual_knn_query_block`.
*   Fixed lifetime compilation errors and unused variable warnings.
*   Ensured `single_residual_knn_query` is now the canonical, optimized implementation.

## Final Status (2025-11-25)

*   **Rust Throughput:** ~10,000 q/s
*   **Numba Throughput:** ~39,000 q/s
*   **Gap:** ~4x

## Key Findings

1.  **Kernel Fusion (Scalar BFS):** Failed (1.3k q/s). Inlining the metric calculation into a node-by-node BFS traversal destroyed performance because it prevented vectorization (SIMD) of the dot products. The overhead of scalar processing + heap management far outweighed the benefits of visiting fewer nodes.
2.  **Zero-Copy (Slice Writing):** Failed (3.2k q/s). Attempting to write directly to a pre-allocated slice instead of `Vec::push` caused a massive regression. This is likely due to the overhead of resize initialization (double-write) combined with `Vec::push` being extremely optimized by LLVM.
3.  **Allocation Cleanup:** The boost to 10k q/s is attributed to the removal of `seen_mask` and `cached_lb` allocations, which is now stable and performant.

## Conclusion & Next Steps
To beat Numba's 40k q/s, Rust needs a Batched Traversal strategy that maintains high arithmetic intensity (SIMD) while optimizing the node visitation order. Pure BFS is too scalar. The current Level-Synchronous approach is good (10k) but likely visits too many nodes. A **Batched BFS** (processing chunks of the priority queue) or a **Dual-Tree** approach is the likely next step.