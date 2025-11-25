# Residual Optimization Gap Analysis & Argumentation

## The Argument: Arithmetic Intensity vs. Memory Latency

We have successfully pushed the Rust implementation from **~4.5k q/s** (regression state) to **~10.3k q/s** (current state). However, the Numba baseline stands firm at **~39k q/s**. This analysis argues that the remaining gap is structural, specifically tied to **how data is gathered** before being processed by our now-vectorized kernel.

### 1. Most Impactful Optimization: SIMD Vectorization (The "Math" Bottleneck)
*   **Change:** Explicit usage of `wide::f32x8` to vectorize `exp`, `sqrt`, and division in `metric.rs`.
*   **Impact:** ~40% speedup (7k -> 10k).
*   **Argument:** The Residual RBF kernel is "transcendentally expensive." Unlike Euclidean distance (simple FMA), RBF involves exponentials and square roots. In scalar code, these stall the pipeline. By using SIMD, we process 8 distances for the cost of roughly 2 scalar ops (latency-wise). This proves that **arithmetic intensity** was a major bottleneck.

### 2. Second Most Impactful: Allocation Removal (The "alloc" Bottleneck)
*   **Change:** Removing `Vec` allocations inside the `single_residual_knn_query` hot loop (replacing `parent_children` vector with direct iterator).
*   **Impact:** Recovered from ~4.5k regression back to ~7k baseline.
*   **Argument:** `malloc` and `free` are system calls (or complex allocator ops). Doing this **per node visited** (roughly 500-1000 times per query * 30k queries = 30M allocs/sec) saturates the allocator lock and destroys cache locality.

### 3. The "Failed" Experiment: BFS / Kernel Fusion
*   **Change:** Switching from Level-Synchronous (batches of nodes) to Batched BFS (priority queue).
*   **Impact:** Regression to ~1.6k q/s.
*   **Argument:**
    1.  **Heap Overhead:** `BinaryHeap` operations are `O(log N)` and involve unpredictable memory access compared to `Vec::push`.
    2.  **Loss of Contiguity:** BFS processes nodes in "quality" order, which is random in memory. Level-Synchronous processes nodes in "layer" order, which effectively batches children comparisons. This allows the CPU prefetcher to work. Numba likely succeeds with BFS because it JIT-compiles the *entire* stack, fusing the heap logic with the metric logic, whereas Rust has function call boundaries.

### 4. The Remaining Gap: The "Gather" Bottleneck
Numba (39k) is still 4x faster than optimized Rust (10k).
*   **Hypothesis:** Numba's JIT might be generating a **fused gather-compute kernel**.
*   **Rust's Problem:** We gather indices into a `p_indices` vector, then pass that to `distances_f32...`. This kernel then has to **indirectly address** `coords` and `v_matrix` (`coords[p_indices[i]]`). This "gather" pattern prevents full SIMD utilization because we are waiting on cache misses for every vector load.
*   **Evidence:** We optimized the math (SIMD), we optimized the control flow (Allocations), but we are still slow. The only thing left is **Data Access**.

## Strategic Recommendation: Dual-Tree or Blocked Layout

To beat Numba, we must stop "chasing pointers."
1.  **Dual-Tree Traversal:** Instead of Query vs. Tree, traverse Query-Batch vs. Tree-Batch. This amortizes the "gather" cost over many queries.
2.  **Data Layout:** Reorganize the tree data so that children of a node are contiguous in memory. This eliminates the `p_indices` gather entirely—we just load `ptr[0..8]`.

**Winner:** The most impactful *immediate* change was **SIMD Vectorization**. The most impactful *future* change will be **Data Layout / Dual-Tree**.
