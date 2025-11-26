# Rust Residual Optimization Report - 2025-11-26

## Objective
Close the performance gap between Rust (~1.3k - 7k q/s) and Python-Numba (~40k q/s) for the Residual Cover Tree traversal.

## Experiments & Findings

### 1. Traversal Strategy: Level-Synchronous vs. Heap BFS
- **Heap BFS:** Implemented a Priority Queue based BFS (similar to Numba's logic).
    - **Result:** Poor performance (~1.3k q/s). High overhead of heap operations per node.
- **Level-Synchronous:** Processes nodes level-by-level, grouping children.
    - **Result:** Superior (~7k q/s baseline). Better cache locality and vectorization potential.

### 2. Kernel Fusion
- **Hypothesis:** Numba is faster because it fuses the RBF metric calculation into the traversal loop.
- **Experiment:** Implemented a manual "Fused" kernel in Rust, bypassing `ResidualMetric` abstraction and `ndarray` views, using `wide` SIMD.
- **Result:** **Slower** (~5.9k q/s). The manual SIMD loop (f32x8) performed worse than the compiler-optimized generic code (`dot_tile_f32` + separate loop).
- **Conclusion:** Abstraction overhead is not the primary bottleneck.

### 3. Level Caching & Allocation
- **Hypothesis:** Numba uses a "Level Cache" (Lower Bound Pruning) effectively. Rust implementation disabled it because per-query allocation of `cached_lb` (size N) was too slow.
- **Optimization C:** **Thread-Local Buffer Reuse**.
    - Modified `batch_residual_knn_query` to process queries in chunks (via `par_chunks`).
    - Allocated `cached_lb` (128KB) once per chunk and passed it down.
    - Re-enabled Level Cache pruning logic.
- **Result:** **~12.8k q/s** (peak).
    - This is the most significant gain (~1.8x speedup over baseline).
    - "MIN" accumulation strategy (`cached_lb = min(cached_lb, parent_lb)`) proved effective.

### 4. Lazy Reset (Optimization D)
- **Hypothesis:** `memset` (fill) of `cached_lb` buffer costs ~8% of time. Lazy reset via generation counter could save this.
- **Experiment:** Implemented `lb_generation` array and checks.
- **Result:** **Regression** (~6.9k q/s). The overhead of checking generations and reading the extra array outweighed the `memset` savings.

### 5. Deduplication (Visited Set)
- **Experiment:** Enabled `visited_nodes` bitset to prevent re-evaluating duplicates.
- **Result:** **Regression** (~8k q/s vs 9.4k q/s). The overhead of bitset maintenance outweighed the savings from avoiding duplicate distance calculations (likely duplicates are rare or cache-hot).

## Final Status
- **Current Rust Performance:** **~9,400 - 12,800 q/s**.
- **Numba Performance:** **~40,000 q/s**.
- **Gap:** ~3x - 4x.

## Future Work
- **SIMD Analysis:** Deep dive into assembly. Numba likely generates efficient AVX-512 for the RBF kernel. Rust might need explicit intrinsics or layout changes (SoA vs AoS) to match.
- **Dynamic Blocking:** Re-visit adaptive tile sizes with the new buffer reuse architecture.
