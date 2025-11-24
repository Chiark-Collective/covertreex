# Residual Status Report: 2025-11-24 (Update 3)

## Final Comparison: Rust vs. Python-Numba

Completed gold standard benchmarks for all three engines (32k points, d=3, k=50, 1024 queries).

### Results Table

| Engine | Implementation | Order | Build Time | Query Time | Throughput | Relative Perf |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **python-numba** | Numba JIT (level cache, dynamic blocks) | Natural | 16.10s | **0.05s** | **21,123 q/s** | **1.0x (Baseline)** |
| **rust-fast** | Rust Native (dense streamer, caps) | Natural | 7.42s | 0.14s | 7,105 q/s | ~0.34x |
| **rust-pcct2** | Rust Native (dense streamer, caps) | Hilbert | **4.20s** | 0.21s | 4,971 q/s | ~0.24x |

### Analysis

1.  **Build Speed:** `rust-pcct2` is the clear winner, building nearly **4x faster** than the Python path and ~1.7x faster than `rust-fast`. The Hilbert ordering effectively sparsifies the conflict graph, reducing construction overhead.
2.  **Query Throughput:** Python-Numba remains significantly faster (~3x faster than `rust-fast`).
    *   The Numba path benefits heavily from **Level Caching** (reusing valid parents from the previous level's search) and **Dynamic Block Sizing** (adapting batch size to active query count).
    *   The Rust engines currently re-evaluate children for every query at every level (despite the new tiling optimization) and use a fixed block size.
3.  **Rust Progress:**
    *   `rust-fast` throughput improved from ~37 q/s to 7,105 q/s today (200x speedup).
    *   The gap between `rust-fast` and `rust-pcct2` queries (~30%) highlights that Hilbert ordering, while great for build, might produce tree structures that are slightly less optimal for the current *random* query access pattern (or interact less favorably with the budget pruning heuristics than a natural/random order).

### Conclusion
The Rust migration is viable for high-performance construction. To match Numba's query speed, we must implement the **Level Cache** mechanism in the Rust traversal.