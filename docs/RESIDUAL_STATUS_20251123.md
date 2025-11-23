# Residual Metric Status Report (2025-11-23)

## Overview

Update on the Rust backend optimization for the Residual Correlation metric.

## Progress

1.  **Optimization Implemented:**
    *   **Branch-and-Bound Pruning:** Implemented standard Cover Tree pruning ($d(q, u) - 2^{j+1} > d_{kth}$) in the Rust traversal.
    *   **Memory Reduction:** Removed per-query $O(N)$ allocation (`visited` vector).
    *   **Panic Fix:** Fixed a critical panic in `ResidualMetric` caused by assuming contiguous memory layout for V-Matrix rows passed from Python. Now uses robust unchecked indexing.

2.  **Performance Results (50k benchmark):**
    *   **Baseline (Before):** ~20 QPS (Est.)
    *   **Current (After):** ~202 QPS
    *   **Speedup:** ~10x
    *   **Comparison:** Still significantly slower than Python/Numba (~36,000 QPS).

3.  **Analysis of the Gap:**
    *   The Rust implementation computes distances one-by-one using scalar loops (auto-vectorized at best).
    *   The Python/Numba implementation likely benefits from highly optimized JIT compilation of the kernel or potentially uses a structural batching approach (processing multiple queries against a node, or vice versa) that is more cache-friendly.
    *   **Hypothesis:** To match Numba, we need to amortize the overhead of the metric computation (specifically the V-Matrix dot product) by processing blocks of queries or using explicit SIMD/BLAS calls.

## Next Steps

*   Branch `optimization/rust-batching` created.
*   Goal: Implement a "Batch Traversal" or "Dual-Tree-like" traversal in Rust where a set of queries descends the tree together, allowing for matrix-vector multiplication (GEMV) or matrix-matrix multiplication (GEMM) for distance computations.
