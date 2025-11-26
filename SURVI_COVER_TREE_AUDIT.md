# Architecture Review: Survi vs. Chiark Cover Tree

**Date:** 2025-11-26
**Subject:** Comparative analysis of the legacy `survi` cover tree implementation versus the new `chiark` (Rust/Numba) engines.

## 1. Overview

| Feature | `survi` (Legacy) | `chiark` (New) |
| :--- | :--- | :--- |
| **Language** | Python (Build) + Numba (Query) | Rust (Build & Query) or Numba (Query) |
| **Tree Storage** | Python Dicts (`children`, `assign`) | Flat Vectors (CSR-like in Rust) |
| **Construction** | Serial, Top-down (Slow) | Parallel, Batch Insertion (Fast) |
| **Memory Layout** | Input Order (Random/Natural) | Hilbert Curve / Z-Order (Cache Optimized) |
| **Kernels** | Hardcoded RBF/Matern52 in Numba | Native RBF/Matern52 + Embedding Support |

## 2. Detailed Comparison

### A. Tree Construction (The Major Bottleneck)
*   **Survi**: Uses a pure Python loop to insert points one by one (`CoverTree.build`). While it batches distance computations via JAX/NumPy callbacks, the overhead of Python object management for tree nodes makes this extremely slow for $N > 10,000$.
*   **Chiark**: Uses a highly optimized **Parallel Batch Insertion** algorithm in Rust. It processes points in chunks, uses multiple threads (`rayon`), and keeps all tree data in contiguous memory vectors. This yields **100x+ faster builds** for RBF kernels.

### B. Query Traversal
*   **Survi (Numba)**:
    *   Requires a costly "freeze" step to convert Python dicts into Numba-friendly arrays.
    *   Hardcodes kernel logic (`_rbf_pair`, `_m52_pair`) inside the JIT function. Adding a new kernel requires modifying the library source.
    *   Queries are parallelized via `prange` but process independently, missing opportunities for shared cache usage.
*   **Chiark (Rust - Hilbert)**:
    *   **Hilbert Ordering**: Reorders the dataset spatially before building. This ensures that points near each other in the tree are also near each other in memory, drastically reducing cache misses during traversal.
    *   **SIMD**: Leverages Rust's explicit control over memory alignment for effective auto-vectorization.
    *   **Batching**: Can process blocks of queries against the tree, amortizing node fetch costs (though the current "latency" benchmark focuses on per-query speed).

### C. Kernel Flexibility & Deep Kernels
*   **Survi**: To support a Deep Kernel (Neural Network), one would either have to write a complex Numba implementation of the forward pass (impractical) or fall back to the unoptimized Python/JAX query path (slow).
*   **Chiark**:
    *   **Native**: Supports RBF and Matern 5/2 directly with high performance.
    *   **Embeddings**: Explicitly designed for Deep Kernels via the `points_override` / embedding strategy. By building the tree on precomputed feature vectors $\Phi(X)$, it transforms the complex Deep Kernel problem into a fast standard metric search.

## 3. Why Swap?

1.  **Scalability**: `survi`'s Python build time is prohibitive for large datasets or online learning loops. `chiark`'s Rust build removes this barrier.
2.  **Maintainability**: `chiark` centralizes complex spatial logic in a rigorously tested Rust crate, removing ad-hoc Numba kernels from the `survi` codebase.
3.  **Performance**: The Hilbert-ordered Rust engine consistently outperforms the "natural order" Numba implementation, especially as $N$ grows, due to superior memory access patterns.

## 4. Recommendation

Proceed with the integration of `covertreex` (Rust engine) into `survi`.
-   Use **Rust Hilbert Engine** as the default.
-   Use **Matern 5/2** (native Rust) for standard GP workloads.
-   Use **Embedding Strategy** (`points_override`) for Neural/Deep Kernels.
