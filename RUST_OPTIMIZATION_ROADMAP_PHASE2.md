# Optimization Roadmap: Residual Cover Tree (Phase 2)

**Date:** 2025-11-26
**Status:** "Euclidean Hybrid" mode is fast (~12.8k q/s) but lags Numba (~40k q/s). "Pure Residual" mode is functional (O(N^2) bug fixed) but slower (~6.6k q/s).

## 1. Primary Objective: Close the Compute Gap (12.8k -> 40k)
The 3x performance disparity between Rust and Numba for the *exact same* traversal logic (Euclidean structure, Residual query) indicates a massive difference in the efficiency of the core distance kernel.

### A. SIMD Assembly Analysis
*   **Hypothesis:** Numba (LLVM) is generating AVX2/AVX-512 FMA instructions for the RBF kernel that are significantly more efficient than the Rust compiler's auto-vectorization.
*   **Action:**
    1.  Disassemble the Numba JIT-compiled kernel.
    2.  Disassemble the Rust `ResidualMetric::distance_sq`.
    3.  Compare instruction mix, register pressure, and loop unrolling.
*   **Target:** Implement explicit `std::simd` or `wide` crate intrinsics if auto-vectorization is failing.

### B. Data Layout (SoA vs AoS)
*   **Current:** Rust uses `ArrayView2` (row-major), effectively Array-of-Structures (AoS) for 3D points.
*   **Numba:** Often optimizes layout or unrolls effectively.
*   **Action:** Experiment with a Structure-of-Arrays (SoA) layout for the points (i.e., `[x0...xn], [y0...yn], [z0...zn]`). This allows loading 8 `x` coordinates into a single AVX register without gathering.

## 2. Secondary Objective: Optimizing "Pure" Residual Trees
We fixed the O(N^2) build bug, achieving a **7.05s** build time for 32k points. However, query throughput dropped to **6.6k q/s** (vs 12.8k q/s on the Euclidean tree).

### A. Tree Quality Analysis
*   **Question:** Why does the tree built with the *correct* metric perform *worse*?
*   **Hypotheses:**
    1.  **Poor Balancing:** The Residual metric (0-1 bounded) might be producing "stringy" or unbalanced trees compared to the spatial Euclidean tree.
    2.  **Expansion Constant:** The base expansion constant (base=2.0) might be too aggressive for a metric bounded at 1.0.
*   **Action:**
    1.  Profile tree depth, branching factor, and "waste" (overlap) for both tree types.
    2.  Experiment with different expansion bases (e.g., 1.5, 1.3) for the Residual metric.

### B. Build Performance
*   **Status:** 7s for 32k points is acceptable but could be faster.
*   **Action:**
    1.  Profile the `batch_insert` parallel implementation.
    2.  Optimize the `compute_mis_greedy` step, which is likely the bottleneck for high-conflict non-Euclidean metrics.

## 3. Architectural Cleanup
*   **Action:** Formalize the `COVERTREEX_RESIDUAL_USE_STATIC_EUCLIDEAN_TREE` configuration. It currently feels like a hidden switch. It should be a first-class citizen in the `Profile` or `Config` object, explicitly selecting the "Build Strategy" (Spatial Proxy vs. Exact Metric).
