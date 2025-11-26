# Project Status Report: Rust Residual Optimization

**Date:** 2025-11-26
**Context:** Closing the performance gap between Rust and Python-Numba backends for the "Residual" Cover Tree traversal.

## 1. Executive Summary
We have achieved a major milestone in **stability and correctness** by fixing a critical performance regression in the Rust tree construction.
*   **Build Fixed:** The "Pure" Residual tree construction (using the actual RBF metric) no longer suffers from $O(N^2)$ complexity. Build times for 32k points have dropped to **~4-6 seconds**, making them faster than the Python reference.
*   **Query Gap:** A significant throughput gap remains. The optimized Rust implementation (~9k q/s) is **~4.2x slower** than the Numba reference (~38k q/s).
*   **Parity Verified:** We confirmed that "Pure" Residual trees and "Hybrid" (Euclidean-built) trees produce identical k-NN results, validating the correctness of the Pure approach.

## 2. Current Benchmarks
**Scenario:** "Gold Standard" - 32,768 Points (3D Gaussian), 1,024 Queries, $k=50$.

| Engine | Configuration | Build Time | Query Throughput | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Python-Numba** | Reference (Natural Order) | ~7.83s | **~38,361 q/s** | **Gold Standard** |
| **Rust Hilbert** | Pure Residual (Hilbert Order) | **~4.30s** | ~8,961 q/s | Best Rust |
| **Rust Hybrid** | Euclidean Build / Residual Query | ~5.83s | ~7,412 q/s | Deprecated* |
| **Rust Natural** | Pure Residual (Natural Order) | ~6.31s | ~7,302 q/s | Baseline |

*\*The "Hybrid" approach offers no query performance benefit and is slightly slower to build than the Hilbert-ordered Pure tree. It is now considered a fallback rather than a primary optimization strategy.*

## 3. Completed Optimizations (Rust)

### A. Construction Fix (The "Star Tree" Bug)
Investigated and resolved a severe performance regression where metrics advertising a `max_distance_hint` (like Residual) triggered a shortcut in the builder.
*   **Issue:** The builder bypassed the hierarchy, attaching all points to the root (degenerate "star" graph).
*   **Impact:** Insertion became $O(N)$, Total Build $O(N^2)$.
*   **Fix:** Removed the flawed optimization. Construction is now properly $O(N \log N)$.

### B. Level-Synchronous Traversal
Moved to a **Level-Synchronous** approach (vs Heap BFS). Processes nodes layer-by-level, improving instruction cache locality.

### C. Thread-Local Buffer Reuse
Eliminated allocation pressure by using `par_chunks` and reusing the `cached_lb` pruning buffer (128KB) across queries in a batch. This yielded a **1.8x** speedup over the naive Rust implementation.

## 4. Dead Ends & Discarded Approaches
*   **Hybrid Construction:** Building the tree with Euclidean distance ($L_2$) and querying with Residual (RBF) was hypothesized to be faster. **Finding:** It is *not* faster to build (vs Hilbert Pure) and offers *no* query throughput advantage.
*   **Fused Kernel (Manual SIMD):** Manual `f32x8` SIMD in Rust was slower than the compiler's auto-vectorization of the generic kernel.
*   **Lazy Buffer Reset:** Using generation counters to avoid `memset` proved slower due to branch overhead.

## 5. Codebase Changes
*   `src/algo/batch/mod.rs`: Removed `max_distance_hint` shortcut.
*   `src/algo.rs`: Implemented chunked batch processing with buffer reuse.
*   `benchmarks/`: Added `residual_pure_vs_hybrid.py` and `residual_natural_vs_hilbert.py` for ongoing regression testing.

## 6. Optimization Roadmap

### Near Term: The "Query Gap"
The 4x throughput disparity (~9k vs ~38k) is the primary remaining issue.
1.  **SIMD Assembly Analysis:** Numba is likely generating superior vector code for the RBF kernel. We need to inspect the generated assembly for `distance_sq`.
2.  **Memory Layout:** Investigate if Numba benefits from specific memory layouts (e.g., contiguous coordinate arrays vs Rust's `ArrayView` slicing).
3.  **Block Query:** The `rust-hilbert` engine uses an experimental block-query path. This path needs to be refined and potentially adopted as the default if it proves robust.

### Long Term
1.  **Metric Pruning:** Explore if the `si_cache` (triangle inequality bounds) is being utilized as effectively in Rust as in Numba.
2.  **Kernel Fusion:** Re-evaluate kernel fusion with a focus on "vertical" fusion (integrating distance calculation directly into the traversal sorting network).

## 7. Final Verified Configuration (Optimized)
Following the investigation into the "Release Mode" performance gap, we have applied a definitive fix to ensure high performance by default.

*   **Root Cause:** The `rust-hilbert` engine relies heavily on auto-vectorization (AVX2/AVX-512) for the Residual RBF kernel. The default `dev` profile (used by `maturin develop`) disabled optimizations, resulting in scalar execution (~9k q/s). Release builds (`--release`) correctly vectorized the kernel (~68k q/s).
*   **Resolution:** We configured `Cargo.toml` to force `opt-level = 3` for the `[profile.dev]` section.
*   **Result:** The default build now achieves **~62,508 q/s**, exceeding the Python-Numba reference (~40k q/s) by **~1.5x**.
*   **Recommendation:** No special build flags are required. Standard installation (`pip install .`) or development builds (`maturin develop`) will now be optimized out-of-the-box.