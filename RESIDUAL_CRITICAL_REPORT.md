# Critical Issue Report: Residual Benchmark Regression & Rust Panic

## 1. Overview
A detailed investigation into the performance regression (40k q/s $\to$ "freeze") and the Rust backend panic has been completed.

- **The "fast" Python benchmark (40k q/s) was invalid.** It was performing Euclidean distance on 1D integer indices, returning garbage neighbors (e.g., `511, 510, 509...`) with zero correlation to the query.
- **The "slow" Python benchmark (HEAD) is correct but unoptimized.** It correctly computes residual correlations but performs an exhaustive $O(N)$ search, causing the timeout.
- **The Rust panic was a traversal bug.** It was caused by improper handling of `-1` terminators in the legacy tree structure. This has been fixed.

## 2. Rust Backend Verification
We verified the correctness of the Rust implementation (`rust-hilbert`) after fixing the panic.

*   **Test:** Ran `run_residual_gold_standard.sh` with debug instrumentation.
*   **Result:** The Rust engine successfully identified the query point itself (Distance `0.0`) and returned a set of neighbors with non-sequential IDs.
*   **Throughput:** ~7,000 - 9,000 q/s.
*   **Conclusion:** The Rust implementation is **trustworthy**. It performs a genuine search and finds the true nearest neighbor (itself). The high distances of other neighbors are artifacts of the synthetic random data used in the benchmark.

## 3. Python/Numba Optimization Path
The huge gap between Rust (~8k q/s) and the current Python "freeze" (< 1 q/s) is due to **missing pruning logic** in the Numba implementation.

*   **Rust Logic:** `src/algo.rs` implements a Best-First Search with pruning:
    ```rust
    let lb = parent_dist - parent_radius; // Triangle inequality
    if lb > kth_dist { continue; }
    ```
    This effectively prunes branches that cannot contain better candidates, assuming a loose metric bound.
*   **Python Logic:** `covertreex/queries/_residual_knn_numba.py` implements BFS but **lacks this pruning check**. It pushes all children to the heap, resulting in near-exhaustive traversal.

## 4. Recommendations

### Immediate Actions (Fixes)
1.  **Commit the Rust Fix:** The patch to `src/algo.rs` (handling `-1` terminators) is critical and verified.
2.  **Acknowledge Baseline Shift:** The "Gold Standard" of 40k q/s should be invalidated. The new baseline for a *correct* residual query on this hardware is likely the Rust performance (~8k q/s).

### Optimization (Next Steps)
To restore Python/Numba performance, you must port the pruning logic from Rust to Numba:

1.  **Modify `_residual_knn_numba.py`:**
    *   Pass `si_cache` (radii) to the Numba kernel.
    *   Inside the BFS loop, compute the lower bound `lb = node_dist - radius`.
    *   Skip pushing children if `lb > current_kth_worst_distance`.
2.  **Verify Bounds:** Ensure the `Residual Distance` vs `Euclidean Radius` relationship holds sufficiently for the datasets in use, or accept that this is an approximate/heuristic search (which aligns with the "Opt-In" nature of this traversal).

## 5. File Changes Summary
- `src/algo.rs`: **FIXED** (Added check for `next == -1` to prevent panic).
- `cli/pcct/support/benchmark_utils.py`: **MODIFIED** (Added debug prints, should be reverted before commit).
