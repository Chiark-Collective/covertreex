# Residual Benchmark Regression & Rust Panic Analysis

## Executive Summary
The massive performance regression (from ~40k queries/sec to "freezing") observed between commit `6e81` and `HEAD` is caused by a logic change in `covertreex/queries/knn.py` that inadvertently forces an unoptimized, exhaustive search path for residual metric queries.

Additionally, the Rust comparison backend (`rust-hilbert`) was panicking due to a traversal bug in `src/algo.rs` when handling singly-linked lists (as produced by the legacy residual tree builder).

## 1. Performance Regression (Python/Numba)

- **Previous Behavior (Fast, Correctness Impacted):** The benchmark ran the "legacy" residual path which built a tree over integer indices and, due to a flag mismatch, fell back to standard Euclidean KNN on those 1D indices. This resulted in extremely fast O(log N) lookups (~40k/s) that were effectively checking numerical index proximity rather than actual residual correlation. **Proof:** Debugging revealed neighbors like `[511, 510, 509...]` for query `16558` with residual distances of `~0.99` (pure noise).
- **Current Behavior (Slow/Freeze, Correctness Improved but Unoptimized):** A recent change forces the usage of the dedicated `residual_knn_query` implementation when the metric is `residual_correlation`. However, this implementation currently lacks pruning logic ("No, if we don't prune, we visit everyone"), resulting in an O(N) exhaustive search. For 32k points and 1k queries, this triggers ~32 million expensive RBF kernel evaluations, causing the observed "freeze" (runtime in minutes vs milliseconds).

### Recommendations
- To restore the previous (fast/invalid) behavior for baselining purposes, revert the condition in `covertreex/queries/knn.py`.
- To fix the performance properly, a pruning bound (e.g. Triangle Inequality with Euclidean wrapper) must be implemented in `residual_knn_search_numba`.

## 2. Rust Panic (rust-hilbert)

- **Issue:** The `rust-hilbert` engine (invoked as a comparator) crashed with `index out of bounds` on `usize::MAX`.
- **Root Cause:** The tree builder (`src/tree.rs`) initializes sibling lists with `-1` terminators. The traversal code in `src/algo.rs` (specifically `dfs`) assumed circular lists or did not check for `-1` explicitly before casting to `usize`. This caused `child` to become `-1` (cast to `usize::MAX`), leading to an out-of-bounds access on the next iteration.
- **Fix:** Applied a check `if next == -1` in `src/algo.rs` loops to safely terminate traversal.
- **Verification:** The benchmark now runs the Rust comparison successfully:
  ```
  pcct | build=3.6216s queries=1024 k=50 time=0.1262s latency=0.1233ms throughput=8,112.5 q/s
  ```