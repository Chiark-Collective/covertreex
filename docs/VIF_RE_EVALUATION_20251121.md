# VIF Re-evaluation: Viability on Large Point Clouds

**Date:** 2025-11-21
**Context:** Re-evaluating the PCCT strategy for Vecchia-Inducing-Points (VIF) on large, low-dimensional datasets (e.g., 3D Point Clouds, $N=200k+$) in light of GPBoost implementation notes and recent scaling benchmarks.

## 1. The Critical Conflict

We have identified a fundamental conflict between the "Gold Standard" implementation (GPBoost C++) and the current PCCT architecture (Python + Numba):

| Feature | GPBoost (C++) | PCCT (Python/Numba) |
| :--- | :--- | :--- |
| **Tree Construction** | Fast ($O(N \log N)$ with low constant). Native C++ `std::vector` / `std::map`. | **Slow** ($D=8, N=200k \approx 10$ mins). dominated by Python interpreter overhead for node/list management. |
| **Update Strategy** | **Rebuild** tree every $2^k$ steps (amortized cost). | **Cannot Rebuild**. 10-minute latency breaks the training loop. |
| **Metric** | Dynamic Residual Correlation (changes every step). | Static Euclidean (built once) or Dynamic (too slow). |

**Conclusion:** The "Lazy Rebuild" strategy used by GPBoost is **currently non-viable** for PCCT on large datasets because our build times are orders of magnitude higher than the C++ reference, despite Numba acceleration for the distance kernels. The "Python Tax" on tree structure manipulation is the bottleneck.

## 2. The "Static Tree" Hypothesis

Given that we cannot rebuild the tree, we must double down on the **"Static Tree, Dynamic Query"** strategy proposed in `VIF_DYNAMIC_METRIC_GUIDE.md`.

### The Approach
1.  **Build Once:** Construct the tree using **Euclidean Distance** ($L_2$) on the static input coordinates ($X$). This takes ~10 minutes for 200k points (currently), but is paid only once at initialization.
2.  **Query Dynamically:** Use the Euclidean tree structure to perform k-NN searches for the **Residual Correlation** metric ($d_{res}$).

### The Risk: Metric Mismatch
The GPBoost paper explicitly states: *"simply using the Euclidean metric in a transformed space is not applicable... corresponds to a non-stationary covariance function"*.

However, we are not using Euclidean distance *as* the metric. We are using the Euclidean *tree structure* to **branch-and-bound** the Residual metric. This works **IF AND ONLY IF**:
$$ d_{res}(A, B) \ge f(d_{eucl}(A, B)) $$
...where $f$ is some monotonic function that allows us to prune a node based on its Euclidean bounding box.

*   **Evidence for Viability:** The current `find_parents_numba` implementation in `covertreex` *already* uses an RBF (Euclidean-based) approximation to prune the search. This suggests that implicitly, we are already relying on the fact that "spatial proximity $\approx$ correlation proximity".
*   **Evidence against:** If the residual process has strong long-range correlations (e.g., low-frequency artifacts not captured by the low-rank inducing points), the Euclidean tree might prune nodes that are actually highly correlated in the residual space.

## 3. Re-evaluation & Recommendation

### A. Immediate Strategy (The "Static" Bet)
We proceed with the **Static Tree** approach. It is the only path that fits within the computational budget of a Python-based training loop.
*   **Action:** Ensure `covertreex.metrics.residual` correctly implements the pruning logic using Euclidean bounds against the current Residual Kernel state.
*   **Optimization:** Investigate why $D=8$ build time is 574s vs $D=64$ is 64s. The "crowding" hypothesis suggests we might need to tune `base` (expansion constant) for low dimensions to reduce tree depth and Python overhead.

### B. The "Plan B" (Engineering Heavy)
If the Static Tree fails (accuracy drops significantly compared to exact brute-force), we must eliminate the Python bottleneck to enable the "Rebuild" strategy.
*   **Requirement:** Port the **entire tree structure and construction logic** to Numba or C++.
*   **Target:** A "Flat Array" Cover Tree (CSR-style), removing all `Dict` and `List` objects.
*   **Est. Effort:** High. This is effectively a rewrite of the core `batch_insert` logic.

### C. The Numba Port (Notes)
The file `numba_cover_tree.py` provided in the notes is a **partial solution**. It accelerates distances but *still uses Python dictionaries* for the tree structure (`Dict[int, List[int]]`). **It will not solve the build-time bottleneck.** It should be treated as a reference for the distance logic, not a drop-in replacement for the tree builder.

## Summary
We cannot afford to copy GPBoost's "Rebuild" strategy. We must make the **Static Tree** work by proving that Euclidean structure is a sufficient proxy for efficient pruning of the Dynamic Residual metric.
