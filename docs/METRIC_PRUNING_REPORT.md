# Metric Pruning Optimization Report

## Overview
This report details the **Metric Pruning** optimization applied to the Residual Correlation Cover Tree (PCCT). The primary objective was to eliminate the $O(N^2)$ bottleneck in the "Parent Search" phase of the tree construction, enabling the system to scale efficiently to larger datasets ($N > 100k$).

## Optimizations Implemented

### 1. Single-Tree Traversal
We replaced the naive "flat scan" parent search (which checked every active query against every existing tree node, roughly $O(N^2)$) with a **Single-Tree Traversal**. This approach uses the hierarchical structure of the existing Cover Tree to prune vast search spaces.

### 2. Triangle Inequality Pruning
The traversal employs standard metric pruning. For a query $q$ and a tree node $p$ with covering radius $r_p$ (stored in `si_cache`), if the distance to the node minus its radius is greater than the current nearest neighbor distance, the entire subtree at $p$ can be safely ignored:
$$ \max(0, d(q, p) - r_p) > d_{best}(q) \implies \text{Prune Subtree} $$

### 3. Numba Acceleration (`_residual_parent_numba.py`)
To make the fine-grained pruning checks efficient in Python, we implemented the traversal logic using **Numba**.
*   **Challenge:** The Residual metric relies on an RBF kernel, usually computed via BLAS `sgemm` in the host backend. Numba cannot easily call these Python/BLAS functions inside a JIT loop.
*   **Solution:** We reconstructed the RBF distance calculation directly inside the Numba kernel ($d(x,y) = \sqrt{1 - e^{-\gamma \|x-y\|^2}}$). This bypasses the Python interpreter overhead entirely for the traversal phase.

### 4. Cleanup (Gate 1 Removal)
The previous "Geometric Gate 1" mechanism was found to be ineffective and was removed to simplify the architecture and test suite.

## Benchmark Results

We compared the optimized **PCCT (Residual Metric)** against the baseline **MLPack (Euclidean Metric)**. Note that PCCT is computing a significantly more complex metric (kernel-based) than MLPack, yet achieves superior scaling.

### Scaling Analysis

| N Points | PCCT Build (Residual) | MLPack Build (Euclidean) | Speedup | PCCT Scaling (vs 4x Data) | MLPack Scaling (vs 4x Data) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **8,192** | 1.70s | 0.81s | 0.48x | - | - |
| **32,768** | 7.76s | 10.00s | **1.29x** | **4.6x** (Near Linear) | 12.3x (Super-Linear) |
| **131,072** | 44.89s | 180.62s | **4.02x** | **5.8x** (Near Linear) | 18.1x (Quadratic) |

*   **Cross-Over:** PCCT becomes faster than the C++ baseline between 8k and 32k points.
*   **Throughput:** PCCT query throughput is consistently **~10x higher** than MLPack (e.g., 11k q/s vs 882 q/s at 128k).

### Work Reduction (32k Dense)
*   **Before Optimization:** ~546 Million kernel evaluations.
*   **After Optimization:** ~17.8 Million kernel evaluations.
*   **Reduction:** **~30x**.

## Reproduction

### 1. Dense Baseline Verification
To run the standard 32k point dense build benchmark:
```bash
./run_dense_benchmark.sh
```

### 2. Scaling Comparison
To reproduce the specific data points from the report, you can use the CLI directly. Ensure `mlpack` bindings are installed for the baseline comparison.

**Example for N=32768:**
```bash
python -m cli.pcct query \
    --metric residual \
    --tree-points 32768 \
    --queries 1024 \
    --batch-size 512 \
    --baseline mlpack \
    --seed 42
```
*(Look for `pcct | build=...` and `baseline[mlpack] | build=...` in the output)*.
