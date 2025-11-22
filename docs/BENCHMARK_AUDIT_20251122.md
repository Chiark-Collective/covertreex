# Benchmark Suite Audit (2025-11-22)

## Executive Summary

The project currently relies on two "canonical" sources of truth for benchmarking and several ad-hoc developer scripts. Confusion regarding performance results often stems from the fact that the automated regression suite **does not** run the "Gold Standard" configuration used for historical claims.

1.  **Historical "Gold Standard"**: Defined in `benchmarks/run_residual_gold_standard.sh`.
2.  **Automated Regression**: Defined in `tools/run_reference_benchmarks.py` (runs different, smaller jobs).
3.  **Developer/Ad-Hoc**: Scripts like `rust_full_residual_benchmark.py` that test the Rust backend directly, often with different architectural assumptions (indices vs coordinates).

---

## 1. The "Gold Standard" (Historical Results)
**File:** `benchmarks/run_residual_gold_standard.sh`

This script is the definitive source for the **24.20s build / 0.046s query** result reported on 2025-11-17. To reproduce historical numbers, this script must be used.

*   **Entry Point:** `python -m cli.pcct query`
*   **Configuration:**
    *   **N (Points)**: 32,768
    *   **D (Dimension)**: 3
    *   **Metric**: Residual
    *   **Batch Order**: Natural
    *   **Chunking/Sparse**: DISABLED (Explicitly unsets sparse traversal and chunking).
*   **Key Insight:** This script targets the Python/Numba path (or mixed path if environment variables allow) but strictly controls the environment to match the conditions of the original result.

## 2. Automated Regression Suite
**File:** `tools/run_reference_benchmarks.py`

This tool is designed for CI/CD or nightly checks to ensure feature stability. **It does NOT run the 32k Gold Standard workload.**

*   **Key Jobs:**
    *   `queries_2048_*`: Quick smoke tests (2k points).
    *   `queries_8192_*`: Medium scaling tests (8k points).
    *   `queries_32768_euclidean_hilbert_grid`: Tests Euclidean metric with advanced build options (Hilbert ordering + Grid conflict graph).
    *   `queries_32768_residual_dense_pairmerge`: Tests Residual metric with "dense pair-merge" streaming.
*   **Purpose:** Checks for regressions in specific subsystems (diagnostics, conflict graph implementations, streaming logic) rather than tracking peak performance of the standard path.

## 3. The Rust Discrepancy (Indices vs Coordinates)
**Files:** `benchmarks/rust_full_residual_benchmark.py`, `benchmarks/rust_knn_benchmark.py`

These ad-hoc scripts reveal a critical architectural divergence between the Python and Rust backends for the Residual metric:

*   **Python/Numba Path:** Builds the tree on **D-dimensional coordinates** (floats).
*   **Rust Residual Path:** Builds the tree on **1-dimensional indices** (integers stored as floats).
    *   The Rust wrapper (`CoverTreeWrapper`) for residual mode expects the primary tree data to be `(N, 1)` arrays of indices.
    *   The actual coordinates (`X`), V-matrix, and other metric data are passed as *separate arguments* to `insert_residual` and `knn_query_residual`.

**Impact:** Toggling `enable_rust=True` in a script designed for the Python path (which passes coordinates) will result in crashes or incorrect behavior, as the Rust backend attempts to interpret coordinates as indices. The main CLI (`cli.pcct`) handles this abstraction, but raw benchmark scripts must handle this data transformation manually.

## 4. Recommendations

1.  **Use `run_residual_gold_standard.sh`** for apples-to-apples historical comparisons.
2.  **Use `tools/run_reference_benchmarks.py`** for validating stability and correctness across different features.
3.  **Treat `rust_*_benchmark.py` scripts as low-level driver tests** for backend development, not system-level benchmarks.
