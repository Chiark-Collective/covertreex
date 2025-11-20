# Residual Sparse Gate Postmortem

**Date:** 2025-11-20
**Status:** Retired / Purged from Main Branch

## Summary

We attempted to optimize the build time of the Residual Correlation metric cover tree by implementing a "Sparse Traversal" strategy. This strategy relied on "Residual Gate 1" (a Whitened Euclidean Distance lower bound) to prune the search space, thereby avoiding expensive exact kernel evaluations.

The experiment failed to deliver performance improvements and, in fact, significantly regressed build times (from ~26s to ~270s on the 32k benchmark).

## Root Cause

1.  **Ineffective Lower Bound (Gate 1):**
    *   The "Whitened Euclidean" proxy metric proved to be an exceptionally loose lower bound for the Residual Correlation metric, particularly at larger radii required for the upper levels of the cover tree.
    *   Analysis of the gate profile revealed that to guarantee correctness (0 false negatives), the gate threshold had to be set so high that it effectively accepted **100% of candidates**.
    *   Telemetry confirmed `traversal_gate1_pruned` was consistently **0**.

2.  **Self-Inflicted Sparse Overhead:**
    *   The "Sparse" traversal engine is designed for scenarios where the candidate set is small (sparse). It involves overheads for managing adjacency lists, shuffling indices, and sorting pairs (`traversal_semisort_ms`) to optimize memory access for sparse operations.
    *   Because the gate failed to prune anything, the sparse engine was forced to process a fully dense dataset (100% density).
    *   Processing a 100% dense dataset with sparse-optimized logic (O(N log N) sorting + management overhead) is significantly slower than the "Dense" baseline, which simply blasts through contiguous memory using blocked BLAS/SGEMM instructions (O(N^2) but with highly optimized constants).

## Conclusion

The "Sparse Gated" approach is a dead end without a significantly tighter mathematical lower bound. Engineering optimizations to the sparse engine (hybrid switching, better sorting) would at best only allow it to fallback to the Dense path, matching the current baseline but adding complexity.

We have purged the `ResidualGate`, `ResidualPolicy`, `GateProfile`, and related sparse-gating machinery from the codebase to reduce technical debt and focus on optimizing the Dense path or finding a fundamentally better bound ("Gate 2").

## Artifacts Removed
*   `covertreex/metrics/residual/policy.py`
*   `tools/build_residual_gate_profile.py`
*   `tools/ingest_residual_gate_profile.py`
*   `covertreex/metrics/residual/_gate_profile_numba.py`
*   Gate-related logic in `core.py` and `host_backend.py`
