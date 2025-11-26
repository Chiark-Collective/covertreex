# Residual Optimization Implementation Plan (Query Path)

**Date:** 2025-11-26
**Objective:** Close the ~7x query throughput gap (6k vs 40k q/s) between Rust and Python-Numba for Residual Cover Trees.

## 1. Status & Analysis (Post-Phase 1)
*   **Completed:** Phase 1 (Compute Efficiency) & Correctness Fixes.
    *   **SIMD:** Implemented unrolled row-major dot product kernels. (SoA was attempted and reverted due to complexity/sparse access patterns).
    *   **Correctness:** Fixed critical bug where `rust-hilbert` returned internal node IDs instead of dataset IDs.
    *   **Layout:** Implemented `inv_order` mapping to ensure `ResidualMetric` accesses reordered (contiguous) memory.
*   **Current Performance:** ~6,000 q/s (Correct Rust) vs ~40,000 q/s (Numba).
*   **Root Cause:** The "Fast" Rust baseline (~9k-13k q/s) was largely illusory due to incorrect indices or lucky memory layout in `natural` order. The correct baseline is ~6k q/s.
*   **Hypothesis:** The remaining gap is purely **Algorithmic**. Numba uses a sophisticated "Scope Streamer" that manages active sets with bitsets, avoiding the massive allocation overhead of `Vec::push` and `visited` sets in the current Rust `single_residual_knn_query`.

## 2. Implementation Phases

### Phase 2: Algorithmic Parity (The Scope Streamer)
*Goal: Eliminate allocation overhead and reduce distance evaluations via heuristics.*

1.  **Dense Bitset & Buffer Reuse:**
    *   **Current:** `visited_nodes: Vec<bool>` allocated per-query (or `HashMap`). `children_nodes: Vec<usize>` allocated per-tile.
    *   **Target:** A reused `BitSet` (simple `Vec<u64>` or fixed buffer) that is reset cheaply (generation ID or fast clear).
    *   **Target:** A fixed-size stack buffer for children to avoid `Vec::push` resizing.
2.  **Budget Heuristic (The "Ladder"):**
    *   **Logic:** Port the Numba `budget` logic. If a branch yields few survivors relative to evaluations, prune it.
    *   *Details:* Track `survivors_count` vs `evals`. If ratio drops below threshold, terminate branch early.
3.  **Dynamic Block Sizing:**
    *   **Logic:** Adjust `stream_tile` size (16/32/64/128) dynamically based on the size of the active frontier. Small frontiers need small blocks to avoid latency; large frontiers benefit from large blocks (SIMD).

### Phase 3: Advanced Traversal Architecture
*Goal: Minimize pointer chasing and maximize prefetch.*

1.  **Batched Priority Queue:**
    *   Replace `BinaryHeap` pushes with blocked updates.
2.  **Node Reordering (BFS Layout):**
    *   Ensure tree nodes are stored in BFS order in memory to improve cache locality during level-wise traversal.

## 3. Execution Plan (Phase 2 Focus)

1.  **Benchmark Baseline:** Run `COMP_ENGINE=rust-natural ./benchmarks/run_residual_gold_standard.sh` to establish the clean ~6k q/s number.
2.  **Bitset/Buffer:** Refactor `single_residual_knn_query` in `src/algo.rs`.
    *   Create a thread-local `SearchContext` struct holding the `BitSet` and `NodeBuffer`.
    *   Replace `visited: Vec<bool>` with `BitSet`.
    *   Benchmark.
3.  **Budgeting:** Add the `budget_schedule` and `yield` checks to the inner loop.
    *   Benchmark.

## 4. Verification Commands

*   **Performance:**
    ```bash
    COMP_ENGINE=rust-natural ./benchmarks/run_residual_gold_standard.sh
    ```
*   **Correctness (CRITICAL):**
    ```bash
    pytest tests/test_residual_parity.py
    ```
    *Must pass after every algorithmic change to ensure index mapping remains correct.*