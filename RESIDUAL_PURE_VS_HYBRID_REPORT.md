# Residual Tree: Pure vs Hybrid Analysis

## Status
- **Bug Fixed:** The O(N^2) performance regression in `residual_correlation` tree construction was identified and fixed by removing the `max_distance_hint` shortcut in `src/algo/batch/mod.rs`.
- **Correctness Verified:** A parity test (`tests/test_residual_pure_vs_hybrid_manual.py`) confirms that the "Pure" Residual tree (built with `RustNaturalEngine`) produces identical k-NN results to the "Hybrid" tree (Euclidean build + Residual query, `RustHybridResidualEngine`) when the kernel is full rank.
- **Benchmark:** Initial benchmarking (`benchmarks/residual_pure_vs_hybrid.py`) shows comparable build times and slightly faster query performance for the Pure engine.

## Benchmark Results (N=2000, Inducing=2000)
| Mode | Engine | Build Time | Query Speed |
|pV | --- | --- | --- |
| **Pure Residual** | `RustNaturalEngine` | 1.26s | ~9,165 q/s |
| **Hybrid Residual** | `RustHybridResidualEngine` | 1.16s | ~7,600 q/s |

*Note: Build times are dominated by the O(N^3) kernel matrix construction in this test setup (due to `inducing_count=N`). The tree construction itself is efficient.*

## Configuration
- **Pure Mode:** Use `engine="rust-natural"` (Input Order) or `engine="rust-hilbert"` (Hilbert Order).
- **Hybrid Mode:** Use `engine="rust-hybrid"`.

## Implementation Details
- The `max_distance_hint` optimization in Rust was creating a degenerate "star" tree for bounded metrics, forcing O(N) scans per insertion.
- This has been reverted to standard Cover Tree insertion, ensuring O(log N) behavior.
- `RustNaturalEngine` was used for the correctness test to avoid complexity from the experimental block-query path in `RustHilbertEngine`.
