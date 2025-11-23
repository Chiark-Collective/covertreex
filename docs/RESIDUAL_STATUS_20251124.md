# Residual Status Report: 2025-11-24

## Rust Fast Path Optimization

Implemented dense scope streamer, budget ladder, radius floor, and scope cap loading in the Rust backend (`rust-fast` engine).

### Changes
- **Algo:** Refactored `single_residual_knn_query` in `src/algo.rs` to use a tiled child loop (`stream_tile=64`), budget ladder (32/64/96), and radius clamping.
- **Caps:** Added `load_scope_caps` in `src/lib.rs` to load JSON scope caps from Python module via `COVERTREEX_RESIDUAL_SCOPE_CAP_PATH` env var.
- **Build:** `maturin develop --release`.

### Benchmark Results (Gold Standard)
- **Dataset:** 32,768 points, d=3, k=50, 1024 queries.
- **Engine:** `rust-fast` (Residual-only Rust implementation).
- **Config:**
  - Caps: `docs/data/residual_scope_caps_32768.json`
  - Budget: 32, 64, 96 (Up 0.6, Down 0.01)
  - Stream Tile: 64 (Default)
- **Result:**
  - **Build:** 7.42s
  - **Query Throughput:** 7,105 q/s (0.14s total)
  - **Latency:** 0.14ms

### Comparison
- **Previous Rust:** ~37 q/s (200x improvement).
- **Python-Numba Gold:** ~22,600 q/s.
- **Gap:** Rust is now within 3x of Numba performance. The remaining gap is likely due to:
  - Lack of "level cache" (Numba reuses valid scope members from parent level).
  - Dynamic block sizing (Numba adjusts query block size based on active queries).
  - Kernel fusion / memory layout differences (Numba kernels are JIT-compiled specifically for the metric).

### Next Steps
- Implement level cache in Rust (requires more complex memory management for the cache).
- Port conflict graph reuse to `batch_insert` for faster build times.
- Profile `rust-fast` to identify hot spots (likely distance computation or heap management).
