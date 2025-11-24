# Rust Residual Heap Parity Plan

Goal: close the gap between the Python/Numba gold traversal and the Rust default (rust-hilbert) by re‑introducing a heap-style residual traversal and running f64/f32 comparisons.

## Snapshot of outstanding differences
- Heap vs streamer: gold uses a single max-heap with bound pruning; Rust uses frontier/streamer with budgets.
- Bounds: gold uses separation invariants; Rust uses `2^(level+1)` caps + radius floor.
- Budget ladder: absent in gold, present in Rust (keeps exploring).
- Precision: gold runs float64; Rust often runs float32.
- Child order & visited: gold simple visited/enqueued; Rust reorders + masked dedupe + cached LB.
- Caps: off in gold; default cap (2.0) in Rust.

## Plan of record
1) Add an opt-in heap-style residual traversal in Rust that mirrors the previous implementation.
2) Gate it via `COVERTREEX_RUST_RESIDUAL_HEAP=1`, leave streamer as default for experiments.
3) Keep caps/budgets off for the heap path to match gold (scope chunk target still 0 for gold).
4) Run gold script with comparison (rust-hilbert + heap path) in both precisions:
   - `COVERTREEX_PRECISION=float64 COVERTREEX_RUST_RESIDUAL_HEAP=1 ./benchmarks/run_residual_gold_standard.sh ...`
   - `COVERTREEX_PRECISION=float32 COVERTREEX_RUST_RESIDUAL_HEAP=1 ./benchmarks/run_residual_gold_standard.sh ...`
5) Collect build/query numbers and update parity notes.

## Acceptance criteria
- Heap path compiles and is selectable via env flag without impacting default streamer.
- Benchmarks run cleanly in f64 and f32 with the heap path.
- Gold script remains unchanged for primary run (Numba only), comparison uses new heap path when env is set.
