# Rust Residual Query Parity Plan (vs. Gold Python/Numba) — 2025-11-24

Goal: Close the ~130× query throughput gap to the gold Numba path on the 32k / d=3 / k=50 / 1,024q workload **without changing the gold configuration**. The gold script now hard-disables the Rust backend (`COVERTREEX_ENABLE_RUST=0`) to keep the reference immutable; Rust parity work must compare against that fixed Python/Numba baseline.

The gold Numba run (via `benchmarks/run_residual_gold_standard.sh`) has these defaults **active**:
- Level cache reuse (`residual_level_cache_batching=True`).
- Dynamic block sizing (`residual_dynamic_query_block=True`) tied to active queries.
- Budget ladder thresholds up=0.015, down=0.002.
- Radius floor = 1e-3.
- Masked dedupe/bitset in the scope streamer.
- Query telemetry emitted every run.
- Caps, gate/prefilter, SGEMM fallback: **off** by default (not set in the gold script).

## Gaps to close (Rust)
1) **Level cache reuse parity**
   - Reuse parent-chain candidates across levels, not just per-level frontier collection.
   - Maintain a “valid parents” cache akin to Numba’s level cache and consult it before child expansion.

2) **Dynamic block sizing (query-aware)**
   - Tie block size to active queries and frontier size (not just child count heuristic).
   - Shrink blocks aggressively when active set is small; expand when large.

3) **Budget ladder effectiveness**
   - Ensure ladder thresholds (0.015 / 0.002) act on cached/pruned sets so survivor counts drop quickly.
   - Track yield per level; stop early when yields stay low.

4) **Pre-distance pruning & radius floor**
   - Apply radius floor (1e-3) and cached parent bounds *before* scheduling tiles.
   - Tighten lower bounds using level cache + radius to skip tile emission.

5) **Masked dedupe correctness/perf**
   - Keep masked append on by default; ensure dedupe counters and reuse of masks across tiles.

6) **Query telemetry (opt-in, default off for perf)**
   - Record: frontier sizes/expansions, lower-bound prunes, cap/radius prunes, masked dedupe hits, distance evals, yield per level.
   - Expose an opt-in flag (env/CLI) and pipe through the Rust bridge to JSONL telemetry, mirroring Numba schema where possible.

## Non-goals for parity
- Caps table derivation/loading (caps are off by default in gold).
- Gate/prefilter.
- SGEMM fallback tuning (Numba doesn’t use it in gold).

## Suggested implementation order
1) Refactor traversal to add a persistent level-parent cache and pre-distance pruning with radius floor.
2) Make block sizing depend on active queries + frontier size (configurable thresholds).
3) Strengthen budget ladder stop using yield per level (with default thresholds).
4) Wire telemetry hooks and FFI plumbing (opt-in) mirroring Numba query stats.
5) Validate on gold workload; keep telemetry off for perf unless requested.
