# Rust Residual Query Parity Plan (vs. Gold Python/Numba) — 2025-11-24

Goal: Close the ~130× query throughput gap to the gold Numba path on the 32k / d=3 / k=50 / 1,024q workload **without changing the gold configuration**. The gold script now hard-disables the Rust backend (`COVERTREEX_ENABLE_RUST=0`) to keep the reference immutable; Rust parity work must compare against that fixed Python/Numba baseline.

Status 2025-11-24 (evening):
- ✅ Rust residual heap now computes and uses a stored `si_cache` (cover radii) during traversal, matching the Python separation bounds.
- ✅ Added `COVERTREEX_RESIDUAL_PARITY=1` toggle that disables budgets/caps, dynamic tiling, and child reordering to mirror the Python gold traversal (stream_tile=1, raw si radii, no cap ladder).
- ⏳ Remaining gaps tracked below.

Update 2025-11-24 (late):
- Rust residual parity path now records query telemetry for both f32 and f64 builds; `COVERTREEX_RUST_QUERY_TELEMETRY=1` produces the same frontier/prune/eval counters as the f32 path.
- Parity flag also forces the residual metric to bypass SIMD/tiled fast paths (or use `COVERTREEX_RESIDUAL_DISABLE_FAST_PATHS=1` explicitly).
- Added an optional traversal visited set (auto-enabled under parity or via `COVERTREEX_RESIDUAL_VISITED_SET=1`) to keep node expansions deterministic when masked dedupe is disabled.

The gold Numba run (via `benchmarks/run_residual_gold_standard.sh`) has these defaults **active**:
- Level cache reuse (`residual_level_cache_batching=True`).
- Dynamic block sizing (`residual_dynamic_query_block=True`) tied to active queries.
- Budget ladder thresholds up=0.015, down=0.002.
- Radius floor = 1e-3.
- Masked dedupe/bitset in the scope streamer.
- Query telemetry emitted every run.
- Caps, gate/prefilter, SGEMM fallback: **off** by default (not set in the gold script).

## Gaps to close (Rust)
1) **Payload / precision parity**
   - ✅ Parity flag now builds residual trees in f64 and keeps the idx↔coord map; still uses index payloads. Optional: add coord payload variant.

2) **Ordering & visited semantics**
   - ✅ Parity mode disables child reordering and masked dedupe; uses insertion order. Consider explicit visited set if duplicates ever appear.

3) **Stop rule & tiling**
   - ✅ Parity mode adds kth/frontier lower-bound stop and forces stream_tile=1.

4) **Caps/budgets fully off (done) but audit side effects**
   - Verify no cap_default leakage from metric; ensure scope caps are ignored in parity.

5) **Telemetry parity**
   - Emit/query counters matching Numba JSONL when parity mode is on (frontier sizes, prunes, eval counts, yields).

6) **Optional fast paths off**
   - Ensure SGEMM/block residual fast paths remain disabled under parity.

Suggested next steps
1) Add f64 build/residual metric path and idx→coord payload support; gate via parity flag.
2) Implement explicit child-order/visited set for parity (no sort/masking), plus kth/frontier early-stop.
3) Add telemetry shim keyed by parity mode to mirror Numba counters.
4) Re-run gold vs. parity-mode Rust on 32k/d=3/k=50 (1,024q) and record in benchmark audit.
