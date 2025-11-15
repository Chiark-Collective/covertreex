# PCCT Build Acceleration Plan — 2025-11-07

This replaces the historical implementation plan. It focuses exclusively on the
post-audit work required to bring 32 k-point PCCT builds—especially the
Numba-powered residual correlation path—down to an acceptable wall-clock without
regressing query latency or existing diagnostics.

---

## 0. Baseline Telemetry (diagnostics off unless noted)

| Workload                             | PCCT Build | Query | Notes / Artefacts |
|-------------------------------------|-----------:|------:|-------------------|
| 8 192 × 1 024 × k=16 (grid)         | 4.15 s     | 0.018 s | `bench_euclidean_grid_8192_20251108.log`, `benchmark_grid_8192_baseline_20251108.jsonl` |
| 32 768 × 1 024 × k=8 (grid)         | 16.75 s    | 0.039 s | `bench_euclidean_grid_32768_20251108*.log`, `benchmark_grid_32768_baseline_20251108{,_run2}.jsonl` |
| 32 768 × 1 024 × k=8 (Euclid. legacy) | 44.22 s  | 0.284 s | `benchmark_euclidean_clamped_20251107_fix*.jsonl` |
| 32 768 × 1 024 × k=8 (Residual)     | 66.25 s    | 0.305 s | `benchmark_residual_32768_default.jsonl`, `run_residual_32768_default.txt` |
| Residual “gold standard” (dense)    | 71.75 s    | 0.272 s | `benchmarks/run_residual_gold_standard.sh`, `bench_residual.log` |

Targets after this sprint:

1. **Euclidean build ≤ 20 s** (steady state, diagnostics off).
2. **Residual build ≤ 40 s** with the clamped pipeline, ≤30 s aspirational with chunking enabled.
3. Maintain query throughput (≥3.5 k q/s) and regression parity on Tier A/B tests.

---

## 1. Grid-Leader Conflict Builder (Euclidean) — **in progress**

**Objective:** Replace dense CSR + full MIS on dominated Euclidean levels with a hash-grid leader election + micro-MIS to cut adjacency+MIS cost by ≥5×.

**Deliverables**
- `build_conflict_graph(..., impl="grid")` path with:
  - Shifted integer grid binning (cell width `~2^ℓ`, ≥3 deterministic shifts).
  - Priority rule (64-bit mix of point id, level, shift) and per-cell leader pick.
  - Optional neighbor-cell micro-MIS using existing Luby kernel on the tiny induced graph.
  - Telemetry counters: `grid_cells`, `leaders_raw`, `leaders_after_mis`, `grid_local_edges`.
- CLI/runtime flag (`COVERTREEX_CONFLICT_GRAPH_IMPL=grid` + `cli.queries --conflict-impl grid`).
- Tier-B integration coverage validating per-level separation and maximality vs brute force for both dense and sparse batches.

**Breakdown / Status**
1. Prototype binning + priority selection in pure NumPy; validate on recorded batch dumps. ✅ *Implemented via `_grid_select_leaders`, exercised in `tests/test_conflict_graph.py::test_grid_conflict_builder_forces_leaders`.*
2. Port to Numba helper (optional) once semantics freeze; ensure determinism given `seed`. ✅ *`covertreex/algo/_grid_numba.py` mirrors `_grid_select_leaders`, and `build_grid_adjacency` now defaults to the Numba fast path whenever `COVERTREEX_ENABLE_NUMBA=1`.*
3. Implement neighbor adjacency generator (3×3 Moore per cell) feeding existing MIS. 🔄 *Current build uses greedy priority filtering among grid leaders; evaluate whether we still need a micro-MIS step after profiling leader counts.*
4. Wire telemetry + JSONL logging; add regression test fixtures. ✅ *`grid_*` counters now flow through conflict timings, batch logger, and diagnostics.*
5. Benchmark 8 192 and 32 768 Euclidean workloads vs dense builder; collect scatter timings. ✅ *`benchmark_grid_32768_{natural,hilbert}.jsonl` capture the steady-state grid builder (scatter **62 ms → 0.63 ms**, `conflict_graph_ms` **83.7 ms → 22.6 ms**) for the two batch-order strategies.*

**Risks & Mitigations**
- *Boundary artifacts*: use multiple shifts + local MIS to restore maximality.
- *Non-Euclidean metrics*: keep feature flagged; fall back automatically outside Euclidean.

---

## 2. Batch Shaping & Adaptive Prefix Growth — **phase 1 completed, phase 2 pending**

**Objective:** Reduce dominated batch density before traversal by ordering inserts and throttling prefix growth based on live domination ratio.

**Components & Status**
1. **Hilbert/Morton batch ordering** – ✅ landed:
   - Config/plumbing merged (`COVERTREEX_BATCH_ORDER{,_SEED}`); `plan_batch_insert` now applies Hilbert or random permutations via the new helper; telemetry surfaces strategy + Hilbert spread in logs.
   - **Update 2025‑11‑07:** Hilbert is now the default order (`_DEFAULT_BATCH_ORDER_STRATEGY`) after the `benchmark_grid_32768_{natural,hilbert}.jsonl` sweep demonstrated a 100× reduction in first-batch scatter (3.95 s → 8.8 ms).
2. **Adaptive prefix scheduler** – 🔄 partially landed:
   - Runtime knobs + adaptive growth logic implemented inside `batch_insert_prefix_doubling`; per-group domination ratios/prefix factors logged.
   - **Update 2025‑11‑07:** Tuned defaults (`prefix_density_low/high=0.15/0.55`, `prefix_growth_small/mid/large=1.25/1.75/2.25`) keep domination ≈1.0 while halving the number of adaptive-prefix batches (see `notes/prefix_schedule_adaptive_experiment.json`). Remaining task: wire the adaptive path into the public benchmark harness so the telemetry lands in the canonical JSONL artefacts.

**Immediate Next Steps**
1. Extend `benchmarks/queries.py` with a toggle that builds via `batch_insert_prefix_doubling` (capturing `prefix_factor` + domination telemetry in the same JSONL stream as the Hilbert/grid runs). ✅ *`--build-mode prefix` now routes builds through `batch_insert_prefix_doubling`; `benchmark_grid_32768_prefix.jsonl` contains the per-group `prefix_factor`/domination telemetry.*
2. Replay the 32 k Euclidean + residual suites under the new defaults (Hilbert + adaptive prefix + grid) and refresh docs/AUDIT with the before/after scatter tables. 🔄 *Euclidean rerun complete (`build=37.7 s`); residual standard run updated to 66.3 s, but the full prefix-mode residual log still times out (>20 min) under the current clamp, so we need chunk-merge tweaks before publishing that artefact.*
3. Document rollout guidance for tiered workloads (when to fall back to natural order / doubling) once the benchmark refresh lands. 🔄 *Docs updated with the new defaults + CLI flag; finalize guidance after the residual prefix artefact lands.*

---

## 3. Residual Two-Gate & Kernel Load Shedding — **in progress**

**Objective:** Halve `_distance_chunk` invocations by inserting a float32 whitening gate ahead of the expensive residual correlation kernel while preserving exact distances.

**Deliverables**
- `ResidualCorrHostData` additions: pre-whitened `v32`, `norm32`, gating parameters (α, ε).
- `residual_gate1_whitened` Numba kernel returning a conservative keep mask using L2 bounds.
- Updated `compute_residual_distances_with_radius` to short-circuit when gate prunes all candidates and to report telemetry counters (`residual_gate1_kept`, `_pruned_ratio`).
- Tests covering:
  - Gate monotonicity (never prunes a true neighbor) against brute-force Python baseline.
  - Numeric stability for large/small radii and pathological covariance inputs.

**Breakdown**
1. Derive safe bound mapping between Euclidean distance in whitened space and residual radius (document assumptions).
2. Implement gating kernel + integration; add feature flag `COVERTREEX_RESIDUAL_GATE1=1` for rollout.
3. Replay residual benchmark logs (both clamped and gold standard) capturing pruned ratios.
4. Profile `_distance_chunk` call counts pre/post to confirm ≥50 % reduction.

**Update 2025‑11‑07:** `ResidualCorrHostData` now stages float32-whitened caches and runtime-configurable `gate1_*` parameters, `compute_residual_distances_with_radius` runs the gate-1 mask ahead of `_distance_chunk`, and traversal telemetry/JSONL logs record `traversal_gate1_{candidates,kept,pruned}` counters. Residual traversal now tracks the observed residual radius per query, `build_conflict_graph` consumes the new ladder (bounded by `COVERTREEX_RESIDUAL_RADIUS_FLOOR`), and `batch_insert` clamps `si_cache` before writing so parent chains never inherit `∞`. The gate remains disabled by default until we replay the 32 k residual suite with the feature enabled.

**Update 2025‑11‑08:** The follow-up `p2k` sweeps (diag0 harness, audit on) are done: every α in [0.75, 10] still throws `_audit_gate1_pruned` as soon as the first dominated batch streams scopes, even with the new radii ladder. The only configuration that completes the eight batches is the intentionally loose `benchmark_residual_gate_p2k_alpha100_diag0{,_run2}.jsonl`, and those logs show `traversal_gate1_pruned=0`, i.e. the gate kept every one of the 1.8 M candidates. No safe-but-useful threshold emerged, so Gate‑1 stays off and we stop iterating until a better residual bound (or per-scope histogram) is derived.

**Calib status 2025‑11‑07 (evening):** Short sweeps (`benchmark_residual_gate_p2k_alpha{2_0,1_5}.jsonl`) show that with the current radii source every dominated scope inherits `radius≈2` and the audit tripwire fires even for conservative thresholds (α≤2.0, margin 0.05, cap 2.5). Forcing the gate on a 32 k clamped run (`benchmark_residual_clamped_gate_20251107.jsonl`) pruned ~99.9 % of candidates, confirming we would be throwing away valid neighbors. Root causes:

- Newly inserted nodes still carry `si_cache=∞` because the conflict builder previously defaulted parentless nodes to `inf` radii; the follow-up patch now falls back to the geometric base so future inserts at least inherit a finite value.
- Even with that fix, the residual traversal currently assigns every parent `level=0`, so the effective radius never drops below ~2; we need either a proper residual level ladder or a metric-specific bound before gate-1 can make meaningful decisions.

**Next steps:** use the new `traversal_scope_chunk_{scans,points,dedupe,saturated}` telemetry to quantify the 16 384-cap hit, rerun the short α/margin sweeps with the diag0 logs (so gate counters + scope stats are captured), and update the audit before re-enabling the gate. Until then keep `COVERTREEX_RESIDUAL_GATE1=0` in long-form benchmarks.

**Risks**
- Over-aggressive pruning: mitigate with unit tests + optional “audit mode” that double-checks gate decisions.
- Memory overhead from cached float32 buffers: reuse existing host data arrays and lazily allocate on first residual config.

---

## 4. Chunk & Degree-Aware Heuristics — **not started**

**Objective:** Keep conflict shard counts and scatter time bounded even when scope chunking is enabled.

**Streams of work**
1. **Degree-aware shard merging**
   - Extend `_chunk_ranges_from_indptr` to use estimated candidate pair counts when merging tails.
   - Emit telemetry: `chunk_pairs_before`, `chunk_pairs_after`, `chunk_merge_iterations`.
2. **Degree cap in `_expand_pairs_directed`**
   - Optional cap `COVERTREEX_DEGREE_CAP` (default off) limiting each node to top-K annulus neighbors before CSR emission.
   - Record average degree before/after cap to ensure MIS maximality remains achievable.
3. **Buffer reuse**
   - Introduce scratch “arena” for `sources/targets/indices` sized by moving max; wipe with lightweight fill kernels.
   - Track `arena_bytes` and reuse hit rate in diagnostics.

**Breakdown**
1. Prototype pair-count estimator (using per-scope membership histograms) and integrate with chunk builder.
2. Add regression tests ensuring shard merging still respects `scope_chunk_target` and `scope_chunk_max_segments`.
3. Implement degree cap with deterministic tie-breaking; validate on Tier-A conflict graph tests.
4. Add buffer pool object; ensure thread-safety within batch insert orchestrator.

**Expected Gains**
- First dominated batch scatter drops from 300+ ms to <100 ms under chunking.
- Smoother RSS usage due to arena reuse (reduces allocator churn noted in logs).

---

## 5. Instrumentation & Benchmark Refresh — **ongoing**

**Objective:** Provide authoritative evidence for auditors once optimisations land.

**Tasks / Status**
1. Extend JSONL writer with new counters (`grid_*`, `gate1_*`, `prefix_factor`, `arena_bytes`). *(partially done — `batch_order_*` + domination ratios already flowing; remaining counters pending their respective features.)*
2. Automate benchmark suite: *(baseline harness exists — `tools/run_reference_benchmarks.py` runs guardrail + 2 048/8 192/32 768 presets and emits CSV + manifest output; wire it into CI and keep presets synced with the latest gold config.)*
   - 2 048 quick check (diagnostics on/off).
   - 8 192 Euclidean/residual.
   - 32 768 Euclidean/residual (clamped + chunked variants).
   - GPBoost baseline comparison (`--baseline gpboost`) post-optimisation.
3. Update `docs/CORE_IMPLEMENTATIONS.md` tables + narrative.
4. Add `AUDIT.md` “response” section summarising which recommendations shipped and their impact.

**Next Benchmarks**
- Once grid builder + adaptive prefix metrics stabilize, rerun 8 192 + 32 768 suites (Euclidean/residual) to quantify improvements and refresh `docs/CORE_IMPLEMENTATIONS.md`.

---

## Execution Order (recommended)

1. **Grid-Leader builder** — unlocks the largest Euclidean win immediately.
2. **Batch shaping & adaptive prefix** — complements #1 and reduces chunk pressure globally (CLI overrides + docs now partially done; awaiting tuned defaults + scope stats).
3. **Residual two-gate** — directly tackles residual build cost.
   - *2025‑11‑08 update:* gate-1 now consumes a lookup table instead of hand-tuned α values. `tools/build_residual_gate_profile.py` writes `docs/data/residual_gate_profile_diag0.json` (2,096,128 samples, 512 bins). Opt into the gate with:

     ```bash
     COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 \
     COVERTREEX_RESIDUAL_GATE1=1 \
     COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH=docs/data/residual_gate_profile_diag0.json \
     COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN=0.02 \
     # optional: COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP=2.0
     ```

     Audit mode stays on while we finish validating the sparse traversal rollout; the feature remains behind flags until we prove the lookup holds for the 32 k chunked builds.
   - **Next steps:**
     1. Re-run the 32 k residual suites (clamped + chunked) with the lookup flags enabled to capture fresh `traversal_gate1_*` counters and wall-clock deltas versus the current baseline.
     2. Add a `--residual-gate lookup` preset (or similar) to `benchmarks/queries` so CI and manual runs can enable the gate without copy/pasting environment variables.
     3. Regenerate the lookup using real 32 k artefacts (or production datasets) alongside the diag0 profile, compare the thresholds, and only then consider enabling the gate by default.
4. **Chunk/degree heuristics** — stabilises chunked runs and prepares for larger datasets.
5. **Instrumentation refresh** — finalize evidence + documentation (grid + batch-order sections added; residual/GPBoost refresh still pending).

Each stage should conclude with targeted tests + benchmark runs before proceeding. Maintain feature flags for rollout, and keep dense builder + legacy batch order available for A/B comparisons until parity is proven.
