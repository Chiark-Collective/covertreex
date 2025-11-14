# Backlog

_Last updated: 2025-11-14._

This list is reprioritised to focus on the configurations that already deliver ≈20 s 32 k builds (dense streamer + Hilbert batches). Items are grouped by the impact they can have on the current winning recipe; jobs that are less urgent (but still valuable) appear in the lower sections.

## A. Guarding the 32 k / <20 s Baseline

### Chunk & degree-aware heuristics
- **Goal:** Keep conflict shard counts bounded under `scope_chunk_target` by merging shards based on pair counts, capping degrees, and reusing arenas; directly reduces scatter/queue time in the <20 s build.
- **Why high leverage:** The current fastest runs still spike when chunked scopes or high-degree nodes appear; these heuristics target exactly those residual hotspots so the <20 s recipe holds across datasets.
- **Status:** Not started; see `PARALLEL_COMPRESSED_PLAN.md §4`.

### Dense residual regression verification
- **Goal:** Now that the dense residual preset hits ≈21 s build / 0.026 s query again, confirm which commits fixed the >2 h regression and tag that SHA so future bisects have a known good point (or repeat the bisect if the fix was accidental).
- **Why high leverage:** Vecchia pipelines rely on the dense residual path; without a clearly documented “good” SHA we can’t prove future slowdowns are new regressions.
- **Status:** `pcct-20251114-105500-6dc4f6` (`git HEAD`, log `artifacts/benchmarks/artifacts/benchmarks/residual_dense_32768_dense_streamer_gold.jsonl`) is the latest ≤22 s run (total traversal ≈18.8 s; dominated `traversal_semisort_ms` median ≈62 ms). Tag this commit once merged so future bisects can anchor on the dense-scope-streamer baseline.
- **Refs:** `docs/journal/2025-11-12_residual_dense_regression.md` (latest CLI snapshot).

### Residual guardrails before 32 k
- **Goal:** Enforce the 4 k Hilbert criteria (≥0.95 whitened coverage, `<1 s` semisort, non-zero gate prunes) before any 32 k reruns so we don’t regress the <20 s preset.
- **Why high leverage:** A single bad run can silently degrade the fast config; automated guardrails ensure every 32 k benchmark still represents the tuned recipe before we publish new numbers.
- **Status:** Criteria documented but not automated.  
- **Refs:** `docs/journal/2025-11.md` (“Phase‑5 Shakedown Guardrails”).

### Instrumentation & benchmark refresh
- **Goal:** Capture `grid_*`, `gate1_*`, `prefix_factor`, `arena_bytes`, and automate the 2 048/8 192/32 768 suites so every tweak to the <20 s config is measurable; refresh `docs/CORE_IMPLEMENTATIONS.md` + add an `AUDIT.md` response section.
- **Why high leverage:** Without complete telemetry and scripted benchmarks, we can’t prove whether a change helps or hurts the winning build; finishing this work locks in observability for every future tweak.
- **Status:** Partially done: batch-order telemetry exists and `tools/residual_scaling_sweep.py` now automates 4 k→64 k runs, but the full telemetry capture + CI wiring remain outstanding.  
- **Refs:** `PARALLEL_COMPRESSED_PLAN.md §5`.

### Grid/prefix benchmark refresh & rollout guidance
- **Goal:** Publish the before/after scatter tables for Hilbert + adaptive prefix + grid at 32 k (Euclidean + residual) and document when to fall back to legacy orders; keeps the winning recipe reproducible.
- **Why high leverage:** The sub-20 s result hinges on this combo; without public artefacts and guidance, the “winning formula” is tribal knowledge and hard to reproduce or compare against.
- **Status:** Euclidean rerun done; residual prefix run still timing out without the chunk tweaks above.

### Completed (A-tier)

#### Scope streamer & budget fixes for dense runs
- **Goal:** Reintroduce a dense scope streamer that scans each 512-point chunk once, honours ≤64 survivors/query, and restores `traversal_semisort_ms ≤ 50 ms`.
- **Outcome:** ✅ `--residual-dense-scope-streamer`, scope bitsets, and dynamic query blocks are now default-on for residual runs. Gold artefact `artifacts/benchmarks/artifacts/benchmarks/residual_dense_32768_dense_streamer_bitset.jsonl` (`pcct-20251114-141549-e500d8`) delivers **≈20.6 s wall / 16.9 s traversal**, and the 4 k guardrail log `…/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl` confirms `<1 s` semisort before 32 k reruns. Disable via CLI/env toggles only for legacy comparisons.

#### Scope streamer hot path → Numba / JIT
- **Goal:** Move the remaining Python loops in `_collect_residual_scopes_streaming_parallel` onto the existing Numba helpers.
- **Outcome:** ✅ Dense scope inserts now route through `residual_scope_append[_masked][_bitset]` by default, so cache-prefetch hits and masked tiles stay inside compiled code. Benchmark `pcct-20251114-141549-e500d8` records `traversal_semisort_ms≈47 ms` (p90 ≈66 ms) with dominated traversal ≈16.9 s, proving the gain.

#### Scope-budget + tile math JIT
- **Goal:** JIT the pure-Python helpers `_compute_dynamic_tile_stride` and `_update_scope_budget_state` (plus the tiny budget arrays they drive) so query-block scheduling scales with smaller tiles.
- **Outcome:** ✅ Both helpers now dispatch into `_residual_scope_numba.py` (`residual_scope_dynamic_tile_stride` / `residual_scope_update_budget_state`) whenever Numba is enabled, with delegation enforced in `tests/test_residual_parents.py::test_*delegates_to_numba`. Dense batches therefore avoid repeated Python loops when tiles shrink below survivor budgets.

#### Level-cache batching & parent chain Numba port
- **Goal:** Batch the level-scope prefetch and parent/chain append paths so cached Hilbert windows, parent guarantees, and next-chain inserts use Numba helpers instead of per-query Python loops.
- **Outcome:** ✅ `_process_level_cache_hits` now batches cached Hilbert windows per level (see `strategies/residual.py:108-220`, `820-905`), `residual_collect_next_chain` replaces the Python loop in both serial and parallel collectors, and the preset ships enabled by default via `--residual-level-cache-batching`. Fresh artefacts (`pcct-20251114-162220-9efaf0` for 32 k, `pcct-20251114-162122-761439` for 4 k) capture the new gold baseline.

## B. Next-Level Optimisations (after A is healthy)

### Gate‑1 rescue plan
- **Goal:** Build a safer lookup from production artefacts, fix the residual level/radius ladders, and re-run audits until `traversal_gate1_pruned>0` without false negatives so gate-on can beat the current dense baseline.
- **Status:** Lookup ingestion exists, but audits still fail immediately and sparse traversal costs >30 s/batch.  
- **Refs:** `docs/journal/2025-11.md` (“Gate‑1 Lookup Experiments”).

### Residual gate reruns + CLI preset
- **Goal:** Once the lookup is trustworthy, ship a CLI preset (`--residual-gate lookup`) and re-run the 32 k residual suites (clamped + chunked) with gate flags enabled.
- **Status:** Blocked on the previous item.  
- **Refs:** `PARALLEL_COMPRESSED_PLAN.md §3`.

### Grid builder: evaluate neighbor-cell micro-MIS
- **Goal:** Decide whether the grid builder still needs the micro-MIS now that greedy priority filtering delivers the <20 s build; avoids unnecessary work in conflict construction.
- **Status:** Pending (step 3 in `PARALLEL_COMPRESSED_PLAN.md §1`).

### JSONL schema cleanup & benchmark automation
- **Goal:** Finalise telemetry naming and script the canonical benchmark runs so improvements to the <20 s config can be validated automatically.
- **Status:** Partially complete; automation not yet wired into CI.  
- **Refs:** `PARALLEL_COMPRESSED_PLAN.md §5`, `docs/residual_metric_notes.md`.

## C. Nice-to-Haves / Deferred

### Neighbor cache plumbing & async refresh worker
- **Goal:** Cache neighbor graphs and refresh them asynchronously to keep LBFGS epochs from blocking, improving overall throughput once tree builds are fast.
- **Status:** Design captured in `notes/cover_tree_async_extension_2025-10-30.md`; implementation + config toggles pending.

### Documentation deltas
- **Goal:** After the benchmark refresh, update `docs/CORE_IMPLEMENTATIONS.md` and summarise outcomes in the archived audit doc.
- **Status:** Blocked on the refreshed artefacts.
