# Backlog

_Last updated: 2025-11-13._

This list is reprioritised to focus on the configurations that already deliver ~30 s 32 k builds. Items are grouped by the impact they can have on the current winning recipe; jobs that are less urgent (but still valuable) appear in the lower sections.

## A. Critical to Improving the 32 k / 30 s Baseline

### Dense residual regression bisect
- **Goal:** Bisect the regression that pushed dense residual builds from ~83 s to >2 h between commits `48334de` and `5030894`, then retag the canonical baseline so future work compares to the healthy build.
- **Status:** Not started; Phase 5 optimisations remain blocked.  
- **Refs:** `docs/journal/2025-11-12_residual_dense_regression.md`.

### Scope streamer & budget fixes for dense runs
- **Goal:** Reintroduce a dense scope streamer that scans each 512-point chunk once, honours ≤64 survivors/query, and restores `traversal_semisort_ms ≤ 50 ms`—the biggest lever left on the now-30 s pipeline.
- **Status:** Design sketched in the regression journal (“Next-Step Optimization Plan”), implementation pending.

### Chunk & degree-aware heuristics
- **Goal:** Keep conflict shard counts bounded under `scope_chunk_target` by merging shards based on pair counts, capping degrees, and reusing arenas; directly reduces scatter/queue time in the 30 s build.
- **Status:** Not started; see `PARALLEL_COMPRESSED_PLAN.md §4`.

### Residual guardrails before 32 k
- **Goal:** Enforce the 4 k Hilbert criteria (≥0.95 whitened coverage, `<1 s` semisort, non-zero gate prunes) before any 32 k reruns so we don’t regress the 30 s preset.
- **Status:** Criteria documented but not automated.  
- **Refs:** `docs/journal/2025-11.md` (“Phase‑5 Shakedown Guardrails”).

### Instrumentation & benchmark refresh
- **Goal:** Capture `grid_*`, `gate1_*`, `prefix_factor`, `arena_bytes`, and automate the 2 048/8 192/32 768 suites so every tweak to the 30 s config is measurable; refresh `docs/CORE_IMPLEMENTATIONS.md` + add an `AUDIT.md` response section.
- **Status:** Partially done (batch-order telemetry exists); rest outstanding.  
- **Refs:** `PARALLEL_COMPRESSED_PLAN.md §5`.

### Grid/prefix benchmark refresh & rollout guidance
- **Goal:** Publish the before/after scatter tables for Hilbert + adaptive prefix + grid at 32 k (Euclidean + residual) and document when to fall back to legacy orders; keeps the winning recipe reproducible.
- **Status:** Euclidean rerun done; residual prefix run still timing out without the chunk tweaks above.

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
- **Goal:** Decide whether the grid builder still needs the micro-MIS now that greedy priority filtering delivers the 30 s build; avoids unnecessary work in conflict construction.
- **Status:** Pending (step 3 in `PARALLEL_COMPRESSED_PLAN.md §1`).

### JSONL schema cleanup & benchmark automation
- **Goal:** Finalise telemetry naming and script the canonical benchmark runs so improvements to the 30 s config can be validated automatically.
- **Status:** Partially complete; automation not yet wired into CI.  
- **Refs:** `PARALLEL_COMPRESSED_PLAN.md §5`, `docs/residual_metric_notes.md`.

## C. Nice-to-Haves / Deferred

### Neighbor cache plumbing & async refresh worker
- **Goal:** Cache neighbor graphs and refresh them asynchronously to keep LBFGS epochs from blocking, improving overall throughput once tree builds are fast.
- **Status:** Design captured in `notes/cover_tree_async_extension_2025-10-30.md`; implementation + config toggles pending.

### Documentation deltas
- **Goal:** After the benchmark refresh, update `docs/CORE_IMPLEMENTATIONS.md` and summarise outcomes in the archived audit doc.
- **Status:** Blocked on the refreshed artefacts.
