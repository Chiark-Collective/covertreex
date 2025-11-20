# Residual Experiment & Telemetry Plan

> **Note:** The experiments regarding "Residual Gate 1" described here have concluded. The gate proved ineffective and has been removed. The codebase now focuses on the optimized dense baseline. See `docs/RESIDUAL_GATE_POSTMORTEM.md`.

This plan translates the findings from `docs/RESIDUAL_SWEEP_BREAKDOWN.md`, `docs/residual_metric_notes.md`, and the associated source (`tools/run_residual_gold_sweep.py`, `covertreex/metrics/residual/*`, `covertreex/telemetry/logs.py`, `profiles/residual-*.yaml`) into structured experiment stories. Every change keeps the audited APIs intact (`cli.pcct`, `Runtime`, `BenchmarkLogWriter`) and treats the gold residual sweep as the reference surface.

## Context and Guardrails

- **Gold baseline.** The dense residual sweep (`dims 2–3`, `N ∈ {8k,16k,32k}`, `k ∈ {15,25,50}`) uses `python -m cli.queries --metric residual --baseline gpboost --batch-size 512 --queries 1024` with the environment from `tools/run_residual_gold_sweep.py`. The best-of-7 numbers in `docs/RESIDUAL_SWEEP_BREAKDOWN.md` plus the 64k probe are non-negotiable regression controls.
- **Guardrail first.** Before touching 32k/64k loads, run the 4k Hilbert preset through `tools/residual_guardrail_check.py` (see `docs/residual_metric_notes.md`). Use the script’s thresholds (`whitened coverage ≥0.95`, `median traversal_semisort_ms ≤1000`, `conflict_pairwise_reused=1`) and enable `--require-gate1-prunes` once the lookup path is turned back on.
- **Telemetry everywhere.** Every CLI/Runtime invocation must point `BenchmarkLogWriter` at `artifacts/benchmarks/...` and keep diagnostics on when exploring new knobs. Summaries come from `python -m cli.pcct telemetry render ...` and the CSV exporter (`tools/export_benchmark_diagnostics.py`). Never rely on stdout milliseconds; diff JSONL artefacts and `summary.json`.
- **Change discipline.** Prefer additive toggles (`profiles/*.yaml`, env vars, CLI overrides) so we can bisect behaviour. Source-level experiments stay scoped (e.g., opt-in kernel backends via `Residual` config) and keep `covertreex.api.Runtime`/CLI contract stable.

## Observability Backbone

1. **Reproduce gold sweep.** Run `python tools/run_residual_gold_sweep.py` on the current commit to regenerate JSONL logs plus `summary.json`. Store the run ID and SHA in `artifacts/benchmarks/residual_gold_sweep/README.md` (new note) for traceability.
2. **Add telemetry rollups.** For each log:
   - `python -m cli.pcct telemetry render <log> --format md --show fields > <log>.md`
   - `python -m cli.pcct telemetry render <log> --format csv > <log>.csv`
   - Collect `traversal_{pairwise,semisort,scope_chunk_*}` statistics plus gate counters into a spreadsheet so we can plot against new experiments.
3. **Guardrail automation.** Use `tools/run_residual_guardrail_suite.py` (wrapper around `tools/residual_guardrail_check.py`) so every guardrail run sets the Hilbert preset environment and stores the JSONL log, summary JSON, and metadata together under `artifacts/benchmarks/guardrails/`.
4. **Diff tooling.** Extend `tools/export_benchmark_diagnostics.py` (CLI option) to accept two log folders and emit “current vs gold” ratios for the key metrics listed in `docs/RESIDUAL_SWEEP_BREAKDOWN.md` so every experiment report references the same diff table.

## Experiment Stories

Each story below is additive; the order matters because later work assumes the telemetry harness and guardrails from earlier stories are already validated. “Runs” always mean both 4k guardrail + the gold triplet `(8k,16k,32k)` unless noted; add the 64k probe whenever an experiment looks promising to characterise scaling pressure.

### Story 0 — Baseline + Dashboard Refresh

**Hypothesis.** The current gold logs remain a faithful representation of main, and we can visualise bottlenecks batch-by-batch to catch regressions quickly.

**Steps.**
1. Launch `tools/run_residual_gold_sweep.py` with `COVERTREEX_ENABLE_DIAGNOSTICS=1` to capture richer telemetry (diagnostics overhead is acceptable during planning).
2. Feed every resulting JSONL file through `cli.pcct telemetry render` and `tools/export_benchmark_diagnostics.py` to produce Markdown/CSV rollups.
3. Pull the important per-stage splits (`traversal_ms`, `traversal_pairwise_ms`, `traversal_kernel_provider_ms`, `traversal_semisort_ms`, `conflict_graph_ms`) into a light dashboard (Jupyter notebook or `tools/plot_residual_sweep.py`). Record medians/p90s per `N`/`k`.
4. Document deltas vs. `docs/RESIDUAL_SWEEP_BREAKDOWN.md` (max ±2 % drift acceptable). If drift exceeds the threshold, file a regression ticket before running other stories.

**Telemetry focus.** End-to-end build time, per-stage ms shares, `dominated` counts, `run_hash` comparison to ensure configs match.

### Story 1 — Residual Gate Lookup Revival

**Hypothesis.** Feeding the lookup tables in `docs/data/residual_gate_profile_32768_caps.json` (and `..._scope8192.json`) into `covertreex/metrics/residual/policy.py` will prune ≥5 % of candidates, translating almost linearly into pairwise ms savings without hurting whitened coverage.

**Preparations.**
1. Inspect the gate profile JSON to confirm bin ranges match the current residual radius distribution (`ResidualGateProfile.bin_edges` in `policy.py`).
2. Enable `Residual(gate1_enabled=True, gate1_lookup=...)` via `profiles/residual-audit.yaml` or CLI flags (`--residual-gate lookup --residual-gate-lookup-path ... --residual-gate-cap ... --residual-gate-margin 0.02`).
3. Run `tools/residual_guardrail_check.py --require-gate1-prunes` on 4k to ensure `traversal_gate1_pruned > 0` and `whitened coverage` stays ≥0.95.

**Experiment matrix.**
1. Sweeps over `gate1_margin ∈ {0.01, 0.02, 0.05}` and `gate1_cap ∈ {0, 4096, 8192}` for 8k/16k/32k. Track `traversal_gate1_pruned` and false negatives (`ResidualGateProfile` telemetry: `profile.false_negative_samples`).
2. Enable `gate1_audit` (audit profile) to log pruned/kept stats; render via `pcct telemetry render --show fields` for quick inspection.
3. If coverage drops in any batch, widen `gate1_margin` or adjust `gate1_keep_pct` until `residual_batch_whitened_pair_share` matches the gold distribution (80 ± 2 % at 16k+).

**Exit criteria.** Achieve ≥7 % reduction in `traversal_kernel_provider_ms` at 16k and 32k with zero regression in `residual_batch_whitened_pair_share`. Document before/after logs and keep gate lookup disabled by default until the guardrail script reports success twice consecutively.

**Status (2025-11-16).** Ratio-aware thresholds landed in `ResidualGateLookup` so new profiles can prune relative to the query radius, but the historical lookup (`docs/data/residual_gate_profile_32768_caps.json`) and the freshly captured guardrail profile both report zeros for radii below ≈0.6, so the gate still prunes nothing by default. Guardrail runs (`guardrail_residual_lookup_ratio*.jsonl`) confirm `traversal_gate1_pruned=0` and the audit still fires as soon as we tighten the thresholds. Next step is to regenerate lookup tables from guardrail data using the logged `max_ratio` values; once we have non-zero ratios for the radii that actually occur, rerun this story before escalating to the 8k/16k/32k sweeps.

**Status (2025-11-17).** Merged the captured guardrail profiles (`artifacts/profiles/guardrail_profile_capture.json`, `gate_audit_guardrail.json`) into `docs/data/residual_gate_profile_guardrail4k_ratio.json` via `tools/ingest_residual_gate_profile.py`. The new payload keeps the 512-bin layout, carries 131 842 samples, and exposes quantile surfaces plus non-zero `max_ratio` values only where the guardrail actually produced radii (≈0.61–1.0). Despite wiring that lookup into `tools/residual_guardrail_check.py` (`guardrail_residual_lookup_ratio_v4.jsonl`, keep/prune percentiles 80/95 and `margin=0.01`), the guardrail still recorded `traversal_gate1_pruned=0`, whitened coverage collapsed to ≈0.22, and `traversal_semisort_ms` ballooned well past the 1 s threshold. Even more aggressive percentiles (95/99, 10/20, etc.) failed to produce prunes because the ratios remain empty for radii <0.6, so the lookup never constrains the low-radius guardrail batches. Parking the story here until we can capture ratios for the missing bins or adjust `ResidualGateLookup` to avoid propagating the large-radius `max_ratio` floor into the low-radius region.

**Status (2025-11-18).** Added a “hybrid” lookup (`docs/data/residual_gate_profile_guardrail4k_ratio_hybrid.json`) by merging the guardrail captures with `docs/data/residual_gate_profile_diag0.json` so bins down to ≈0.42 have non-zero ratios (`samples_total≈2.2 M`). Running the guardrail with that file plus moderate thresholds (`--residual-gate-keep-pct 80 --residual-gate-prune-pct 95`, log `guardrail_residual_lookup_hybrid_wide.jsonl`) still yielded `traversal_gate1_pruned=0` and — worse — every batch inserted all 512 candidates (`dominated=0`, `selected=512`, conflict edges=0). Tightening the thresholds (`keep/prune 10/30`, log `guardrail_residual_lookup_hybrid_loose.jsonl`) did not help; the conflict graph stayed empty and the guardrail script aborted before emitting a summary. This suggests the lookup path is collapsing dominance entirely rather than pruning, so the next debugging step is to inspect `compute_residual_distances_with_radius` / gate telemetry to understand why conflict edges disappear as soon as the lookup is enabled.

### Story 2 — Sparse Traversal + Scope Caps

**Hypothesis.** Enabling sparse traversal (`COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1`) with tuned `scope_chunk_target` and pair-merge reuse reduces tile counts per batch once the residual gate is working, particularly beyond 32k.

**Preparations.**
1. Start from `profiles/residual-fast.yaml` (sparse traversal already on) but override `scope_chunk_target` with `{4096, 8192, 12288}`.
2. Ensure `residual_scope_member_limit` and `scope_conflict_buffer_reuse` match the dense baseline so comparisons stay apples-to-apples.

**Experiment steps.**
1. Run the guardrail preset with each chunk target; confirm `traversal_scope_chunk_saturated` tracks with the intended budget and no batch hits `scope_chunk_points == 0`.
2. Launch `(8k,16k,32k,64k)` runs with the best guardrail chunk target while toggling `COVERTREEX_ENABLE_SPARSE_TRAVERSAL` between 0/1 to isolate sparse traversal’s effect.
3. Capture `traversal_scope_chunk_points`, `traversal_scope_chunk_batches`, `traversal_scope_chunk_segments`, `scope_chunk_pair_merge` metrics per log. Look for reduced `traversal_pairwise_ms` share even if traversal bookkeeping increases slightly.
4. Combine with the gate (Story 1) to see whether pruned candidates drop scope chunk saturation earlier than in the gold run.

**Exit criteria.** Show that `scope_chunk_target=8192` + gate reduces total build time by ≥10 % at 32k while keeping `conflict_graph_ms` within +5 % of the dense baseline. Produce a short write-up summarising chunk telemetry deltas.

**Status (2025-11-17).** Guardrail sweeps with sparse traversal enabled (`tools/residual_guardrail_check.py --enable-sparse-traversal`) across `scope_chunk_target ∈ {4096, 8192, 12288}` all dominated the scope budget, so `traversal_scope_chunk_points` and `traversal_scope_chunk_saturated` stayed pegged at 16 384/512 for every dominated batch. The 4k target (`guardrail_sparse_scope_target4096.jsonl`) delivered the lowest medians so far (`traversal_semisort_ms≈25 ms`, `traversal_ms≈71 ms`) versus the dense baseline (`guardrail_residual_default_postratio.jsonl`, `semisort≈29.5 ms`, `traversal≈94.8 ms`). Larger targets regressed semisort timings (33–37 ms) without unlocking additional survivors. All runs still fail the guardrail coverage requirement (whitened coverage = 0 because Gate 1 remains off), so the improvements are scoped to traversal bookkeeping only; parking Story 2 until the gate lookup is healthy enough to keep scope counts below the guardrail cap.

### Story 3 — Residual Kernel Provider / Stream Tile Tuning

**Hypothesis.** Adjusting `Residual.stream_tile`, chunk sizes, and backend caching in `covertreex/metrics/residual/host_backend.py` can shave 10–20 % off `traversal_kernel_provider_ms` once traversal counts are lower.

**Experiment steps.**
1. Instrument `host_backend.py` (add optional telemetry hooks guarded by a flag to log SGEMM time per call) so JSONL rows also capture `traversal_kernel_provider_tile_ms` and cache hit rates. Respect existing APIs by extending telemetry via `BenchmarkLogWriter` metadata.
2. Sweep `residual_stream_tile ∈ {64, 128, 256, 512}` and `residual_chunk_size ∈ {512, 1024}` across 16k and 32k (dense baseline). Record how `traversal_kernel_provider_pairs` and `residual_batch_whitened_time_share` change.
3. Prototype a “prefetch” mode where we call `kernel_provider` with wider blocks (precache) based on `scope_chunk_target`. Measure memory impact (RSS from `BenchmarkLogWriter`) and time savings.
4. If hardware allows, stub `TreeBackend.gpu` so we can re-run a subset with GPU SGEMM; capture telemetry to compare host vs GPU for the same run hash (optional stretch goal).

**Exit criteria.** Document the stream tile/chunk pair that minimises `traversal_kernel_provider_ms` without hurting memory. Highlight whether gains stack with Stories 1–2.

**Status (2025-11-17).** Guardrail probes isolated the impact of `residual_stream_tile` (`64` baseline) and `residual_chunk_size` (`512` baseline). Running `tools/residual_guardrail_check.py` with `--residual-stream-tile {128,256,512}` and `--residual-chunk-size 1024` kept the scope/chunk telemetry unchanged (still saturating because Gate 1 is off), but kernel timings shifted:

- `stream_tile=128` (`guardrail_kernel_stream128.jsonl`) lowered median `traversal_kernel_provider_ms` from **3.16 ms → 2.65 ms** (~16 % drop) and shaved `traversal_semisort_ms` to 22 ms, but total `traversal_ms` crept up (median 93 ms vs 72 ms) because conflict bookkeeping spent longer per batch.
- `stream_tile=256/512` regressed kernel time (median 3.27 ms and 4.69 ms respectively) and inflated wall time without improving pair coverage, so we should cap tiles at ≤128 for the guardrail load.
- `chunk_size=1024` (`guardrail_kernel_chunk1024.jsonl`) also helped kernel time (median 2.35 ms) but increased `traversal_ms` variance past 100 ms, likely because larger chunks amplify scope saturation when the gate is off.

Because the gate and sparse traversal stories are still blocked, we do not yet have a 16k/32k replay to prove throughput wins translate beyond the guardrail. Parking this story after noting “stream_tile=128 + chunk_size=512” as the most promising combo so far; revisit once Gate 1 and scope budgets are healthy so the kernel savings reflect in end-to-end wall time.

### Story 4 — Batch Order + Prefix Schedule Experiments

**Hypothesis.** Switching from `batch_order=natural` to Hilbert plus adaptive prefix scheduling reduces early batch variance, improving dominance onset and lowering total traversal work.

**Experiment steps.**
1. Use `python -m cli.pcct query --profile residual-audit --batch-order hilbert --prefix-schedule adaptive ...` for the guardrail and validate telemetry.
2. Record `prefix_schedule` metadata, `prefix_density_{low,high}`, and `dominant` counters per batch (available in JSONL).
3. Compare the first 8 batches’ `traversal_pairwise_ms` and survivor counts between natural/doubling vs Hilbert/adaptive for 16k and 32k.
4. Combine best prefix settings with gate + sparse traversal to see if dominance begins earlier, further shrinking candidate counts.

**Exit criteria.** Achieve ≥5 % reduction in total build time at 32k solely from ordering, or demonstrate no benefit and backstop with telemetry to explain why.

### Story 5 — Warm-Up and Runtime Health Checks

**Hypothesis.** Ensuring `warmup_scope_builder()` (`covertreex/algo/_scope_numba.py:281`) and residual append kernels (`_residual_scope_numba.py`) are warmed before the first batch prevents JIT spikes that skew telemetry and degrade reproducibility.

**Experiment steps.**
1. Add a CLI/runtime option `--warmup-kernels` that calls `warmup_scope_builder()` plus residual warmups during `Runtime.activate()`.
2. Run the guardrail twice: once cold, once with warmup, capturing `batch_index=0` telemetry (`traversal_ms`, `traversal_semisort_ms`, `residual_batch_*` shares). Expect dramatic drop in first-batch latency when warmed.
3. Apply the same warmup to the gold sweep and verify `build_total_ms` sums align more closely between reruns (lower variance).
4. Combine with Stories 1–4 to ensure the warmup path does not hide regressions (i.e., confirm run hashes differ because of config, not random warmup state).

**Exit criteria.** Document variance reduction (e.g., stdev of first-batch `traversal_ms` <5 %) and note any overhead the warmup introduces. Decide whether to bake it into profiles or keep as a guardrail-only option.

### Story 6 — Telemetry Writer Throughput

**Hypothesis.** Buffering `BenchmarkLogWriter` output or moving telemetry flushes off the hot path can reclaim ≤1 s per 64-batch run without losing visibility.

**Experiment steps.**
1. Introduce a feature flag (`--telemetry-buffer B`) that batches JSON lines in memory before writing to disk; ensure `BenchmarkLogWriter` still flushes on exit (`__enter__/__exit__` in `covertreex/telemetry/logs.py`).
2. Benchmark 8k/16k/32k with buffer sizes `{1 (default), 8, 32}` and capture `BenchmarkLogWriter`’s self-reported `write_ms` (add this metric if missing).
3. Check that log integrity remains intact (no partial lines) by validating with `pcct telemetry render`.
4. Decide whether telemetry buffering is worth the complexity; even if savings are small, we gain evidence and can fall back to synchronous writes by default.

**Exit criteria.** Demonstrate measurable (≥0.5 s) wall-clock improvement on 64 batches without losing telemetry fidelity. If improvements are negligible, document the finding so future agents do not revisit it.

## Reporting Expectations

- Every experiment bundle (stories above) should end in a short note under `docs/journal/` summarising:
  - CLI/Runtime command + env.
  - Links to JSONL/Markdown/CSV telemetry artefacts.
  - Key metric deltas vs. the refreshed gold baseline.
- Keep a living checklist at the top of this doc (update as stories complete) so reviewers can see progress at a glance.
- Never delete or overwrite artefacts; add timestamps or run IDs to new logs so the audit trail stays intact.
