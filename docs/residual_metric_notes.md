# Residual-Correlation Metric Integration (2025-11)

_Status snapshot (2025-11-17)._ Dense Hilbert 32 k residual builds now complete in **≈17.8 s** wall time using the pair-count shard-merging + buffer-reuse defaults (`pcct-20251114-214845-7df1be`, log `artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`, `median traversal_semisort_ms≈36 ms`, p90 ≈59 ms). The previous level-cache log (`pcct-20251114-215010-fa37f4`, `artifacts/benchmarks/residual_dense_32768_gold_best5_run4.jsonl`) still serves as the regression control (best run 18.65 s). The 4 k guardrail (`pcct-20251114-105559-b7f965`, log `artifacts/benchmarks/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl`) stays under the `<1 s` requirement (`median traversal_semisort_ms≈41.7 ms`) but still records `whitened_block_pairs_sum=0` / `traversal_gate1_pruned=0`, so rerun once the lookup + whitening path is re-enabled. Use `--no-residual-*`, `--no-scope-chunk-pair-merge`, or `--no-scope-conflict-buffer-reuse` only when you need to reproduce historical telemetry.

> Historical log entries that used to live in this file are now indexed under [`docs/journal/2025-11.md`](journal/2025-11.md). Keep the high-level guidance here current and push dated telemetry into the journal.

**Telemetry playbook (Phase 5).**
- Always start with a 4 096-point Hilbert preset (`--tree-points 4096`, otherwise matching the 32 k CLI flags) and capture both the JSONL log and CSV summary under `artifacts/benchmarks/`. Verify `whitened_block_pairs_sum / kernel_provider_pairs_sum ≥ 0.95`, `traversal_semisort_ms` < 1 s median, and `conflict_pairwise_reused=1` for every batch before escalating.
- Only after the 4 k run looks healthy should we launch the 32 k replay. Use the same environment (thread caps, gating knobs, chunk target 8 192) so we can diff telemetry directly. If the 32 k run still exceeds 1 h wall-clock or `traversal_semisort_ms` fails to collapse, halt and investigate before attempting further gate experiments.
- Dump rich telemetry every time: CLI already streams per-plan stats, but we additionally persist JSONL + CSV via `--log-file` / `tools/export_benchmark_diagnostics.py` so auditors can diff runs without rerunning the workload.
- Conflict telemetry now tracks the degree-cap heuristics and scratch arena footprint: JSONL batches surface `conflict_degree_cap`, `conflict_degree_pruned_pairs`, and `conflict_arena_bytes` alongside the existing chunk counters so we can prove when `--degree-cap` / `COVERTREEX_DEGREE_CAP` kicks in. Leave the cap at `0` unless you explicitly need to bound scope fanout; flag any non-zero `degree_pruned_pairs` in the run notes so we can correlate antichains with throughput.

**Automation helper.** `tools/residual_guardrail_check.py` now wraps the guardrail preset, runs the 4 k benchmark (or reuses an existing log via `--skip-run`), and enforces the thresholds above (`coverage ≥ 0.95`, `median traversal_semisort_ms ≤ 1000`, `conflict_pairwise_reused=1`). Pass `--require-gate1-prunes` once the lookup path is healthy; use `--extra-cli-args -- …` to forward feature toggles that mirror the current gold preset. The script exits non-zero when any gate fails, so CI jobs or local workflows can block 32 k reruns automatically.

**Reference benchmark harness.** Run `python tools/run_reference_benchmarks.py` to capture the guardrail plus 2 048/8 192/32 768 suites (Euclidean + residual) in one go. Logs land under `artifacts/benchmarks/reference/<timestamp>/`, CSV summaries are emitted via `tools/export_benchmark_diagnostics.py`, and the manifest `manifest.json` records every job + CLI/env snapshot. Use `--jobs` to rerun a subset or `--list-jobs` to inspect presets; CI can point the script at a fixed output directory and enable `--skip-existing` to avoid duplicate runs.

Gate-on reruns are still in-progress: `traversal_gate1_candidates≈2.33×10^8` yet `traversal_gate1_pruned=0`, so cache/prefilter work remains before we can claim build-time wins.

**Historical regression snapshot (2025-11-12).**
- `artifacts/benchmarks/residual_phase05_hilbert_4k.jsonl` — eight dominated batches, **114 s** build, **median traversal_ms=13.3 s**, **median semisort_ms=7.53 s**, whitened/kernels = **0.875×**, gate candidates **7.34 M**, gate pruned **0**. Scope telemetry shows 3 584 saturated queries despite the dataset being half the cap, proving that we are still scanning virtually every tree point per batch.
- `artifacts/benchmarks/residual_phase05_hilbert.jsonl` — 123 dominated batches, **median traversal_ms=59.4 s**, **median semisort_ms=28.0 s**, **total traversal 7 277 s (~2 h)**. Coverage (0.979×) only looks healthy because the streaming path dwarfs the per-batch 512×512 pairwise matrix; the gate again prunes nothing. This is strictly worse than the November 10 dense run and should be treated as a failure case, not a baseline.
- Historical action item (2025-11-12): do not run 32 k again until the 4 k shakedown hits ≥0.95 coverage, `<1 s` semisort, and a non-zero gate prune count. Archive every JSONL/CSV pair alongside a short note explaining whether the run passed the shakedown; delete or quarantine anything that looks like the runs above.

This note summarizes the current host-side implementation for the Vecchia residual-correlation metric inside `covertreex`.

## Host Caches & Configuration

We introduced `ResidualCorrHostData` in `covertreex/metrics/residual.py`. It packages the host-resident artefacts supplied by the VIF pipeline:

- `v_matrix` — the low-rank factors \( V = L_{mm}^{-1} K(X, U) \) for all training points. As of 2025‑11‑12 the host cache stores this (and every downstream traversal buffer) in float32; audits and Gate‑1 can request a float64 view via `backend.v_matrix_view(np.float64)` which lazily materialises / memoises the wider copy when the policy demands it.
- `p_diag` — per-point residual diagonals \( p_i = \max(K_{x_ix_i} - \|V_i\|^2, 10^{-9}) \). `p_diag`, `kernel_diag`, and `v_norm_sq` now follow the same float32 staging.
- `kernel_diag` — raw kernel diagonals \( K_{x_ix_i} \) (provides a fallback for bound computations).
- `kernel_provider(rows, cols)` — a callable that returns the raw kernel slice \( K(X_{rows}, X_{cols}) \) over integer dataset indices. The host backend now stages a float32, C-contiguous copy of the dataset plus squared norms and evaluates these tiles via SGEMM + the \( \|x\|^2 + \|y\|^2 - 2 x y^\top \) identity, so residual traversal no longer allocates the (rows × cols × dim) broadcast tensor per chunk.
- `point_decoder` — optional decoder that maps tree payloads to dataset indices (defaults to treating them as integer ids).
- `chunk_size` — preferred batch size for host streaming (defaults to 512).

**Precision policy.** Traversal caches, the per-batch pairwise matrix, and the conflict-graph inputs are all float32 by default, which cuts memory residency roughly in half for the ≥32 k Hilbert suites. The only float64 materialisations that remain are (a) the factorisation/eigen problems inside the backend builder and (b) the explicit `*_view(np.float64)` calls made by Gate‑1 audits or by researchers that need double precision diagnostics. Conflict-graph reuse, CLI telemetry, and the new regression tests assert that `ResidualTraversalCache.pairwise.dtype == np.float32` so we cannot silently drift back to float64. Gate audits stay strict: when `gate1_audit` is enabled the runtime replays pruned chunks through `_audit_gate1_pruned`, records any false negatives in `ResidualGateProfile.false_negative_samples`, and raises immediately if a survivor was pruned, so the float32 staging cannot hide precision regressions.

`configure_residual_correlation(...)` installs the residual metric hooks. We intentionally keep Euclidean metrics untouched: the residual path is only active when `COVERTREEX_METRIC=residual_correlation` and custom caches are registered.

## Traversal Path

### Early-Exit Parent Search

- `_residual_find_parents` (in `covertreex/algo/traverse.py`) streams the tree indices in `chunk_size` tiles.
- For each chunk, we request the raw kernel block and feed it to the chunk kernel (`compute_distance_chunk` from `metrics/_residual_numba.py`).
- The kernel accumulates `V_i · V_j` incrementally, uses cached \( \|V_i\|^2 \) and \( p_i \) to bound the residual correlation, and aborts if the best possible distance still exceeds the caller’s current best. This replicates the residual bound from `_ResidualCorrBackend.lower_bound_train`.
- We track the minimum distance per query, yielding the same parent as the dense path.
- **2025‑11‑11 update.** Parent search now routes every chunk through `compute_residual_distances_with_radius(..., force_whitened=True)` using a shared `ResidualWorkspace`. When Gate‑1 is enabled (`COVERTREEX_RESIDUAL_GATE1=1`) the whitened SGEMM covers nearly all candidate pairs before any kernel tiles are requested. On a 4 096×8 synthetic backend (512 queries, chunk size 256) we measured 1 124 864 whitened pairs vs **528** kernel pairs (99.95 % coverage), whereas the pre-change Hilbert 32 k sweep plateaued at 0.43× coverage because every parent chunk streamed the raw kernel block.
- Gate audits stay strict: if `_audit_gate1_pruned` detects a false negative we immediately rescan that chunk via the float32 kernel provider (`_residual_parent_kernel_block`), so the overall traversal keeps SGEMM coverage without ever returning an incorrect parent.
- If Gate‑1 stays disabled, the code still emits the whitened counters (providing 50/50 coverage so dashboards stay sensible), but enabling the gate is now the recommended way to keep kernel calls proportional to true survivors.

### Streaming Scope Assembly

- `_collect_residual_scopes_streaming` reuses the chunk kernel to gather per-query conflict scopes.
- For each parent, we stream candidate tree nodes, apply the residual bound (lower bound via kernel diagonal) and exact distance checks only for survivors, and accumulate dataset indices into the scope. Parent chains (`Next`) are appended afterwards, ensuring deterministic ordering (descending level then index).
- Radii are derived the same way as the Euclidean path: \( \max(2^{\ell_i+1}, S_i) \).
- While streaming we now record the **observed maximum residual distance** per query into `TraversalResult.residual_cache.scope_radii`. These measurements form the “residual radius ladder” used during conflict-graph construction: when the metric is residual-correlation, `build_conflict_graph` swaps the Euclidean fallback \( \max(2^{\ell_i+1}, S_i) \) for the observed value (bounded below by `COVERTREEX_RESIDUAL_RADIUS_FLOOR`, default `1e-3`). This keeps newly inserted nodes from inheriting `radius≈2` regardless of depth and feeds Gate‑1 with radii that reflect the actual residual distances seen during streaming.
- `_collect_residual_scopes_streaming` now tracks chunk-level telemetry (chunks scanned, points touched, dedupe hits, saturation flags). These counters flow into `TraversalTimings` (`traversal_scope_chunk_{scans,points,dedupe,saturated}`) and the JSONL writer so profiling the 16 384-cap hit is straightforward.
- Per-level scope caches remember up to `scope_limit` tree positions from the most recent query at each level. Later queries with the same parent level prefetch those nodes first (`traversal_scope_cache_{prefetch,hits}` capture how many were re-checked and re-used) before the chunk streamer emits new kernel tiles.
- When `COVERTREEX_SCOPE_CHUNK_TARGET>0`, the same value now caps **both** the scope size and the number of tree points we are willing to scan per query. Once the scan budget is exhausted we mark the scope as saturated, stop requesting new kernel tiles, and rely on the telemetry above to highlight the truncation.
- Practical impact: the 32 768-point Hilbert run with `COVERTREEX_SCOPE_CHUNK_TARGET=8192` (`pcct-20251110-105526-68dddf`, log `artifacts/benchmarks/residual_sparse_streamer_20251110115526.jsonl`) now finishes in **≈493 s build / 0.027 s query (37.6 k q/s)** with the scan budget tripping on 48 of 64 dominated batches (`traversal_scope_chunk_points` per batch ≈4.19 M = 512 queries × 8 192 cap, `traversal_scope_chunk_saturated=48`). The earlier cap-only replay (no scan guard) took 900.2 s, so reuse plus the Numba streamer cuts ~45 % off the sparse path without touching the conflict graph.
- For smaller Hilbert sweeps we now collapse the JSONL telemetry with `tools/export_benchmark_diagnostics.py`. Example: `python tools/export_benchmark_diagnostics.py --output artifacts/benchmarks/residual_budget_diagnostics.csv artifacts/benchmarks/residual_{dense,sparse}_budget_4096.jsonl` yields a two-row CSV showing (a) dense residual, budgets disabled → **76.49 s** build with zero budget counters, (b) sparse streamer with `scope_chunk_target=4096` & schedule `1 024,2 048,4 096` → **103.37 s** build, total budget amplification `3.73×`, zero early-terminates, and ≥93.7 % of batches reusing cached pairwise blocks. Hand this CSV (plus the log paths) to anyone auditing adaptive-scan progress.

### Telemetry: “pairs” vs. “milliseconds”

- `traversal_whitened_block_pairs` counts how many whitened SGEMM entries we evaluated (rows × columns per tile). The companion `traversal_whitened_block_ms` gives the wall-clock we spent inside the SGEMM helper.
- Even when `COVERTREEX_RESIDUAL_GATE1=0`, the dense/parallel traversal path now calls `compute_residual_distances_with_radius(..., force_whitened=True)` with the shared `ResidualWorkspace`, so every chunk registers SGEMM work before the kernel fallback. This keeps the `whitened_block_*` counters meaningful for both serial (gate-on) and dense (gate-off) sweeps.
- `traversal_kernel_provider_pairs` / `_ms` track the exact kernel tiles the traversal requested from the host backend. Because the same `ResidualDistanceTelemetry` instance now wraps the dense parent finder, the streaming helpers, and the pairwise cache builder, every residual batch reports comparable totals even when the gate is disabled.
- `tools/export_benchmark_diagnostics.py` exposes the fields in notebook-friendly form: new columns such as `whitened_block_pairs_per_batch`, `whitened_block_ms_per_call`, `kernel_provider_ms_per_pair`, and `whitened_to_kernel_pair_ratio` make it trivial to histogram coverage or prove that SGEMM dominates the traversal budget. Use these summaries when you need to justify ≥80 % “whitened pairs” coverage or to spot batches that still lean on the slow kernel fallback.
- `python -m cli.queries --metric residual ...` now prints the same counters once the run completes, so we can sanity-check coverage without parsing the JSONL. The block lists per-batch medians/p90s for `whitened_block_pairs` vs. `kernel_provider_pairs`, total milliseconds for each path, the aggregate + per-batch coverage ratios, and a final “pair/time mix” sentence. Read that last line as: how much of the traversal operated on whitened SGEMM entries (“pairs”) versus raw kernel tiles, and how the corresponding wall-clock budget split (“milliseconds”). When auditors ask for the ≥80 % coverage proof, quote the CLI summary directly.
- Residual batches emit `traversal_scope_radius_{obs,initial,limit,cap_values}_*` summaries plus `traversal_scope_radius_cap_hits` and `*_delta_{mean,max}` in every JSONL record. Those stats let you inspect how aggressive the ladder currently is (observed maxima) versus the limits you feed into traversal.
- Optional per-scope caps can rein in traversal radii before the gate runs. Provide a JSON table via `COVERTREEX_RESIDUAL_SCOPE_CAPS_PATH=/path/to/caps.json` (schema `{"schema": 1, "default": <optional>, "levels": {"0": 0.5, "1": 0.75}}`) and/or a global fallback with `COVERTREEX_RESIDUAL_SCOPE_CAP_DEFAULT`. The caps clamp the runtime radius but still honour `COVERTREEX_RESIDUAL_RADIUS_FLOOR`; cap hits show up in the JSON logs so you can confirm how often each level is being throttled.
- `docs/data/residual_scope_caps_scope8192.json` captured the legacy flat 1.25 cap, while `docs/data/residual_scope_caps_32768.json` now records the per-level medians (+0.05 margin) from the November 8 32 768-point replay. Derive fresh tables without hand-editing JSON by adding `--residual-scope-cap-output /path/to/out.json` (plus optional `--residual-scope-cap-percentile` / `--residual-scope-cap-margin`) to `cli.queries`; the new `ResidualScopeCapRecorder` streams per-level ladder samples directly from each batch insert, annotates the active `run_id`, and emits summaries at shutdown.
- Residual gate profiles recorded via `tools/build_residual_gate_profile.py` now ship with embedded metadata (`run_id`, seeds, dataset size, kernel hyperparameters, etc.). The builder writes to `artifacts/profiles/residual_profile_<run_id>.json` by default; override with an explicit `output` path or pass `--run-id` to control the identifier if you need deterministic filenames.

**Coverage snapshot recipe.** To capture the ≥80 % whitened-pair coverage target for the Hilbert 32 k suite, run the telemetry-enabled CLI and then summarise the JSONL:

```bash
python -m cli.queries --metric residual --dimension 8 --tree-points 32768 --batch-size 512 --queries 256 --k 8 --seed 0 \
  --log-file artifacts/residual_phase01_hilbert.jsonl
python tools/export_benchmark_diagnostics.py --coverage-threshold 0.8 \
  --output artifacts/benchmarks/residual_phase01_hilbert.csv \
  artifacts/residual_phase01_hilbert.jsonl
```

The CSV now includes `whitened_to_kernel_pair_ratio` plus per-batch `whitened_block_pairs` counts so reviewers can confirm the SGEMM path dominates the traversal budget.

- 2025‑11‑11 — Gate preset replay (`pcct-20251111-211123-2f711f`, `artifacts/benchmarks/residual_phase01_hilbert_gate.jsonl`) produced `whitened_to_kernel_pair_ratio=0.9785` (`tools/export_benchmark_diagnostics.py` wrote the summary to `artifacts/benchmarks/residual_phase01_hilbert_gate.csv`). Scope-cap derivation lives at `artifacts/benchmarks/residual_scope_caps_gate.json`.

## Conflict Graph Pipeline

### Pairwise Matrix

- Inside `build_conflict_graph`, when residual mode is active we decode dataset ids for the batch and materialise the full \( n \times n \) matrix once per batch. That matrix now feeds every downstream stage (dense, segmented, and filter) when available; we fall back to on-demand kernel streaming only when the cache is absent.

### Adjacency Filter

- The dense adjacency builder (`_build_dense_adjacency`) now accepts an optional `residual_pairwise` matrix. When provided, the Numba helper receives the residual distances directly.
- When the cached matrix is available we also reuse it during the post-build radius filter, so pruning simply indexes into that \(n\times n\) buffer. Only when the cache is missing (e.g., legacy dense traversal) do we stream kernel tiles through `compute_residual_distances_from_kernel`.
- The segmented builder piggybacks on the same residual matrix (used when `COVERTREEX_CONFLICT_GRAPH_IMPL=segmented`).
- Chunk telemetry now records pair-wise saturation as well: JSONL batches expose `conflict_scope_chunk_pair_cap`, `conflict_scope_chunk_pairs_before`, `conflict_scope_chunk_pairs_after`, **and** `conflict_scope_chunk_pair_merges`, making it obvious when `_chunk_ranges_from_indptr` fuses or re-merges shards because the estimated pair budget is tiny even if membership volume looks large.
- Set `COVERTREEX_SCOPE_CHUNK_PAIR_MERGE=1` / `--scope-chunk-pair-merge` to enable the new pair-count-aware shard merging heuristics, and `COVERTREEX_SCOPE_CONFLICT_BUFFER_REUSE=1` / `--scope-conflict-buffer-reuse` to reuse the Numba scope-builder buffers (see the `conflict_arena_bytes` counter for confirmation).

## Chunk Kernel (Numba)

`covertreex/metrics/_residual_numba.py` houses the Numba implementation. Given:

- query factor `v_query`, chunk factors `v_chunk`
- cached radii/diagonals `p_i`, `p_chunk`
- raw kernel entries

It emits both distances and a mask indicating which entries fall below a caller-specified radius. We expose `compute_residual_distances_with_radius` in `metrics/residual.py` so traversal and conflict graph code paths can reuse the same accelerated helper with CPU fallback.

### Gate‑1 (whitened Euclidean bound)

- `ResidualCorrHostData` materialises a float32-whitened copy of `v_matrix` (`gate_v32` + `gate_norm32`) plus telemetry counters and runtime knobs (`gate1_alpha`, `gate1_margin`, `gate1_eps`, `gate1_audit`, `gate1_band_eps`, `gate1_keep_pct`, `gate1_prune_pct`). Gate‑1 now ships **disabled by default**; you must export `COVERTREEX_RESIDUAL_GATE1=1` (or use the CLI `--residual-gate` preset) to opt in.
- `compute_residual_distances_with_radius` now evaluates a **two-threshold, gray-band** gate: per-radius bins supply both “keep” and “prune” quantiles, the keep side shrinks by `(1−margin)`, the prune side expands by `(1+margin)`, and any whitened distance within `band_eps` of the prune threshold is forced into the gray band for an exact kernel call. Only entries above `prune+band_eps` are discarded.
- Per-batch telemetry (`traversal_gate1_{candidates,kept,pruned}`, `traversal_gate1_ms`) still flows through traversal timings; audit mode replays `_distance_chunk` on the pruned rows and raises immediately if a true neighbour slips through.
- The gate remains fully opt-in (`COVERTREEX_RESIDUAL_GATE1=1`) and guarded by the new env knobs `COVERTREEX_RESIDUAL_GATE1_{BAND_EPS,KEEP_PCT,PRUNE_PCT}` so we can sweep gray-band widths independently from α/ε.
- Lookup-driven runs now consume quantile envelopes (p95/p99/p99.9 by default). As we refresh `docs/data/residual_gate_profile_32768_caps.json` we can simply change `KEEP_PCT/PRUNE_PCT` to pick the envelope we want without rebuilding the tree.
- Gate-on reruns should finally show `traversal_gate1_pruned>0` once we regenerate the lookup with the new quantile schema; until that happens we keep the feature disabled by default and only use it for explicit experiments.

## Gate Profile Capture & JSONL Ingestion

- `cli.queries` exposes three new knobs for residual runs:
  - `--residual-gate-profile-path=/path/to/profile.json` records the per-run Gate‑1 samples (same structure as `ResidualGateProfile.dump`).
  - `--residual-gate-profile-bins=N` overrides the bin count (default 512).
  - `--residual-gate-profile-log=/path/to/profile_logs.jsonl` appends the recorded profile payload (plus run metadata, runtime snapshot, and the batch log path) to a JSONL stream.
- The CLI auto-creates a timestamped profile JSON under `artifacts/profiles/` when you supply `--residual-gate-profile-log` but omit `--residual-gate-profile-path`, so a single run now yields both the binwise lookup and a normalized JSONL line without any manual file juggling.
- `tools/ingest_residual_gate_profile.py` replaces the ad-hoc rebuild workflow. Point it at one or more JSON/JSONL profile dumps and it will:
  1. Validate that all inputs share identical `radius_bin_edges`.
  2. Merge the `max_whitened`, `max_ratio`, and `counts` arrays via element-wise maxima / sums.
  3. Merge **quantile envelopes** per percentile (taking the per-bin max across runs) and stamp `quantile_{counts,totals}` so we can audit coverage later.
  4. Stamp aggregate metadata (`sources`, `run_ids`, overrides) and write a ready-to-use lookup JSON.

Example:

```bash
# Capture a Hilbert run with streamer + scope cap, profile log + JSON file
uv run python -m cli.queries --metric residual \
  --tree-points 32768 --queries 1024 --k 8 --batch-size 512 \
  --residual-gate-profile-log profiles/residual_gate_runs.jsonl \
  --residual-gate-profile-bins 512 \
  --residual-gate lookup

# Merge multiple runs into a single lookup artefact
uv run python tools/ingest_residual_gate_profile.py \
  artifacts/profiles/residual_gate_runs.jsonl \
  --output docs/data/residual_gate_profile_32768_caps.json \
  --metadata corpus=hilbert32k scope_cap=8192
```

The resulting lookup mirrors `ResidualGateProfile.dump` but includes additional provenance (sources, run ids, runtime snapshot). `COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH` can then point at the merged JSON without re-running the expensive sparse traversal.

### Calibration & Lookup Table

- `tools/build_residual_gate_profile.py` builds an empirical profile from a synthetic residual workload. It reproduces the diag0 harness (2048 points, dimension 8, seed 42) and samples every pair, recording (residual distance, whitened distance) into evenly spaced radius bins. Use `--quantiles 95,99,99.9 --quantile-sample-cap 4096` (defaults shown) to control which percentiles are preserved in the on-disk lookup.
- The 32 768-point capture (`docs/data/residual_gate_profile_32768_caps.json`) is now the default lookup used by `COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH` and the CLI presets. The file was refreshed on **2025‑11‑10** via a full sparse run with `COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH=$PWD/docs/data/residual_gate_profile_32768_caps.json` (log `artifacts/benchmarks/residual_sparse_streamer_profile_20251110122936.jsonl`, run id `pcct-20251110-112936-0d2a25`), so it reflects the radius ladder and telemetry we see with the Numba streamer + 8 192 scan cap. Use the original diag0 artefact (`docs/data/residual_gate_profile_diag0.json`) only for quick smoke tests.
- To regenerate or explore alternative datasets, run for example:

  ```bash
  python tools/build_residual_gate_profile.py docs/data/residual_gate_profile_diag0.json \
      --tree-points 2048 --dimension 8 --seed 42 --bins 512 --pair-chunk 64
  ```

- At runtime you can consume the lookup by setting:

  ```bash
  export COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1
  export COVERTREEX_RESIDUAL_GATE1=1
  export COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH=docs/data/residual_gate_profile_32768_caps.json
  export COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN=0.02   # optional safety buffer
  export COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP=10.0      # optional cap for huge radii
  ```

  (You can omit the cap to let the lookup see larger radii; we cap it when we want deterministic thresholds even if the residual ladder spikes.) The lookup supplies per-radius thresholds directly, so `residual_gate1_alpha` becomes a fallback rather than the primary tuning knob.

- The CLI now takes care of those flags for you: `python -m cli.queries --metric residual --residual-gate lookup ...` wires sparse traversal, enables the gate, and points at `docs/data/residual_gate_profile_32768_caps.json`. Override the path (`--residual-gate-lookup-path`), margin (`--residual-gate-margin`), or radius cap (`--residual-gate-cap`) as needed. Use `--residual-gate off` to explicitly keep the gate disabled during experiments.

- Gate‑1 still defaults to off globally—we only flip it on in telemetry or experimental runs until we finish the sparse traversal rollout and confirm the lookup holds for larger corpora. When you opt in, make sure `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` is set; otherwise the dense traversal path bypasses the streaming helper and the gate never engages.
- Regardless of whether Gate‑1 is enabled, residual batches now materialise the whitened cache so the grid builder can operate in the same space as the Euclidean runs. If the default cell width feels too loose/tight, set `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` (>1 tightens the cells, <1 loosens them; default `1.0`) to dial in `grid_*` telemetry before capturing regression artefacts.

- `docs/data/residual_gate_profile_scope8192.json` captures a larger synthetic workload (8 192 points, 33 550 336 samples, same 512 bins) for comparison. Relative to the diag0 artefact the median threshold is +9.66, the 90th percentile delta is +19.93, the maximum absolute delta is ≈61.9, and 387/512 bins have higher maxima. `docs/data/residual_gate_profile_32768_caps.json` replaces those synthetic probes with the full 32 768-point capture (caps enabled, audit on) from 2025‑11‑08 so gate experiments can finally reference a workload that matches the production corpus.

- Latest 32 k gate replay (pcct-20251110-112456-69dfdd, log `artifacts/benchmarks/residual_sparse_gate_20251110124256.jsonl`): chunk target 8 192, lookup path pointing at the refreshed profile, audit on. Build cost is unchanged (≈1.3 k–3.0 k ms dominated batches) and `traversal_gate1_pruned` remains zero, so the lookup is now safe but still conservative. Once the new per-level cache heuristics land we can rerun this experiment to confirm pruning >0 without tripping the audit.
- Setting `COVERTREEX_RESIDUAL_PREFILTER=1` enables the lookup-driven gate as a chunk-level SIMD prefilter (defaults: diag0 lookup, margin 0.02, radius cap 10). Override the lookup via `COVERTREEX_RESIDUAL_PREFILTER_LOOKUP_PATH`—for example, point it at `docs/data/residual_gate_profile_32768_caps.json` to reuse the November 8 corpus. Combining the prefilter with the scan cap and per-level cache (`benchmark_residual_cache_prefilter_20251108.jsonl`) yields **700.71 s build / 0.027 s query (37.6 k q/s)** while keeping `traversal_scope_chunk_points≈4.19 M` and logging `traversal_scope_cache_prefetch≈2.1 M` on saturated batches.

## Tests

- `tests/test_metrics.py` exercises the chunk kernel (distance + radius masks) and validates that residual distances computed via kernel reuse match the dense path.
- `tests/test_traverse.py` now includes a residual sparse traversal regression (dense vs. streamed scopes) to ensure parents/levels/scopes remain consistent.
- `tests/test_conflict_graph.py` gained a residual parity check to confirm dense Euclidean and residual-aware models produce identical CSR structures.

## Residual Grid Scale Sweep (2025‑11‑09)

**Objective.** Measure whether the new `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` knob changes the dominated residual workload enough to impact build time, and capture telemetry deltas that auditors can compare against future runs.

**Setup.** `python -m cli.queries --dimension 8 --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --metric residual --baseline gpboost --seed 42 --log-file <artifact>` with

- `COVERTREEX_BACKEND=numpy`, `COVERTREEX_ENABLE_NUMBA=1`
- `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`, `COVERTREEX_PREFIX_SCHEDULE=doubling`
- diagnostics on, dense traversal (`COVERTREEX_SCOPE_CHUNK_TARGET=0` from default gold-standard harness)
- swept `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` ∈ {0.75, 1.50}

Artifacts live under `artifacts/benchmarks/` and share the run IDs below.

| Scale | Log file | Run ID | PCCT Build (s) | Throughput (q/s) | Grid leaders (mean) | Grid local edges (mean) | Traversal ms (p50) | Conflict ms (p50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | `residual_grid_scale0p75_20251109180103.jsonl` | `pcct-20251109-170103-ce1129` | 56.10 | 28 802 | 510.17 | 130 115 | 737 ms | 69.6 ms |
| 1.50 | `residual_grid_scale1p50_20251109180226.jsonl` | `pcct-20251109-170226-df7a7f` | 55.73 | 28 698 | 511.70 | 130 713 | 711 ms | 69.9 ms |

**Observations.**

- The grid stays fully saturated (domination ratio ≈0.984) regardless of scale: every batch emits ~1.5 k cells, ~510 leaders, and zero conflict edges, so MIS time remains negligible.
- Tightening the scale increases local grid edges by <0.5 % and shaves ~25 ms off the traversal median, but the overall build time is bounded by traversal (0.7–1.4 s per dominated batch) rather than grid work.
- RSS peaks around 1.53–1.60 GB in both runs; memory pressure doesn’t change with the scale adjustments.
- The reference dense run on the new default lookup (`residual_grid_default_20251109200738.jsonl`) still reports **63.39 s build / 0.035 s query (29.5 k q/s)** with `traversal_gate1_* = 0`, so switching the default from `diag0` to `residual_gate_profile_32768_caps.json` leaves the grid telemetry unchanged while keeping the harness aligned with the production corpus.

**Traversal-focused follow-ups.** Grid tuning alone cannot deliver material build wins while traversal dominates:

1. **Gate‑1 rollout.** The sweep still shows `traversal_gate1_* = 0`; enabling the lookup-backed gate (with `COVERTREEX_RESIDUAL_GATE1=1` plus the diag0 profile) should prune candidates before `_distance_chunk` and directly attack the 0.7–1.4 s dominated batches.
2. **Scope trimming / cache hits.** Saturated batches hit zero cache hits at the start and only ~344/512 by the tail. Investing in per-level cache prefetch heuristics (or raising `residual_scope_cap_default`) should reduce the number of fully scanned scopes.
3. **Sparse traversal gating.** Re-enabling the sparse traversal path once the bucketed CSR builder lands (see AUDIT.B) will remove the O(n log n) semisort tail (`traversal_semisort_ms` still >20 ms median) and keep dominated scopes bounded without relying on scope chunk caps.

Until those traversal improvements ship, keep the residual grid in place for determinism (leader telemetry + MIS seeds) but don’t expect the whiten scale alone to replicate the Euclidean build-time gains.

### Gate‑1 Experiments (2025‑11‑09 evening)

**Goal.** Re-enable the float32 Gate‑1 (lookup-backed) and confirm whether it can prune enough dominated candidates to lower traversal time without violating correctness. Runs used the same 32 768‑point benchmark as above.

> **Implementation note (2025‑11‑09):** `_collect_residual_scopes_streaming()` now streams scope memberships through the same bucketed CSR builder (`build_scope_csr_from_pairs`) that the conflict graph uses. This gets rid of the quadratic Python `list.extend` + `tuple` churn and guarantees the CSR output matches the deterministic tuple-of-tuples representation the Euclidean paths already emit. The change doesn’t fix the long “semisort” timings by itself—those counters still include the actual residual distance scans—but it removes a source of Python overhead and makes the sparse diagnostics comparable to the dense path.

| Config | Key env knobs | Outcome |
| --- | --- | --- |
| `residual_gate_lookup_sparse_scale1p00_20251109191729.jsonl` | `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1`, `COVERTREEX_RESIDUAL_GATE1=1`, lookup margin 0.02, audit on | Gate tripped audit in the second batch (`RuntimeError: Residual gate pruned a candidate that lies within the requested radius`). Only the first batch logged, `traversal_gate1_*` counters stayed at 0 because the failure happened during the audit replay. |
| `..._margin0p5_20251109191744.jsonl` | Same but lookup margin 0.5 | Audit still failed immediately. Loosening the margin alone does not prevent false negatives with the current lookup profile. |
| `residual_gate_lookup_sparse_noaudit_scale1p00_20251109191757.jsonl` | Same as the first run, but `COVERTREEX_RESIDUAL_GATE1_AUDIT=0` (unsafe) | Run completed but took **>34 minutes** (our harness killed the CLI after ~2 050 s). Telemetry: `traversal_semisort_ms` averaged **31.2 s** per batch (sparse traversal is still sorting scope members), Gate‑1 processed ~5.5×10^8 candidates but only pruned ~7.6×10^6 (~1.4 %), and the gate itself cost ~2.85 s per batch. Whatever pruning we gained was dwarfed by the sparse semisort overhead. |
| `residual_gate_dense_scale1p00_20251109195236.jsonl` | Gate enabled, sparse traversal **disabled** | Gate counters remained zero for all 64 batches; as expected, the dense traversal path never calls the gate and behavior matches the baseline dense run. |

**Lessons.**

1. **Audit fails immediately.** Even the “loose” lookup margin (0.5) still prunes legitimate neighbors on the first dominated batch. We need either a safer lookup profile (probably drawn from the 32 k corpus) or an adaptive bound that accounts for the per-batch radius ladder before we can ship Gate‑1.
2. **Sparse traversal must be fixed first.** With audit off, Gate‑1 prunes millions of candidates, but the sparse traversal’s semisort dominates runtime (dozens of seconds per batch). The bucketed CSR builder from AUDIT.B is a prerequisite; otherwise enabling sparse traversal regresses build time by 40–60×.
3. **Gate on dense traversal is a no-op.** Without the streaming helper the gate never runs, so toggling `COVERTREEX_RESIDUAL_GATE1` alone does nothing in production today.

**Traversal notes.** To make Gate‑1 viable we need the bucketed CSR path (so sparse scopes don’t explode), a lookup derived from the latest 32 k telemetry (to keep audit happy), and probably a cheaper gate kernel (current gate spent ~2.85 s/batch processing 5.5×10^8 candidates). Only after those land will Gate‑1 have a chance of chipping away at the 0.7–1.4 s dominated batches we see in the dense runs.

## Operational Notes

- Residual mode requires `backend.name == "numpy"` (NumPy backend) until the GPU/JAX kernels are ported.
- The decoder must map tree payloads to dataset indices; if trees store transformed buffers, supply a custom `point_decoder` when creating `ResidualCorrHostData`.
- The chunk kernel honours `chunk_size`; tune it to balance host-side streaming vs. cache reuse.
- `COVERTREEX_RESIDUAL_STREAM_TILE` (CLI `--residual-stream-tile`) caps the dense streamer’s tile size. The runtime now defaults to `min(scope_limit, backend.chunk_size)` (64 entries when Gate‑1 is off), and the override lets auditors reproduce the historical 512-entry scans or probe smaller tiles without code edits.
- Setting `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1` now engages residual streaming automatically when the metric is residual-correlation; otherwise, traversal falls back to the dense mask path.
- **Conflict fallback (2025‑11‑11).** When the traversal cache is missing we now stream adjacency rows through the same whitened helper rather than calling `_rbf_kernel` directly. Each `(source, target)` chunk inherits the staged float32 workspace, Gate‑1 prunes obviously distant edges, and only the surviving edges trigger SGEMM kernel tiles before we compare against `min(radii_source, radii_target)`.
