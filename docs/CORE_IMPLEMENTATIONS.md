# Core Implementations & Benchmark Snapshot (2025-11-10)

_This file is a to-be-maintained reference. Keep the benchmark table and the code listings in sync with the repository state whenever core algorithms change._

## Performance Summary (CPU, Numba-enabled)

> GPU/JAX execution is currently disabled. All timings below use the NumPy backend with `COVERTREEX_BACKEND=numpy` and `COVERTREEX_ENABLE_NUMBA=1`.

> Use `python tools/run_reference_benchmarks.py` to regenerate the 2 048 / 8 192 / 32 768 artefacts below. The harness writes JSONL logs and CSV summaries under `artifacts/benchmarks/reference/<timestamp>/` (manifest included).

### Quick Benchmark — 2 048 tree pts / 512 queries / k=8

| Implementation               | Build Time (s) | Query Time (s) | Throughput (q/s) | Notes |
|------------------------------|----------------|----------------|------------------|-------|
| PCCT (Numba, diagnostics off)| 0.366          | 0.097          | 5 261            | `COVERTREEX_ENABLE_DIAGNOSTICS=0`; diagnostics-on run: 0.373 s / 0.098 s |
| Sequential baseline          | 2.25           | 0.024          | 21 001           | In-repo compressed cover tree |
| GPBoost Numba baseline       | 0.292          | 0.519          | 987              | Numba port of the GPBoost cover tree |
| External CoverTree baseline  | 1.00           | 1.215          | 421              | `pip install -e '.[baseline]'` |

_Command:_
```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=0 \
UV_CACHE_DIR=$PWD/.uv-cache \
uv run python -m cli.queries \
  --dimension 8 --tree-points 2048 \
  --batch-size 128 --queries 512 --k 8 \
  --seed 42 --baseline gpboost
```

_Set `COVERTREEX_ENABLE_DIAGNOSTICS=1` to collect the instrumentation counters (adds ~7 ms to the build in this configuration)._

### Telemetry artefacts (default output paths)

All benchmark entrypoints now stamp a unique `run_id` and write structured telemetry under `artifacts/` unless you explicitly opt out:

- `cli.queries` creates `artifacts/benchmarks/queries_<run_id>.jsonl` automatically. Use `--log-file custom.jsonl` to override the location or `--no-log-file` to suppress JSONL output altogether.
- `cli.runtime_breakdown` writes CSV summaries to `artifacts/benchmarks/runtime_breakdown_<run_id>.csv` unless you pass `--no-csv-output`. Override with `--csv-output /path/to/file.csv` if you need a fixed path (the `cli.runtime_breakdown` shim remains for backwards compatibility).
- `benchmarks.batch_ops` emits a JSON summary (`artifacts/benchmarks/batch_ops_<run_id>.json`) by default. Pass `--log-json custom.json` to change the destination or `--no-log-json` to skip it.

These files include the runtime configuration snapshot (`runtime_*` keys) so you can correlate logs back to the exact backend/precision/strategy selections that were active for that run.

### Scaling Snapshot — CPU builds (diagnostics on)

| Workload (tree pts / queries / k) | PCCT Build (s) | PCCT Query (s) | PCCT q/s | Sequential Build (s) | Sequential q/s | GPBoost Build (s) | GPBoost q/s | External Build (s) | External q/s |
|-----------------------------------|----------------|----------------|----------|----------------------|----------------|-------------------|-------------|--------------------|---------------|
| 8 192 / 1 024 / 16                | 4.15           | 0.018          | 57 660   | 33.65               | 5 327         | 0.75              | 285         | 14.14              | 122           |
| 32 768 / 1 024 / 8 (Euclidean)    | 16.75          | 0.039          | 25 973   | —                   | —             | 3.10              | 65.1        | —                  | —             |
| 32 768 / 1 024 / 8 (Residual, dense)** | 17.8          | 0.026          | 39 000   | —                   | —             | 2.51              | 91.6        | —                  | —             |
| 32 768 / 1 024 / 8 (Residual, sparse streamer)** | 493          | 0.027          | 37 600   | —                   | —             | 2.51              | 91.6        | —                  | —             |

_*GPBoost remains Euclidean-only; the baseline numbers in the residual row are provided for throughput context only._

_**Hilbert batches, diagnostics on, `COVERTREEX_SCOPE_CHUNK_TARGET=0` (dense) or 8 192 (sparse); dense path adds pair-count shard merging + buffer reuse. Logs: `pcct-20251114-214845-7df1be` (`residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`) and `pcct-20251110-105526-68dddf` (sparse streamer)._ 

**Residual build status (2025-11-17).**
- Dense residual (gate off, dense scope streamer + masked append + level-cache batching + bitsets + pair-count shard merging + buffer reuse) is the current gold baseline: `pcct-20251114-214845-7df1be` reports **≈17.8 s build**, `traversal_semisort_ms≈36 ms` (p90 ≈59 ms), and `pcct | throughput≈39 k q/s`. Log: `artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`.
- The 4 k guardrail (`pcct-20251114-105559-b7f965`, log `artifacts/benchmarks/residual_phase05_hilbert_4k_dense_streamer_gold.jsonl`) clocks `traversal_semisort_ms≈41.7 ms` (still `<1 s`) but currently logs zero whitened coverage / gate prunes, so rerun the preset once Gate‑1 is re-enabled.
- Sparse residual + streamer + cap remains scan-cap dominated (historical `pcct-20251110-105526-68dddf` sits at ≈493 s); rerunning it with the dense scope streamer enabled is on the backlog.

**Recommended CLI defaults (dense streamer + batching, 2025-11-17).** Use the command below for the fastest audited residual numbers; it mirrors the log above and explicitly keeps the dense scope streamer, bitsets, level-cache batching, pair-count shard merging, and buffer reuse enabled. Pass `--no-residual-*`, `--no-scope-chunk-pair-merge`, or `--no-scope-conflict-buffer-reuse` if you need to reproduce the legacy behaviour.

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline none \
  --residual-dense-scope-streamer \
  --residual-scope-bitset \
  --residual-level-cache-batching \
  --log-file artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl
```

**Gold-standard residual benchmark (dense streamer + batching + pair merge).** When publishing new numbers or triaging regressions, start from `pcct-20251114-214845-7df1be` (`artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`). Diagnostics remain on by default so telemetry stays comparable; disable them explicitly if you need apples-to-apples with the historical diag0 logs.

**Historical residual build status (2025-11-10).**
- Dense residual (gate off) remains the wall-clock leader: `pcct-20251110-105043-101bb9` reports ≈257 s build, `traversal_ms≈1.23 s`.
- Sparse residual + streamer + cap trims the old 581 s baseline down to ≈493 s but is still scan-cap dominated (`traversal_scope_chunk_saturated` on 48/64 batches).
- Sparse residual + profile capture `pcct-20251110-112936-0d2a25` mirrors the streamer run and feeds the refreshed lookup at `docs/data/residual_gate_profile_32768_caps.json`.
- Sparse residual + gate audit `pcct-20251110-112456-69dfdd` stays clean but `traversal_gate1_pruned=0`, so further cache/prefilter work is needed before we can claim gate-on speedups.
- Every residual run can now emit a reusable Gate‑1 profile without reruns: `--residual-gate-profile-log profiles/residual_gate_runs.jsonl` appends a JSONL line per run (auto-generating the underlying profile JSON if needed), and `tools/ingest_residual_gate_profile.py` merges those lines into lookup files under `docs/data/`.
- Residual traversal caches, conflict inputs, and telemetry now default to float32 staging (with opt-in float64 views for audits), cutting traversal/cache residency roughly in half. `ResidualTraversalCache.pairwise.dtype == np.float32` is enforced in tests, so any drift back to float64 will fail CI.
- Phase 5 deterministic selection (2025‑11‑12) swapped all residual/Euclidean scope builders to `select_topk_by_level` (argpartition + stable ties) and moved the limit logic into the Numba CSR path, so `traversal_semisort_ms` is expected to approach zero once the telemetry sweep is re-run. Because the latest Hilbert 32 k replay still exceeded an hour wall-clock, every residual audit now starts with a 4 k-point dry run (same flags, smaller `--tree-points`) and only scales to 32 k after the small run reports ≥95 % whitened coverage, `<1 s` median semisort, and `conflict_pairwise_reused=1` across batches.
- Per-run telemetry is summarised with `python tools/export_benchmark_diagnostics.py --output artifacts/benchmarks/residual_budget_diagnostics.csv artifacts/benchmarks/residual_{dense,sparse}_budget_4096.jsonl`: the dense control (no chunk target) records a **76.49 s** build while the budgeted sparse streamer (chunk target 4 096, schedule `1 024/2 048/4 096`) lands at **103.37 s** with budget amplification `3.73×` and ≥93.7 % pairwise reuse after the warm-up batch. Point perf reviews at that CSV instead of console output.

**Historical CLI defaults (pre-dense-streamer, 2025-11-10).** Use the dense traversal recipe from `pcct-20251110-105043-101bb9` whenever you need the fastest audited numbers from the maskopt_v2 era. The command below leaves the gate off, keeps scopes unclamped, and mirrors every setting from the 257 s / 0.027 s build:

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=doubling \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file artifacts/benchmarks/residual_dense_32768_best.jsonl
```

Treat this command as the default when publishing numbers or triaging regressions; any change that enables the sparse streamer, gate, or scope caps should be considered experimental. The CLI still defaults to Euclidean + Hilbert when no flags are provided, so pass the exact arguments above whenever you need the historically best residual timings.

The 32 768-point run currently logs PCCT and the GPBoost baseline; sequential/external baselines are still pending optimisations to keep runtime manageable at that scale.

### Historical gold-standard residual benchmark (default path)

Fresh artefacts: `pcct-20251110-105043-101bb9` (dense) and `pcct-20251110-105526-68dddf` (sparse streamer + cap) capture the residual rows above with `COVERTREEX_BATCH_ORDER=hilbert`, `COVERTREEX_PREFIX_SCHEDULE=doubling`, `COVERTREEX_CONFLICT_GRAPH_IMPL=grid`, diagnostics on, and Numba enabled. Dense traversal leaves `COVERTREEX_SCOPE_CHUNK_TARGET=0`; sparse traversal sets it to 8 192.

These runs supersede the historical **56.09 s / 0.229 s (4 469 q/s)** reference from 2025‑11‑06. Keep that older log only for pre-grid comparisons; use the November 10 artefacts when auditing regressions on the streamer, scan cap, or refreshed gate lookup. To regenerate the legacy “gold standard” (natural batch order, diagnostics off) for continuity, run:

```
./benchmarks/run_residual_gold_standard.sh [optional_log_path]
```

By default the script writes `bench_residual.log` in the repo root, **forces the Python/Numba path** (`COVERTREEX_ENABLE_RUST=0`) and sets `COVERTREEX_ENABLE_NUMBA=1`, `COVERTREEX_BATCH_ORDER=natural`, `COVERTREEX_PREFIX_SCHEDULE=doubling`, `COVERTREEX_SCOPE_CHUNK_TARGET=0`, and `COVERTREEX_ENABLE_DIAGNOSTICS=0` so the output stays comparable across machines. An optional comparison pass runs after the gold run with `COMP_ENGINE` (default: `rust-hilbert`); set `COMP_ENGINE=none` to skip. Treat the gold log as the reference artefact when auditing regressions or publishing updated numbers (refer back to the 2025‑11‑06 56.09 s run when you need the historical pre-grid baseline).

To reproduce the clamped adjacency run captured on 2025‑11‑07 (matching `benchmark_residual_clamped_20251107_fix_run2.jsonl`), invoke:

```
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
UV_CACHE_DIR=$PWD/.uv-cache \
uv run python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --metric residual --baseline none
```

The JSONL log now lands in `artifacts/benchmarks/queries_<run_id>.jsonl` automatically; pass `--log-file benchmark_residual_clamped_20251107_fix_run2.jsonl` if you need to mirror the historical file layout.

> **Baseline note:** The `_diag0` artefacts from 2025‑11‑07 (`benchmark_residual_clamped_20251107_diag0.jsonl`, `benchmark_residual_chunked_20251107_diag0.jsonl`) are the preferred reference logs whenever you need to compare against GPBoost or older “*_fix_run2” runs. They disable diagnostics to match the baseline settings but still include the new `gate1_*` and `traversal_scope_chunk_{scans,points,dedupe,saturated}` counters, so keep using them for apples-to-apples regressions going forward.

### Phase 5 Regression Warning (2025-11-12)

- The 4 k-point Hilbert shakedown (`artifacts/benchmarks/residual_phase05_hilbert_4k.jsonl`) now takes **114 s wall-clock** across eight dominated batches. Median `traversal_ms` sits at **13.3 s**, `traversal_semisort_ms` hits **7.53 s (p50)** / **11.0 s (p90)** / **17.5 s (max)**, and whitened coverage collapses to **0.875×** because `compute_residual_pairwise_matrix` still streams the dense 512×512 kernel for every batch. Gate telemetry recorded **7.34 M candidates / 0 pruned**, so enabling the lookup with the current cap/margin buys us nothing other than extra bookkeeping.
- The so-called “Phase 5 validation” on the full Hilbert 32 k preset (`artifacts/benchmarks/residual_phase05_hilbert.jsonl`) is even worse: **123 batches**, **median traversal_ms≈59.4 s**, **median semisort_ms≈28.0 s**, and **total traversal time 7 277 s (~2.0 h)**. Coverage looks fine on paper (0.979×) purely because the streamer dwarfs the per-batch pairwise cache, not because the gate/selection changes helped.
- Conclusion: the current deterministic-selection release is a regression factory. Until the gate actually prunes and semisort collapses to the promised “near-zero”, stick to the dense defaults above. Any sparse/gate experiment must include a 4 k-point dry run (same CLI flags, smaller `--tree-points`) and a CSV dump via `tools/export_benchmark_diagnostics.py` so we can reject regressions before burning another hour on the 32 k suite.

_Command (8 192 row):_
```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
python -m cli.queries \
  --dimension 8 --tree-points 8192 \
  --batch-size 256 --queries 1024 --k 16 \
  --seed 12345 --baseline gpboost \
  --log-file benchmark_grid_8192_baseline_20251108.jsonl

# 32k Euclidean vs GPBoost
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file benchmark_grid_32768_baseline_20251108.jsonl

# 32k residual vs GPBoost (baseline remains Euclidean-only)
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --metric residual
```
Residual commands automatically run with `COVERTREEX_PREFIX_SCHEDULE=doubling`, Gate‑1 lookup enabled, and chunking disabled unless you override those env vars.

To capture warm-up versus steady-state timings for plotting, append `--csv-output runtime_breakdown_metrics.csv` when running `cli.runtime_breakdown`.

## Current Observations & Hypotheses

### Euclidean metric (NumPy backend)

- Fresh 32 768-point Hilbert+grid reruns (dimension 8, batch 512, 1 024 queries, k = 8, seed 42) now land at **16.75 s build / 0.039 s query (~26 k q/s)** for PCCT with NumPy+Numba (`bench_euclidean_grid_32768_20251108*.log` + `benchmark_grid_32768_baseline_20251108{,_run2}.jsonl`). The paired GPBoost baseline clocks in at **≈3.10 s build / 15.72 s query (65 q/s)**, so PCCT sustains ~400× higher steady-state throughput in this configuration.
- The 8 192-point suite (`--batch-size 256`, `k=16`, seed 12345) records **4.15 s build / 0.018 s query (57.7 k q/s)** for PCCT versus GPBoost’s **0.75 s / 3.59 s (285 q/s)**, captured in `bench_euclidean_grid_8192_20251108.log` + `benchmark_grid_8192_baseline_20251108.jsonl`.
- Batch logs show `traversal_ms` clustering between ≈0.25–0.49 s on dominated 32 k batches while `conflict_graph_ms` remains within 7–19 ms and MIS stays sub-0.2 ms. With the journal pipeline eliminating repeated array slices, traversal/mask assembly is once again the limiting phase at scale even under the grid builder.
- Conflict-graph builders now live in `conflict_graph_builders.py`; dense vs segmented vs residual paths report distinct telemetry. Dense remains the default until scope chunk limits (see Next Steps) are dialled in.
- GPBoost’s cover tree baseline is sequential and Euclidean-only (no conflict graphs, no MIS). We continue to keep comparisons on the CPU/NumPy backend so wall-clock deltas stay reproducible without JAX/GPU variability.

These November 8 artefacts supersede the 37.7 s / 0.262 s Hilbert+grid numbers from earlier in the week; treat them as the current Euclidean “gold standard” for PCCT until another build drops below ~15 s while keeping telemetry comparable.

### Residual-correlation metric (synthetic RBF caches, 2025-11-09)

- **Current best (clamped, scope chunking off).** Re-running the 32 768×1 024×k=8 workload with Hilbert ordering, the grid conflict builder, diagnostics on, and no scope chunking now lands at **57.80 s build / 0.028 s query (36.2 k q/s)** for PCCT while the GPBoost baseline remains at **≈2.68 s build / 10.50 s query (97.5 q/s)**. Artefacts: `benchmark_residual_clamped_20251109.log` and `artifacts/benchmarks/benchmark_residual_clamped_20251109.jsonl`.
- **Chunked traversal (scope cap 8 192).** Enabling sparse traversal with `COVERTREEX_SCOPE_CHUNK_TARGET=8192` keeps query time identical (0.028 s) but increases build time to **727.44 s** because each dominated batch now scans ≈4.19 M points in 512 chunks before conflict filtering; adjacency scatter drops to ≈18 ms. Logs: `benchmark_residual_scope8192_20251109.log` + `artifacts/benchmarks/benchmark_residual_scope8192_20251109.jsonl`.
- The historical unclamped Hilbert run from 2025‑11‑07 (`benchmark_residual_32768_default.jsonl` / `run_residual_32768_default.txt`) is still useful for regression tracking (66.25 s build / 0.305 s query). Likewise, the earlier chunked-with-prefilter sweeps (`benchmark_residual_cache_prefilter_20251108.jsonl`, `benchmark_residual_scopecap_20251108.jsonl`) describe how lookup-driven prefilters impact traversal telemetry.
- Per-batch telemetry for the new clamped run shows dominated batches averaging **~0.76 s traversal / 93 ms conflict graph / 70 ms adjacency scatter** while leaving Gate‑1 counters at zero; conflict scopes regularly swell past 16 M members because chunking is disabled. The chunked run trades build time for tighter memory bounds (max scope shard 8 192 members, 512 segments per dominated batch) and keeps adjacency scatter bounded.
- The residual adjacency filter still recomputes pairwise kernels even when `residual_pairwise` is cached from traversal; linking those surfaces remains the most obvious follow-up before we attempt to rely on Gate‑1 lookups.

**Commands to reproduce**

Current best (clamped, diagnostics on, `scope_chunk_target=0`):

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_TARGET=0 \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
UV_CACHE_DIR=$PWD/.uv-cache \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file benchmark_residual_clamped_20251109.jsonl
```

Chunked traversal (sparse traversal + scan cap):

```
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
COVERTREEX_PREFIX_SCHEDULE=adaptive \
COVERTREEX_SCOPE_CHUNK_TARGET=8192 \
COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS=256 \
UV_CACHE_DIR=$PWD/.uv-cache \
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file benchmark_residual_scope8192_20251109.jsonl
```

These commands emit the same JSONL telemetry referenced above (see `artifacts/benchmarks/`) and print the console summaries captured in the paired `.log` files.

## Residual-Correlation Metric Benchmark Status (2025-11-06)
## PCCT Reference Implementation (2025-11-10)

### Source map
- `covertreex/algo/traverse.py` — parent search, dense vs sparse traversal toggles, scope cache/ladder telemetry.
- `covertreex/algo/conflict/` — conflict graph builders (`grid`, `segmented`, `sparse`) plus adjacency filters and MIS strategies.
- `covertreex/metrics/residual.py` + `_residual_numba.py` — host caches, chunk kernels, Gate‑1 machinery, and the bucketed CSR streamer.
- `cli/queries.py` + `benchmarks/` helpers — reference entrypoints for benchmarking, logging, and artefact stamping.

### Runtime configuration knobs
| Knob | Description | Default | Notes |
|------|-------------|---------|-------|
| `COVERTREEX_ENABLE_NUMBA` | Enables Numba chunk kernels & CSR streaming | `1` | Leave on for all reference builds.
| `COVERTREEX_ENABLE_SPARSE_TRAVERSAL` | Switches traversal to scope chunk streamer | `0` | Turn on for scan-cap experiments.
| `COVERTREEX_SCOPE_CHUNK_TARGET` | Cap on scope size and scan budget | `0` (dense) / `8192` (sparse recipes) | Telemetry: `traversal_scope_chunk_*`.
| `COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH` | Gate‑1 lookup table | `docs/data/residual_gate_profile_32768_caps.json` | Refreshed 2025‑11‑10 from run `pcct-20251110-112936-0d2a25`.
| `COVERTREEX_RESIDUAL_GATE1` | Enables runtime gate | `0` | `cli.queries --residual-gate lookup` flips this on plus audit by default.
| `COVERTREEX_CONFLICT_GRAPH_IMPL` | Conflict builder (`grid`, `segmented`, `sparse`) | `grid` | Grid remains the audited baseline.
| `COVERTREEX_BATCH_ORDER` | Batch ordering (`hilbert`, `natural`, …) | `hilbert` | Hilbert used for 32 k gold runs.
| `COVERTREEX_PREFIX_SCHEDULE` | Level advance schedule | `adaptive` (Euclidean) / `doubling` (residual) | Residual path forces `doubling` automatically.

### Reference command set
```
# Dense Euclidean baseline (diagnostics on)
COVERTREEX_BACKEND=numpy \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_ENABLE_DIAGNOSTICS=1 \
COVERTREEX_CONFLICT_GRAPH_IMPL=grid \
COVERTREEX_BATCH_ORDER=hilbert \
python -m cli.queries \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost

# Dense residual control (gate off)
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
python -m cli.queries --metric residual \
  --dimension 8 --tree-points 32768 \
  --batch-size 512 --queries 1024 --k 8 \
  --seed 42 --baseline gpboost \
  --log-file artifacts/benchmarks/residual_dense_<ts>.jsonl

# Sparse residual + streamer + cap 8 192
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=8192 \
python -m cli.queries --metric residual ...

# Sparse residual + gate audit
COVERTREEX_RESIDUAL_GATE1=1 \
COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH=docs/data/residual_gate_profile_32768_caps.json \
python -m cli.queries --metric residual --residual-gate lookup --residual-gate-audit 1 ...

# Gate profile refresh (profiling run)
COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH=docs/data/residual_gate_profile_32768_caps.json \
python -m cli.queries --metric residual --log-file artifacts/benchmarks/residual_sparse_streamer_profile_<ts>.jsonl
```

For scripted legacy regressions, `./benchmarks/run_residual_gold_standard.sh` still produces the natural-order, diagnostics-off log used before the grid refresh; keep it for continuity but prefer the Hilbert recipes above.

### Telemetry expectations (Hilbert 32 k, diagnostics on)
- **Dense residual (gate off).** Run `pcct-20251110-105043-101bb9` reports ≈257 s build / 0.027 s query, `traversal_ms≈1.23 s`, `conflict_graph_ms≈2.51 s`, `traversal_gate1_* = 0`.
- **Sparse residual + streamer.** `pcct-20251110-105526-68dddf` shows ≈493 s build / 0.027 s query, `traversal_ms≈8.65 s`, `conflict_graph_ms≈0.038 s`, `traversal_scope_chunk_saturated=48/64`.
- **Sparse residual + profile capture.** `pcct-20251110-112936-0d2a25` mirrors streamer timings and feeds the lookup refresh.
- **Sparse residual + gate audit.** `pcct-20251110-112456-69dfdd` keeps audit clean with `traversal_gate1_candidates≈2.33×10^8` and `traversal_gate1_pruned=0`; gate-on reruns remain open until pruning appears.

### Euclidean observations (updated 2025-11-10)
- 32 768 / 1 024 / k=8 Hilbert+grid runs land at **16.75 s build / 0.039 s query (~26 k q/s)** for PCCT vs **≈3.10 s build / 15.7 s query** for the GPBoost baseline (NumPy backend, diagnostics on). Logs: `benchmark_grid_32768_baseline_20251108{,_run2}.jsonl`.
- 8 192 / 1 024 / k=16 runs record **4.15 s build / 0.018 s query (~57.7 k q/s)** for PCCT vs **0.75 s / 3.59 s** for GPBoost (`benchmark_grid_8192_baseline_20251108.jsonl`).
- Dominated batches show `conflict_graph_ms≤20 ms` while traversal remains the limiter even with the grid builder; no MIS regressions observed.

### Residual-correlation benchmark status (2025-11-10)
- Dense residual remains the fastest audited build (≈257 s). Use it as the regression baseline until Gate‑1 pruning shows measurable wins.
- Sparse residual + streamer saves ~88 s vs the Python path but is still scan-cap dominated; further cache/prefilter work targets a ≥10 % reduction from the 493 s baseline.
- Gate‑1 lookup refresh completed on November 10 (`docs/data/residual_gate_profile_32768_caps.json`). Gate-on runs are safe but prune zero candidates, so AUDIT follow-up focuses on cache ladders and lookup ingestion tooling.
- Legacy artefacts (`benchmark_residual_clamped_20251107*.jsonl`, `benchmark_residual_scopecap_20251108.jsonl`) remain useful for historical comparisons, but new work should reference the `pcct-20251110-*` logs listed above.

### Key source excerpt (ResidualCorrHostData)
```python
@dataclass(frozen=True)
class ResidualCorrHostData:
    v_matrix: np.ndarray
    p_diag: np.ndarray
    kernel_diag: np.ndarray
    kernel_provider: KernelProvider
    point_decoder: PointDecoder = _default_point_decoder
    chunk_size: int = 512
    v_norm_sq: np.ndarray | None = None
    gate_v32: np.ndarray | None = None
    gate_norm32: np.ndarray | None = None
    gate1_enabled: bool | None = None
    gate1_alpha: float | None = None
    gate1_margin: float | None = None
    gate1_eps: float | None = None
    gate1_audit: bool | None = None
    gate_stats: ResidualGateTelemetry = field(default_factory=ResidualGateTelemetry)
```

`compute_residual_distances_with_radius(...)` invokes the Gate‑1 mask when enabled, records telemetry (`candidates`, `kept`, `pruned`), and falls back to the chunk kernel for survivors. Audit mode re-evaluates pruned rows with the exact kernel to ensure no true neighbours are dropped.

### Lookup & ingestion status
- Default lookup: `docs/data/residual_gate_profile_32768_caps.json` (captured via `pcct-20251110-112936-0d2a25`). Set `COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH` to this file for all reference builds.
- Synthetic fallback: `docs/data/residual_gate_profile_diag0.json` (`tools/build_residual_gate_profile.py`), useful for small regression tests.
- TODO: add a JSONL ingestion helper that converts streamed telemetry into gate profiles automatically (Plan step 2 in `docs/RESIDUAL_PCCT_PLAN.md`).
