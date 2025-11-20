# Residual PCCT Build Optimisation — Living Plan

> **Note:** The "Sparse Gated" strategy (Residual Gate 1) described in this plan was implemented, evaluated, and found ineffective (0% pruning). It has been removed from the codebase as of Nov 2025 in favor of an optimized "Dense" baseline. See `docs/RESIDUAL_GATE_POSTMORTEM.md` for details.

_Last updated: 2025-11-10_

This file tracks ongoing work on the residual-correlation PCCT build pipeline. It captures the current state, recent experiments, and the next concrete steps so anyone can pick up the thread quickly.

## Current State (2025-11-10)

- **Dense control (gate off).** `python -m cli.queries --metric residual --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --baseline gpboost --seed 42` (`pcct-20251110-105043-101bb9`, log `artifacts/benchmarks/artifacts/benchmarks/residual_dense_20251110115043.jsonl`) delivers **≈257 s build / 0.027 s query (37.6 k q/s)** with diagnostics on. `traversal_gate1_*` counters remain zero because dense traversal never consults the lookup.
- **Sparse traversal + Numba scope streamer.** `_collect_residual_scopes_streaming()` is now fully Numba-backed and writes directly into the bucketed CSR builder. Run `pcct-20251110-105526-68dddf` (log `.../residual_sparse_streamer_20251110115526.jsonl`) shows per-batch `traversal_ms≈8.65 s` (down from 30 s+) and total build wall-clock **≈493 s**—an 80 s drop vs the previous Python-bound streamer. 48/64 dominated batches still saturate the 8 192 scan cap, so sparse remains slower than dense.
- **Adaptive scope budgets.** `COVERTREEX_SCOPE_BUDGET_*` now drives the streaming traversal. Hilbert 32 k runs emit `traversal_scope_budget_{start,final,escalations,early_terminate}` telemetry, and capped scans stop early once two consecutive survivor ratios fall below the configured `DOWN_THRESH`. For the 4 096-point sanity sweep we parsed `residual_{dense,sparse}_budget_4096.jsonl` with `python tools/export_benchmark_diagnostics.py --output artifacts/benchmarks/residual_budget_diagnostics.csv ...`: dense residual builds land at **76.49 s** with budgets disabled, while sparse + `scope_chunk_target=4096` finishes in **103.37 s** and reports average budget amplification `3.73×` (14.68 M final vs 3.93 M start) plus ≥93.7 % pairwise reuse after the first batch. That CSV is the artefact we hand off to perf reviewers.
- **Residual pairwise reuse enforced.** `build_conflict_graph` now refuses to run without the cached pairwise block from traversal. CLI prints a one-line warning if a residual build somehow loses the cache, and `conflict_pairwise_reused` stays `1` for every residual batch.
- **Gate‑1 lookup refresh (opt-in only).** `docs/data/residual_gate_profile_32768_caps.json` was regenerated from the sparse profile capture `pcct-20251110-112936-0d2a25`, aligning the lookup with the current 32 k radius ladder. Gate‑1 now ships **disabled by default**; set `COVERTREEX_RESIDUAL_GATE1=1` (or use `--residual-gate ...`) to experiment with the lookup.
- **Gate‑on audit.** Validation run `pcct-20251110-112456-69dfdd` (same sparse config, lookup enabled, audit on) stayed clean: `traversal_gate1_candidates≈2.33×10^8`, `traversal_gate1_pruned=0`. Because pruning headroom remains unproven, the gate is paused as a default optimisation and tracked as a follow-up experiment only.
- **Lookup ingestion helper.** `cli.queries` now exposes `--residual-gate-profile-log` (plus `--residual-gate-profile-path/bins`) so any residual run can stream its Gate‑1 samples to JSONL, and `tools/ingest_residual_gate_profile.py` merges those telemetry files into a lookup JSON without replaying the build. The refreshed `docs/data/residual_gate_profile_32768_caps.json` was regenerated through this flow for traceability.

## Recent Experiments

1. **Grid scale sweep (0.75 vs 1.50) — 2025-11-09.** Adjusting `COVERTREEX_RESIDUAL_GRID_WHITEN_SCALE` nudged grid telemetry but left traversal_ms unchanged (0.7–1.4 s per dominated batch). Logs: `residual_grid_scale0p75_20251109180103.jsonl`, `residual_grid_scale1p50_20251109180226.jsonl`.
2. **Gate lookup stress (pre-refresh).** Margin sweeps (0.02, 0.50) failed the audit immediately; disabling auditing produced runs with `traversal_semisort_ms>30 s` and only ~1.4 % pruning. Logs: `residual_gate_lookup_sparse_scale1p00_*.jsonl`.
3. **Bucketed CSR streaming.** The new `_collect_residual_scopes_streaming()` emits `(owner, member)` pairs straight into the CSR builder via Numba, keeping sparse and dense outputs byte-identical (tests cover both paths). `pcct-20251110-105526-68dddf` is the canonical telemetry sample.
4. **Lookup refresh (AUDIT.C).** Profile capture `pcct-20251110-112936-0d2a25` and audit `pcct-20251110-112456-69dfdd` closed the lookup blocker. `docs/residual_metric_notes.md` and this plan now document provenance.

## Next Steps

1. **Adaptive sparse scan stop (AUDIT §2).** Implement the budget-based early-exit described in `AUDIT.md` so `_collect_residual_scopes_streaming()` can end hopeless scans once the survivor rate collapses. Success: non-zero `traversal_scope_budget_early_terminate` and ≥10 % drop in `traversal_scope_chunk_saturated` on Hilbert 32 k runs.
2. **Residual pairwise reuse (AUDIT §3A).** Thread the cached pairwise blocks from traversal into every adjacency/radius path so we never recompute kernels. Add `conflict_pairwise_reused` telemetry and fail loudly when the cache is missing.
3. **Gate experiments (back-burner).** Keep the lookup tooling ready, but only revisit once (1) and (2) land. When we do, regenerate the 32 k profile with quantile envelopes and rerun sparse Hilbert with audits to check for `traversal_gate1_pruned>0`.

## Handy Commands

```bash
# Dense baseline (diagnostics on)
python -m cli.queries \
  --dimension 8 --tree-points 32768 --batch-size 512 \
  --queries 1024 --k 8 --metric residual \
  --baseline gpboost --seed 42 \
  --log-file artifacts/benchmarks/residual_dense_<ts>.jsonl

# Sparse traversal + streamer + cap
COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 \
COVERTREEX_ENABLE_NUMBA=1 \
COVERTREEX_SCOPE_CHUNK_TARGET=8192 \
python -m cli.queries --metric residual ...

# Refresh gate profile from runtime log
COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH=docs/data/residual_gate_profile_32768_caps.json \
python -m cli.queries --metric residual ... --log-file artifacts/benchmarks/residual_sparse_streamer_profile_<ts>.jsonl

# Synthetic fallback (until ingestion tooling exists)
python tools/build_residual_gate_profile.py docs/data/residual_gate_profile_diag0.json \
  --tree-points 2048 --dimension 8 --seed 42 --bins 512 --pair-chunk 64
```

## 2025-11-10 — Sparse Streamer Benchmarks & CPU Utilisation Notes

### Runs captured

- **Dense control (pcct-20251110-105043-101bb9).** `python -m cli.queries --metric residual --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --baseline gpboost --seed 42 --log-file artifacts/benchmarks/artifacts/benchmarks/residual_dense_20251110115043.jsonl`
  - Build wall-clock: 257 s; per-batch `traversal_ms≈1.23 s`, `traversal_semisort_ms≈0.32 s`, `conflict_graph_ms≈2.51 s`.
  - Scope chunk telemetry stays zero; dense traversal ignores `scope_chunk_target`. Gate counters remain zero.
- **Sparse streamer + cap (pcct-20251110-105526-68dddf).** `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=1 COVERTREEX_ENABLE_NUMBA=1 COVERTREEX_SCOPE_CHUNK_TARGET=8192 python -m cli.queries --metric residual ... --log-file artifacts/benchmarks/artifacts/benchmarks/residual_sparse_streamer_20251110115526.jsonl`.
  - Build wall-clock: 493 s (down from ~581 s pre-streamer). Median `traversal_ms≈8.65 s`, `conflict_graph_ms≈0.038 s`.
  - Still scan-cap dominated: 48/64 batches report `traversal_scope_chunk_saturated=1`.
- **Sparse streamer + profile capture (pcct-20251110-112936-0d2a25).** Same config; used to refresh the lookup and archive per-level radius ladders.
- **Sparse streamer + gate (pcct-20251110-112456-69dfdd).** Gate audit stayed clean; `traversal_gate1_candidates≈2.33×10^8`, `traversal_gate1_pruned=0`. Confirms lookup alignment but shows no pruning yet.
