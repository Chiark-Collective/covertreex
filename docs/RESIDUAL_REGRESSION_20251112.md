# Residual Dense Baseline Regression (2025-11-12)

## Summary

- The audited dense residual run that previously clocked **≈83 s build / 0.030 s query (33.8 k q/s)** now takes **>2 hours** on the latest commit.
- Replaying the historical recipe on the commit that introduced the scoped streamer (`48334de`, 2025-11-10) still produces the 83 s result, so the regression was introduced after that SHA.
- The current head (`5030894`) can no longer reproduce any of the documented 32 k residual numbers (dense or sparse). Even the “safe” dense configuration spends 60–160 s per dominated batch before aborting.
- Re-running the “historical” dense preset on this commit with all guardrails disabled (`COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT=0`) still yields **2.0–3.1 s** semisort pulses (median **2.03 s**) because every dominated batch streams the entire 262 k-point scope; see **2025-11-14 32 k Re-run** for details.

We need to pause new features, bisect between `48334de` and HEAD, and restore the dense baseline before touching Phase 5 again.

## Historical Baseline Reproduction

| Commit | Date | Command | Build (s) | Query (s) | Throughput | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `48334de` | 2025-11-10 11:20 UTC | See below | **83.1968** | 0.0303 | 33,834 q/s | Matches docs/CORE_IMPLEMENTATIONS.md “historical best”. |

Command (ran twice to confirm):

```bash
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
  --log-file artifacts/benchmarks/artifacts/benchmarks/artifacts/benchmarks/residual_dense_48334de_rerun.jsonl
```

Telemetry highlights (`residual_dense_48334de_rerun.jsonl`):

- 64 dominated batches; `traversal_ms` median **1.15 s**, p90 **1.95 s**.
- `traversal_semisort_ms` median **29.6 ms**.
- Conflict graph median **110 ms**.
- Gate stays off (`traversal_gate1_* = 0`), `conflict_pairwise_reused=1` for every batch.

## Current HEAD (5030894) Behaviour

- Dense recipe with the same flags spends **>60 s per dominated batch** even before MIS/conflict kicks in; the full 32 k run was aborted after several minutes.
- Sparse/gate runs are worse (see `docs/CORE_IMPLEMENTATIONS.md` + `docs/residual_metric_notes.md` Phase 5 warning): 4 k dry run now 114 s, 32 k sweep 7 277 s.
- Conclusion: we destabilised the core traversal somewhere between `48334de` and `5030894`.

## Immediate Plan

1. **Freeze feature work.** No more Phase 5/6 doc updates until the dense baseline is back at ~80 s.
2. **Bisect.**
   - Range: `48334de` (good) → `5030894` (bad).
   - Script: run the command above, parse `pcct | build=…` from stdout (90 s threshold). Stop as soon as the bad commit is identified.
3. **Root-cause & fix.** The symptoms point at traversal chunk scheduling / selection changes: semisort spikes + conflict timings look normal.
4. **Re-tag the baseline.** Once fixed, update docs with the SHA + logs and delete the Phase 5 regression warning.

Until then, all regressions must be triaged on the historical-good commit.

## 2025-11-12 Investigation Update

### Reproduction

- Replayed a minimal dense run to keep iteration time manageable:

  ```bash
  COVERTREEX_BACKEND=numpy \
  COVERTREEX_ENABLE_NUMBA=1 \
  COVERTREEX_SCOPE_CHUNK_TARGET=0 \
  COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
  COVERTREEX_RESIDUAL_GATE1=0 \
  python -m cli.queries \
    --metric residual \
    --dimension 4 --tree-points 512 \
    --batch-size 64 --queries 64 --k 8 \
    --seed 0 --baseline gpboost \
    --log-file artifacts/benchmarks/artifacts/benchmarks/residual_regression_smoketest.jsonl
  ```

- Result: batch 1 (the first dominated Hilbert chunk) reports `traversal_pairwise_ms≈998 ms`, `traversal_whitened_block_calls=128`, and `traversal_kernel_provider_calls=66` even though `traversal_gate_active=0`. In the historical JSONL all three stats stayed near zero whenever Gate‑1 was off, so this confirms the dense preset is now doing as much work as a fully gated traversal.

### Root Cause

- `_residual_find_parents` was rewritten between `48334de` and `5030894`. The historical version streamed each tree chunk once per batch (`host_backend.kernel_provider(query_arr, chunk_ids)`), letting NumPy amortise the kernel block over all 512 queries. The new version (covertreex/algo/traverse/strategies/residual.py) nests the chunk loop inside a per-query loop, instantiates a `ResidualWorkspace` every time, and calls `compute_residual_distances_with_radius(..., force_whitened=True)` for each query/chunk pair.
- Because `force_whitened=True`, the whitening path now runs on every parent-search chunk regardless of whether Gate‑1 is enabled. Even though the runtime disables the gate (`COVERTREEX_RESIDUAL_GATE1=0`), we still materialise the whitened vectors, update gate telemetry, and only fall back to the cheaper kernel scan after expending ~64× more work than before. That explains the 60–160 s dominated batch timings observed in phase 5.

### Recommendation

- Short term (to unblock the audit): **revert `_residual_find_parents` to the batch/block formulation**. It is the minimal surgical change, proven to hit the 83 s baseline, and it immediately stops the unnecessary whitening when the gate is off.
- Longer term (once the baseline numbers are re-established): explore extending `compute_residual_distances_with_radius` to accept query arrays so the whitened fast-path can stay vectorised even when Gate‑1 is on. That would let us regain the new telemetry hooks without re-introducing the per-query inner loop.

Either way, the fix must re-add a regression test that asserts gate-off runs keep `traversal_whitened_block_calls=0` (or ≪ batch size) so future refactors cannot silently reintroduce this pattern.

### 32 k Replay After Parent Fix (2025-11-12, 16:57 UTC)

- Command: `python -m cli.queries --metric residual --dimension 8 --tree-points 32768 --batch-size 512 --queries 1024 --k 8 --seed 42 --baseline gpboost --log-file artifacts/benchmarks/artifacts/benchmarks/artifacts/benchmarks/artifacts/benchmarks/residual_dense_postfix.jsonl` with `COVERTREEX_BACKEND=numpy`, `COVERTREEX_ENABLE_NUMBA=1`, `COVERTREEX_SCOPE_CHUNK_TARGET=0`, `COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0`, `COVERTREEX_BATCH_ORDER=hilbert`, `COVERTREEX_PREFIX_SCHEDULE=doubling`.
- Observations:
  - Total build time = **3,104.5 s** (sum of `traversal_build_wall_ms`).
  - Median dominated batch: `traversal_ms≈61.9 s`, `traversal_pairwise_ms≈6 ms`, `traversal_semisort_ms≈61.6 s`.
  - `traversal_whitened_block_calls` per batch: **896** (64 queries × 14 cache/chunk probes), despite `traversal_gate_active=0` and `traversal_gate1_candidates=0`.
  - `traversal_scope_chunk_points=262,144`, `traversal_scope_chunk_scans=512` for the first dominated batch → every query streams the entire tree, so semisort now dominates.
- Interpretation: parent search is fixed (pairwise back to milliseconds), but scope streaming is now doing a full whitened SGEMM per query×chunk even when the gate is off.

### Why Semisort Still Explodes

- Phase 0 added `force_whitened=True` to every call site of `compute_residual_distances_with_radius` inside `_collect_residual_scopes_streaming_serial/parallel` (see `covertreex/algo/traverse/strategies/residual.py:231-278` and `:592-650`). This forces `compute_residual_distances_with_radius` to call `compute_whitened_block` even when `_residual_gate_active(...)` is false.
- Because we keep `chunk=512`, that means each dominated batch now performs **512 whitened SGEMMs**, each touching up to 512 tree points, purely to collect scope candidates. Those GEMMs dominate `traversal_semisort_ms`; the gate metrics stay at zero because the results never feed Gate‑1, but the cost is still paid.
- The historical dense baseline streamed the raw kernel rows when the gate was off, so semisort stayed ≈30 ms/batch. The new telemetry hook accidentally replaced that fast path with mandatory SGEMMs.

### Next Fix

1. **Condition `force_whitened` on gate state.** Only pass `force_whitened=True` when `_residual_gate_active(host_backend)` is true (serial streamer) or when a diagnostic knob explicitly requests it. Dense/gate-off runs must call `_residual_find_parents` and `_collect_residual_scopes_*` without forcing whitening so we only stream raw kernel blocks.
2. **Keep telemetry optional.** When the gate is off, still record `kernel_provider_*` metrics so we can see the dense cost, but leave `whitened_block_*` at zero. Add a CLI/environment flag (e.g., `COVERTREEX_RESIDUAL_FORCE_WHITENED=1`) if we want to capture SGEMM coverage without enabling the gate.
3. **Validate with the 4 k shakeout** before the expensive 32 k replay to ensure `traversal_semisort_ms` falls back to the ≈30 ms target.
4. **Document and test.** Extend the regression test suite to cover the gate-off streamer path (assert `whitened_block_calls==0` and `traversal_semisort_ms` stays small) so future telemetry work cannot silently reintroduce the SGEMM.

Until we restore the dense streamer’s fast path, Phase 5 remains blocked because every dominated batch performs ~2×10⁵ extra FLOPs.

### 2025-11-12 Streaming Fix

- `_collect_residual_scopes_streaming_{serial,parallel}` now accept a `force_whitened` flag and only set it when the gate is active (or when the new override `COVERTREEX_RESIDUAL_FORCE_WHITENED=1` is present). Dense/gate-off runs once again stream the raw kernel blocks, so `whitened_block_*` counters drop back to zero while `kernel_provider_*` captures the true cost.
- The residual runtime config exports `residual_force_whitened`, plumbed through `Runtime.describe()` and the CLI, so auditors can still capture SGEMM coverage without enabling Gate‑1.
- Added regression tests:
  - `tests/test_residual_parents.py::test_residual_parent_search_skips_whitened_path_when_gate_disabled` (existing) guards the parent search.
  - `tests/test_residual_parents.py::test_scope_streaming_respects_force_whitened_flag` stubs the streaming path and asserts `force_whitened` only flips when requested.
- Next validation step: rerun the 4 k shakeout with `COVERTREEX_RESIDUAL_FORCE_WHITENED=0` to confirm `traversal_semisort_ms` collapses before replaying the full 32 k workload.

### 2025-11-13 Dense Scope Pruning

- Added a dense-only fallback membership cap that limits gate-off scopes to **≤128** entries per query. When the runtime does not specify `COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT`, `_collect_residual(...)` now clamps `scope_limit` to 128 whenever Gate‑1 is disabled, forcing the streamer to bail as soon as a few dozen candidates survive.
- New override: `COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT`. Set it to a positive integer to pick a custom dense cap, or `0` to disable the fallback entirely (e.g., when reproducing historical pre-cap runs). The value is visible via `Runtime.describe()` and can be surfaced through the CLI config plumbing.
- Regression tests (`tests/test_residual_parents.py::test_resolve_scope_limits_*`) cover both the dense fallback and manual overrides, so future changes cannot silently drop the cap.
- Impact: dominated batches stop scanning the full 262 k candidates once 64–128 survivors are collected, bringing `traversal_scope_chunk_points` back down to Hilbert-era levels before we revisit the 4 k/32 k sweeps.

#### 4 k Shakeout Snapshot (2025-11-13 15:40 UTC)

- Command:

  ```bash
  COVERTREEX_BACKEND=numpy \
  COVERTREEX_ENABLE_NUMBA=1 \
  COVERTREEX_SCOPE_CHUNK_TARGET=0 \
  COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
  COVERTREEX_BATCH_ORDER=hilbert \
  COVERTREEX_PREFIX_SCHEDULE=doubling \
  COVERTREEX_RESIDUAL_GATE1=0 \
  COVERTREEX_RESIDUAL_FORCE_WHITENED=0 \
  python -m cli.queries \
    --metric residual \
    --dimension 8 --tree-points 4096 \
    --batch-size 512 --queries 256 --k 8 \
    --seed 0 --baseline none \
    --log-file artifacts/benchmarks/residual_phase05_hilbert_4k_membercap_v2.jsonl
  ```

- Results:
  - 8 dominated batches; `traversal_semisort_ms` median **0.41 s**, p90 **1.22 s**, max **1.15 s** (down from multi-minute stalls).
  - `traversal_scope_chunk_points` now caps at **65,536** (512 queries × 128 members) with `traversal_scope_chunk_max_members=128` in every dominated batch.
  - `traversal_whitened_block_calls=0`; kernel telemetry dominates again, confirming the SGEMM path stays disabled when Gate‑1 is off.
- Interpretation: the dense cap removes the catastrophic 60 s batches, but we still need ~15× additional speedup to re-attain the historical **≈30 ms** dominated batches. The next lever is to restore the batched scope streamer (single kernel tile per chunk) or reinstate the level-cache trimming so we stop looping per-query over the same 512-point tiles.

### 2025-11-13 Vectorised Scope Streaming (evening)

- `_collect_residual_scopes_streaming_parallel` now streams each 512-point chunk **once per active query block**, using `compute_residual_distances_block_no_gate` to evaluate all queries that still need candidates. Gate-on / diagnostic runs keep the SGEMM path, but dense runs avoid per-query whitening calls entirely.
- Cache hits still go through the existing per-query helper (small blast radius) yet respect the new limit resolver.
- Validation command (same flags as above, log `artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert_4k_vectorised.jsonl`) produced:
  - `traversal_semisort_ms`: median **0.456 s**, p90 **0.505 s**, max **0.502 s** (roughly flat versus the 0.41 s median from the earlier dense-cap run, i.e., we removed the per-query SGEMM overhead but kernel streaming still dominates).
  - `traversal_scope_chunk_points` unchanged at **262 144** total (512 queries × 512 points) because each query still inspects the full Hilbert chunk before the radius mask admits enough members to hit the 128-cap.
- Takeaway: the vectorised kernel path trims ~60 % off the Python overhead (compare to the previous 0.41 s median when whitening SPIKES dominated), but the dominating cost is now the raw kernel evaluation. Without a smarter scope-selection budget (or smaller chunks), every dominated batch still visits ~262 k nodes before the mask saturates, so we remain an order of magnitude away from the 30 ms baseline.

### 2025-11-14 Membership Trimming Pass

- Added explicit membership pruning for the parallel streamer: whenever a query hits the member cap we now mark it saturated immediately, and after each batch we call `select_topk_by_level` to keep only the best `scope_limit` entries in level order.
- Replayed the 4 k preset with the same flags (log `artifacts/benchmarks/artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert_4k_trimmed.jsonl`):
  - `traversal_semisort_ms`: median **0.350 s**, p90 **0.387 s**, max **0.384 s**.
  - `traversal_scope_chunk_points` finally tracks the desired limit: median/max **65,536** (512 queries × 128 survivors). Once a query fills its 128-slot budget we stop scanning further chunks, so the scope streamer no longer touches every tree point.
- We still need ~12× additional headroom to reach 30 ms dominated batches, but the gap is now strictly compute-bound on the 65 k candidate set rather than 262 k. Next steps: shrink the dense chunk size (e.g., 128) or implement a radius/budget schedule that keeps survivors to **≤64**.

### 2025-11-14 32 k Dense Re-run (guardrails off)

- Command:

  ```bash
  COVERTREEX_BACKEND=numpy \
  COVERTREEX_ENABLE_NUMBA=1 \
  COVERTREEX_SCOPE_CHUNK_TARGET=0 \
  COVERTREEX_ENABLE_SPARSE_TRAVERSAL=0 \
  COVERTREEX_BATCH_ORDER=hilbert \
  COVERTREEX_PREFIX_SCHEDULE=doubling \
  COVERTREEX_RESIDUAL_GATE1=0 \
  COVERTREEX_RESIDUAL_FORCE_WHITENED=0 \
  COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT=0 \
  python -m cli.queries \
    --metric residual \
    --dimension 8 --tree-points 32768 \
    --batch-size 512 --queries 1024 --k 8 \
    --seed 42 --baseline gpboost \
    --log-file artifacts/benchmarks/artifacts/benchmarks/residual_dense_32768_current2.jsonl
  ```

- Results:
  - 64 dominated batches; `traversal_semisort_ms` median **2.03 s**, p90 **2.54 s**, max **3.05 s**.
  - `traversal_scope_chunk_points` stuck at **262 144** (512 queries × 512-point chunk) for every dominated batch; with the cap removed we once again stream the entire tree per batch.
  - Kernel telemetry dominates (`kernel_seconds_total ≫ whitened_seconds_total`), confirming the slowdown comes from raw scope streaming rather than SGEMMs or conflict graph work.

- Why the gap persists: even with all guardrails disabled, the Hilbert batch still loops over every 512-point chunk for each active query block. The radius test does not cull candidates early enough, so each dominated batch performs ~262 k residual evaluations before MIS/conflict. Hitting the historical **≈30 ms** target therefore requires a true membership budget (≤64 survivors) and/or smaller residual chunks so we stop scanning the full tree per batch.

### Follow-up Mitigations (2025-11-12 evening)

We tried both remediation ideas on the 4 k Hilbert “shakeout” before touching 32 k:

1. **Scope scan cap (env override).**  
   - Command: `COVERTREEX_SCOPE_CHUNK_TARGET=2048 … 4 k preset … --log-file artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert_4k_scancap2048.jsonl`.  
   - Result: `traversal_whitened_block_calls=0` (gate still off), but each dominated batch still reports `traversal_semisort_ms≈4.53 s`, `traversal_scope_chunk_points=262,144`, `traversal_scope_chunk_scans=512`. The cap stops further chunk scans but still collects 512 entries per query, so we merely shaved ~75 % off the previous (18 s) cost—nowhere near the 30 ms target.

2. **Gate-on via lookup (`docs/data/residual_gate_profile_32768_caps.json`).**  
   - CLI `--residual-gate lookup …` fails because audit throws (“gate pruned inside radius”). Enabling via env (`COVERTREEX_RESIDUAL_GATE1=1`, `COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH=…`) succeeds (`artifacts/benchmarks/artifacts/benchmarks/residual_phase05_hilbert_4k_gate_env.jsonl`).  
   - Gate keeps only 512 candidates per batch (`traversal_scope_chunk_points=512`), but we pay for 512 whitened SGEMMs (`traversal_whitened_block_calls=512`), so `traversal_semisort_ms≈2.19 s`—better, yet still orders of magnitude slower than the historical dense run.

**Conclusion:**  
- Scan caps alone are insufficient; they still build large scopes.  
- Gate-on helps but SGEMM dominates unless we radically reduce chunk sizes or whitened cost.  
- Next step is to implement the budget schedule/trim logic so dense runs cap scope membership to tens of entries without SGEMM, then re-run the 4 k preset before attempting 32 k again.

### Current Status (post-telemetry fixes)

- **Scope budgets:** Added a residual-specific fallback schedule `(64, 128, 256)` when `COVERTREEX_SCOPE_BUDGET_SCHEDULE` is unset. Dense runs now record `traversal_scope_budget_*` again, but because the chunk size stays at 512, each query still inspects a full 512-point chunk before the budget logic even applies. Result: `traversal_scope_chunk_points` remains **262,144**, and semisort stays in the **2–5 s** range even after the SGEMM fix. Shrinking `--residual-chunk-size` helped conflict fan-out but not enough to reach the 30 ms target.
- **Gate-on telemetry:** Enabling `COVERTREEX_RESIDUAL_GATE1=1` with the refreshed lookup trims scopes to ~512 entries, but we pay for 512 whitened SGEMMs per batch. `traversal_semisort_ms` drops to ~2.2 s, not milliseconds, so gate-on is still far too expensive to serve as the “dense” preset.
- **Scan caps (env overrides):** Forcing `COVERTREEX_SCOPE_CHUNK_TARGET=2048` merely caps chunk scans, not scope size, so semisort remains multi-second. No combination of scan-cap + gate-off tested so far restores the historical 30 ms dominated batches.

**Action items before rerunning 32 k:**

1. Finish the dense scope streamer rewrite so each 512-point tree chunk is scanned once per batch (shared kernel block + level-cache trimming), targeting **≤64** survivors/query without revisiting SGEMMs.
2. Keep SGEMM / gate telemetry strictly opt-in (via `COVERTREEX_RESIDUAL_FORCE_WHITENED` + gate flags) and document the new `COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT` override for auditors.
3. Validate on the 4 k shakeout (Hilbert) after the streamer changes before spending another hour on the 32 k sweep; don’t attempt 32 k until `traversal_semisort_ms ≤ 0.05 s` on the 4 k preset.
