# Residual Dense Baseline Regression (2025-11-12)

## Summary

- The audited dense residual run that previously clocked **≈83 s build / 0.030 s query (33.8 k q/s)** now takes **>2 hours** on the latest commit.
- Replaying the historical recipe on the commit that introduced the scoped streamer (`48334de`, 2025-11-10) still produces the 83 s result, so the regression was introduced after that SHA.
- The current head (`5030894`) can no longer reproduce any of the documented 32 k residual numbers (dense or sparse). Even the “safe” dense configuration spends 60–160 s per dominated batch before aborting.

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
