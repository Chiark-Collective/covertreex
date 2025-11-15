# Residual Gold Rerun Note (2025-11-15)

## Provenance

- The current gold artefact (`artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold.jsonl`, `run_id=pcct-20251114-214845-7df1be`) was introduced in commit `a0402c7` (“Add guardrail tooling and overview report”, 2025-11-15). The later `stable` tag (`e128a31`) simply refreshed the documentation to point at the same run.

## Reproduction attempts

1. **Typer CLI (`pcct query …`)** — Ran the dense preset via the new surface (`--metric residual`, Hilbert batches, dense scope streamer, etc.) but left the conflict backend at its default (`grid`) and sparse traversal disabled. As a result, `conflict_scope_chunk_pair_merges` stayed at zero and traversal averaged ≈340 ms/batch. Wall clock via `time` landed at **27.439 s** (`run_id=pcct-20251115-195422-e03d5a`, log `artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold_rerun2.jsonl`). This configuration mismatch explains the slower build.

2. **Legacy `cli.queries` command** — Replayed the exact recipe documented alongside the gold log (env vars for dense conflict graph, sparse traversal on, pair-count shard merging + buffer reuse). `time` reported **25.877 s** (`run_id=pcct-20251115-195931-184755`, log `artifacts/benchmarks/residual_dense_32768_dense_streamer_pairmerge_gold_cli_rerun3.jsonl`). Telemetry matches the gold configuration (same shard-merge heuristics, Hilbert batches, diagnostics on), but traversal_ms averages ≈382 ms/batch versus ≈256 ms in the November 14 artefact, so the slower wall clock appears to be host/runtime variance rather than a code change.

## Takeaway

- The 17.8 s “gold” artefact remains valid; it was captured on `a0402c7` with the command embedded in `docs/CORE_IMPLEMENTATIONS.md`. Our Typer rerun differed in configuration, and even the corrected legacy rerun shows the same telemetry but higher traversal timings on this machine.
- Keep referencing the original command/env combo when comparing regressions, and prefer the legacy `cli.queries` shim (or add equivalent toggles to the Typer surface) if you need to match the gold preset byte-for-byte. For future reruns, note that host load can swing traversal medians considerably, so capture CPU affinity/thermal data if exact wall clock parity is required.
