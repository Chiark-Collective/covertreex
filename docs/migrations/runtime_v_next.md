# Runtime vNext Migration Guide

Stage 7 of the DX/UX plan finalised the profile-first CLI, telemetry schema v2, and deterministic
`SeedPack` threading described in `AUDIT.md`. This document helps existing scripts migrate from the
legacy `cli.queries` entrypoint and `COVERTREEX_*` environment variables to the new surfaces without
losing reproducibility.

## 1. Swap entrypoints

| Legacy command | Replacement |
| --- | --- |
| `python -m cli.queries ...` | `python -m cli.pcct query --profile default ...` |
| `python -m cli.queries ... --no-log-file` | `python -m cli.pcct query --profile default --no-log-file ...` |
| `python -m benchmarks.queries ...` | `python -m cli.pcct query ...` |

`python -m cli.pcct legacy-query` remains available for one release and simply re-exports the old
Typer app. It prints a warning that links back to this guide; use it only as a temporary bridge while
you port automation to the profile workflow.

## 2. Adopt profiles + overrides

1. Inspect curated profiles to find the closest match:

   ```bash
   python -m cli.pcct profile list
   python -m cli.pcct profile describe residual-fast --format yaml
   ```

2. Invoke every CLI command with `--profile NAME`.
3. Apply overrides through dot-path syntax (`--set diagnostics.enabled=false`,
   `--set residual.scope_member_limit=32768`). Values follow YAML parsing rules, so booleans,
   numbers, and quoted strings work as expected.
4. When scripting, mirror the same behaviour by calling
   `Runtime.from_profile("NAME", overrides=[...])` and `runtime.activate()`.

## 3. Map environment variables to overrides

Environment variables still work through `RuntimeModel.from_env()`, but the recommended path is to
express the same settings through CLI flags or dot-path overrides. Use the table below as a cheat
sheet:

| Environment variable | CLI flag | Dot-path override |
| --- | --- | --- |
| `COVERTREEX_BACKEND` | `--backend BACKEND` | `backend=BACKEND` |
| `COVERTREEX_PRECISION` | `--precision float32|float64` | `precision=float32` |
| `COVERTREEX_ENABLE_NUMBA=0/1` | `--enable-numba / --disable-numba` | `enable_numba=true/false` |
| `COVERTREEX_SCOPE_CHUNK_TARGET` | `--scope-chunk-target 2048` | `scope_chunk_target=2048` |
| `COVERTREEX_SCOPE_SEGMENT_DEDUPE` | `--scope-segment-dedupe/--no-scope-segment-dedupe` | `scope_segment_dedupe=true/false` |
| `COVERTREEX_CONFLICT_GRAPH` | `--conflict-graph dense|segmented|grid|auto` | `conflict_graph_impl=grid` |
| `COVERTREEX_RESIDUAL_GATE_LOOKUP_PATH` | `--residual-gate-lookup-path PATH` | `residual.gate1_lookup_path=PATH` |
| `COVERTREEX_RESIDUAL_SCOPE_CAP_PATH` | `--residual-scope-caps PATH` | `residual.scope_cap_path=PATH` |
| `COVERTREEX_GLOBAL_SEED` | `--global-seed N` | `seeds.global=N` |
| `COVERTREEX_BATCH_ORDER_SEED` | `--batch-order-seed N` | `seeds.batch_order=N` |
| `COVERTREEX_MIS_SEED` | `--mis-seed N` | `seeds.mis=N` |

Add new overrides to the CLI with multiple `--set` arguments or a YAML patch file if you prefer
checked-in configurations.

## 4. Telemetry + determinism

- Every `pcct` subcommand now emits a `covertreex.benchmark_batch.v2` JSONL file unless
  `--no-log-file` is passed. The header includes both the `runtime_digest` and the `seed_pack`, so
  identical runs share the same `run_hash`.
- Use `python -m cli.pcct telemetry render LOG.jsonl --format md|csv|json --show fields` to inspect
  runs. The renderer validates schema IDs and produces summaries that can be pasted into audits.
- Prefer setting `--global-seed` (or `seeds.global=...` via overrides) and letting the `SeedPack`
  derive per-channel seeds deterministically. Direct overrides remain available for experiments.

## 5. Verifying the migration

1. Run your existing command with `python -m cli.pcct legacy-query ... --log-file LEGACY.jsonl`.
2. Run the profile-based equivalent.
3. Compare `run_hash` values (or diff the `runtime` snapshots) to confirm the configuration matches.
4. Capture the updated commands and overrides in `docs/examples/profile_workflows.md` or the
   reference scripts for future contributors.

Questions? See `docs/CLI.md` for the full flag reference and `docs/API.md` for runtime context APIs.
