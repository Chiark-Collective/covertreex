# CLI Reference

`pcct` is the [Typer](https://typer.tiangolo.com/) command-line interface for the parallel
compressed cover tree. Invoke it with `python -m cli.pcct ...` (or `uv run python -m cli.pcct`)
to access subcommands for benchmarking and profile inspection. The legacy
`python -m cli.queries` entrypoint still works, but it now prints a compatibility warning before
delegating to `pcct query` so users notice the new surface.

Every run continues to emit batch-level telemetry (`artifacts/benchmarks/*.jsonl`) unless
`--no-log-file` is passed, ensuring reproductions stay audit-friendly.

## Usage

```bash
# inspect the root app and subcommands
python -m cli.pcct --help

# inspect all query options (grouped via Typer help panels)
python -m cli.pcct query --help

# dense Euclidean sweep
python -m cli.pcct query --dimension 8 --tree-points 4096 --queries 1024 --k 8

# residual audit with gate lookup and scope cap capture
python -m cli.pcct query \
  --metric residual \
  --dimension 8 --tree-points 32768 --queries 1024 --batch-size 512 --k 8 \
  --residual-gate lookup \
  --residual-gate-lookup-path docs/data/residual_gate_profile_32768_caps.json \
  --residual-scope-cap-output artifacts/residual_scope_caps.json \
  --baseline both

# build-only mode
python -m cli.pcct build --dimension 8 --tree-points 65536 --batch-size 1024 --profile default

# repeat benchmark runs and summarise latencies
python -m cli.pcct benchmark --repeat 5 --dimension 8 --tree-points 8192 --queries 1024

# run environment diagnostics for a profile
python -m cli.pcct doctor --profile default

# render a telemetry log as Markdown and list the recorded fields
python -m cli.pcct telemetry render artifacts/benchmarks/queries_run.jsonl --format md --show fields

# list registered traversal/conflict/metric plugins
python -m cli.pcct plugins
```

Need the original firehose of runtime flags? `python -m cli.pcct legacy-query ...` re-exports the
legacy Typer app so existing scripts keep working while the new `pcct query` command focuses on
profile-driven ergonomics. `python -m cli.queries ...` remains available for one release as well.

### Profile-first workflow

Stage 7 standardises on the curated YAML profiles under `profiles/`. For most runs:

1. Inspect the available presets (`python -m cli.pcct profile list`, `... profile describe NAME`).
2. Select a profile when invoking any subcommand (`pcct query --profile residual-fast`).
3. Apply dot-path overrides with `--set PATH=VALUE` (strings use YAML syntax, so `true/false`, quoted
   strings, and numbers work).
4. Record telemetry (default; disable via `--no-log-file`) and render it later with
   `pcct telemetry render ... --format md`.

If you previously relied on environment variables, consult `docs/migrations/runtime_v_next.md` for a
mapping between `COVERTREEX_*` settings, CLI flags, and profile override paths.

### Profile inspection

Stage 2 introduced curated YAML profiles (see `profiles/*.yaml`). Use the profile subcommands to
discover them without reading files manually:

```bash
# list each profile with workload/tags metadata
python -m cli.pcct profile list

# show the runtime payload for a specific profile
python -m cli.pcct profile describe residual-fast --format json
```

### Doctor command

`python -m cli.pcct doctor` validates your environment against the selected profile. The command
checks for Numba/JAX availability, verifies that the artifact root is writable (so telemetry can be
emitted), and reports BLAS/Numba thread settings plus platform metadata. When the runtime requests
Numba or JAX features that are missing, `pcct doctor` surfaces warnings; pass
`--fail-on-warning` to turn those into non-zero exit codes for CI.

### Plugin registry

Use `python -m cli.pcct plugins` to see the traversal, conflict, and metric strategies currently
registered. The listing surfaces each plugin’s module and predicate/factory so you can confirm that
entry-point overrides loaded as expected (helpful when downstream packages ship custom strategies).

### Build command

`python -m cli.pcct build` constructs a tree (batch or prefix mode), records telemetry for each batch,
and optionally exports the structure as a `.npz` artifact (`--export-tree`). The command accepts the
same dataset/profile knobs as `pcct query` so you can prebuild large trees before querying them in
other processes.

### Benchmark command

`python -m cli.pcct benchmark` repeats the query workflow multiple times (controlled via
`--repeat`/`--seed-step`) and prints aggregate latency/build summaries. Each iteration still emits
its own telemetry file, while the final summary highlights mean/median/best latencies so you can
spot regressions quickly.

### Runtime activation & contexts

`cli.runtime.runtime_from_args()` mirrors `covertreex.api.Runtime`, so the CLI never mutates
`os.environ`. Each invocation builds a `Runtime` from the flags, activates a `RuntimeContext` once,
and threads that context through telemetry, residual helpers, and benchmarking utilities:

```python
cli_runtime = runtime_from_args(args)
with cli_runtime.activate() as context:
    telemetry = initialise_cli_telemetry(..., context=context)
    benchmark_knn_latency(..., context=context)
```

Helper modules (`cli.queries.telemetry`, `cli.queries.benchmark`, tools under `tools/`) accept a
`context` keyword for every operation so tests and scripts can opt in to explicit, isolated runtime
configuration rather than relying on globals.

### Profiles & overrides

Use `--profile` to start from curated presets (see `profiles/*.yaml`) and `--set PATH=VALUE` to tweak
individual fields without juggling dozens of flags:

```bash
python -m cli.pcct query \
  --profile residual-fast \
  --set diagnostics.enabled=true \
  --set residual.scope_member_limit=32768

# force deterministic seeds via the SeedPack flags
python -m cli.pcct query --profile default --global-seed 123
python -m cli.pcct query --profile residual-fast --residual-grid-seed 777
```

Overrides use dot-path syntax that mirrors the nested `RuntimeModel`. Values are parsed with YAML
semantics, so `true/false`, numbers, and quoted strings all work as expected.
Seed overrides feed the `SeedPack` channels directly (`seeds.global`, `seeds.mis`, `seeds.batch_order`,
`seeds.residual_grid`), making deterministic reruns straightforward.

## Flag groups

| Panel | Purpose | Highlights |
| --- | --- | --- |
| **Benchmark shape** | Controls dataset geometry and build style. | `--dimension`, `--tree-points`, `--batch-size`, `--queries`, `--k`, `--seed`, `--build-mode`. |
| **Runtime controls** | Mirrors `covertreex.api.Runtime` knobs. | `--profile`, `--set PATH=VALUE`, `--backend`, `--precision`, `--device/-d`, `--enable-numba/--disable-numba`, `--conflict-graph`, `--scope-chunk-target`, `--batch-order`, `--prefix-*`, `--global-seed`, `--mis-seed`, `--residual-grid-seed`. |
| **Residual metric** | Synthetic backend + traversal caps. | `--residual-lengthscale`, `--residual-variance`, `--residual-inducing`, `--residual-stream-tile`, `--residual-scope-member-limit`, `--residual-scope-caps`, `--residual-scope-cap-output`, `--residual-force-whitened`. |
| **Gate & prefilter** | Gate-1/prefilter lookup management. | `--residual-gate` (off/lookup), `--residual-gate-lookup-path`, `--residual-gate-margin/cap`, `--residual-gate-alpha/eps/band-eps`, `--residual-gate-profile-*`, `--residual-prefilter*`. |
| **Telemetry & baselines** | Output paths + comparisons. | `--log-file`, `--no-log-file`, `--baseline` (none/sequential/gpboost/external/both/all). |

Every flag can also be driven via `covertreex.api.Runtime` for programmatic scenarios; the CLI simply exposes them without touching environment variables.

## Telemetry & reproducibility

- By default, runs emit JSONL telemetry containing traversal/conflict timings, scope budgets, kernel vs. whitened counters, MIS iterations, and RSS deltas. Set `--log-file` to control the destination or `--no-log-file` to disable.
- Residual runs can additionally emit per-level scope caps (`--residual-scope-cap-output`) and gate lookup profiles (`--residual-gate-profile-log`).
- The helper `cli.queries.telemetry.initialise_cli_telemetry()` centralises log writer initialisation so the CLI and any future tools share the same behaviour. The Typer subcommand `pcct telemetry render LOG.jsonl --format md|csv|json --show fields` renders the resulting JSONL in human-friendly formats.
- `python -m cli.pcct telemetry render ...` turns the JSONL stream into JSON/Markdown/CSV summaries (see `docs/telemetry.md` for schema details and sample output).

Refer to the JSONL schema in `covertreex/telemetry/schemas.py` for column definitions. For a tour of the knobs and when to enable them, see `docs/RESIDUAL_REGRESSION_20251112.md`.

### Legacy compatibility & migration

`python -m cli.pcct legacy-query` exists for one release and simply re-exports the old Typer app so
existing scripts can bridge to the profile-first workflow. It emits a warning describing how to
switch to `pcct query`. See `docs/migrations/runtime_v_next.md` for a step-by-step migration checklist
covering flag/ENV replacements, profile selection, override syntax, and telemetry rendering.
