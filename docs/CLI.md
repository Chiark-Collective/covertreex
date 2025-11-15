# CLI Reference

`python -m cli.queries` is implemented with [Typer](https://typer.tiangolo.com/) so every runtime knob is accessible via a flag and documented in `--help`. The CLI always writes batch-level telemetry (`artifacts/benchmarks/*.jsonl`) unless `--no-log-file` is passed, ensuring every run is reproducible.

## Usage

```bash
# inspect all options (grouped via Typer help panels)
python -m cli.queries --help

# dense Euclidean sweep
python -m cli.queries --dimension 8 --tree-points 4096 --queries 1024 --k 8

# residual audit with gate lookup and scope cap capture
python -m cli.queries \
  --metric residual \
  --dimension 8 --tree-points 32768 --queries 1024 --batch-size 512 --k 8 \
  --residual-gate lookup \
  --residual-gate-lookup-path docs/data/residual_gate_profile_32768_caps.json \
  --residual-scope-cap-output artifacts/residual_scope_caps.json \
  --baseline both
```

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
python -m cli.queries \
  --profile residual-fast \
  --set diagnostics.enabled=true \
  --set residual.scope_member_limit=32768
```

Overrides use dot-path syntax that mirrors the nested `RuntimeModel`. Values are parsed with YAML
semantics, so `true/false`, numbers, and quoted strings all work as expected.

## Flag groups

| Panel | Purpose | Highlights |
| --- | --- | --- |
| **Benchmark shape** | Controls dataset geometry and build style. | `--dimension`, `--tree-points`, `--batch-size`, `--queries`, `--k`, `--seed`, `--build-mode`. |
| **Runtime controls** | Mirrors `covertreex.api.Runtime` knobs. | `--profile`, `--set PATH=VALUE`, `--backend`, `--precision`, `--device/-d`, `--enable-numba/--disable-numba`, `--conflict-graph`, `--scope-chunk-target`, `--batch-order`, `--prefix-*`, `--mis-seed`. |
| **Residual metric** | Synthetic backend + traversal caps. | `--residual-lengthscale`, `--residual-variance`, `--residual-inducing`, `--residual-stream-tile`, `--residual-scope-member-limit`, `--residual-scope-caps`, `--residual-scope-cap-output`, `--residual-force-whitened`. |
| **Gate & prefilter** | Gate-1/prefilter lookup management. | `--residual-gate` (off/lookup), `--residual-gate-lookup-path`, `--residual-gate-margin/cap`, `--residual-gate-alpha/eps/band-eps`, `--residual-gate-profile-*`, `--residual-prefilter*`. |
| **Telemetry & baselines** | Output paths + comparisons. | `--log-file`, `--no-log-file`, `--baseline` (none/sequential/gpboost/external/both/all). |

Every flag can also be driven via `covertreex.api.Runtime` for programmatic scenarios; the CLI simply exposes them without touching environment variables.

## Telemetry & reproducibility

- By default, runs emit JSONL telemetry containing traversal/conflict timings, scope budgets, kernel vs. whitened counters, MIS iterations, and RSS deltas. Set `--log-file` to control the destination or `--no-log-file` to disable.
- Residual runs can additionally emit per-level scope caps (`--residual-scope-cap-output`) and gate lookup profiles (`--residual-gate-profile-log`).
- The helper `cli.queries.telemetry.initialise_cli_telemetry()` centralises log writer initialisation so the CLI and any future tools share the same behaviour.

Refer to the JSONL schema in `covertreex/telemetry/schemas.py` for column definitions. For a tour of the knobs and when to enable them, see `docs/RESIDUAL_REGRESSION_20251112.md`.
