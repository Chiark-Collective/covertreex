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

## Flag groups

| Panel | Purpose | Highlights |
| --- | --- | --- |
| **Benchmark shape** | Controls dataset geometry and build style. | `--dimension`, `--tree-points`, `--batch-size`, `--queries`, `--k`, `--seed`, `--build-mode`. |
| **Runtime controls** | Mirrors `covertreex.api.Runtime` knobs. | `--backend`, `--precision`, `--device/-d`, `--enable-numba/--disable-numba`, `--conflict-graph`, `--scope-chunk-target`, `--batch-order`, `--prefix-*`, `--mis-seed`. |
| **Residual metric** | Synthetic backend + traversal caps. | `--residual-lengthscale`, `--residual-variance`, `--residual-inducing`, `--residual-stream-tile`, `--residual-scope-member-limit`, `--residual-scope-caps`, `--residual-scope-cap-output`, `--residual-force-whitened`. |
| **Gate & prefilter** | Gate-1/prefilter lookup management. | `--residual-gate` (off/lookup), `--residual-gate-lookup-path`, `--residual-gate-margin/cap`, `--residual-gate-alpha/eps/band-eps`, `--residual-gate-profile-*`, `--residual-prefilter*`. |
| **Telemetry & baselines** | Output paths + comparisons. | `--log-file`, `--no-log-file`, `--baseline` (none/sequential/gpboost/external/both/all). |

Every flag can also be driven via `covertreex.api.Runtime` for programmatic scenarios; the CLI simply exposes them without touching environment variables.

## Telemetry & reproducibility

- By default, runs emit JSONL telemetry containing traversal/conflict timings, scope budgets, kernel vs. whitened counters, MIS iterations, and RSS deltas. Set `--log-file` to control the destination or `--no-log-file` to disable.
- Residual runs can additionally emit per-level scope caps (`--residual-scope-cap-output`) and gate lookup profiles (`--residual-gate-profile-log`).
- The helper `cli.queries.telemetry.initialise_cli_telemetry()` centralises log writer initialisation so the CLI and any future tools share the same behaviour.

Refer to the JSONL schema in `covertreex/telemetry/schemas.py` for column definitions. For a tour of the knobs and when to enable them, see `docs/RESIDUAL_REGRESSION_20251112.md`.
