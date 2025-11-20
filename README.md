# Covertreex

Parallel compressed cover tree (PCCT) library engineered for Vecchia-style Gaussian process pipelines. The current focus is an efficient **CPU + NumPy/Numba** implementation; GPU/JAX execution has been intentionally disabled until the CPU path meets the audited performance envelope. This repository is the implementation companion to `PARALLEL_COMPRESSED_PLAN.md`.

> Status: housekeeping phase. Expect rapid iteration across runtime knobs, telemetry, and traversal optimisations, but avoid destructive changes (see `AGENTS.md`).

## Getting Started

Within a Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

The default backend is `numpy` (CPU). Optional acceleration hooks leverage `numba` when the `numba` extra is installed.

The Typer-powered benchmark CLI surfaces every runtime knob so runs are reproducible and discoverable via curated profiles:

```bash
python -m cli.pcct --help

# List curated profiles and inspect their payloads
python -m cli.pcct profile list
python -m cli.pcct profile describe residual-gold --format json

# Profile-driven residual sweep with overrides
python -m cli.pcct query \
  --profile residual-gold \
  --dimension 8 --tree-points 32768 --queries 1024 \
  --batch-size 512 --k 8 --baseline both \
  --set diagnostics.enabled=true \
  --set residual.scope_member_limit=32768

# Build-only (no query phase)
python -m cli.pcct build --dimension 8 --tree-points 65536 --batch-size 1024 --profile default

# Aggregate multiple runs and summarise latency
python -m cli.pcct benchmark --repeat 5 --dimension 8 --tree-points 8192 --queries 1024

# Environment guardrails
python -m cli.pcct doctor --profile default
```

`python -m cli.queries` remains available for one release and issues a compatibility warning
before dispatching to `pcct query`, and `python -m cli.pcct legacy-query ...` exposes the full
legacy flag firehose for scripts that still depend on it. For a guided checklist that maps legacy
flags/environment variables to the new profile-driven workflow, see
`docs/migrations/runtime_v_next.md`. Every invocation writes JSONL telemetry under
`artifacts/benchmarks/` unless `--no-log-file` is passed, making audits deterministic, and
`python -m cli.pcct telemetry render ...` can turn those logs into JSON/Markdown/CSV summaries.

## Profile-driven workflows

Profiles under `profiles/*.yaml` capture the supported runtime combinations (default dense,
residual-fast, audit, CPU-debug, etc.). Use `Runtime.from_profile("default", overrides=[...])` or the
CLI snippets above to load one configuration and tweak dot-path overrides in a reproducible way. The
new examples in `docs/examples/profile_workflows.md` illustrate how to:

- Build and query with the same `RuntimeContext` from Python without mutating globals.
- Record telemetry for every batch and render summaries for audit trails.
- Derive deterministic `SeedPack` values (global, batch order, MIS, residual grid) so multiple runs
  can be compared by hash alone.

## Runtime Controls

The `covertreex.api.Runtime` façade now mirrors every CLI flag (backend, precision, scope caps, residual gates, prefilters, stream tiling, etc.). For non-CLI consumers, configuration can still be driven by environment variables read during import, but the CLI is the source of truth for supported knobs. Highlights:

- `Runtime(...).activate()` installs an in-process configuration without mutating `os.environ`.
- All residual settings (membership caps, forced whitening, etc.) are addressable via CLI flags or `Runtime` keyword arguments.
- CLI runs always emit batch-level telemetry (`BenchmarkLogWriter`) unless `--no-log-file` is passed, so reproductions include scope budgets, kernel/whitened counters, and resource snapshots. Render those logs with `pcct telemetry render ...` for Markdown/JSON/CSV views.
- Seed handling is unified through `Runtime.seeds` / `SeedPack`; reuse the same values (or `--global-seed`) to regenerate runs that should match byte-for-byte.

Residual metrics remain lazy: call `configure_residual_correlation(...)` after import to supply Vecchia backends; `reset_residual_metric()` clears hooks for tests. Use `covertreex.config.describe_runtime()` or the CLI JSONL headers to inspect the active runtime.

## High-Level API

The `covertreex.api` façade keeps ergonomics tight while still delegating to the same batch kernels:

```python
from covertreex.api import PCCT, Runtime, Residual

rt = Runtime(
    backend="numpy",
    precision="float64",
    conflict_graph="grid",
    batch_order="hilbert",
    residual=Residual(),
)

points = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
tree = PCCT(rt).fit(points)
tree = PCCT(rt, tree).insert([[2.5, 2.5]])
indices, distances = PCCT(rt, tree).knn([[0.25, 0.25]], k=2, return_distances=True)
```

`Runtime.activate()` installs the configuration without fiddling with environment variables, and every `PCCT` method returns immutable `PCCTree` instances so downstream pipelines can treat updates as pure functions.

## Benchmarks

Smoke benchmarks live under `benchmarks/` and the Typer CLI (`python -m cli.pcct query`) emits structured telemetry by default (`artifacts/benchmarks/*.jsonl`). Example:

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv run python -m benchmarks.batch_ops insert --dimension 8 --batch-size 64 --batches 4
```

The k-NN benchmark supports baseline comparisons against the in-repo sequential tree, PyPI's `covertree`, and the mlpack cover-tree implementation:

```bash
# Optional extras for the external baselines
pip install -e '.[baseline]'

UV_CACHE_DIR=$PWD/.uv-cache uv run python -m cli.pcct query \
  --dimension 8 --tree-points 2048 --queries 512 --k 8 --baseline cover
```

Baseline outputs list build/query timings, throughput, and slowdown ratios alongside the PCCT
measurements so you can quantify gains from the compressed parallel design. The legacy
`python -m benchmarks.queries` shim still works, and `python -m cli.queries` now prints a
compatibility notice before delegating to the Typer `pcct` subcommand so scripts share one runtime
configuration layer.

CLI documentation (flag reference, input/output description, telemetry schema) lives in `docs/CLI.md`.

Additional profile-driven walkthroughs live in `docs/examples/profile_workflows.md`, and migration
notes for legacy scripts are in `docs/migrations/runtime_v_next.md`. When you run with `--metric residual`
and omit `--profile`, the CLI automatically loads the dense `residual-gold` preset that matches
`benchmarks/run_residual_gold_standard.sh`. Pass `--profile residual-fast` (or another residual profile)
only when you explicitly need the throughput/sparse variants. If you just want a pure distance-function
variant without Vecchia backends or gate/whitening plumbing, opt into `--metric residual-lite`. That mode
maps to the registered `residual_correlation_lite` metric and simply computes correlation distances from
the payload vectors, so it stays self-contained and strictly opt-in.

For larger sweeps (or to exercise mlpack on a toy problem) use the automated runner in `tools/baseline_matrix.py`.
It shells out to `cli.pcct query`, samples CPU/RAM via `psutil`, and appends JSONL rows under `artifacts/`. To compare
the gold residual run against external cover trees, invoke the helper twice: first with `--baseline-mode none` to
capture the pure PCCT numbers, and again with `--baseline-mode cover` (PyPI + mlpack). Because the script defaults to
`residual-gold` whenever `--metric residual` is supplied, both runs match `benchmarks/run_residual_gold_standard.sh`.

```bash
# Toy Euclidean comparison against PyPI + mlpack cover trees
python tools/baseline_matrix.py \
  --profile default \
  --metric euclidean \
  --dimension 3 \
  --tree-points 512 \
  --queries 64 \
  --k 4 \
  --baseline-mode cover \
  --output artifacts/benchmarks/baseline_toy.jsonl

# Residual-only sweeps reuse the same CLI plumbing (PCCT-only gold reference)
python tools/baseline_matrix.py \
  --profile residual-gold \
  --metric residual \
  --dimension 6 \
  --tree-points 2048 \
  --queries 256 \
  --k 8 \
  --baseline-mode none

# …and to capture the matching mlpack/PyPI baselines for the same shape:
python tools/baseline_matrix.py \
  --profile residual-gold \
  --metric residual \
  --dimension 6 \
  --tree-points 2048 \
  --queries 256 \
  --k 8 \
  --baseline-mode cover
```

Each JSONL entry includes the CLI command, PCCT timings, external baseline timings (PyPI + mlpack when
`--baseline-mode cover` is used), and resource telemetry (wall, CPU seconds, and RSS watermark).

### Agents / Contributors

Automated agents or contributors must follow `AGENTS.md`: do **not** remove code paths, telemetry, or historical artefacts unless explicitly requested. Prefer additive feature flags and keep runs reproducible.

## Reference Material

- `PARALLEL_COMPRESSED_PLAN.md` &mdash; architecture, milestones, and testing ladder.
- `docs/API.md` &mdash; public runtime/PCCT façade usage and CLI entrypoints.
- `notes/` &mdash; upstream context and domain-specific constraints gathered during planning.
