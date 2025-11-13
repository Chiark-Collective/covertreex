# Covertreex

Parallel compressed cover tree (PCCT) library engineered for Vecchia-style Gaussian process pipelines. The current focus is an efficient **CPU + NumPy/Numba** implementation; GPU/JAX execution has been intentionally disabled until the CPU path meets the audited performance envelope. This repository is the implementation companion to `PARALLEL_COMPRESSED_PLAN.md`.

> Status: housekeeping phase. Expect rapid iteration across runtime knobs, telemetry, and traversal optimisations, but avoid destructive changes (see `AGENTS.md`).

## Getting Started

Within a Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

The default backend is `numpy` (CPU). Optional acceleration hooks leverage `numba` when the `numba` extra is installed.

The Typer-powered benchmark CLI surfaces every runtime knob so runs are reproducible:

```bash
python -m cli.queries --help

# Example residual sweep
python -m cli.queries \
  --metric residual --dimension 8 --tree-points 32768 --queries 1024 \
  --batch-size 512 --k 8 --baseline both
```

Every invocation writes JSONL telemetry under `artifacts/benchmarks/` unless `--no-log-file` is passed, making audits deterministic.

## Runtime Controls

The `covertreex.api.Runtime` façade now mirrors every CLI flag (backend, precision, scope caps, residual gates, prefilters, stream tiling, etc.). For non-CLI consumers, configuration can still be driven by environment variables read during import, but the CLI is the source of truth for supported knobs. Highlights:

- `Runtime(...).activate()` installs an in-process configuration without mutating `os.environ`.
- All residual gate/prefilter settings (lookup paths, margins, audits, membership caps, forced whitening, etc.) are addressable via CLI flags or `Runtime` keyword arguments.
- CLI runs always emit batch-level telemetry (`BenchmarkLogWriter`) unless `--no-log-file` is passed, so reproductions include scope budgets, kernel/whitened counters, and resource snapshots.

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
    residual=Residual(gate1_enabled=True, gate1_alpha=2.0),
)

points = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
tree = PCCT(rt).fit(points)
tree = PCCT(rt, tree).insert([[2.5, 2.5]])
indices, distances = PCCT(rt, tree).knn([[0.25, 0.25]], k=2, return_distances=True)
```

`Runtime.activate()` installs the configuration without fiddling with environment variables, and every `PCCT` method returns immutable `PCCTree` instances so downstream pipelines can treat updates as pure functions.

## Benchmarks

Smoke benchmarks live under `benchmarks/` and the Typer CLI (`python -m cli.queries`) emits structured telemetry by default (`artifacts/benchmarks/*.jsonl`). Example:

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv run python -m benchmarks.batch_ops insert --dimension 8 --batch-size 64 --batches 4
```

The k-NN benchmark supports baseline comparisons against both the in-repo sequential tree and the optional external CoverTree implementation:

```bash
# Optional extras for the external baseline
pip install -e '.[baseline]'

UV_CACHE_DIR=$PWD/.uv-cache uv run python -m cli.queries \
  --dimension 8 --tree-points 2048 --queries 512 --k 8 --baseline both
```

Baseline outputs list build/query timings, throughput, and slowdown ratios alongside the PCCT measurements so you can quantify gains from the compressed parallel design. The legacy `python -m benchmarks.queries` shim still works, but the supported entrypoint now lives under `cli/` so scripts share one runtime configuration layer.

CLI documentation (flag reference, input/output description, telemetry schema) lives in `docs/CLI.md`.

### Agents / Contributors

Automated agents or contributors must follow `AGENTS.md`: do **not** remove code paths, telemetry, or historical artefacts unless explicitly requested. Prefer additive feature flags and keep runs reproducible.

## Reference Material

- `PARALLEL_COMPRESSED_PLAN.md` &mdash; architecture, milestones, and testing ladder.
- `docs/API.md` &mdash; public runtime/PCCT façade usage and CLI entrypoints.
- `notes/` &mdash; upstream context and domain-specific constraints gathered during planning.
