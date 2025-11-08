# Covertreex

Parallel compressed cover tree (PCCT) library engineered for Vecchia-style Gaussian process pipelines. The current focus is an efficient **CPU + Numba** implementation; GPU/JAX execution has been intentionally disabled until the CPU path reaches the desired performance envelope. This is the implementation companion to the plan captured in `PARALLEL_COMPRESSED_PLAN.md`.

> Status: scaffolding in progress. Expect rapid iteration across backends, persistence utilities, and traversal/insertion kernels.

## Getting Started

Within a Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

The default backend is `jax.numpy` (`jnp`). Optional acceleration hooks leverage `numba` when the `numba` extra is installed.

## Runtime Controls

Configuration is driven by environment variables consumed when `covertreex` is imported:

- `COVERTREEX_DEVICE` is ignored for now—execution is forced onto CPU even if GPUs are present.
- `COVERTREEX_ENABLE_NUMBA=1` enables the Numba-backed MIS kernels (recommended).
- `COVERTREEX_ENABLE_DIAGNOSTICS=0` disables resource polling (wall/CPU/RSS/GPU) in the operation logs to minimise benchmarking overhead.
- `COVERTREEX_METRIC=residual_correlation` switches the library to the residual-correlation metric (pairwise handler configured via `covertreex.configure_residual_metric`).

Residual metrics are wired up lazily: call `covertreex.configure_residual_metric(pairwise=..., pointwise=...)` after import to supply the Vecchia residual-correlation provider; `reset_residual_metric()` clears the hooks (useful for tests).

Use `covertreex.config.describe_runtime()` to inspect the active settings.

## High-Level API

The new `covertreex.api` façade keeps ergonomics tight while still delegating to the same batch kernels:

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

Smoke benchmarks live under `benchmarks/` and now emit structured telemetry:

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv run python -m benchmarks.batch_ops insert --dimension 8 --batch-size 64 --batches 4
```

The k-NN benchmark supports baseline comparisons against both the in-repo sequential tree and the optional external CoverTree implementation:

```bash
# Optional extras for the external baseline
pip install -e '.[baseline]'

UV_CACHE_DIR=$PWD/.uv-cache uv run python -m benchmarks.queries \
  --dimension 8 --tree-points 2048 --queries 512 --k 8 --baseline both
```

Baseline outputs list build/query timings, throughput, and slowdown ratios alongside the PCCT measurements so you can quantify gains from the compressed parallel design.

## Reference Material

- `PARALLEL_COMPRESSED_PLAN.md` &mdash; architecture, milestones, and testing ladder.
- `notes/` &mdash; upstream context and domain-specific constraints gathered during planning.
