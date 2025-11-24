# Cover Tree CLI

The `cli.pcct` module provides a command-line interface for building and querying Parallel Compressed Cover Trees (PCCT).

## Engines

The CLI supports multiple execution engines via the `--engine` flag.

### `python-numba` (Default)
The reference implementation. Features full telemetry, dynamic query block sizing, and level caching.
- **Best for:** Maximum query throughput, debugging, telemetry analysis.
- **Trade-offs:** Slower build times.

### `rust-natural` (formerly `rust-fast`)
A high-performance Rust implementation using the natural (input) order of points.
- **Best for:** Fast builds and moderate query throughput without reordering data.
- **Trade-offs:** Slower queries than `python-numba`, no telemetry.

### `rust-hilbert` (formerly `rust-pcct2`)
A Rust implementation that reorders points using a Hilbert curve (Morton order) to optimize build speed.
- **Best for:** **Fastest build times** (up to 4x faster than Python).
- **Trade-offs:** Queries are currently slower than `rust-natural` and `python-numba`.

## Usage Examples

### Gold Standard Residual Benchmark
Run the standard residual benchmark (32k points, d=3, k=50, 1024 queries) with the default engine:
```bash
python -m cli.pcct query \
  --metric residual \
  --tree-points 32768 \
  --dimension 3 \
  --queries 1024 \
  --k 50
```

### High-Performance Rust Build (Natural Order)
Use the `rust-natural` engine for faster builds than Python, with decent query speed:
```bash
export COVERTREEX_RESIDUAL_SCOPE_CAP_PATH="docs/data/residual_scope_caps_32768.json"
python -m cli.pcct query \
  --engine rust-natural \
  --metric residual \
  --tree-points 32768 \
  --dimension 3 \
  --queries 1024 \
  --k 50
```

### Fastest Build (Hilbert Order)
Use the `rust-hilbert` engine for the absolute fastest construction time:
```bash
export COVERTREEX_RESIDUAL_SCOPE_CAP_PATH="docs/data/residual_scope_caps_32768.json"
python -m cli.pcct query \
  --engine rust-hilbert \
  --metric residual \
  --tree-points 32768 \
  --dimension 3 \
  --queries 1024 \
  --k 50
```

## Optimization Flags

- `COVERTREEX_RESIDUAL_SCOPE_CAP_PATH`: Path to a JSON file defining per-level radius caps (crucial for performance).
- `COVERTREEX_RESIDUAL_BUDGET_SCHEDULE`: Comma-separated list of budget thresholds (e.g., "32,64,96").
- `COVERTREEX_RESIDUAL_STREAM_TILE`: Batch size for child processing in the Rust streamer (default: 64).