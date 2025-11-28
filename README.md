# Covertreex

High-performance cover tree library for k-NN queries, optimized for Vecchia-style Gaussian process pipelines.

[![PyPI version](https://badge.fury.io/py/covertreex.svg)](https://pypi.org/project/covertreex/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **Very fast**, parallel implementation following [Parallel Cover Trees and their Applications](https://cs.ucr.edu/~ygu/papers/SPAA22/covertree.pdf)
- **Hybrid Python/Numba + Rust** implementation
- **AVX2 SIMD optimized** dot products in Rust backend
- **Residual correlation metric** with RBF and Matérn 5/2 kernels
- **Hilbert curve ordering** for cache-efficient tree construction

## Installation

```bash
pip install covertreex
```

## Quick Start

### Basic Euclidean k-NN

```python
import numpy as np
from covertreex import CoverTree

# Build tree from points
points = np.random.randn(10000, 3)
tree = CoverTree().fit(points)

# Query k nearest neighbors
query_points = np.random.randn(100, 3)
neighbors = tree.knn(query_points, k=10)

# With distances
neighbors, distances = tree.knn(query_points, k=10, return_distances=True)
```

### Residual Correlation Metric (Vecchia GP)

For Gaussian process applications with Vecchia approximations:

```python
import numpy as np
from covertreex import ResidualCoverTree

coords = np.random.randn(10000, 3).astype(np.float32)
tree = ResidualCoverTree(
    coords,
    variance=1.0,
    lengthscale=1.0,
    inducing_count=512,
)

# Query all points
neighbors = tree.knn(k=50)

# Predecessor constraint (for Vecchia): neighbor j must have j < query i
neighbors = tree.knn(k=50, predecessor_mode=True)
```

### Engine Selection

```python
# ResidualCoverTree defaults to rust-hilbert (fastest)
tree = ResidualCoverTree(coords, engine="rust-hilbert")
tree = ResidualCoverTree(coords, engine="rust-natural")

# For CoverTree, use Runtime
from covertreex import Runtime
runtime = Runtime(engine="rust-hilbert")
tree = CoverTree(runtime).fit(points)
```

### Profile Presets

```python
from covertreex import Runtime

# Load predefined configuration
runtime = Runtime.from_profile("residual-gold")

# With overrides
runtime = Runtime.from_profile("residual-gold", overrides=["seeds.global_seed=42"])
```

Available profiles: `default`, `residual-gold`, `residual-fast`, `residual-audit`, `cpu-debug`

## Performance

Residual correlation k-NN benchmark (AMD Ryzen 9 9950X, N=32k, D=3, k=50):

| Engine | Build Time | Query Throughput |
|--------|------------|------------------|
| python-numba | 7.0s | 40,000 q/s |
| **rust-hilbert** | **0.9s** | **44,000 q/s** |

## API Reference

### CoverTree

Main interface for building trees and running k-NN queries.

```python
CoverTree(runtime: Runtime = Runtime())
    .fit(points) -> tree              # Build tree from points
    .knn(queries, k=10) -> indices    # Find k nearest neighbors
    .knn(queries, k=10, return_distances=True) -> (indices, distances)
```

### Runtime

Configuration for backend, metric, and engine selection.

```python
Runtime(
    engine="rust-hilbert",           # Execution engine
    metric="residual_correlation",   # Distance metric
    backend="numpy",                 # Array backend
    precision="float32",             # Float precision
)
```

### Residual Backend

For Vecchia GP residual correlation:

```python
from covertreex.metrics.residual import build_residual_backend

backend = build_residual_backend(
    coords,                    # (n, d) spatial coordinates
    seed=42,                   # Random seed
    inducing_count=512,        # Number of inducing points
    variance=1.0,              # Kernel variance σ²
    lengthscale=1.0,           # Kernel lengthscale ℓ
    kernel_type=0,             # 0=RBF, 1=Matérn 5/2
)
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Build Rust backend
maturin develop --release

# Run tests
pytest

# Lint
ruff check covertreex
```

## CLI (Testing)

A CLI is included for benchmarking and testing:

```bash
python -m cli.pcct --help
python -m cli.pcct query --engine rust-hilbert --tree-points 32768 --k 50
```

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
