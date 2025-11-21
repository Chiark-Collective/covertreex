# Residual-only Rust fast path

This documents the residual-only index-tree build that bypasses the PCCT
Hilbert/conflict graph and uses the Rust backend end-to-end.

- **What it is:** `covertreex.metrics.residual.fast_build.build_fast_residual_tree`
  builds a `CoverTreeWrapper` that stores 1-D dataset ids and uses the residual
  metric for both insert and query. No Hilbert ordering, no CLI telemetry, no
  gate/policy plumbing.
- **When to use:** Only when you care about residual-correlation queries and
  want minimal build overhead. It will not serve Euclidean workloads or reuse
  existing PCCT trees.
- **Parallelism:** Build and query are Rayon-parallel across candidates/queries,
  but the per-level conflict detection still evaluates residual distances, so
  pathological kernels can dominate build time.
- **Example (n=50k, d=3, k=50, 1024 queries, commit 5c63111):**
  - float32: build ~1.5s, query ~6.2s (164.8 q/s)
  - float64: build ~1.5s, query ~5.9s (174.0 q/s)
  The PCCT Hilbert path on the same commit was ~31–33s to build.
- **Limitations:** No telemetry/gating, no batch ordering/prefix scheduling,
  residual-only metric, 1-D payloads (dataset ids). If you need those features,
  use the PCCT CLI path instead.

## Usage

```python
import numpy as np
from covertreex.metrics.residual.fast_build import build_fast_residual_tree

points = np.random.default_rng(0).normal(size=(50_000, 3)).astype(np.float32)
tree, node_to_dataset, backend = build_fast_residual_tree(points, seed=0, inducing_count=512)

# Query (CoverTreeWrapper API)
import covertreex_backend as cb
query_idx = np.arange(1024, dtype=np.int64)
indices, dists = tree.knn_query_residual(
    query_idx, node_to_dataset, backend.v_matrix, backend.p_diag, backend.kernel_points_f32,
    float(getattr(backend, "rbf_variance", 1.0)),
    np.asarray(getattr(backend, "rbf_lengthscale", np.ones(points.shape[1], dtype=np.float32)), dtype=np.float32),
    50
)
```
