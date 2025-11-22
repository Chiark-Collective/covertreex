# Residual-only Rust fast path

This documents the residual-only index-tree build that bypasses the PCCT
Hilbert ordering and uses the Rust backend end-to-end. Conflict handling runs
inside the Rust crate with a squared-distance fast path and chunked inserts so
we can parallelise without exploding the conflict graph.

- **What it is:** `covertreex.metrics.residual.fast_build.build_fast_residual_tree`
  builds a `CoverTreeWrapper` that stores 1-D dataset ids and uses the residual
  metric for both insert and query. No Hilbert ordering, no CLI telemetry, no
  gate/policy plumbing. Inserts are chunked (`chunk_size` controls the maximum
  residual batch per conflict-graph pass).
- **When to use:** Only when you care about residual-correlation queries and
  want minimal build overhead. It will not serve Euclidean workloads or reuse
  existing PCCT trees.
- **Parallelism:** Build and query are Rayon-parallel across candidates/queries.
  Coarse levels short-circuit when the radius already exceeds the residual
  metric bound (√2), and conflict detection uses squared distances to avoid
  unnecessary square-roots.
- **Latest benchmark (2025-11-22, float32):**
  - N=50k, d=3, k=50, 1024 queries, `chunk_size=2048`, seed=0 → build 24.71s,
    query 31.56s (32.5 q/s).
  - N=10k, d=3, k=50, 1024 queries, `chunk_size=512`, seed=0 → build 1.10s.
  Benchmark command for the 50k run (release build of `covertreex_backend`):

```
python - <<'PY'
import time, numpy as np
from covertreex.metrics.residual.fast_build import build_fast_residual_tree

N = 50_000
d = 3
k = 50
queries = 1_024
chunk_size = 2_048
rng = np.random.default_rng(0)
points = rng.normal(size=(N, d)).astype(np.float32)
start = time.perf_counter()
tree, node_to_dataset, backend = build_fast_residual_tree(
    points, seed=0, inducing_count=512, chunk_size=chunk_size
)
build = time.perf_counter() - start
q_idx = np.arange(queries, dtype=np.int64)
q_start = time.perf_counter()
tree.knn_query_residual(
    q_idx,
    node_to_dataset,
    backend.v_matrix,
    backend.p_diag,
    getattr(backend, "kernel_points_f32", backend.v_matrix),
    float(getattr(backend, "rbf_variance", 1.0)),
    np.asarray(getattr(backend, "rbf_lengthscale", np.ones(d, dtype=np.float32)), dtype=np.float32),
    k,
)
query = time.perf_counter() - q_start
print(f"build={build:.4f}s query={query:.4f}s qps={queries/query:.1f}")
PY
```
- **Limitations:** No telemetry/gating, no batch ordering/prefix scheduling,
  residual-only metric, 1-D payloads (dataset ids). If you need those features,
  use the PCCT CLI path instead.

## Usage

```python
import numpy as np
from covertreex.metrics.residual.fast_build import build_fast_residual_tree

points = np.random.default_rng(0).normal(size=(50_000, 3)).astype(np.float32)
tree, node_to_dataset, backend = build_fast_residual_tree(
    points, seed=0, inducing_count=512, chunk_size=2048
)

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

## CLI quick start

Use the cover-tree CLI with the Rust fast engine to exercise this path end to
end (residual metric only):

```
python -m cli.pcct.query --metric residual --engine rust-fast --tree-points 50000 --dimension 3 --queries 1024 --k 50 --residual-chunk-size 2048
```
