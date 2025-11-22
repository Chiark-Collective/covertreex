# Residual build/query cheat sheet (2025-11-22)

Quick references for the fastest reproducible residual builds on this branch and how the conflict graph factors in.

## Current best-performing runs

- **Python/Numba residual (PCCT path)**
  - Command:
    ```bash
    python - <<'PY'
    import time, numpy as np, dataclasses
    from covertreex import config as cx_config
    from covertreex.engine import build_tree
    from covertreex.metrics.residual import build_residual_backend, configure_residual_correlation

    N=50_000; d=3; k=50; queries=1_024; seed=0; chunk_size=512
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(N, d)).astype(np.float32)
    backend = build_residual_backend(points, seed=seed, inducing_count=512, variance=1.0, lengthscale=1.0, chunk_size=chunk_size)
    rt = dataclasses.replace(cx_config.runtime_config(), metric='residual_correlation', engine='python-numba', enable_numba=True, enable_rust=False, precision='float32')
    ctx = cx_config.configure_runtime(rt)
    configure_residual_correlation(backend, context=ctx)
    start = time.perf_counter();
    tree = build_tree(points.astype(np.float64), runtime=rt, context=ctx, residual_backend=backend,
                      residual_params={'variance':1.0,'lengthscale':1.0,'inducing_count':512,'chunk_size':chunk_size});
    build = time.perf_counter() - start
    q_idx = np.arange(queries, dtype=np.int64).reshape(-1,1)
    q_start = time.perf_counter(); tree.knn(q_idx, k=k, return_distances=False, context=ctx); query = time.perf_counter() - q_start
    print(f"python-numba residual: build={build:.4f}s query={query:.4f}s qps={queries/query:.1f}")
    PY
    ```
  - Result (2025-11-22): **build 12.52s, query 0.037s, 27.7k q/s**.

- **Rust fast residual (index-tree path)**
  - Command:
    ```bash
    python - <<'PY'
    import time, numpy as np
    from covertreex.metrics.residual.fast_build import build_fast_residual_tree

    N=50_000; d=3; k=50; queries=1_024; chunk_size=2_048; seed=0
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(N, d)).astype(np.float32)
    start = time.perf_counter();
    tree, node_to_dataset, backend = build_fast_residual_tree(points, seed=seed, inducing_count=512, chunk_size=chunk_size)
    build = time.perf_counter() - start
    q_idx = np.arange(queries, dtype=np.int64)
    q_start = time.perf_counter();
    tree.knn_query_residual(
        q_idx,
        node_to_dataset,
        backend.v_matrix,
        backend.p_diag,
        getattr(backend, 'kernel_points_f32', backend.v_matrix),
        float(getattr(backend, 'rbf_variance', 1.0)),
        np.asarray(getattr(backend, 'rbf_lengthscale', np.ones(d, dtype=np.float32)), dtype=np.float32),
        k,
    )
    query = time.perf_counter() - q_start
    print(f"rust-fast residual: build={build:.4f}s query={query:.4f}s qps={queries/query:.1f}")
    PY
    ```
  - Result (2025-11-22): **build 24.71s, query 31.56s, 32.5 q/s** with `chunk_size=2048`.

## Conflict-graph significance

- The PCCT (python-numba) path builds via a conflict graph to preserve cover-tree separation. Batch size and Hilbert ordering keep the graph sparse; chunking the backend kernels (`residual_chunk_size`) mainly affects residual cache sizing.
- The Rust fast path previously bulk-inserted indices without conflicts; now we chunk inserts and run a lightweight conflict pass inside Rust. The `chunk_size` parameter caps the number of points per conflict-graph pass: smaller chunks reduce conflict density but add more passes; larger chunks (e.g., 2048) give fewer passes but heavier per-pass work.
- Residual metric is bounded (√2). We now exploit that bound plus squared-distance checks to skip expensive evaluations at coarse levels, reducing conflict-graph work especially for large radii.

## Notes
- The historic rust-only baseline (commit 5c63111) reported build ~1.5s and query ~6s (165 q/s) at N=50k; current code paths are slower because we now preserve cover-tree invariants via conflict checks. This doc tracks the current optimization thread for closing that gap.
- Always rebuild the backend before benchmarking: `maturin develop --release`.
- Use the commands above verbatim for reproducibility; adjust `chunk_size` to probe the build/query trade-off.
