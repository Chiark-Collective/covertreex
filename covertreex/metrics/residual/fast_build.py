from __future__ import annotations

import numpy as np

from .host_backend import build_residual_backend


def build_fast_residual_tree(
    points: np.ndarray,
    *,
    seed: int = 0,
    variance: float = 1.0,
    lengthscale: float = 1.0,
    inducing_count: int = 512,
    chunk_size: int = 512,
):
    """
    Build a residual-only CoverTree using the Rust backend and index payloads.

    This bypasses the PCCT Hilbert/conflict-graph pipeline and stores 1-D
    dataset indices; it is suitable when you only need residual-correlation
    queries and want minimal build overhead. The returned tree and mapping can
    be passed directly to `CoverTreeWrapper.knn_query_residual`.

    `chunk_size` controls both the backend kernel chunking and the maximum
    number of points processed per conflict-graph pass during tree insertion.
    """

    try:
        import covertreex_backend  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("covertreex_backend is not installed.") from exc

    host_backend = build_residual_backend(
        np.asarray(points, dtype=np.float64),
        seed=seed,
        inducing_count=inducing_count,
        variance=float(variance),
        lengthscale=float(lengthscale),
        chunk_size=chunk_size,
    )
    dtype = np.float32 if host_backend.v_matrix.dtype == np.float32 else np.float64
    coords = np.asarray(getattr(host_backend, "kernel_points_f32", host_backend.v_matrix), dtype=dtype)
    v_matrix = np.asarray(host_backend.v_matrix, dtype=dtype)
    p_diag = np.asarray(host_backend.p_diag, dtype=dtype)

    rbf_var = float(getattr(host_backend, "rbf_variance", variance))
    rbf_ls = np.asarray(
        getattr(host_backend, "rbf_lengthscale", np.ones(coords.shape[1], dtype=dtype)),
        dtype=dtype,
    )

    dummy = np.empty((0, 1), dtype=dtype)
    empty_i64 = np.empty(0, dtype=np.int64)
    tree = covertreex_backend.CoverTreeWrapper(dummy, empty_i64, empty_i64, empty_i64, empty_i64, -20, 20)

    indices_all = np.arange(host_backend.num_points, dtype=dtype).reshape(-1, 1)
    tree.insert_residual(indices_all, v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size)

    node_to_dataset = np.arange(host_backend.num_points, dtype=np.int64).tolist()
    return tree, node_to_dataset, host_backend


__all__ = ["build_fast_residual_tree"]
