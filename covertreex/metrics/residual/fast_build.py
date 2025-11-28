from __future__ import annotations

import numpy as np

from .host_backend import build_residual_backend


def _hilbert_order(coords: np.ndarray) -> np.ndarray:
    """Compute Hilbert-curve-like ordering for spatial locality."""
    try:
        import covertreex_backend
        return np.asarray(covertreex_backend.hilbert_order(coords), dtype=np.int64)
    except (ImportError, AttributeError):
        # Fallback: Morton-like ordering using bit interleaving
        n, d = coords.shape
        # Normalize to [0, 1] range
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # Avoid division by zero
        normalized = (coords - mins) / ranges

        # Quantize to integers (use 10 bits per dimension)
        bits = 10
        quantized = (normalized * ((1 << bits) - 1)).astype(np.int64)

        # Simple Z-order (Morton) curve: interleave bits
        morton = np.zeros(n, dtype=np.int64)
        for dim in range(d):
            for bit in range(bits):
                morton |= ((quantized[:, dim] >> bit) & 1) << (bit * d + dim)

        return np.argsort(morton)


def build_fast_residual_tree(
    points: np.ndarray,
    *,
    seed: int = 0,
    variance: float = 1.0,
    lengthscale: float = 1.0,
    inducing_count: int = 512,
    chunk_size: int = 512,
    use_hilbert_order: bool = True,
):
    """
    Build a residual-only CoverTree using the Rust backend and index payloads.

    This bypasses the PCCT Hilbert/conflict-graph pipeline and stores 1-D
    dataset indices; it is suitable when you only need residual-correlation
    queries and want minimal build overhead. The returned tree and mapping can
    be passed directly to `CoverTreeWrapper.knn_query_residual`.

    `chunk_size` controls both the backend kernel chunking and the maximum
    number of points processed per conflict-graph pass during tree insertion.

    `use_hilbert_order` enables Hilbert curve ordering for faster builds via
    spatial locality. Default True.
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

    # For predecessor_mode to work efficiently, we need natural ordering
    # (Hilbert ordering scatters dataset indices, making predecessor pruning ineffective)
    n_points = host_backend.num_points

    dummy = np.empty((0, 1), dtype=dtype)
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_i32 = np.empty(0, dtype=np.int32)
    tree = covertreex_backend.CoverTreeWrapper(dummy, empty_i64, empty_i64, empty_i64, empty_i32, -20, 20)

    # Insert in natural order - required for predecessor_mode to work with subtree pruning
    indices_all = np.arange(n_points, dtype=dtype).reshape(-1, 1)
    tree.insert_residual(indices_all, v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size)

    # Identity mapping: tree node i corresponds to dataset index i
    node_to_dataset = np.arange(n_points, dtype=np.int64).tolist()
    return tree, node_to_dataset, host_backend


__all__ = ["build_fast_residual_tree"]
