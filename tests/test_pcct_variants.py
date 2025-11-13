from __future__ import annotations

import numpy as np
import pytest

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import PCCT, Runtime
from covertreex.metrics import build_residual_backend, configure_residual_correlation

POINTS = np.array([[float(i), float(i) * 0.5] for i in range(10)], dtype=np.float64)
QUERIES = np.array([[0.1, 0.05], [4.9, 2.45], [8.05, 4.025]], dtype=np.float64)
K = 3


@pytest.fixture(autouse=True)
def reset_runtime_context() -> None:
    cx_config.reset_runtime_context()
    yield
    cx_config.reset_runtime_context()


def _knn_result(runtime: Runtime) -> tuple[np.ndarray, np.ndarray]:
    tree = PCCT(runtime).fit(POINTS, mis_seed=123)
    indices, distances = PCCT(runtime, tree).knn(
        QUERIES,
        k=K,
        return_distances=True,
    )
    return np.asarray(indices), np.asarray(distances, dtype=np.float64)


def _base_runtime() -> Runtime:
    return Runtime(
        backend="numpy",
        precision="float64",
        metric="euclidean",
        enable_numba=False,
        enable_sparse_traversal=False,
    )


def test_knn_consistency_across_batch_orders() -> None:
    baseline = _base_runtime().with_updates(batch_order="natural")
    hilbert = baseline.with_updates(batch_order="hilbert")

    idx_a, dist_a = _knn_result(baseline)
    idx_b, dist_b = _knn_result(hilbert)

    np.testing.assert_array_equal(idx_b, idx_a)
    np.testing.assert_allclose(dist_b, dist_a)


def _assert_knn_equal(a_idx: np.ndarray, a_dist: np.ndarray, b_idx: np.ndarray, b_dist: np.ndarray) -> None:
    np.testing.assert_allclose(a_dist, b_dist)
    np.testing.assert_array_equal(a_idx, b_idx)


def test_knn_consistency_enable_numba_toggle() -> None:
    baseline = _base_runtime().with_updates(enable_numba=False)
    numba = baseline.with_updates(enable_numba=True)

    idx_a, dist_a = _knn_result(baseline)
    idx_b, dist_b = _knn_result(numba)

    _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)


def test_knn_consistency_sparse_toggle() -> None:
    dense_runtime = _base_runtime().with_updates(enable_sparse_traversal=False)
    sparse_runtime = dense_runtime.with_updates(enable_sparse_traversal=True)

    idx_a, dist_a = _knn_result(dense_runtime)
    idx_b, dist_b = _knn_result(sparse_runtime)

    _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)


def test_knn_consistency_residual_whitened_toggle() -> None:
    residual_backend = build_residual_backend(
        POINTS,
        seed=42,
        inducing_count=16,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=16,
    )
    configure_residual_correlation(residual_backend)
    runtime_base = Runtime(
        backend="numpy",
        precision="float64",
        metric="residual_correlation",
        enable_numba=True,
        enable_sparse_traversal=True,
    )
    try:
        dense = runtime_base.with_updates(residual_force_whitened=False)
        whitened = runtime_base.with_updates(residual_force_whitened=True)

        idx_a, dist_a = _knn_result(dense)
        idx_b, dist_b = _knn_result(whitened)

        _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)
    finally:
        reset_residual_metric()
