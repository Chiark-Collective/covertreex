from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import PCCT, Runtime
from covertreex.metrics import (
    build_residual_backend,
    configure_residual_correlation,
)

POINTS = np.array([[float(i), float(i) * 0.5] for i in range(10)], dtype=np.float64)
QUERIES = np.array([[0.1, 0.05], [4.9, 2.45], [8.05, 4.025]], dtype=np.float64)
K = 3


@pytest.fixture(autouse=True)
def reset_runtime_context() -> None:
    cx_config.reset_runtime_context()
    yield
    cx_config.reset_runtime_context()


@pytest.fixture
def residual_backend_config() -> None:
    backend = build_residual_backend(
        POINTS,
        seed=7,
        inducing_count=16,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=16,
    )
    configure_residual_correlation(backend)
    yield
    reset_residual_metric()


def _knn_result(runtime: Runtime, *, queries: np.ndarray | None = None, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    query_batch = QUERIES if queries is None else np.asarray(queries, dtype=np.float64)
    tree = PCCT(runtime).fit(POINTS, mis_seed=123)
    indices, distances = PCCT(runtime, tree).knn(
        query_batch,
        k=k,
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


def test_knn_consistency_batch_order() -> None:
    baseline = _base_runtime().with_updates(batch_order="natural")
    hilbert = baseline.with_updates(batch_order="hilbert")

    idx_a, dist_a = _knn_result(baseline)
    idx_b, dist_b = _knn_result(hilbert)

    _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)


def test_knn_consistency_sparse_toggle() -> None:
    dense_runtime = _base_runtime().with_updates(enable_sparse_traversal=False)
    sparse_runtime = dense_runtime.with_updates(enable_sparse_traversal=True)

    idx_a, dist_a = _knn_result(dense_runtime)
    idx_b, dist_b = _knn_result(sparse_runtime)

    _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)


@pytest.mark.usefixtures("residual_backend_config")
def test_knn_consistency_residual_whitened_toggle() -> None:
    runtime_base = Runtime(
        backend="numpy",
        precision="float64",
        metric="residual_correlation",
        enable_numba=True,
        enable_sparse_traversal=True,
    )

    dense = runtime_base.with_updates(residual_force_whitened=False)
    whitened = runtime_base.with_updates(residual_force_whitened=True)

    idx_a, dist_a = _knn_result(dense)
    idx_b, dist_b = _knn_result(whitened)

    _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)


@pytest.mark.usefixtures("residual_backend_config")
def test_knn_consistency_residual_scope_member_limit() -> None:
    runtime_base = Runtime(
        backend="numpy",
        precision="float64",
        metric="residual_correlation",
        enable_numba=True,
        enable_sparse_traversal=True,
        residual_scope_member_limit=64,
    )
    uncapped = runtime_base.with_updates(residual_scope_member_limit=0)

    idx_a, dist_a = _knn_result(runtime_base)
    idx_b, dist_b = _knn_result(uncapped)

    _assert_knn_equal(idx_a, dist_a, idx_b, dist_b)


def _residual_runtime(**overrides: Any) -> Runtime:
    base_kwargs: Dict[str, Any] = {
        "backend": "numpy",
        "precision": "float64",
        "metric": "residual_correlation",
        "enable_numba": True,
        "enable_sparse_traversal": False,
        "residual_dense_scope_streamer": True,
        "residual_scope_bitset": True,
    }
    base_kwargs.update(overrides)
    return Runtime(**base_kwargs)


@pytest.mark.usefixtures("residual_backend_config")
def test_knn_consistency_residual_bitset_toggle() -> None:
    runtime_on = _residual_runtime(residual_scope_bitset=True)
    runtime_off = runtime_on.with_updates(residual_scope_bitset=False)

    idx_on, dist_on = _knn_result(runtime_on)
    idx_off, dist_off = _knn_result(runtime_off)

    _assert_knn_equal(idx_on, dist_on, idx_off, dist_off)


@pytest.mark.usefixtures("residual_backend_config")
def test_knn_consistency_residual_dense_streamer_toggle() -> None:
    runtime_dense = _residual_runtime(residual_dense_scope_streamer=True)
    runtime_serial = runtime_dense.with_updates(residual_dense_scope_streamer=False)

    idx_dense, dist_dense = _knn_result(runtime_dense)
    idx_serial, dist_serial = _knn_result(runtime_serial)

    _assert_knn_equal(idx_dense, dist_dense, idx_serial, dist_serial)


@pytest.mark.usefixtures("residual_backend_config")
def test_knn_consistency_residual_masked_append_toggle() -> None:
    runtime_masked = _residual_runtime(residual_masked_scope_append=True)
    runtime_legacy = runtime_masked.with_updates(residual_masked_scope_append=False)

    idx_masked, dist_masked = _knn_result(runtime_masked)
    idx_legacy, dist_legacy = _knn_result(runtime_legacy)

    _assert_knn_equal(idx_masked, dist_masked, idx_legacy, dist_legacy)


@pytest.mark.usefixtures("residual_backend_config")
def test_knn_consistency_residual_sparse_toggle() -> None:
    dense_runtime = _residual_runtime(enable_sparse_traversal=False)
    sparse_runtime = dense_runtime.with_updates(enable_sparse_traversal=True)

    idx_dense, dist_dense = _knn_result(dense_runtime)
    idx_sparse, dist_sparse = _knn_result(sparse_runtime)

    _assert_knn_equal(idx_dense, dist_dense, idx_sparse, dist_sparse)
