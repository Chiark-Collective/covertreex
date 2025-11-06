import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.core.metrics import (
    available_metrics,
    configure_residual_metric,
    get_metric,
    MetricRegistry,
    Metric,
    reset_residual_metric,
)
from covertreex.core.tree import DEFAULT_BACKEND, TreeBackend
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
    compute_residual_distances_with_radius,
    compute_residual_distances,
    compute_residual_distances_with_kernel,
    compute_residual_distances_from_kernel,
    compute_residual_lower_bounds_from_kernel,
    get_residual_backend,
    set_residual_backend,
)


def test_euclidean_pairwise_matches_manual():
    backend = DEFAULT_BACKEND
    metric = get_metric("euclidean")
    lhs = backend.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=backend.default_float)
    rhs = backend.asarray([[1.0, 1.0], [2.0, 1.0]], dtype=backend.default_float)

    distances = metric.pairwise(backend, lhs, rhs)
    manual = jnp.sqrt(jnp.sum((lhs[:, None, :] - rhs[None, :, :]) ** 2, axis=-1))

    assert distances.shape == (2, 2)
    assert jnp.allclose(distances, manual)


def test_euclidean_pointwise_supports_vector_inputs():
    backend = DEFAULT_BACKEND
    metric = get_metric()
    lhs = backend.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=backend.default_float)
    rhs = backend.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=backend.default_float)

    result = metric.pointwise(backend, lhs, rhs)
    expected = jnp.array([0.0, jnp.sqrt(2.0)], dtype=backend.default_float)

    assert result.shape == (2,)
    assert jnp.allclose(result, expected)


def test_metric_registry_registers_and_retrieves():
    backend = TreeBackend.jax(precision="float32")

    called = {"pairwise": False, "pointwise": False}

    def _pairwise(bk: TreeBackend, lhs, rhs):
        called["pairwise"] = True
        return bk.asarray([[0.0]], dtype=bk.default_float)

    def _pointwise(bk: TreeBackend, lhs, rhs):
        called["pointwise"] = True
        return bk.asarray([0.0], dtype=bk.default_float)

    registry = MetricRegistry()
    custom_metric = Metric("custom", _pairwise, _pointwise)
    registry.register(custom_metric)

    metric = registry.get("custom")
    metric.pairwise(backend, [[0.0]], [[0.0]])
    metric.pointwise(backend, [0.0], [0.0])

    assert called["pairwise"] and called["pointwise"]
    assert "custom" in registry.names()


def test_get_metric_unknown_raises():
    with pytest.raises(KeyError):
        get_metric("not-a-metric")


def test_available_metrics_contains_euclidean():
    names = available_metrics()
    assert "euclidean" in names
    assert "residual_correlation" in names


def test_residual_metric_requires_configuration():
    backend = DEFAULT_BACKEND
    metric = get_metric("residual_correlation")
    lhs = backend.asarray([[0.0, 0.0]], dtype=backend.default_float)
    rhs = backend.asarray([[1.0, 1.0]], dtype=backend.default_float)

    with pytest.raises(RuntimeError):
        metric.pairwise(backend, lhs, rhs)

    def _pairwise(bk: TreeBackend, l, r):
        return bk.asarray([[42.0]], dtype=bk.default_float)

    configure_residual_metric(pairwise=_pairwise)
    metric = get_metric("residual_correlation")
    result = metric.pairwise(backend, lhs, rhs)
    assert result.shape == (1, 1)
    assert pytest.approx(float(result[0, 0])) == 42.0

    point = metric.pointwise(backend, lhs[0], rhs[0])
    if point.shape == ():
        assert pytest.approx(float(point)) == 42.0
    else:
        assert point.shape == (1,)
        assert pytest.approx(float(point[0])) == 42.0

    reset_residual_metric()


def test_residual_distance_chunk_respects_radius():
    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.6, 0.3],
            [0.4, 0.7],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.8, 1.0, 1.2], dtype=np.float64)
    kernel_diag = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    kernel_full = np.array(
        [
            [1.0, 0.6, 0.5],
            [0.6, 1.1, 0.55],
            [0.5, 0.55, 1.2],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
    )
    configure_residual_correlation(backend)
    host_backend = get_residual_backend()

    query_index = 0
    chunk_indices = np.array([1, 2], dtype=np.int64)
    kernel_row = kernel_provider(np.array([query_index], dtype=np.int64), chunk_indices)[0]

    distances_full = compute_residual_distances_from_kernel(
        host_backend,
        np.array([query_index], dtype=np.int64),
        chunk_indices,
        kernel_row[None, :],
    )[0]

    radius = 0.25
    distances, mask = compute_residual_distances_with_radius(
        host_backend,
        query_index,
        chunk_indices,
        kernel_row,
        radius,
    )

    assert distances.shape == mask.shape
    assert mask.dtype == np.uint8
    for idx, flag in enumerate(mask):
        if flag:
            assert np.isclose(distances[idx], distances_full[idx], atol=1e-9)
        else:
            assert distances[idx] > radius - 1e-9

    radius_large = 1.0
    distances_large, mask_large = compute_residual_distances_with_radius(
        host_backend,
        query_index,
        chunk_indices,
        kernel_row,
        radius_large,
    )
    assert np.all(mask_large == 1)
    assert np.allclose(distances_large, distances_full, atol=1e-9)

    reset_residual_metric()


def test_residual_correlation_helper_computes_distances():
    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 1.1],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.9, 1.1, 0.7], dtype=np.float64)
    kernel_diag = np.array([1.2, 1.3, 1.1], dtype=np.float64)

    kernel_full = np.array(
        [
            [1.2, 0.8, 0.6],
            [0.8, 1.3, 0.5],
            [0.6, 0.5, 1.1],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=1,
    )
    configure_residual_correlation(backend)

    metric = get_metric("residual_correlation")
    tree_backend = DEFAULT_BACKEND
    lhs = tree_backend.asarray([0, 2], dtype=tree_backend.default_int)
    rhs = tree_backend.asarray([1], dtype=tree_backend.default_int)

    distances = metric.pairwise(tree_backend, lhs, rhs)
    distances_np = np.asarray(tree_backend.to_numpy(distances), dtype=np.float64)

    # manual residual correlation computation
    lhs_indices = np.array([0, 2], dtype=np.int64)
    rhs_indices = np.array([1], dtype=np.int64)
    v_lhs = v_matrix[lhs_indices]
    v_rhs = v_matrix[rhs_indices]
    p_lhs = p_diag[lhs_indices]
    p_rhs = p_diag[rhs_indices]
    kernel_vals = kernel_full[np.ix_(lhs_indices, rhs_indices)]
    corr = (kernel_vals - v_lhs @ v_rhs.T) / np.sqrt(np.maximum(p_lhs[:, None] * p_rhs[None, :], 1e-18))
    corr = np.clip(corr, -1.0, 1.0)
    expected = np.sqrt(np.maximum(1.0 - np.abs(corr), 0.0))

    assert distances_np.shape == expected.shape
    assert np.allclose(distances_np, expected, atol=1e-9)

    # pointwise convenience
    pointwise = metric.pointwise(tree_backend, lhs[0], rhs[0])
    pointwise_val = float(tree_backend.to_numpy(pointwise))
    assert np.isclose(pointwise_val, expected[0, 0], atol=1e-9)

    reset_residual_metric()


def test_residual_distance_and_bounds_reuse_kernel_blocks():
    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.8, 0.4],
            [0.3, 0.7],
            [0.5, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.5, 0.7, 0.6, 0.9], dtype=np.float64)
    kernel_diag = np.array([1.1, 1.05, 0.95, 1.2], dtype=np.float64)
    kernel_all = np.array(
        [
            [1.1, 0.4, 0.3, 0.2],
            [0.4, 1.05, 0.25, 0.3],
            [0.3, 0.25, 0.95, 0.35],
            [0.2, 0.3, 0.35, 1.2],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_all[np.ix_(rows, cols)]

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
    )
    configure_residual_correlation(backend)
    host_backend = get_residual_backend()

    lhs = np.array([0, 1], dtype=np.int64)
    rhs = np.array([2, 3], dtype=np.int64)
    kernel_block = kernel_provider(lhs, rhs)

    via_provider, auto_kernel = compute_residual_distances_with_kernel(host_backend, lhs, rhs)
    via_kernel = compute_residual_distances_from_kernel(host_backend, lhs, rhs, kernel_block)
    assert np.allclose(via_provider, via_kernel, atol=1e-9)
    assert np.allclose(auto_kernel, kernel_block, atol=1e-12)

    lower_bounds = compute_residual_lower_bounds_from_kernel(host_backend, lhs, rhs, kernel_block)
    diag_lhs = kernel_diag[lhs]
    diag_rhs = kernel_diag[rhs]
    manual_ratio = kernel_block / np.sqrt(np.maximum(diag_lhs[:, None] * diag_rhs[None, :], 1e-9 * 1e-9))
    manual_ratio = np.clip(manual_ratio, -1.0, 1.0)
    manual_bound = np.sqrt(np.maximum(1.0 - manual_ratio, 0.0))
    assert np.allclose(lower_bounds, manual_bound, atol=1e-12)

    reset_residual_metric()
