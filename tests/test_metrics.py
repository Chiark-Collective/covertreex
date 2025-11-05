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
