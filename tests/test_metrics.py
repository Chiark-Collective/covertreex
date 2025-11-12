import json

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
from covertreex.core.tree import TreeBackend, get_runtime_backend
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    ResidualDistanceTelemetry,
    ResidualGateLookup,
    ResidualGateProfile,
    ResidualWorkspace,
    build_residual_backend,
    compute_whitened_block,
    configure_residual_correlation,
    compute_residual_distances_with_radius,
    compute_residual_distances,
    compute_residual_distances_with_kernel,
    compute_residual_distances_from_kernel,
    compute_residual_lower_bounds_from_kernel,
    compute_residual_pairwise_matrix,
    get_residual_backend,
    set_residual_backend,
)


def test_euclidean_pairwise_matches_manual():
    backend = get_runtime_backend()
    metric = get_metric("euclidean")
    lhs = backend.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=backend.default_float)
    rhs = backend.asarray([[1.0, 1.0], [2.0, 1.0]], dtype=backend.default_float)

    distances = metric.pairwise(backend, lhs, rhs)
    manual = jnp.sqrt(jnp.sum((lhs[:, None, :] - rhs[None, :, :]) ** 2, axis=-1))

    assert distances.shape == (2, 2)
    assert jnp.allclose(distances, manual)


def test_euclidean_pointwise_supports_vector_inputs():
    backend = get_runtime_backend()
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
    backend = get_runtime_backend()
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


def test_sgemm_kernel_provider_matches_reference():
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=np.float64,
    )
    variance = 1.3
    lengthscale = 0.7
    backend = build_residual_backend(
        points,
        seed=0,
        inducing_count=points.shape[0],
        variance=variance,
        lengthscale=lengthscale,
    )
    rows = np.array([0, 3], dtype=np.int64)
    cols = np.array([1, 2], dtype=np.int64)
    kernel_block = backend.kernel_provider(rows, cols)

    diff = points[rows][:, None, :] - points[cols][None, :, :]
    sq = np.sum(diff * diff, axis=2)
    denom = max(lengthscale, 1e-12)
    expected = variance * np.exp(-0.5 * sq / (denom * denom))

    assert kernel_block.dtype == np.float32
    assert np.allclose(kernel_block, expected.astype(np.float32), atol=1e-6)
    assert backend.kernel_points_f32 is not None
    assert backend.kernel_row_norms_f32 is not None

    reset_residual_metric()


def test_force_whitened_records_pairs_when_gate_disabled():
    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.6, 0.2, 0.1],
            [0.3, 0.7, 0.5],
            [0.9, 0.1, 0.4],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.8, 0.9, 1.1], dtype=np.float64)
    kernel_full = np.array(
        [
            [1.0, 0.4, 0.3],
            [0.4, 1.1, 0.2],
            [0.3, 0.2, 1.2],
        ],
        dtype=np.float64,
    )
    kernel_diag = np.diag(kernel_full).copy()

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
    )
    configure_residual_correlation(backend)
    host_backend = get_residual_backend()

    query_index = 0
    chunk_indices = np.array([1, 2], dtype=np.int64)
    kernel_row = kernel_provider(np.array([query_index], dtype=np.int64), chunk_indices)[0]
    workspace = ResidualWorkspace(max_queries=1, max_chunk=chunk_indices.size)

    base_distances, base_mask = compute_residual_distances_with_radius(
        host_backend,
        query_index,
        chunk_indices,
        kernel_row,
        radius=0.5,
        workspace=workspace,
    )

    telemetry = ResidualDistanceTelemetry()
    forced_distances, forced_mask = compute_residual_distances_with_radius(
        host_backend,
        query_index,
        chunk_indices,
        kernel_row,
        radius=0.5,
        workspace=workspace,
        telemetry=telemetry,
        force_whitened=True,
    )

    assert np.allclose(base_distances, forced_distances)
    assert np.array_equal(base_mask, forced_mask)
    assert telemetry.whitened_calls == 1
    assert telemetry.whitened_pairs == chunk_indices.size
    assert telemetry.kernel_pairs == 0

    reset_residual_metric()


def test_residual_gate1_prunes_far_candidates():
    class RecordingKernel:
        def __init__(self, full):
            self.full = full
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
            self.recording = False

        def __call__(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
            if self.recording:
                self.calls.append((tuple(rows.tolist()), tuple(cols.tolist())))
            return self.full[np.ix_(rows, cols)]

    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.5, 0.0],
            [0.4, 0.1],
            [0.1, 0.7],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p_diag = np.full(4, 0.9, dtype=np.float64)
    kernel_full = np.array(
        [
            [1.0, 0.6, 0.2, 0.1],
            [0.6, 1.0, 0.25, 0.2],
            [0.2, 0.25, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    kernel_diag = np.diag(kernel_full).copy()
    provider = RecordingKernel(kernel_full)

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=provider,
        chunk_size=3,
    )
    object.__setattr__(backend, "gate1_enabled", True)
    object.__setattr__(backend, "gate1_alpha", 0.4)
    object.__setattr__(backend, "gate1_margin", 0.0)
    object.__setattr__(backend, "gate1_radius_cap", 1.0)
    object.__setattr__(backend, "gate1_band_eps", 0.0)
    object.__setattr__(backend, "gate1_eps", 1e-6)
    configure_residual_correlation(backend)
    host_backend = get_residual_backend()

    custom_whitened = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.8, 0.0],
            [1.2, 0.0],
        ],
        dtype=np.float32,
    )
    norms32 = np.sqrt(np.sum(custom_whitened * custom_whitened, axis=1)).astype(np.float32)
    object.__setattr__(host_backend, "gate_v32", custom_whitened)
    object.__setattr__(host_backend, "gate_norm32", norms32)

    chunk_ids = np.array([1, 2, 3], dtype=np.int64)
    radius = 0.5
    workspace = ResidualWorkspace(max_queries=1, max_chunk=chunk_ids.size)

    provider.recording = True
    distances_lazy, mask_lazy = compute_residual_distances_with_radius(
        host_backend,
        query_index=0,
        chunk_indices=chunk_ids,
        kernel_row=None,
        radius=radius,
        workspace=workspace,
    )
    provider.recording = False

    assert provider.calls == [((0,), (1,))]

    full_kernel_row = provider(np.array([0], dtype=np.int64), chunk_ids)[0]
    distances_full, mask_full = compute_residual_distances_with_radius(
        host_backend,
        query_index=0,
        chunk_indices=chunk_ids,
        kernel_row=full_kernel_row,
        radius=radius,
        workspace=workspace,
    )

    assert np.array_equal(mask_lazy, mask_full)
    assert np.allclose(distances_lazy, distances_full, atol=1e-9)
    assert mask_lazy.dtype == np.uint8

    reset_residual_metric()


def test_residual_gate_audit_remains_clean(tmp_path):
    class RecordingKernel:
        def __init__(self, full):
            self.full = full

        def __call__(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
            return self.full[np.ix_(rows, cols)]

    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.5, 0.0],
            [0.4, 0.1],
            [0.1, 0.7],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p_diag = np.full(4, 0.9, dtype=np.float64)
    kernel_full = np.array(
        [
            [1.0, 0.6, 0.2, 0.1],
            [0.6, 1.0, 0.25, 0.2],
            [0.2, 0.25, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    kernel_diag = np.diag(kernel_full).copy()
    provider = RecordingKernel(kernel_full)

    profile_path = tmp_path / "gate_profile.json"
    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=provider,
        chunk_size=3,
    )
    object.__setattr__(backend, "gate1_enabled", True)
    object.__setattr__(backend, "gate1_alpha", 0.4)
    object.__setattr__(backend, "gate1_margin", 0.0)
    object.__setattr__(backend, "gate1_radius_cap", 1.0)
    object.__setattr__(backend, "gate1_band_eps", 0.0)
    object.__setattr__(backend, "gate1_eps", 1e-6)
    object.__setattr__(backend, "gate1_audit", True)
    object.__setattr__(backend, "gate_profile_path", str(profile_path))
    object.__setattr__(backend, "gate_profile_bins", 8)
    configure_residual_correlation(backend)
    host_backend = get_residual_backend()

    custom_whitened = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.8, 0.0],
            [1.2, 0.0],
        ],
        dtype=np.float32,
    )
    norms32 = np.sqrt(np.sum(custom_whitened * custom_whitened, axis=1)).astype(np.float32)
    object.__setattr__(host_backend, "gate_v32", custom_whitened)
    object.__setattr__(host_backend, "gate_norm32", norms32)

    chunk_ids = np.array([1, 2, 3], dtype=np.int64)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=chunk_ids.size)

    compute_residual_distances_with_radius(
        host_backend,
        query_index=0,
        chunk_indices=chunk_ids,
        kernel_row=None,
        radius=0.5,
        workspace=workspace,
    )

    profile = host_backend.gate_profile
    assert isinstance(profile, ResidualGateProfile)
    assert profile.false_negative_samples == 0

    reset_residual_metric()


def test_residual_pairwise_matrix_records_telemetry():
    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.6, 0.1],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([1.0, 0.9], dtype=np.float64)
    kernel_full = np.array(
        [
            [1.0, 0.5],
            [0.5, 1.1],
        ],
        dtype=np.float64,
    )

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return kernel_full[np.ix_(rows, cols)]

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=np.diag(kernel_full),
        kernel_provider=kernel_provider,
        chunk_size=1,
    )
    configure_residual_correlation(backend)

    telemetry = ResidualDistanceTelemetry()
    batch_indices = np.array([0, 1], dtype=np.int64)
    matrix = compute_residual_pairwise_matrix(
        host_backend=backend,
        batch_indices=batch_indices,
        telemetry=telemetry,
    )

    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32
    assert telemetry.kernel_calls == 2
    assert telemetry.kernel_pairs == 4
    assert telemetry.whitened_pairs == 0
    assert telemetry.kernel_seconds >= 0.0

    reset_residual_metric()


def test_residual_backend_uses_float32_staging():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    backend = build_residual_backend(
        points,
        seed=0,
        inducing_count=2,
        variance=1.0,
        lengthscale=1.0,
    )

    assert backend.v_matrix.dtype == np.float32
    assert backend.p_diag.dtype == np.float32
    assert backend.kernel_diag.dtype == np.float32
    assert backend.v_norm_sq.dtype == np.float32
    assert backend.v_matrix_f64 is None

    _ = backend.v_matrix_view(np.float64)
    assert backend.v_matrix_f64 is not None

    reset_residual_metric()




def test_compute_whitened_block_matches_reference():
    reset_residual_metric()
    set_residual_backend(None)

    v_matrix = np.array(
        [
            [0.3, 0.1, 0.5],
            [0.2, 0.6, 0.4],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    p_diag = np.full(3, 0.8, dtype=np.float64)
    kernel_diag = np.ones(3, dtype=np.float64)

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        # unused but required by the backend
        return np.eye(len(rows), len(cols), dtype=np.float64)

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=3,
    )
    object.__setattr__(backend, "gate1_enabled", True)
    configure_residual_correlation(backend)
    host_backend = get_residual_backend()

    queries = np.array([0, 1], dtype=np.int64)
    chunk_ids = np.array([1, 2], dtype=np.int64)
    workspace = ResidualWorkspace(max_queries=queries.size, max_chunk=chunk_ids.size)

    block = compute_whitened_block(
        host_backend,
        query_indices=queries,
        chunk_indices=chunk_ids,
        workspace=workspace,
    )

    ref_rows = []
    for q in queries:
        q_vec = host_backend.gate_v32[q]
        chunk_vec = host_backend.gate_v32[chunk_ids]
        diff = chunk_vec - q_vec
        ref_rows.append(np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64)))
    reference = np.vstack(ref_rows).astype(np.float32)

    assert block.shape == (queries.size, chunk_ids.size)
    assert block.dtype == np.float32
    assert np.allclose(block, reference, atol=1e-6)

    reset_residual_metric()


def test_residual_gate_profile_records_samples(tmp_path):
    profile_path = tmp_path / "gate_profile.json"
    profile = ResidualGateProfile.create(bins=4, radius_max=1.0, path=str(profile_path), radius_eps=1e-6)
    residuals = np.array([0.05, 0.15, 0.35], dtype=np.float64)
    whitened = np.array([0.01, 0.08, 0.2], dtype=np.float64)
    mask = np.array([1, 1, 0], dtype=np.uint8)
    profile.record_chunk(
        residual_distances=residuals,
        whitened_distances=whitened,
        inclusion_mask=mask,
    )
    profile.annotate_metadata(run_id="test-profile", tree_points=8)
    profile.dump()
    payload = json.loads(profile_path.read_text())
    assert payload["schema"] == 2
    assert payload["samples_total"] == 2
    assert max(payload["max_whitened"]) >= 0.08
    assert payload["metadata"]["run_id"] == "test-profile"
    assert "quantiles" in payload and payload["quantiles"]
    first_series = next(iter(payload["quantiles"].values()))
    assert len(first_series) == payload["bins"]
    assert len(payload.get("quantile_counts", [])) == payload["bins"]


def test_residual_gate_lookup_thresholds_monotonic(tmp_path):
    profile_path = tmp_path / "gate_profile.json"
    profile = ResidualGateProfile.create(bins=3, radius_max=1.0, path=str(profile_path), radius_eps=1e-6)
    residuals = np.array([0.05, 0.2, 0.4, 0.8], dtype=np.float64)
    whitened = np.array([0.02, 0.1, 0.15, 0.3], dtype=np.float64)
    mask = np.array([1, 1, 1, 1], dtype=np.uint8)
    profile.record_chunk(
        residual_distances=residuals,
        whitened_distances=whitened,
        inclusion_mask=mask,
    )
    profile.dump()
    lookup = ResidualGateLookup.load(str(profile_path), margin=0.01, keep_pct=95.0, prune_pct=99.9)
    keep_small, prune_small = lookup.thresholds(0.1)
    keep_large, prune_large = lookup.thresholds(0.7)
    assert keep_large >= keep_small
    assert prune_large >= prune_small


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
    tree_backend = get_runtime_backend()
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
