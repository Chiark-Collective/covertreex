import os

import pytest

from covertreex import config as cx_config


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "COVERTREEX_BACKEND",
        "COVERTREEX_PRECISION",
        "COVERTREEX_DEVICE",
        "COVERTREEX_ENABLE_NUMBA",
        "COVERTREEX_ENABLE_SPARSE_TRAVERSAL",
        "COVERTREEX_ENABLE_DIAGNOSTICS",
        "COVERTREEX_LOG_LEVEL",
        "COVERTREEX_MIS_SEED",
        "COVERTREEX_CONFLICT_GRAPH_IMPL",
        "COVERTREEX_SCOPE_SEGMENT_DEDUP",
        "COVERTREEX_SCOPE_CHUNK_TARGET",
        "COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS",
        "COVERTREEX_METRIC",
        "COVERTREEX_RESIDUAL_GATE1",
        "COVERTREEX_RESIDUAL_GATE1_ALPHA",
        "COVERTREEX_RESIDUAL_GATE1_MARGIN",
        "COVERTREEX_RESIDUAL_GATE1_EPS",
        "COVERTREEX_RESIDUAL_GATE1_AUDIT",
        "COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP",
        "COVERTREEX_RESIDUAL_RADIUS_FLOOR",
        "COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH",
        "COVERTREEX_RESIDUAL_GATE1_PROFILE_BINS",
        "COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH",
        "COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN",
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_runtime_config_defaults(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()

    assert runtime.backend == "numpy"
    assert runtime.precision == "float64"
    assert runtime.jax_enable_x64 is True
    assert runtime.devices == ()
    assert runtime.primary_platform is None
    assert runtime.enable_sparse_traversal is False
    assert runtime.scope_chunk_target == 0
    assert runtime.metric == "euclidean"
    assert runtime.batch_order_strategy == "hilbert"
    assert runtime.batch_order_seed is None
    assert runtime.prefix_schedule == "adaptive"
    assert runtime.prefix_density_low == pytest.approx(0.15)
    assert runtime.prefix_density_high == pytest.approx(0.55)
    assert runtime.prefix_growth_small == pytest.approx(1.25)
    assert runtime.prefix_growth_mid == pytest.approx(1.75)
    assert runtime.prefix_growth_large == pytest.approx(2.25)
    assert runtime.residual_gate1_enabled is False
    assert runtime.residual_gate1_alpha == pytest.approx(4.0)
    assert runtime.residual_gate1_margin == pytest.approx(0.05)
    assert runtime.residual_gate1_eps == pytest.approx(1e-6)
    assert runtime.residual_gate1_audit is False
    assert runtime.residual_gate1_radius_cap == pytest.approx(1.0)
    assert runtime.residual_radius_floor == pytest.approx(1e-3)
    assert runtime.residual_gate1_profile_path is None
    assert runtime.residual_gate1_profile_bins == 256
    assert runtime.residual_gate1_lookup_path is None
    assert runtime.residual_gate1_lookup_margin == pytest.approx(0.02)


def test_runtime_context_uses_numpy_backend_by_default(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    cx_config.reset_runtime_context()

    backend = cx_config.runtime_context().get_backend()

    assert backend.name == "numpy"


def test_precision_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_PRECISION", "float32")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()

    assert runtime.precision == "float32"
    assert runtime.jax_enable_x64 is False


def test_device_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_BACKEND", "jax")
    # Request an unlikely device to trigger fallback logic.
    monkeypatch.setenv("COVERTREEX_DEVICE", "gpu:99")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()

    assert runtime.primary_platform == "cpu"


def test_invalid_backend(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_BACKEND", "invalid-backend")
    cx_config.reset_runtime_config_cache()

    with pytest.raises(ValueError):
        cx_config.runtime_config()


def test_mis_seed_parsing(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_MIS_SEED", "123")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.mis_seed == 123


def test_disable_diagnostics_flag(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_ENABLE_DIAGNOSTICS", "0")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.enable_diagnostics is False


def test_runtime_config_from_env_matches_cached(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_LOG_LEVEL", "warning")
    cx_config.reset_runtime_config_cache()

    direct = cx_config.RuntimeConfig.from_env()
    cached = cx_config.runtime_config()

    assert direct == cached
    assert direct.log_level == "WARNING"
    assert cached.log_level == "WARNING"


def test_describe_runtime_reports_expected_fields(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    cx_config.reset_runtime_config_cache()

    summary = cx_config.describe_runtime()

    assert summary["backend"] == "numpy"
    assert summary["precision"] == "float64"
    assert summary["log_level"] == "INFO"
    assert summary["enable_sparse_traversal"] is False
    assert summary["enable_diagnostics"] is True
    assert summary["jax_enable_x64"] is True
    assert "primary_platform" in summary
    assert summary["primary_platform"] is None
    assert summary["conflict_graph_impl"] == "dense"
    assert summary["scope_chunk_target"] == 0
    assert summary["scope_chunk_max_segments"] == 512
    assert summary["metric"] == "euclidean"
    assert summary["residual_gate1_enabled"] is False
    assert summary["residual_gate1_alpha"] == pytest.approx(4.0)
    assert summary["residual_gate1_margin"] == pytest.approx(0.05)
    assert summary["residual_gate1_eps"] == pytest.approx(1e-6)
    assert summary["residual_gate1_radius_cap"] == pytest.approx(1.0)
    assert summary["residual_radius_floor"] == pytest.approx(1e-3)
    assert summary["residual_gate1_profile_path"] is None
    assert summary["residual_gate1_profile_bins"] == 256
    assert summary["residual_gate1_lookup_path"] is None
    assert summary["residual_gate1_lookup_margin"] == pytest.approx(0.02)


def test_conflict_graph_impl_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "segmented")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.conflict_graph_impl == "segmented"


def test_conflict_graph_impl_grid(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "grid")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.conflict_graph_impl == "grid"


def test_batch_order_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_BATCH_ORDER", "hilbert")
    monkeypatch.setenv("COVERTREEX_BATCH_ORDER_SEED", "42")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.batch_order_strategy == "hilbert"
    assert runtime.batch_order_seed == 42


def test_prefix_schedule_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_PREFIX_SCHEDULE", "adaptive")
    monkeypatch.setenv("COVERTREEX_PREFIX_DENSITY_LOW", "0.1")
    monkeypatch.setenv("COVERTREEX_PREFIX_DENSITY_HIGH", "0.7")
    monkeypatch.setenv("COVERTREEX_PREFIX_GROWTH_SMALL", "0.4")
    monkeypatch.setenv("COVERTREEX_PREFIX_GROWTH_MID", "1.2")
    monkeypatch.setenv("COVERTREEX_PREFIX_GROWTH_LARGE", "1.8")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.prefix_schedule == "adaptive"
    assert runtime.prefix_density_low == pytest.approx(0.1)
    assert runtime.prefix_density_high == pytest.approx(0.7)
    assert runtime.prefix_growth_small == pytest.approx(0.4)
    assert runtime.prefix_growth_mid == pytest.approx(1.2)
    assert runtime.prefix_growth_large == pytest.approx(1.8)


def test_scope_segment_dedupe_toggle(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_SCOPE_SEGMENT_DEDUP", "0")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.scope_segment_dedupe is False


def test_scope_chunk_target_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "16384")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.scope_chunk_target == 16_384


def test_scope_chunk_target_disable(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_TARGET", "0")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.scope_chunk_target == 0


def test_scope_chunk_max_segments_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS", "128")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.scope_chunk_max_segments == 128


def test_scope_chunk_max_segments_disable(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS", "0")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.scope_chunk_max_segments == 0


def test_metric_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_METRIC", "residual_correlation")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.metric == "residual_correlation"


def test_sparse_traversal_toggle(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_ENABLE_SPARSE_TRAVERSAL", "1")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.enable_sparse_traversal is True


def test_residual_gate1_env_overrides(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1", "1")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_ALPHA", "1.5")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_MARGIN", "0.2")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_EPS", "0.001")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_AUDIT", "1")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_RADIUS_CAP", "0.75")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_RADIUS_FLOOR", "0.25")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_PROFILE_PATH", "/tmp/profile.json")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_PROFILE_BINS", "64")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_LOOKUP_PATH", "/tmp/lookup.json")
    monkeypatch.setenv("COVERTREEX_RESIDUAL_GATE1_LOOKUP_MARGIN", "0.5")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.residual_gate1_enabled is True
    assert runtime.residual_gate1_alpha == pytest.approx(1.5)
    assert runtime.residual_gate1_margin == pytest.approx(0.2)
    assert runtime.residual_gate1_eps == pytest.approx(0.001)
    assert runtime.residual_gate1_audit is True
    assert runtime.residual_gate1_radius_cap == pytest.approx(0.75)
    assert runtime.residual_radius_floor == pytest.approx(0.25)
    assert runtime.residual_gate1_profile_path == "/tmp/profile.json"
    assert runtime.residual_gate1_profile_bins == 64
    assert runtime.residual_gate1_lookup_path == "/tmp/lookup.json"
    assert runtime.residual_gate1_lookup_margin == pytest.approx(0.5)
