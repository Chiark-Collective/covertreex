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


def test_conflict_graph_impl_override(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("COVERTREEX_CONFLICT_GRAPH_IMPL", "segmented")
    cx_config.reset_runtime_config_cache()

    runtime = cx_config.runtime_config()
    assert runtime.conflict_graph_impl == "segmented"


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
