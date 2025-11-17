import pytest

from covertreex.api import Runtime
from profiles import (
    ProfileNotFoundError,
    apply_overrides_to_model,
    available_profiles,
    load_profile,
    parse_override_expression,
    parse_override_expressions,
)


def test_available_profiles_exposes_expected_entries() -> None:
    names = set(available_profiles())
    assert {"default", "cpu-debug", "residual-fast", "residual-gold"}.issubset(names)


def test_load_profile_returns_runtime_model() -> None:
    model = load_profile("cpu-debug")
    assert model.backend == "numpy"
    assert model.diagnostics.enabled is True
    assert model.residual.gate1_enabled is False


def test_load_profile_missing_raises() -> None:
    with pytest.raises(ProfileNotFoundError):
        load_profile("does-not-exist")


def test_override_parser_supports_numbers() -> None:
    key, value = parse_override_expression("residual.gate1_alpha=2.5")
    assert key == "residual.gate1_alpha"
    assert value == pytest.approx(2.5)


def test_apply_overrides_to_model_updates_nested_fields() -> None:
    model = load_profile("default")
    overrides = parse_override_expressions(
        ["diagnostics.enabled=true", "residual.gate1_enabled=true"]
    )
    updated = apply_overrides_to_model(model, overrides)
    assert updated.diagnostics.enabled is True
    assert updated.residual.gate1_enabled is True


def test_runtime_from_profile_accepts_override_expressions() -> None:
    runtime = Runtime.from_profile("default", overrides=["enable_sparse_traversal=true"])
    config = runtime.to_config()
    assert config.enable_sparse_traversal is True
