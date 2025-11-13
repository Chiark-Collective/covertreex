from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from covertreex import config as cx_config

from cli.queries.telemetry import (
    CLITelemetryHandles,
    ResidualTraversalTelemetry,
    initialise_cli_telemetry,
)


class _DummyTimings:
    whitened_block_pairs = 1024.0
    whitened_block_seconds = 0.25
    whitened_block_calls = 8
    kernel_provider_pairs = 64.0
    kernel_provider_seconds = 0.05
    kernel_provider_calls = 2


class _DummyPlan:
    def __init__(self, *, pairwise_reused: int) -> None:
        traversal = type("Traversal", (), {"timings": _DummyTimings()})()
        conflict_timings = type("ConflictTimings", (), {"pairwise_reused": pairwise_reused})()
        conflict_graph = type("ConflictGraph", (), {"timings": conflict_timings})()
        self.traversal = traversal
        self.conflict_graph = conflict_graph


def test_summary_reports_pairwise_reuse_line() -> None:
    telemetry = ResidualTraversalTelemetry()
    plan = _DummyPlan(pairwise_reused=1)
    telemetry.observe_plan(plan, batch_index=0, batch_size=10)

    lines = telemetry.render_summary()

    assert lines[-1] == "  conflict pairwise reuse: 1/1 batches (100.0%)"


def test_observe_plan_raises_when_pairwise_missing(capsys: pytest.CaptureFixture[str]) -> None:
    telemetry = ResidualTraversalTelemetry()
    plan = _DummyPlan(pairwise_reused=0)

    with pytest.raises(RuntimeError, match="conflict_pairwise_reused"):
        telemetry.observe_plan(plan, batch_index=3, batch_size=64)

    captured = capsys.readouterr()
    assert "conflict_pairwise_reused=0" in captured.err


def test_initialise_cli_telemetry_creates_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERTREEX_ARTIFACT_ROOT", str(tmp_path))
    args = SimpleNamespace(
        metric="euclidean",
        no_log_file=False,
        log_file=None,
        residual_scope_cap_output=None,
        residual_scope_cap_percentile=0.5,
        residual_scope_cap_margin=0.05,
        tree_points=32,
        batch_size=8,
        seed=0,
        build_mode="batch",
    )

    handles = initialise_cli_telemetry(
        args=args,
        run_id="pcct-test",
        runtime_snapshot={"backend": "numpy"},
        log_metadata={"benchmark": "cli.queries"},
    )
    assert handles.log_path is not None
    assert Path(handles.log_path).exists()
    handles.close()


def test_initialise_cli_telemetry_scope_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERTREEX_ARTIFACT_ROOT", str(tmp_path))
    cx_config.reset_runtime_context()
    args = SimpleNamespace(
        metric="residual",
        no_log_file=True,
        log_file=None,
        residual_scope_cap_output="caps.json",
        residual_scope_cap_percentile=0.5,
        residual_scope_cap_margin=0.05,
        residual_scope_cap_default=1.0,
        tree_points=16,
        batch_size=4,
        seed=0,
        build_mode="batch",
    )

    handles = initialise_cli_telemetry(
        args=args,
        run_id="pcct-test",
        runtime_snapshot={"backend": "numpy"},
        log_metadata={"benchmark": "cli.queries"},
    )
    assert handles.scope_cap_recorder is not None
    handles.close()
