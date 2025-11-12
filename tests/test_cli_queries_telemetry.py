from __future__ import annotations

import pytest

from cli.queries.telemetry import ResidualTraversalTelemetry


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
