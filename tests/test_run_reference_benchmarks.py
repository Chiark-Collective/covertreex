from __future__ import annotations

import sys
from pathlib import Path

from tools import run_reference_benchmarks as rrb


def test_query_job_build_command(tmp_path: Path) -> None:
    job = rrb.BenchmarkJob(
        name="unit_query",
        metric="euclidean",
        tree_points=2048,
        runner="queries",
        cli_args=("--batch-order", "hilbert"),
    )
    log_path = tmp_path / "unit.jsonl"
    cmd = job.build_command(log_path, python_executable="/usr/bin/python")
    assert cmd[:4] == ["/usr/bin/python", "-m", "cli.queries", "--metric"]
    assert "--batch-order" in cmd
    assert "--residual-gate" in cmd
    assert str(log_path) in cmd


def test_guardrail_job_build_command(tmp_path: Path) -> None:
    job = rrb.BenchmarkJob(
        name="guard",
        runner="guardrail",
        cli_args=("--skip-run",),
    )
    log_path = tmp_path / "guard.jsonl"
    cmd = job.build_command(log_path, python_executable=sys.executable)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("residual_guardrail_check.py")
    assert "--log-file" in cmd
    assert str(log_path) in cmd


def test_select_jobs_handles_unknown() -> None:
    jobs = rrb._jobs()
    selected = list(rrb._select_jobs(jobs, ["queries_2048_diag_on"]))
    assert selected and selected[0].name == "queries_2048_diag_on"
    try:
        list(rrb._select_jobs(jobs, ["missing_job"]))
    except ValueError as exc:
        assert "Unknown job" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unknown job")
