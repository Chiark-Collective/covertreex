from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.queries.app import app


def test_cli_allows_euclidean_with_gate_off(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    invoked = {}

    def fake_run_queries(opts) -> None:  # type: ignore[override]
        invoked["metric"] = opts.metric

    monkeypatch.setattr("cli.queries.app.run_queries", fake_run_queries)
    log_file = tmp_path / "dummy.jsonl"
    result = runner.invoke(
        app,
        [
            "--metric",
            "euclidean",
            "--tree-points",
            "8",
            "--batch-size",
            "4",
            "--queries",
            "8",
            "--k",
            "1",
            "--seed",
            "0",
            "--baseline",
            "none",
            "--log-file",
            str(log_file),
            "--residual-gate",
            "off",
        ],
    )
    assert result.exit_code == 0
    assert invoked["metric"] == "euclidean"
