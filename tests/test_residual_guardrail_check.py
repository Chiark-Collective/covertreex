from __future__ import annotations

from pathlib import Path

import json

from tools import residual_guardrail_check as guardrail


def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "log.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_parse_guardrail_metrics(tmp_path: Path) -> None:
    records = [
        {"dominated": 0, "traversal_whitened_block_pairs": 0},
        {
            "dominated": 1,
            "traversal_semisort_ms": 20.0,
            "traversal_whitened_block_pairs": 950,
            "traversal_kernel_provider_pairs": 1000,
            "traversal_gate1_pruned": 5,
            "conflict_pairwise_reused": 1,
        },
        {
            "dominated": 1,
            "traversal_semisort_ms": 40.0,
            "traversal_whitened_block_pairs": 900,
            "traversal_kernel_provider_pairs": 1000,
            "traversal_gate1_pruned": 7,
            "conflict_pairwise_reused": 1,
        },
    ]
    path = _write_log(tmp_path, records)
    metrics = guardrail.parse_guardrail_metrics(path)
    assert metrics.dominated_batches == 2
    assert metrics.semisort_median_ms == 30.0
    assert metrics.whitened_coverage == 0.925
    assert metrics.gate1_pruned_total == 12
    assert metrics.all_pairwise_reused


def test_evaluate_metrics_reports_failures() -> None:
    metrics = guardrail.GuardrailMetrics(
        dominated_batches=3,
        whitened_pairs_sum=80,
        kernel_pairs_sum=200,
        semisort_median_ms=1500.0,
        gate1_pruned_total=0,
        pairwise_reused_batches=1,
    )
    failures = guardrail.evaluate_metrics(
        metrics,
        min_whitened_coverage=0.95,
        max_median_semisort_ms=1000.0,
        require_gate1_prunes=True,
        require_pairwise_reuse=True,
        min_dominated_batches=5,
    )
    assert len(failures) == 5
    assert "dominated batches" in failures[0]
    assert "whitened coverage" in failures[1]
    assert "median traversal_semisort_ms" in failures[2]
    assert "gate1_pruned" in failures[3]
