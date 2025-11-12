import pytest

from tools import export_benchmark_diagnostics as diag


def test_aggregate_requires_pairwise_reuse_for_residual() -> None:
    records = [
        {"metric": "residual", "conflict_pairwise_reused": 1},
        {"metric": "residual", "conflict_pairwise_reused": 0},
    ]

    with pytest.raises(ValueError, match="conflict_pairwise_reused"):
        diag._aggregate(records)


def test_aggregate_allows_non_residual_without_reuse() -> None:
    records = [
        {"metric": "euclidean", "conflict_pairwise_reused": 0},
        {"metric": "euclidean", "conflict_pairwise_reused": 1},
    ]

    aggregates = diag._aggregate(records)

    assert aggregates["metric"] == "euclidean"
    assert aggregates["pairwise_reused_batches"] == 1
    assert aggregates["pairwise_reused_ratio"] == 0.5
