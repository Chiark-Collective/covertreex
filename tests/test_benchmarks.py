import json
import numpy as np
import sys
import pytest

jax = pytest.importorskip("jax")

from benchmarks.batch_ops import benchmark_delete, benchmark_insert
from benchmarks.queries import (
    BenchmarkLogWriter,
    _build_tree,
    benchmark_knn_latency,
    run_baseline_comparisons,
)
from benchmarks import runtime_breakdown
from covertreex import config as cx_config
from covertreex.baseline import has_gpboost_cover_tree


@pytest.fixture(autouse=True)
def _ensure_euclidean_metric(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COVERTREEX_METRIC", "euclidean")
    cx_config.reset_runtime_config_cache()
    yield
    cx_config.reset_runtime_config_cache()


def test_benchmark_insert_delete_smoke():
    tree, insert_result = benchmark_insert(
        dimension=3,
        batch_size=4,
        batches=2,
        seed=0,
    )
    assert tree.num_points == insert_result.points_processed
    assert insert_result.mode == "insert"
    assert insert_result.batches == 2

    tree, delete_result = benchmark_delete(
        tree,
        batch_size=2,
        batches=1,
        seed=1,
    )
    assert delete_result.mode == "delete"
    assert delete_result.batches == 1
    assert delete_result.points_processed == 2


def test_benchmark_knn_latency_smoke():
    _, result = benchmark_knn_latency(
        dimension=3,
        tree_points=32,
        query_count=4,
        k=2,
        batch_size=8,
        seed=0,
    )
    assert result.queries == 4
    assert result.k == 2
    assert result.latency_ms >= 0.0
    assert result.build_seconds is not None


def test_run_baseline_comparisons_sequential():
    points = np.random.default_rng(0).normal(size=(16, 3))
    queries = np.random.default_rng(1).normal(size=(4, 3))
    results = run_baseline_comparisons(points, queries, k=2, mode="sequential")
    assert len(results) == 1
    baseline = results[0]
    assert baseline.name == "sequential"
    assert baseline.latency_ms >= 0.0


def test_run_baseline_comparisons_gpboost():
    if not has_gpboost_cover_tree():
        pytest.skip("GPBoost baseline requires numba")
    points = np.random.default_rng(0).normal(size=(16, 3))
    queries = np.random.default_rng(1).normal(size=(4, 3))
    results = run_baseline_comparisons(points, queries, k=2, mode="gpboost")
    assert len(results) == 1
    baseline = results[0]
    assert baseline.name == "gpboost"
    assert baseline.latency_ms >= 0.0


def test_runtime_breakdown_csv_output(tmp_path, monkeypatch):
    csv_path = tmp_path / "metrics.csv"
    png_path = tmp_path / "plot.png"
    monkeypatch.setenv("COVERTREEX_BACKEND", "numpy")
    monkeypatch.setenv("COVERTREEX_ENABLE_NUMBA", "1")
    cx_config.reset_runtime_config_cache()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    argv = [
        "runtime_breakdown",
        "--dimension",
        "2",
        "--tree-points",
        "16",
        "--batch-size",
        "4",
        "--queries",
        "8",
        "--k",
        "2",
        "--seed",
        "5",
        "--output",
        str(png_path),
        "--csv-output",
        str(csv_path),
        "--skip-external",
        "--skip-gpboost",
        "--skip-jax",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runtime_breakdown.main()
    cx_config.reset_runtime_config_cache()
    assert csv_path.exists()
    contents = csv_path.read_text().strip().splitlines()
    chunk_tail = ",".join(
        [
            "traversal_chunk_segments_warmup",
            "traversal_chunk_segments_steady",
            "traversal_chunk_emitted_warmup",
            "traversal_chunk_emitted_steady",
            "traversal_chunk_max_members_warmup",
            "traversal_chunk_max_members_steady",
            "conflict_chunk_segments_warmup",
            "conflict_chunk_segments_steady",
            "conflict_chunk_emitted_warmup",
            "conflict_chunk_emitted_steady",
            "conflict_chunk_max_members_warmup",
            "conflict_chunk_max_members_steady",
        ]
    )
    expected_tail = (
        "label,build_warmup_seconds,build_steady_seconds,build_total_seconds," "query_warmup_seconds,"
        "query_steady_seconds,build_cpu_seconds,build_cpu_utilisation," "build_rss_delta_bytes,"
        "build_max_rss_bytes,query_cpu_seconds,query_cpu_utilisation," "query_rss_delta_bytes,"
        "query_max_rss_bytes," + chunk_tail
    )
    header = contents[0]
    if header.startswith("run,"):
        assert header == f"run,{expected_tail}"
        data_rows = [row.split(",", 1)[1] for row in contents[1:] if "," in row]
    else:
        assert header == expected_tail
        data_rows = contents[1:]
    assert any(row.startswith("PCCT") for row in data_rows)


def test_benchmark_log_writer_emits_json(tmp_path):
    log_path = tmp_path / "batches.jsonl"
    writer = BenchmarkLogWriter(str(log_path))
    try:
        _build_tree(
            dimension=2,
            tree_points=8,
            batch_size=4,
            seed=0,
            log_writer=writer,
        )
    finally:
        writer.close()

    contents = log_path.read_text().strip().splitlines()
    assert contents
    first_entry = json.loads(contents[0])
    assert first_entry["batch_index"] == 0
    assert "traversal_ms" in first_entry
    assert "rss_bytes" in first_entry or "rss_delta_bytes" in first_entry
