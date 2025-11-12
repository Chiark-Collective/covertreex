from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.ingest_residual_gate_profile import (
    load_profile_records,
    merge_profile_records,
)


def _sample_record(run_id: str, max_scale: float) -> dict:
    edges = np.linspace(0.0, 1.0, 5).tolist()
    max_whitened = (np.array([0.1, 0.2, 0.3, 0.4]) * max_scale).tolist()
    max_ratio = (np.array([1.0, 1.1, 1.2, 1.3]) * max_scale).tolist()
    counts = [10, 20, 30, 40]
    return {
        "schema_id": "covertreex.residual_gate_profile.v1",
        "run_id": run_id,
        "radius_bin_edges": edges,
        "max_whitened": max_whitened,
        "max_ratio": max_ratio,
        "counts": counts,
        "samples_total": sum(counts),
        "metadata": {"run_id": run_id, "tree_points": 32768},
    }


def test_merge_profile_records(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    records = [
        _sample_record("run-a", 1.0),
        _sample_record("run-b", 1.5),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    loaded = load_profile_records([str(path)])
    assert len(loaded) == 2
    payload = merge_profile_records(loaded, metadata_overrides={"dataset": "hilbert32k"})
    assert payload["bins"] == 4
    assert payload["samples_total"] == sum(r["samples_total"] for r in records)
    assert payload["metadata"]["dataset"] == "hilbert32k"
    # max_whitened should reflect the larger scale record (monotonic cumulative)
    assert np.isclose(payload["max_whitened"][-1], 0.6)
    assert set(payload["metadata"]["run_ids"]) == {"run-a", "run-b"}
