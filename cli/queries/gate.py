from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from covertreex.telemetry.schemas import RESIDUAL_GATE_PROFILE_SCHEMA_ID


def _append_gate_profile_log(
    *,
    profile_json_path: str | None,
    profile_log_path: str | None,
    run_id: str,
    log_metadata: Dict[str, Any],
    runtime_snapshot: Dict[str, Any],
    batch_log_path: str | None,
) -> None:
    if not profile_json_path or not profile_log_path:
        return
    profile_path = Path(profile_json_path)
    if not profile_path.exists():
        print(f"[queries] gate profile path {profile_json_path} not found; skipping JSONL emission")
        return
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[queries] failed to read gate profile {profile_json_path}: {exc}")
        return
    if not isinstance(payload, dict) or "radius_bin_edges" not in payload:
        print(f"[queries] gate profile {profile_json_path} missing expected fields; skipping")
        return
    record = dict(payload)
    record["schema_id"] = RESIDUAL_GATE_PROFILE_SCHEMA_ID
    record["run_id"] = run_id
    record["profile_path"] = str(profile_path)
    if batch_log_path:
        record["batch_log_path"] = batch_log_path
    metadata = dict(record.get("metadata", {}))
    metadata.setdefault("run_id", run_id)
    for key, value in log_metadata.items():
        metadata.setdefault(key, value)
    metadata.setdefault("runtime_snapshot", runtime_snapshot)
    record["metadata"] = metadata
    log_path = Path(profile_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")
    print(f"[queries] appended gate profile snapshot to {log_path}")


__all__ = ["_append_gate_profile_log"]
