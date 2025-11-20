#!/usr/bin/env python3
"""Run the residual 4k guardrail with curated env + telemetry summaries.

This helper wraps ``tools/residual_guardrail_check.py`` so every run:
1. Uses the documented Hilbert preset environment (dense traversal, gate off).
2. Writes both the JSONL telemetry and the guardrail summary JSON under one folder.
3. Captures the command/env metadata for later audits.

Additional cli.queries overrides can be forwarded via ``--forward-cli-args``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
GUARDRAIL_SCRIPT = TOOLS_DIR / "residual_guardrail_check.py"

DEFAULT_ENV = {
    "COVERTREEX_BACKEND": "numpy",
    "COVERTREEX_ENABLE_NUMBA": "1",
    "COVERTREEX_SCOPE_CHUNK_TARGET": "0",
    "COVERTREEX_ENABLE_SPARSE_TRAVERSAL": "0",
    "COVERTREEX_BATCH_ORDER": "hilbert",
    "COVERTREEX_PREFIX_SCHEDULE": "doubling",
    "COVERTREEX_RESIDUAL_FORCE_WHITENED": "0",
}


def _default_output_dir() -> Path:
    return REPO_ROOT / "artifacts" / "benchmarks" / "guardrails"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_env_overrides(values: Sequence[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid env override '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid env override '{item}', empty key")
        overrides[key] = value
    return overrides


def _gather_cli_args(defaults: Sequence[str], forward_chunks: Sequence[str]) -> List[str]:
    cli_args = list(defaults)
    for chunk in forward_chunks:
        cli_args.extend(shlex.split(chunk))
    return cli_args


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional run identifier; defaults to guardrail_<timestamp>",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--tree-points", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument("--min-whitened-coverage", type=float, default=0.95)
    parser.add_argument("--max-median-semisort-ms", type=float, default=1000.0)
    parser.add_argument("--min-dominated-batches", type=int, default=5)
    parser.add_argument(
        "--allow-missing-pairwise-reuse",
        action="store_true",
        help="Permit conflict_pairwise_reused=0 batches.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only evaluate an existing log/summary (no cli.queries invocation).",
    )
    parser.add_argument(
        "--forward-cli-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Extra cli.queries args (quote the chunk; parsed with shlex).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment overrides applied on top of the defaults.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    run_id = args.run_id or f"guardrail_residual_{timestamp}"
    log_path = output_dir / f"{run_id}.jsonl"
    summary_path = output_dir / f"{run_id}_summary.json"
    metadata_path = output_dir / f"{run_id}_metadata.json"

    cli_args = _gather_cli_args((), args.forward_cli_args)

    cmd: List[str] = [
        args.python,
        str(GUARDRAIL_SCRIPT),
        "--log-file",
        str(log_path),
        "--summary-json",
        str(summary_path),
        "--dimension",
        str(args.dimension),
        "--tree-points",
        str(args.tree_points),
        "--queries",
        str(args.queries),
        "--batch-size",
        str(args.batch_size),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--baseline",
        args.baseline,
        "--min-whitened-coverage",
        str(args.min_whitened_coverage),
        "--max-median-semisort-ms",
        str(args.max_median_semisort_ms),
        "--min-dominated-batches",
        str(args.min_dominated_batches),
    ]
    if args.allow_missing_pairwise_reuse:
        cmd.append("--no-require-pairwise-reuse")
    if args.skip_run:
        cmd.append("--skip-run")
    if cli_args:
        cmd.extend(["--extra-cli-args", *cli_args])

    env_overrides = _parse_env_overrides(args.env)
    proc_env = os.environ.copy()
    proc_env.update(DEFAULT_ENV)
    proc_env.update(env_overrides)
    env_snapshot = {key: proc_env[key] for key in DEFAULT_ENV}
    env_snapshot.update(env_overrides)

    print(f"[guardrail-suite] log={log_path} summary={summary_path}")
    print(f"[guardrail-suite] env overrides: {json.dumps(env_snapshot, indent=2)}")
    print(f"[guardrail-suite] running: {shlex.join(cmd)}")
    result = subprocess.run(cmd, env=proc_env)
    return_code = result.returncode

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "log_file": str(log_path),
        "summary_file": str(summary_path),
        "command": cmd,
        "env": env_snapshot,
        "parameters": {
            "dimension": args.dimension,
            "tree_points": args.tree_points,
            "queries": args.queries,
            "batch_size": args.batch_size,
            "k": args.k,
            "seed": args.seed,
            "baseline": args.baseline,
            "min_whitened_coverage": args.min_whitened_coverage,
            "max_median_semisort_ms": args.max_median_semisort_ms,
            "min_dominated_batches": args.min_dominated_batches,
            "allow_missing_pairwise_reuse": args.allow_missing_pairwise_reuse,
            "skip_run": args.skip_run,
            "forward_cli_args": cli_args,
        },
        "return_code": return_code,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[guardrail-suite] metadata written to {metadata_path}")
    if return_code != 0:
        print(f"[guardrail-suite] guardrail script exited with {return_code}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
