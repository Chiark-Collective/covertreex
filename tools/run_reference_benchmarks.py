#!/usr/bin/env python3
"""Automate the reference PCCT benchmark suite and emit JSONL/CSV artefacts.

The default suite runs:
- Residual 4 k guardrail (via tools/residual_guardrail_check.py)
- 2 048-point quick check (diagnostics off + on)
- 8 192-point Euclidean + residual scaling runs
- 32 768-point Euclidean (Hilbert + grid) and residual dense pair-merge runs

Use --list-jobs to inspect the available presets and --jobs to run a subset.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DEFAULT_OUTPUT_BASE = REPO_ROOT / "artifacts" / "benchmarks" / "reference"
DEFAULT_ENV = {
    "COVERTREEX_BACKEND": "numpy",
    "COVERTREEX_ENABLE_NUMBA": "1",
}


@dataclass(frozen=True)
class BenchmarkJob:
    """Description of a benchmark run driven by cli.queries or guardrail helper."""

    name: str
    runner: str = "queries"  # "queries" or "guardrail"
    metric: str = "euclidean"
    tree_points: int = 0
    dimension: int = 8
    queries: int = 1024
    batch_size: int = 512
    k: int = 8
    baseline: str = "none"
    cli_args: Sequence[str] = field(default_factory=tuple)
    env: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def build_command(self, log_path: Path, python_executable: str) -> List[str]:
        if self.runner == "guardrail":
            cmd = [
                python_executable,
                str(TOOLS_DIR / "residual_guardrail_check.py"),
                "--log-file",
                str(log_path),
            ]
            cmd.extend(self.cli_args)
            return cmd
        if not self.metric:
            raise ValueError(f"Benchmark job {self.name} missing metric for cli.queries run")
        cmd = [
            python_executable,
            "-m",
            "cli.pcct",
            "query",
            "--metric",
            self.metric,
            "--dimension",
            str(self.dimension),
            "--tree-points",
            str(self.tree_points),
            "--batch-size",
            str(self.batch_size),
            "--queries",
            str(self.queries),
            "--k",
            str(self.k),
            "--seed",
            "42",
            "--baseline",
            self.baseline,
            "--log-file",
            str(log_path),
        ]
        cmd.extend(self.cli_args)
        return cmd

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "runner": self.runner,
            "metric": self.metric,
            "tree_points": self.tree_points,
            "dimension": self.dimension,
            "queries": self.queries,
            "batch_size": self.batch_size,
            "k": self.k,
            "baseline": self.baseline,
            "cli_args": list(self.cli_args),
            "env": self.env,
            "description": self.description,
        }


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_BASE / timestamp


def _jobs() -> Dict[str, BenchmarkJob]:
    return {
        job.name: job
        for job in [
            BenchmarkJob(
                name="guardrail_residual_4k",
                runner="guardrail",
                cli_args=(
                    "--min-whitened-coverage",
                    "0.95",
                    "--max-median-semisort-ms",
                    "1000",
                ),
                description="Phase-5 residual guardrail (4096 Hilbert preset via guardrail tool).",
            ),
            BenchmarkJob(
                name="queries_2048_diag_off",
                metric="euclidean",
                tree_points=2048,
                batch_size=128,
                queries=512,
                k=8,
                baseline="gpboost",
                env={"COVERTREEX_ENABLE_DIAGNOSTICS": "0"},
                description="2 048-point quick check, diagnostics disabled.",
            ),
            BenchmarkJob(
                name="queries_2048_diag_on",
                metric="euclidean",
                tree_points=2048,
                batch_size=128,
                queries=512,
                k=8,
                baseline="gpboost",
                env={"COVERTREEX_ENABLE_DIAGNOSTICS": "1"},
                description="2 048-point quick check, diagnostics enabled.",
            ),
            BenchmarkJob(
                name="queries_8192_euclidean",
                metric="euclidean",
                tree_points=8192,
                k=16,
                baseline="gpboost",
                env={
                    "COVERTREEX_ENABLE_DIAGNOSTICS": "1",
                    "COVERTREEX_CONFLICT_GRAPH_IMPL": "grid",
                    "COVERTREEX_BATCH_ORDER": "hilbert",
                },
                description="8 192-point Euclidean run (Hilbert + grid builder).",
                cli_args=("--conflict-impl", "grid", "--batch-order", "hilbert"),
            ),
            BenchmarkJob(
                name="queries_8192_residual",
                metric="residual",
                tree_points=8192,
                k=16,
                baseline="gpboost",
                env={
                    "COVERTREEX_ENABLE_DIAGNOSTICS": "1",
                    "COVERTREEX_SCOPE_CHUNK_TARGET": "0",
                },
                description="8 192-point residual dense streamer (diagnostics on).",
                cli_args=("--residual-scope-bitset",),
            ),
            BenchmarkJob(
                name="queries_32768_euclidean_hilbert_grid",
                metric="euclidean",
                tree_points=32768,
                k=8,
                baseline="gpboost",
                env={
                    "COVERTREEX_ENABLE_DIAGNOSTICS": "1",
                    "COVERTREEX_CONFLICT_GRAPH_IMPL": "grid",
                    "COVERTREEX_BATCH_ORDER": "hilbert",
                },
                description="32 768-point Euclidean run (Hilbert batches + grid conflict builder).",
                cli_args=("--conflict-impl", "grid", "--batch-order", "hilbert"),
            ),
            BenchmarkJob(
                name="queries_32768_residual_dense_pairmerge",
                metric="residual",
                tree_points=32768,
                k=8,
                baseline="none",
                env={
                    "COVERTREEX_ENABLE_DIAGNOSTICS": "1",
                    "COVERTREEX_SCOPE_CHUNK_TARGET": "0",
                    "COVERTREEX_ENABLE_SPARSE_TRAVERSAL": "0",
                },
                description="32 768-point residual dense streamer (pair-merge defaults).",
                cli_args=(
                    "--residual-scope-bitset",
                    "--residual-dense-scope-streamer",
                    "--residual-masked-scope-append",
                ),
            ),
            BenchmarkJob(
                name="gold_standard_32k",
                metric="residual",
                tree_points=32768,
                batch_size=512,
                queries=1024,
                k=50,
                baseline="gpboost",
                env={
                    "COVERTREEX_ENABLE_NUMBA": "1",
                    "COVERTREEX_SCOPE_CHUNK_TARGET": "0",
                    # Explicitly unset/disable sparse traversal to match gold standard script
                    "COVERTREEX_ENABLE_SPARSE_TRAVERSAL": "0", 
                    "COVERTREEX_BATCH_ORDER": "natural",
                    "COVERTREEX_PREFIX_SCHEDULE": "doubling",
                    "COVERTREEX_ENABLE_DIAGNOSTICS": "0",
                },
                description="Gold Standard Residual Benchmark (32k points, d=3, k=50). Matches run_residual_gold_standard.sh.",
            ),
        ]
    }


def _export_csv(log_path: Path, csv_path: Path, python_executable: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable,
        str(TOOLS_DIR / "export_benchmark_diagnostics.py"),
        "--output",
        str(csv_path),
        str(log_path),
    ]
    print(f"[export] {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_job(
    job: BenchmarkJob,
    log_path: Path,
    csv_path: Path,
    env: Dict[str, str],
    python_executable: str,
    skip_existing: bool,
) -> Dict[str, str]:
    if skip_existing and log_path.exists():
        print(f"[skip] {job.name} (log exists at {log_path})")
        if csv_path.exists():
            return {"log": str(log_path), "csv": str(csv_path), "skipped": True}
    cmd = job.build_command(log_path, python_executable)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {job.name}: {shlex.join(cmd)}")
    env_vars = os.environ.copy()
    env_vars.update(DEFAULT_ENV)
    env_vars.update(job.env)
    env_vars.update(env)
    subprocess.run(cmd, check=True, env=env_vars, cwd=str(REPO_ROOT))
    _export_csv(log_path, csv_path, python_executable)
    return {"log": str(log_path), "csv": str(csv_path), "skipped": False}


def _select_jobs(all_jobs: Dict[str, BenchmarkJob], names: Sequence[str] | None) -> Iterable[BenchmarkJob]:
    if not names:
        return all_jobs.values()
    selected = []
    for name in names:
        if name not in all_jobs:
            raise ValueError(f"Unknown job '{name}'. Use --list-jobs to inspect available presets.")
        selected.append(all_jobs[name])
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory for logs/CSVs (default: timestamped under artifacts/benchmarks/reference)")
    parser.add_argument("--jobs", type=str, help="Comma-separated list of job names to run")
    parser.add_argument("--list-jobs", action="store_true", help="List available job presets and exit")
    parser.add_argument("--skip-existing", action="store_true", help="Skip jobs whose log files already exist")
    parser.add_argument("--summary", type=Path, help="Optional path for the manifest JSON (default: <output>/manifest.json)")
    args = parser.parse_args(argv)

    jobs = _jobs()
    if args.list_jobs:
        for job in jobs.values():
            print(f"{job.name:35s} {job.description}")
        return 0

    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (args.summary or (output_dir / "manifest.json")).resolve()

    selected_names = [name.strip() for name in args.jobs.split(",")] if args.jobs else None
    python_executable = sys.executable
    manifest: List[Dict[str, object]] = []

    for job in _select_jobs(jobs, selected_names):
        log_path = output_dir / f"{job.name}.jsonl"
        csv_path = output_dir / f"{job.name}.csv"
        result = _run_job(
            job=job,
            log_path=log_path,
            csv_path=csv_path,
            env={},
            python_executable=python_executable,
            skip_existing=args.skip_existing,
        )
        manifest_entry = job.metadata()
        manifest_entry.update(result)
        manifest.append(manifest_entry)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"output_dir": str(output_dir), "jobs": manifest}, handle, indent=2)
    print(f"[summary] wrote manifest to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
