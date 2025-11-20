#!/usr/bin/env python3
"""Run the 4k residual guardrail benchmark and enforce the documented thresholds.

Usage examples:

    # Run the default 4k Hilbert preset and verify thresholds
    python tools/residual_guardrail_check.py

    # Reuse an existing log file (skip rerunning cli.pcct query)
    python tools/residual_guardrail_check.py --skip-run --log-file artifacts/benchmarks/guardrails/latest.jsonl

    # Forward additional cli.pcct arguments after "--"
    python tools/residual_guardrail_check.py --extra-cli-args -- --residual-scope-bitset
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class GuardrailMetrics:
    dominated_batches: int
    whitened_pairs_sum: float
    kernel_pairs_sum: float
    semisort_median_ms: float
    pairwise_reused_batches: int

    @property
    def whitened_coverage(self) -> float:
        if self.kernel_pairs_sum <= 0:
            return 0.0
        return float(self.whitened_pairs_sum) / float(self.kernel_pairs_sum)

    @property
    def all_pairwise_reused(self) -> bool:
        return self.dominated_batches > 0 and self.pairwise_reused_batches == self.dominated_batches

    @property
    def semisort_median_seconds(self) -> float:
        return self.semisort_median_ms / 1000.0


def _default_log_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("artifacts/benchmarks/guardrails") / f"residual_guardrail_{timestamp}.jsonl"


def _build_cli_command(args: argparse.Namespace, log_path: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "cli.pcct",
        "query",
        "--metric",
        "residual",
        "--dimension",
        str(args.dimension),
        "--tree-points",
        str(args.tree_points),
        "--batch-size",
        str(args.batch_size),
        "--queries",
        str(args.queries),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--baseline",
        args.baseline,
        "--log-file",
        str(log_path),
    ]
    cmd.extend(args.extra_cli_args)
    return cmd


def _load_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_guardrail_metrics(path: Path) -> GuardrailMetrics:
    dominated = 0
    semisort: List[float] = []
    whitened = 0.0
    kernel = 0.0
    pairwise_reused = 0
    for record in _load_records(path):
        if not record.get("dominated"):
            continue
        dominated += 1
        semisort.append(float(record.get("traversal_semisort_ms", 0.0)))
        whitened += float(record.get("traversal_whitened_block_pairs", 0.0))
        kernel += float(record.get("traversal_kernel_provider_pairs", 0.0))
        if record.get("conflict_pairwise_reused"):
            pairwise_reused += 1
    if not dominated:
        raise ValueError(f"No dominated batches found in guardrail log {path}")
    return GuardrailMetrics(
        dominated_batches=dominated,
        whitened_pairs_sum=whitened,
        kernel_pairs_sum=kernel,
        semisort_median_ms=statistics.median(semisort),
        pairwise_reused_batches=pairwise_reused,
    )


def evaluate_metrics(
    metrics: GuardrailMetrics,
    *,
    min_whitened_coverage: float,
    max_median_semisort_ms: float,
    require_pairwise_reuse: bool,
    min_dominated_batches: int,
) -> List[str]:
    failures: List[str] = []
    if metrics.dominated_batches < min_dominated_batches:
        failures.append(
            f"expected at least {min_dominated_batches} dominated batches, "
            f"got {metrics.dominated_batches}"
        )
    if metrics.whitened_coverage < min_whitened_coverage:
        failures.append(
            f"whitened coverage {metrics.whitened_coverage:.3f} "
            f"< minimum {min_whitened_coverage:.3f}"
        )
    if metrics.semisort_median_ms > max_median_semisort_ms:
        failures.append(
            f"median traversal_semisort_ms {metrics.semisort_median_ms:.2f} "
            f"> maximum {max_median_semisort_ms:.2f}"
        )
    if require_pairwise_reuse and not metrics.all_pairwise_reused:
        failures.append(
            f"conflict_pairwise_reused=1 for {metrics.pairwise_reused_batches}/"
            f"{metrics.dominated_batches} dominated batches"
        )
    return failures


def _write_summary(path: Path, metrics: GuardrailMetrics) -> None:
    payload = {
        "dominated_batches": metrics.dominated_batches,
        "whitened_pairs_sum": metrics.whitened_pairs_sum,
        "kernel_pairs_sum": metrics.kernel_pairs_sum,
        "whitened_coverage": metrics.whitened_coverage,
        "semisort_median_ms": metrics.semisort_median_ms,
        "semisort_median_seconds": metrics.semisort_median_seconds,
        "pairwise_reused_batches": metrics.pairwise_reused_batches,
        "all_pairwise_reused": metrics.all_pairwise_reused,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[guardrail] summary written to {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, help="Telemetry output (defaults under artifacts/benchmarks)")
    parser.add_argument("--skip-run", action="store_true", help="Do not run cli.pcct query, only validate the log")
    parser.add_argument("--tree-points", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument(
        "--extra-cli-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to cli.pcct query (prefix with '--' to terminate parser options)",
    )
    parser.add_argument(
        "--min-whitened-coverage",
        type=float,
        default=0.95,
        help="Minimum whitened_block_pairs_sum / kernel_provider_pairs_sum ratio",
    )
    parser.add_argument(
        "--max-median-semisort-ms",
        type=float,
        default=1000.0,
        help="Maximum allowed median traversal_semisort_ms (in milliseconds)",
    )
    parser.add_argument(
        "--min-dominated-batches",
        type=int,
        default=5,
        help="Minimum dominated batch count required for the guardrail run",
    )
    parser.add_argument(
        "--no-require-pairwise-reuse",
        action="store_true",
        help="Allow any conflict_pairwise_reused=0 batches (default: fail)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write a JSON summary of the guardrail metrics",
    )
    args = parser.parse_args(argv)

    log_path = args.log_file or _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_run:
        if not log_path.exists():
            raise FileNotFoundError(f"--skip-run set but log file not found: {log_path}")
    else:
        cli_cmd = _build_cli_command(args, log_path)
        print(f"[guardrail] running cli.pcct query -> {' '.join(cli_cmd)}")
        subprocess.run(cli_cmd, check=True)

    metrics = parse_guardrail_metrics(log_path)
    print(
        "[guardrail] dominated_batches=%d coverage=%.3f semisort_median=%.2fms "
        "pairwise_reused=%d"
        % (
            metrics.dominated_batches,
            metrics.whitened_coverage,
            metrics.semisort_median_ms,
            metrics.pairwise_reused_batches,
        )
    )

    failures = evaluate_metrics(
        metrics,
        min_whitened_coverage=args.min_whitened_coverage,
        max_median_semisort_ms=args.max_median_semisort_ms,
        require_pairwise_reuse=not args.no_require_pairwise_reuse,
        min_dominated_batches=args.min_dominated_batches,
    )

    if args.summary_json:
        _write_summary(args.summary_json, metrics)

    if failures:
        for issue in failures:
            print(f"[guardrail] FAIL: {issue}")
        return 1

    print("[guardrail] guardrail checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
