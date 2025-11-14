#!/usr/bin/env python3
"""Run residual benchmarks across multiple tree sizes."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_SIZES = [4096, 8192, 16384, 32768, 49152, 65536]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated list of tree sizes to benchmark (default: %(default)s)",
    )
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("scaling"),
        help="Relative directory (under artifacts/benchmarks) for logs (default: %(default)s)",
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default="residual_scaling",
        help="Filename prefix for per-run logs (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-cli-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to cli.queries after the preset flags.",
    )
    return parser.parse_args()


def _summary_from_log(path: Path) -> tuple[float, float, float, int]:
    semis: List[float] = []
    kernels: List[float] = []
    scope_points: List[float] = []
    total_traversal = 0.0
    dominated_batches = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("dominated", 0) > 0:
                dominated_batches += 1
                semis.append(float(payload.get("traversal_semisort_ms", 0.0)))
                kernels.append(float(payload.get("traversal_kernel_provider_ms", 0.0)))
                scope_points.append(float(payload.get("traversal_scope_chunk_points", 0.0)))
                total_traversal += float(payload.get("traversal_ms", 0.0))
    if not dominated_batches:
        return 0.0, 0.0, 0.0, 0
    return (
        statistics.median(semis),
        statistics.median(kernels),
        statistics.median(scope_points),
        int(total_traversal),
    )


def _run_size(size: int, args: argparse.Namespace) -> Path:
    relative_path = args.log_dir / f"{args.log_prefix}_{size}.jsonl"
    cli_cmd = [
        sys.executable,
        "-m",
        "cli.queries",
        "--metric",
        "residual",
        "--dimension",
        str(args.dimension),
        "--tree-points",
        str(size),
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
        str(relative_path),
    ]
    cli_cmd.extend(args.extra_cli_args)
    subprocess.run(cli_cmd, check=True)
    absolute_path = Path("artifacts/benchmarks") / relative_path
    return absolute_path


def main() -> None:
    args = parse_args()
    tree_sizes: Iterable[int] = [int(s.strip()) for s in args.tree_sizes.split(",") if s.strip()]
    summaries: List[tuple[int, float, float, float, int]] = []
    for size in tree_sizes:
        print(f"\n[scaling] running tree_points={size}")
        log_path = _run_size(size, args)
        semis, kernel, scope_pts, total = _summary_from_log(log_path)
        if total:
            summaries.append((size, semis, kernel, scope_pts, total))
            print(
                f"[scaling] size={size} semisort_med={semis:.3f}ms kernel_med={kernel:.3f}ms "
                f"scope_pts_med={scope_pts:.0f} total_traversal_ms={total}"
            )
        else:
            print(f"[scaling] warning: no dominated batches recorded for size={size}")
    if summaries:
        print("\n[scaling] summary (tree_points, semisort_ms_med, kernel_ms_med, scope_points_med, total_traversal_ms)")
        for entry in summaries:
            size, semis, kernel, scope_pts, total = entry
            print(f"  {size:6d}  {semis:9.3f}  {kernel:9.3f}  {scope_pts:8.0f}  {total:10d}")


if __name__ == "__main__":
    main()
