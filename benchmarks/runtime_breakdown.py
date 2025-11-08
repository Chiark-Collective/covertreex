from __future__ import annotations

import argparse
import contextlib
import csv
import os
import resource
import time
from dataclasses import dataclass, field, replace
from statistics import mean, median
from typing import Dict, List, Optional

import numpy as np
from numpy.random import default_rng

from benchmarks.runtime_cli import runtime_from_args
from covertreex import config as cx_config
from covertreex.algo import batch_insert
from covertreex.algo.mis import NUMBA_AVAILABLE
from covertreex.core.tree import PCCTree, get_runtime_backend
from covertreex.queries.knn import knn
from covertreex.baseline import (
    BaselineCoverTree,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
)


try:  # pragma: no cover - plotting exercised in manual workflows
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - guarded runtime path
    raise SystemExit(
        "matplotlib is required for runtime breakdown plots. "
        "Install it with `pip install matplotlib`."
    ) from exc


@dataclass(frozen=True)
class ImplementationResult:
    label: str
    build_seconds: float
    query_seconds: float
    segments: Dict[str, float]
    build_warmup_seconds: float
    build_steady_seconds: float
    query_warmup_seconds: float
    query_steady_seconds: float
    build_cpu_seconds: float
    build_cpu_utilisation: float
    build_rss_delta_bytes: Optional[int]
    build_max_rss_bytes: Optional[int]
    query_cpu_seconds: float
    query_cpu_utilisation: float
    query_rss_delta_bytes: Optional[int]
    query_max_rss_bytes: Optional[int]
    notes: Optional[str] = None
    chunk_metrics: Dict[str, float] = field(default_factory=dict)


def _generate_dataset(
    *,
    dimension: int,
    tree_points: int,
    queries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = default_rng(seed)
    points = rng.normal(loc=0.0, scale=1.0, size=(tree_points, dimension))
    queries_arr = rng.normal(loc=0.0, scale=1.0, size=(queries, dimension))
    return points.astype(np.float64), queries_arr.astype(np.float64)


def _block_until_ready(value: object) -> None:
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


@dataclass(frozen=True)
class _ResourceSnapshot:
    wall: float
    cpu_user: float
    cpu_system: float
    rss_bytes: Optional[int]
    max_rss_bytes: Optional[int]


def _read_rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.readline().strip().split()
        if len(fields) < 2:
            return None
        rss_pages = int(fields[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(rss_pages * page_size)
    except (OSError, ValueError, IndexError):
        return None


def _resource_snapshot() -> _ResourceSnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss * 1024) if usage.ru_maxrss else None
    rss_bytes = _read_rss_bytes()
    return _ResourceSnapshot(
        wall=time.perf_counter(),
        cpu_user=float(usage.ru_utime),
        cpu_system=float(usage.ru_stime),
        rss_bytes=rss_bytes,
        max_rss_bytes=max_rss,
    )


def _resource_delta(
    start: _ResourceSnapshot, end: _ResourceSnapshot
) -> tuple[float, float, Optional[int], Optional[int]]:
    wall_seconds = max(0.0, end.wall - start.wall)
    cpu_seconds = max(
        0.0,
        (end.cpu_user - start.cpu_user) + (end.cpu_system - start.cpu_system),
    )
    rss_delta = None
    if start.rss_bytes is not None and end.rss_bytes is not None:
        rss_delta = end.rss_bytes - start.rss_bytes
    max_rss_candidates = [
        value for value in (start.max_rss_bytes, end.max_rss_bytes) if value is not None
    ]
    max_rss = max(max_rss_candidates) if max_rss_candidates else None
    return wall_seconds, cpu_seconds, rss_delta, max_rss


def _compute_cpu_utilisation(cpu_seconds: float, wall_seconds: float) -> float:
    if wall_seconds <= 0.0:
        return 0.0
    return cpu_seconds / wall_seconds


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    if value == 0:
        return "0B"
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(int(np.log(abs_value) / np.log(1024)) if abs_value > 0 else 0, len(units) - 1)
    scaled = abs_value / (1024**idx)
    return f"{sign}{scaled:.2f}{units[idx]}"


def _format_resource_summary(
    cpu_seconds: float,
    cpu_utilisation: float,
    rss_delta: Optional[int],
    max_rss: Optional[int],
) -> str:
    return (
        f"cpu={cpu_seconds:.3f}s "
        f"util={cpu_utilisation:.2f}x "
        f"rssΔ={_format_bytes(rss_delta)} "
        f"max={_format_bytes(max_rss)}"
    )


def _print_multi_run_summary(all_results: List[List[ImplementationResult]]) -> None:
    labels = sorted({res.label for run in all_results for res in run})
    print("=== Aggregate Summary ===")
    for label in labels:
        label_results: List[ImplementationResult] = []
        for run in all_results:
            match = next((res for res in run if res.label == label), None)
            if match is not None:
                label_results.append(match)
        if not label_results:
            continue

        build_totals = [res.build_seconds for res in label_results]
        build_warmups = [res.build_warmup_seconds for res in label_results]
        query_totals = [res.query_seconds for res in label_results]

        print(f"{label}:")
        print(
            f"  build total: first={build_totals[0]:.3f}s "
            f"mean={mean(build_totals):.3f}s median={median(build_totals):.3f}s "
            f"min={min(build_totals):.3f}s max={max(build_totals):.3f}s"
        )
        print(
            f"  build warmup: first={build_warmups[0]:.3f}s "
            f"mean={mean(build_warmups):.3f}s median={median(build_warmups):.3f}s "
            f"min={min(build_warmups):.3f}s max={max(build_warmups):.3f}s"
        )
        print(
            f"  query steady: first={query_totals[0]:.3f}s "
            f"mean={mean(query_totals):.3f}s median={median(query_totals):.3f}s "
            f"min={min(query_totals):.3f}s max={max(query_totals):.3f}s"
        )


@contextlib.contextmanager
def _temporary_runtime_override(**overrides: object) -> None:
    base_context = cx_config.runtime_context()
    updated_config = replace(base_context.config, **overrides)
    cx_config.configure_runtime(updated_config)
    try:
        yield
    finally:
        cx_config.set_runtime_context(base_context)


def _chunk_points(points: np.ndarray, batch_size: int) -> List[np.ndarray]:
    return [points[idx : idx + batch_size] for idx in range(0, points.shape[0], batch_size)]


def _summarise_batches(batch_metrics: List[Dict[str, float]], build_seconds: float) -> Dict[str, float]:
    totals = {
        "pairwise": 0.0,
        "mask": 0.0,
        "chain": 0.0,
        "nonzero": 0.0,
        "sort": 0.0,
        "assemble": 0.0,
        "semisort": 0.0,
        "conflict_graph": 0.0,
        "mis": 0.0,
    }
    for metrics in batch_metrics:
        for key in totals:
            totals[key] += metrics.get(key, 0.0)

    semisort_residual = max(
        0.0,
        totals["semisort"]
        - (
            totals["chain"]
            + totals["nonzero"]
            + totals["sort"]
            + totals["assemble"]
        ),
    )

    segments = {
        "pairwise": totals["pairwise"],
        "mask": totals["mask"],
        "chain": totals["chain"],
        "nonzero": totals["nonzero"],
        "sort": totals["sort"],
        "assemble": totals["assemble"],
        "semisort_residual": semisort_residual,
        "conflict_graph": totals["conflict_graph"],
        "mis": totals["mis"],
    }

    accounted = sum(segments.values())
    segments["other"] = max(0.0, build_seconds - accounted)
    return segments


def _summarise_chunk_metrics(batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    def _phases(key: str) -> tuple[float, float]:
        if not batch_metrics:
            return 0.0, 0.0
        values = [metrics.get(key, 0.0) for metrics in batch_metrics]
        warmup = values[0]
        if len(values) > 1:
            steady = float(sum(values[1:])) / float(len(values) - 1)
        else:
            steady = warmup
        return warmup, steady

    summary: Dict[str, float] = {}
    metric_fields = (
        ("traversal", "segments"),
        ("traversal", "emitted"),
        ("traversal", "max_members"),
        ("conflict", "segments"),
        ("conflict", "emitted"),
        ("conflict", "max_members"),
    )
    for prefix, field in metric_fields:
        key = f"{prefix}_chunk_{field}"
        warmup, steady = _phases(key)
        summary[f"{key}_warmup"] = warmup
        summary[f"{key}_steady"] = steady
    return summary


def _run_pcct_variant(
    *,
    label: str,
    points: np.ndarray,
    queries: np.ndarray,
    batch_size: int,
    k: int,
    seed: int,
    enable_numba: bool,
) -> ImplementationResult:
    notes: Optional[str] = None
    if enable_numba and not NUMBA_AVAILABLE:
        notes = "Numba backend unavailable; falling back to JAX MIS."

    with _temporary_runtime_override(enable_numba=enable_numba):
        backend = get_runtime_backend()
        tree = PCCTree.empty(dimension=int(points.shape[1]), backend=backend)
        batch_metrics: List[Dict[str, float]] = []
        batch_totals: List[float] = []

        batches = _chunk_points(points, batch_size)
        build_resource_start = _resource_snapshot()
        for idx, batch_np in enumerate(batches):
            if batch_np.size == 0:
                continue
            batch = backend.asarray(batch_np, dtype=backend.default_float)
            batch_start = time.perf_counter()
            tree, plan = batch_insert(tree, batch, mis_seed=seed + idx)
            batch_totals.append(time.perf_counter() - batch_start)
            timing = plan.traversal.timings
            batch_metrics.append(
                {
                    "pairwise": float(timing.pairwise_seconds),
                    "mask": float(timing.mask_seconds),
                    "chain": float(timing.chain_seconds),
                    "nonzero": float(timing.nonzero_seconds),
                    "sort": float(timing.sort_seconds),
                    "assemble": float(timing.assemble_seconds),
                    "semisort": float(timing.semisort_seconds),
                    "conflict_graph": float(plan.timings.conflict_graph_seconds),
                    "mis": float(plan.timings.mis_seconds),
                    "traversal_chunk_segments": float(timing.scope_chunk_segments),
                    "traversal_chunk_emitted": float(timing.scope_chunk_emitted),
                    "traversal_chunk_max_members": float(timing.scope_chunk_max_members),
                    "conflict_chunk_segments": float(
                        plan.conflict_graph.timings.scope_chunk_segments
                    ),
                    "conflict_chunk_emitted": float(
                        plan.conflict_graph.timings.scope_chunk_emitted
                    ),
                    "conflict_chunk_max_members": float(
                        plan.conflict_graph.timings.scope_chunk_max_members
                    ),
                }
            )

        build_resource_end = _resource_snapshot()
        build_wall_seconds, build_cpu_seconds, build_rss_delta, build_max_rss = _resource_delta(
            build_resource_start, build_resource_end
        )
        build_seconds = float(sum(batch_totals))
        if batch_totals:
            build_warmup = batch_totals[0]
            build_steady = (
                float(sum(batch_totals[1:])) / (len(batch_totals) - 1)
                if len(batch_totals) > 1
                else build_warmup
            )
        else:
            build_warmup = 0.0
            build_steady = 0.0

        query_array = backend.asarray(queries, dtype=backend.default_float)
        warm_start = time.perf_counter()
        indices, distances = knn(tree, query_array, k=k, return_distances=True)
        _block_until_ready(indices)
        _block_until_ready(distances)
        query_warmup = time.perf_counter() - warm_start
        query_resource_start = _resource_snapshot()
        steady_start = time.perf_counter()
        indices, distances = knn(tree, query_array, k=k, return_distances=True)
        _block_until_ready(indices)
        _block_until_ready(distances)
        query_steady = time.perf_counter() - steady_start
        query_resource_end = _resource_snapshot()
        (
            query_wall_seconds,
            query_cpu_seconds,
            query_rss_delta,
            query_max_rss,
        ) = _resource_delta(query_resource_start, query_resource_end)

    segments = _summarise_batches(batch_metrics, build_seconds)
    chunk_summary = _summarise_chunk_metrics(batch_metrics)
    return ImplementationResult(
        label=label,
        build_seconds=build_seconds,
        query_seconds=query_steady,
        segments=segments,
        build_warmup_seconds=build_warmup,
        build_steady_seconds=build_steady,
        query_warmup_seconds=query_warmup,
        query_steady_seconds=query_steady,
        build_cpu_seconds=build_cpu_seconds,
        build_cpu_utilisation=_compute_cpu_utilisation(build_cpu_seconds, build_wall_seconds),
        build_rss_delta_bytes=build_rss_delta,
        build_max_rss_bytes=build_max_rss,
        query_cpu_seconds=query_cpu_seconds,
        query_cpu_utilisation=_compute_cpu_utilisation(query_cpu_seconds, query_wall_seconds),
        query_rss_delta_bytes=query_rss_delta,
        query_max_rss_bytes=query_max_rss,
        notes=notes,
        chunk_metrics=chunk_summary,
    )


def _run_sequential_baseline(
    *,
    label: str,
    points: np.ndarray,
    queries: np.ndarray,
    k: int,
    constructor,
) -> ImplementationResult:
    build_resource_start = _resource_snapshot()
    start_build = time.perf_counter()
    tree = constructor.from_points(points)
    build_seconds = time.perf_counter() - start_build
    build_resource_end = _resource_snapshot()
    build_wall_seconds, build_cpu_seconds, build_rss_delta, build_max_rss = _resource_delta(
        build_resource_start, build_resource_end
    )

    query_resource_start = _resource_snapshot()
    start_query = time.perf_counter()
    tree.knn(queries, k=k, return_distances=False)
    query_seconds = time.perf_counter() - start_query
    query_resource_end = _resource_snapshot()
    (
        query_wall_seconds,
        query_cpu_seconds,
        query_rss_delta,
        query_max_rss,
    ) = _resource_delta(query_resource_start, query_resource_end)

    segments = {"other": build_seconds}
    return ImplementationResult(
        label=label,
        build_seconds=build_seconds,
        query_seconds=query_seconds,
        segments=segments,
        build_warmup_seconds=build_seconds,
        build_steady_seconds=build_seconds,
        query_warmup_seconds=query_seconds,
        query_steady_seconds=query_seconds,
        build_cpu_seconds=build_cpu_seconds,
        build_cpu_utilisation=_compute_cpu_utilisation(build_cpu_seconds, build_wall_seconds),
        build_rss_delta_bytes=build_rss_delta,
        build_max_rss_bytes=build_max_rss,
        query_cpu_seconds=query_cpu_seconds,
        query_cpu_utilisation=_compute_cpu_utilisation(query_cpu_seconds, query_wall_seconds),
        query_rss_delta_bytes=query_rss_delta,
        query_max_rss_bytes=query_max_rss,
    )


def _plot_results(results: List[ImplementationResult], *, output: Optional[str], show: bool) -> None:
    segment_order = [
        "pairwise",
        "mask",
        "chain",
        "nonzero",
        "sort",
        "assemble",
        "semisort_residual",
        "conflict_graph",
        "mis",
        "other",
    ]
    segment_labels = {
        "pairwise": "Pairwise Distances",
        "mask": "Radius Mask",
        "chain": "Next-Chain Merge",
        "nonzero": "Scope Nonzero",
        "sort": "Scope Sort",
        "assemble": "Scope Assemble",
        "semisort_residual": "Semisort Residual",
        "conflict_graph": "Conflict Graph",
        "mis": "MIS",
        "other": "Other",
    }
    colors = {
        "pairwise": "#1f77b4",
        "mask": "#ff7f0e",
        "chain": "#2ca02c",
        "nonzero": "#d62728",
        "sort": "#9467bd",
        "assemble": "#8c564b",
        "semisort_residual": "#e377c2",
        "conflict_graph": "#7f7f7f",
        "mis": "#bcbd22",
        "other": "#17becf",
    }

    labels = [res.label for res in results]
    x = np.arange(len(labels))
    width = 0.6

    fig, (ax_runtime, ax_query) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    bottom = np.zeros(len(results))
    for segment in segment_order:
        heights = np.array([res.segments.get(segment, 0.0) for res in results])
        if np.allclose(heights, 0.0):
            continue
        ax_runtime.bar(
            x,
            heights,
            width,
            bottom=bottom,
            label=segment_labels[segment],
            color=colors.get(segment),
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += heights

    ax_runtime.set_ylabel("Build Time (s)")
    ax_runtime.set_title("Cover Tree Build Runtime Breakdown")
    ax_runtime.grid(axis="y", alpha=0.3)
    ax_runtime.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    query_seconds = [res.query_steady_seconds for res in results]
    ax_query.bar(
        x,
        query_seconds,
        width,
        color="#6c6f7f",
        edgecolor="black",
        linewidth=0.3,
    )
    ax_query.set_ylabel("Query Time (s)")
    ax_query.set_xlabel("Implementation")
    ax_query.grid(axis="y", alpha=0.3)
    ax_query.set_xticks(x)
    ax_query.set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved runtime breakdown to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate runtime breakdown plots for cover tree implementations."
    )
    parser.add_argument("--dimension", type=int, default=8, help="Dimensionality of points.")
    parser.add_argument("--tree-points", type=int, default=2048, help="Number of points to insert.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for PCCT insertions.")
    parser.add_argument("--queries", type=int, default=512, help="Number of query points.")
    parser.add_argument("--k", type=int, default=8, help="Number of neighbours for k-NN.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used across runs.")
    parser.add_argument(
        "--output",
        type=str,
        default="runtime_breakdown.png",
        help="Path to save the generated plot (set to '' to skip saving).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    parser.add_argument(
        "--skip-numba",
        action="store_true",
        help="Skip the Numba-enabled PCCT run.",
    )
    parser.add_argument(
        "--skip-jax",
        action="store_true",
        help="Skip the JAX-backed PCCT run.",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        default=True,
        help="Skip the external cover tree baseline if installed (default: skipped).",
    )
    parser.add_argument(
        "--include-external",
        action="store_false",
        dest="skip_external",
        help="Include the external cover tree baseline.",
    )
    parser.add_argument(
        "--skip-gpboost",
        action="store_true",
        help="Skip the GPBoost cover tree baseline if numba is installed.",
    )
    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        default=True,
        help="Skip the sequential cover tree baseline (default: skipped).",
    )
    parser.add_argument(
        "--include-sequential",
        action="store_false",
        dest="skip_sequential",
        help="Include the sequential cover tree baseline.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="",
        help="Optional path to write warm-up vs steady-state metrics as CSV.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated benchmark runs (default: 1).",
    )
    parser.add_argument(
        "--backend",
        choices=("jax", "numpy"),
        default="jax",
        help="Runtime backend to activate before benchmarking.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help="Floating-point precision for runtime computations.",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean", "residual"),
        default="euclidean",
        help="Metric to benchmark (residual currently unsupported).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.metric == "residual":
        raise ValueError("runtime_breakdown currently supports only the euclidean metric.")
    runtime_from_args(args).activate()

    points_np, queries_np = _generate_dataset(
        dimension=args.dimension,
        tree_points=args.tree_points,
        queries=args.queries,
        seed=args.seed,
    )

    runs = max(1, args.runs)
    all_results: List[List[ImplementationResult]] = []

    for run_idx in range(runs):
        if runs > 1:
            print(f"=== Run {run_idx + 1}/{runs} ===")
        run_seed = args.seed + run_idx
        run_results: List[ImplementationResult] = []

        backend = get_runtime_backend()
        if not args.skip_jax and backend.name == "jax":
            run_results.append(
                _run_pcct_variant(
                    label="PCCT (JAX)",
                    points=points_np,
                    queries=queries_np,
                    batch_size=args.batch_size,
                    k=args.k,
                    seed=run_seed,
                    enable_numba=False,
                )
            )

        if not args.skip_numba:
            run_results.append(
                _run_pcct_variant(
                    label="PCCT (Numba)",
                    points=points_np,
                    queries=queries_np,
                    batch_size=args.batch_size,
                    k=args.k,
                    seed=run_seed,
                    enable_numba=True,
                )
            )

        if not args.skip_sequential:
            run_results.append(
                _run_sequential_baseline(
                    label="Sequential Baseline",
                    points=points_np,
                    queries=queries_np,
                    k=args.k,
                    constructor=BaselineCoverTree,
                )
            )

        if not args.skip_gpboost:
            if has_gpboost_cover_tree():
                run_results.append(
                    _run_sequential_baseline(
                        label="GPBoost CoverTree",
                        points=points_np,
                        queries=queries_np,
                        k=args.k,
                        constructor=GPBoostCoverTreeBaseline,
                    )
                )
            elif run_idx == 0:
                print("GPBoost cover tree baseline unavailable; skipping.")

        if not args.skip_external and has_external_cover_tree():
            run_results.append(
                _run_sequential_baseline(
                    label="External CoverTree",
                    points=points_np,
                    queries=queries_np,
                    k=args.k,
                    constructor=ExternalCoverTreeBaseline,
                )
            )
        elif (not args.skip_external) and run_idx == 0:
            print("External cover tree baseline unavailable; skipping.")

        for res in run_results:
            notes = f" ({res.notes})" if res.notes else ""
            print(
                f"{res.label}: "
                f"build={res.build_seconds:.3f}s "
                f"[{_format_resource_summary(res.build_cpu_seconds, res.build_cpu_utilisation, res.build_rss_delta_bytes, res.build_max_rss_bytes)}] "
                f"query={res.query_seconds:.3f}s "
                f"[{_format_resource_summary(res.query_cpu_seconds, res.query_cpu_utilisation, res.query_rss_delta_bytes, res.query_max_rss_bytes)}]"
                f"{notes}"
            )

        all_results.append(run_results)

    # Use the last run for plotting to keep behaviour predictable
    results = all_results[-1] if all_results else []

    output_path = args.output or None
    _plot_results(results, output=output_path, show=args.show)

    if args.csv_output:
        chunk_fieldnames = [
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
        fieldnames = [
            "run",
            "label",
            "build_warmup_seconds",
            "build_steady_seconds",
            "build_total_seconds",
            "query_warmup_seconds",
            "query_steady_seconds",
            "build_cpu_seconds",
            "build_cpu_utilisation",
            "build_rss_delta_bytes",
            "build_max_rss_bytes",
            "query_cpu_seconds",
            "query_cpu_utilisation",
            "query_rss_delta_bytes",
            "query_max_rss_bytes",
        ] + chunk_fieldnames
        with open(args.csv_output, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for run_idx, run in enumerate(all_results, start=1):
                for res in run:
                    row = {
                        "run": run_idx,
                        "label": res.label,
                        "build_warmup_seconds": res.build_warmup_seconds,
                        "build_steady_seconds": res.build_steady_seconds,
                        "build_total_seconds": res.build_seconds,
                        "query_warmup_seconds": res.query_warmup_seconds,
                        "query_steady_seconds": res.query_steady_seconds,
                        "build_cpu_seconds": res.build_cpu_seconds,
                        "build_cpu_utilisation": res.build_cpu_utilisation,
                        "build_rss_delta_bytes": res.build_rss_delta_bytes
                        if res.build_rss_delta_bytes is not None
                        else "",
                        "build_max_rss_bytes": res.build_max_rss_bytes
                        if res.build_max_rss_bytes is not None
                        else "",
                        "query_cpu_seconds": res.query_cpu_seconds,
                        "query_cpu_utilisation": res.query_cpu_utilisation,
                        "query_rss_delta_bytes": res.query_rss_delta_bytes
                        if res.query_rss_delta_bytes is not None
                        else "",
                        "query_max_rss_bytes": res.query_max_rss_bytes
                        if res.query_max_rss_bytes is not None
                        else "",
                    }
                    for chunk_field in chunk_fieldnames:
                        row[chunk_field] = res.chunk_metrics.get(chunk_field, "")
                    writer.writerow(row)

    if runs > 1:
        _print_multi_run_summary(all_results)


if __name__ == "__main__":
    main()
