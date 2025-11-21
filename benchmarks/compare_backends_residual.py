"""
Compare residual-correlation performance between the python/Numba path and the
Rust backend for float32/float64 at n=50k, d=3. Both paths share the same
dataset, residual backend, and query indices so we isolate implementation cost.
"""

from __future__ import annotations

import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import logging

# Ensure repository root is on sys.path for local module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

import covertreex
from covertreex.core.tree import PCCTree
from covertreex.metrics.residual.host_backend import build_residual_backend
from covertreex.metrics.residual.core import configure_residual_correlation
from covertreex.algo.batch_insert import batch_insert
from covertreex.queries.knn import knn
from tests.utils.datasets import gaussian_points  # noqa: E402


DATASET_SIZE = 50_000
DIMENSION = 3
K = 50
QUERIES = 1_024
BATCH_SIZE = 512
SEED = 0
RANK = 16
logging.basicConfig(level=logging.WARNING)


@dataclass
class BenchmarkResult:
    build_seconds: float
    query_seconds: float
    qps: float


def prepare_tree(
    dtype_label: str,
    points: np.ndarray,
    backend,
    query_indices: np.ndarray,
) -> tuple[PCCTree, np.ndarray, covertreex.config.RuntimeConfig, float]:
    dtype = np.float32 if dtype_label == "float32" else np.float64

    runtime_cfg = covertreex.config.runtime_config()
    cfg_common = dataclasses.replace(
        runtime_cfg,
        metric="residual_correlation",
        precision=dtype_label,
        enable_numba=True,
        enable_rust=False,
        enable_sparse_traversal=False,
        enable_diagnostics=False,
        log_level="WARNING",
        batch_order_strategy="natural",
        residual_use_static_euclidean_tree=False,
    )
    covertreex.config.configure_runtime(cfg_common)
    object.__setattr__(backend, "rbf_variance", 1.0)
    object.__setattr__(backend, "rbf_lengthscale", np.ones(DIMENSION, dtype=np.float32))
    configure_residual_correlation(backend)

    tree = PCCTree.empty(dimension=DIMENSION)
    batch_points = points.astype(dtype, copy=False)

    start = time.perf_counter()
    tree, _ = batch_insert(tree, batch_points)
    build_seconds = time.perf_counter() - start

    if tree.top_levels is not None:
        min_scale = int(np.min(tree.top_levels)) if tree.top_levels.size else -20
        max_scale = int(np.max(tree.top_levels)) if tree.top_levels.size else 20
        object.__setattr__(tree, "min_scale", min_scale)
        object.__setattr__(tree, "max_scale", max_scale)

    queries = np.asarray(query_indices, dtype=np.int64).reshape(-1, 1)
    return tree, queries, cfg_common, build_seconds


def run_query(
    tree: PCCTree,
    queries: np.ndarray,
    cfg: covertreex.config.RuntimeConfig,
    backend,
) -> BenchmarkResult:
    covertreex.config.configure_runtime(cfg)
    configure_residual_correlation(backend)
    start = time.perf_counter()
    knn(tree, queries, k=K)
    query_seconds = time.perf_counter() - start
    qps = QUERIES / query_seconds if query_seconds > 0 else float("nan")
    return BenchmarkResult(build_seconds=0.0, query_seconds=query_seconds, qps=qps)


def main() -> None:
    dtype_labels = ["float32", "float64"]
    results: Dict[Tuple[str, str], BenchmarkResult] = {}
    rng = np.random.default_rng(SEED)
    points = gaussian_points(rng, DATASET_SIZE, DIMENSION, dtype=np.float64)
    query_rng = np.random.default_rng(SEED + 1)
    query_indices = query_rng.integers(
        0,
        DATASET_SIZE,
        size=QUERIES,
        endpoint=False,
        dtype=np.int64,
    )

    for label in dtype_labels:
        print(f"\n===== dtype={label} =====")

        backend = build_residual_backend(
            points,
            seed=SEED,
            inducing_count=512,
            variance=1.0,
            lengthscale=1.0,
            chunk_size=512,
        )

        tree, queries, cfg_common, build_seconds = prepare_tree(label, points, backend, query_indices)

        py_result = run_query(tree, queries, cfg_common, backend)
        py_result = dataclasses.replace(py_result, build_seconds=build_seconds)
        print(
            f"[python] build={py_result.build_seconds:.4f}s "
            f"query={py_result.query_seconds:.4f}s ({py_result.qps:,.1f} q/s)"
        )
        results[(label, "python")] = py_result

        cfg_rust = dataclasses.replace(cfg_common, enable_rust=True)
        rust_result = run_query(tree, queries, cfg_rust, backend)
        rust_result = dataclasses.replace(rust_result, build_seconds=build_seconds)
        print(
            f"[rust] build={rust_result.build_seconds:.4f}s "
            f"query={rust_result.query_seconds:.4f}s ({rust_result.qps:,.1f} q/s)"
        )
        results[(label, "rust")] = rust_result

    print("\n===== summary =====")
    print(
        f"{'dtype':<10} | {'build_s':>10} | {'py_qps':>10} | {'rust_qps':>10} | {'speedup_q':>10}"
    )
    print("-" * 60)
    for label in dtype_labels:
        py_res = results[(label, "python")]
        rust_res = results[(label, "rust")]
        speedup = rust_res.qps / py_res.qps if py_res.qps > 0 else float("nan")
        print(
            f"{label:<10} | {py_res.build_seconds:.4f} | {py_res.qps:,.1f} | {rust_res.qps:,.1f} | {speedup:.2f}"
        )


if __name__ == "__main__":
    main()
