#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from covertreex import config as cx_config
from covertreex.core.metrics import reset_residual_metric
from covertreex.metrics.residual import (
    ResidualGateProfile,
    compute_residual_distances_from_kernel,
)
from benchmarks.queries import _build_residual_backend, _generate_points_numpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a residual-gate profile by sampling pairwise residual distances.",
    )
    parser.add_argument("output", type=Path, help="Where to write the JSON profile.")
    parser.add_argument("--tree-points", type=int, default=2048, help="Number of synthetic points to sample.")
    parser.add_argument("--dimension", type=int, default=8, help="Dimensionality of each point.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the synthetic dataset.")
    parser.add_argument("--bins", type=int, default=512, help="Number of radius bins to store in the profile.")
    parser.add_argument("--pair-chunk", type=int, default=64, help="Number of rows to process per chunk when sampling pairs.")
    parser.add_argument("--residual-lengthscale", type=float, default=1.0, help="RBF kernel lengthscale (matches benchmarks).")
    parser.add_argument("--residual-variance", type=float, default=1.0, help="RBF kernel variance (matches benchmarks).")
    parser.add_argument("--inducing", type=int, default=512, help="Number of inducing points for the residual backend.")
    return parser.parse_args()


def _build_backend(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    points = _generate_points_numpy(rng, args.tree_points, args.dimension)
    backend = _build_residual_backend(
        points,
        seed=args.seed,
        inducing_count=args.inducing,
        variance=args.residual_variance,
        lengthscale=args.residual_lengthscale,
        chunk_size=512,
    )
    return backend


def _record_pairs(backend, profile: ResidualGateProfile, pair_chunk: int) -> None:
    total = backend.num_points
    indices = np.arange(total, dtype=np.int64)
    whitened = np.asarray(backend.gate_v32, dtype=np.float64)
    for start in range(0, total, pair_chunk):
        stop = min(start + pair_chunk, total)
        rows = indices[start:stop]
        if rows.size == 0:
            continue
        kernel_block = backend.kernel_provider(rows, indices)
        residual_block = compute_residual_distances_from_kernel(
            backend,
            rows,
            indices,
            kernel_block,
        )
        whitened_rows = whitened[rows]
        diff = whitened_rows[:, None, :] - whitened[None, :, :]
        whitened_block = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64))
        for offset, row_idx in enumerate(rows):
            col_start = row_idx + 1
            if col_start >= total:
                continue
            residual_slice = residual_block[offset, col_start:]
            whitened_slice = whitened_block[offset, col_start:]
            if residual_slice.size == 0:
                continue
            mask = np.ones_like(residual_slice, dtype=np.uint8)
            profile.record_chunk(
                residual_distances=residual_slice,
                whitened_distances=whitened_slice,
                inclusion_mask=mask,
            )


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COVERTREEX_ENABLE_DIAGNOSTICS", "0")
    os.environ.setdefault("COVERTREEX_RESIDUAL_GATE1", "1")
    cx_config.reset_runtime_config_cache()
    backend = _build_backend(args)
    profile = ResidualGateProfile.create(
        bins=int(args.bins),
        radius_max=1.0,
        path=str(output),
        radius_eps=cx_config.runtime_config().residual_radius_floor,
    )
    _record_pairs(backend, profile, pair_chunk=int(args.pair_chunk))
    profile.dump(str(output), force=True)
    print(
        f"wrote {output} | samples={profile.samples_total} false_negatives={profile.false_negative_samples}"
    )
    reset_residual_metric()


if __name__ == "__main__":
    main()
