from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched k-NN query latency for the PCCT implementation."
    )
    parser.add_argument("--dimension", type=int, default=8, help="Dimensionality of points.")
    parser.add_argument(
        "--tree-points",
        type=int,
        default=16_384,
        help="Number of points to populate the tree with before querying.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used while constructing the tree.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1024,
        help="Number of query points to evaluate.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of neighbours to request.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier propagated to telemetry artifacts (default: auto-generated).",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean", "residual"),
        default="euclidean",
        help="Distance metric to benchmark (configures residual caches when 'residual').",
    )
    parser.add_argument(
        "--residual-lengthscale",
        type=float,
        default=1.0,
        help="RBF kernel lengthscale for synthetic residual caches.",
    )
    parser.add_argument(
        "--residual-variance",
        type=float,
        default=1.0,
        help="RBF kernel variance for synthetic residual caches.",
    )
    parser.add_argument(
        "--residual-inducing",
        type=int,
        default=512,
        help="Number of inducing points to use when building residual caches.",
    )
    parser.add_argument(
        "--residual-chunk-size",
        type=int,
        default=512,
        help="Chunk size for residual kernel streaming.",
    )
    parser.add_argument(
        "--baseline",
        choices=("none", "sequential", "gpboost", "external", "both", "all"),
        default="none",
        help=(
            "Include baseline comparisons. Install '.[baseline]' for the external library and "
            "'numba' extra for the GPBoost baseline. Options: 'sequential', 'gpboost', "
            "'external', 'both' (sequential + external), 'all' (sequential + gpboost + external)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write per-batch telemetry as JSON lines to the specified path.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable JSONL batch telemetry output (enabled by default).",
    )
    parser.add_argument(
        "--batch-order",
        choices=("natural", "random", "hilbert"),
        default=None,
        help="Override COVERTREEX_BATCH_ORDER for this run.",
    )
    parser.add_argument(
        "--batch-order-seed",
        type=int,
        default=None,
        help="Override COVERTREEX_BATCH_ORDER_SEED for this run.",
    )
    parser.add_argument(
        "--prefix-schedule",
        choices=("doubling", "adaptive"),
        default=None,
        help="Override COVERTREEX_PREFIX_SCHEDULE for this run.",
    )
    parser.add_argument(
        "--build-mode",
        choices=("batch", "prefix"),
        default="batch",
        help="Choose the tree construction strategy (standard batch inserts or prefix doubling).",
    )
    parser.add_argument(
        "--residual-gate",
        choices=("off", "lookup"),
        default=None,
        help="Residual-only: opt into the experimental Gate-1 path (default stays off). 'lookup' wires sparse traversal + lookup table, 'off' keeps the gate disabled.",
    )
    parser.add_argument(
        "--residual-gate-lookup-path",
        type=str,
        default="docs/data/residual_gate_profile_diag0.json",
        help="Lookup JSON used when --residual-gate=lookup (default: diag0 profile).",
    )
    parser.add_argument(
        "--residual-gate-margin",
        type=float,
        default=0.02,
        help="Safety margin added to lookup thresholds when --residual-gate=lookup.",
    )
    parser.add_argument(
        "--residual-gate-cap",
        type=float,
        default=0.0,
        help="Optional radius cap passed to the lookup preset (0 keeps existing env/default).",
    )
    parser.add_argument(
        "--residual-gate-profile-path",
        type=str,
        default=None,
        help="Residual-only: record Gate-1 profile samples to this JSON file (defaults to timestamped artefact when --residual-gate-profile-log is set).",
    )
    parser.add_argument(
        "--residual-gate-profile-bins",
        type=int,
        default=512,
        help="Residual-only: number of bins to use when recording Gate-1 profiles (default: 512).",
    )
    parser.add_argument(
        "--residual-gate-profile-log",
        type=str,
        default=None,
        help="Residual-only: append the recorded Gate-1 profile payload (including metadata) to this JSONL file for offline ingestion.",
    )
    parser.add_argument(
        "--residual-scope-caps",
        type=str,
        default=None,
        help="Residual-only: JSON file describing per-level scope radius caps.",
    )
    parser.add_argument(
        "--residual-scope-cap-default",
        type=float,
        default=None,
        help="Residual-only: fallback radius cap applied when no per-level cap matches.",
    )
    parser.add_argument(
        "--residual-scope-cap-output",
        type=str,
        default=None,
        help="Residual-only: write derived per-level scope caps to this JSON file.",
    )
    parser.add_argument(
        "--residual-scope-cap-percentile",
        type=float,
        default=0.5,
        help="Quantile (0-1) used when deriving new scope caps (default: median).",
    )
    parser.add_argument(
        "--residual-scope-cap-margin",
        type=float,
        default=0.05,
        help="Safety margin added to the sampled percentile when deriving scope caps.",
    )
    return parser.parse_args()


__all__ = ["_parse_args"]
