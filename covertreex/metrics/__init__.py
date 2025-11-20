"""Metric-specific helpers for Covertreex."""

from .residual import (
    ResidualCorrHostData,
    build_residual_backend,
    compute_residual_distance_single,
    compute_residual_distances,
    compute_residual_distances_with_radius,
    compute_residual_distances_from_kernel,
    compute_residual_distances_with_kernel,
    compute_residual_lower_bounds_from_kernel,
    configure_residual_correlation,
    decode_indices,
    get_residual_backend,
    set_residual_backend,
)

__all__ = [
    "ResidualCorrHostData",
    "build_residual_backend",
    "compute_residual_distance_single",
    "compute_residual_distances",
    "compute_residual_distances_with_radius",
    "compute_residual_distances_from_kernel",
    "compute_residual_distances_with_kernel",
    "compute_residual_lower_bounds_from_kernel",
    "configure_residual_correlation",
    "decode_indices",
    "get_residual_backend",
    "set_residual_backend",
]
