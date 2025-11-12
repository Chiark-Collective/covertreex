from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency mirrors residual kernels
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when numba is unavailable
    njit = None  # type: ignore
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _update_quantile_reservoir_impl(
        bin_indices: np.ndarray,
        values: np.ndarray,
        samples: np.ndarray,
        sample_counts: np.ndarray,
        total_counts: np.ndarray,
    ) -> None:
        capacity = samples.shape[1]
        if capacity <= 0:
            return
        num_bins = samples.shape[0]
        for idx in range(values.size):
            bin_idx = int(bin_indices[idx])
            if bin_idx < 0 or bin_idx >= num_bins:
                continue
            total = total_counts[bin_idx] + 1
            total_counts[bin_idx] = total
            count = sample_counts[bin_idx]
            value = values[idx]
            if count < capacity:
                samples[bin_idx, count] = value
                sample_counts[bin_idx] = count + 1
                continue
            draw = np.random.random()
            if draw * total >= capacity:
                continue
            slot = int(np.random.random() * capacity)
            if slot < capacity:
                samples[bin_idx, slot] = value


def update_quantile_reservoir(
    bin_indices: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray,
    sample_counts: np.ndarray,
    total_counts: np.ndarray,
) -> None:
    """Reservoir-sample whitened distances per radius bin.

    Parameters
    ----------
    bin_indices:
        Radius-bin indices (shape (N,), dtype=int64).
    values:
        Whitened distances aligned with ``bin_indices`` (float32/float64).
    samples:
        Array shaped (bins, capacity) that stores the sampled values per bin.
    sample_counts:
        Number of filled entries per bin (<= capacity).
    total_counts:
        Total number of samples observed per bin.
    """

    if bin_indices.size == 0 or values.size == 0:
        return
    if samples.size == 0:
        return
    if _NUMBA_AVAILABLE and _update_quantile_reservoir_impl is not None:  # type: ignore
        _update_quantile_reservoir_impl(  # type: ignore[misc]
            bin_indices,
            values,
            samples,
            sample_counts,
            total_counts,
        )
        return

    # Python fallback (slower, but only hit when numba is absent).
    import random

    capacity = samples.shape[1]
    if capacity <= 0:
        return
    num_bins = samples.shape[0]
    for idx in range(values.size):
        bin_idx = int(bin_indices[idx])
        if bin_idx < 0 or bin_idx >= num_bins:
            continue
        total = total_counts[bin_idx] + 1
        total_counts[bin_idx] = total
        count = sample_counts[bin_idx]
        value = float(values[idx])
        if count < capacity:
            samples[bin_idx, count] = value
            sample_counts[bin_idx] = count + 1
            continue
        if random.random() * total >= capacity:
            continue
        slot = int(random.random() * capacity)
        if slot < capacity:
            samples[bin_idx, slot] = value


__all__ = ["update_quantile_reservoir"]
