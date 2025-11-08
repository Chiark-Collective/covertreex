from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from covertreex import config as cx_config
from covertreex.core.tree import TreeBackend

from .strategy import compute_batch_order


def prepare_batch_points(
    *,
    backend: TreeBackend,
    batch_points: Any,
    runtime: cx_config.RuntimeConfig,
    apply_batch_order: bool,
) -> tuple[Any, Optional[np.ndarray], Dict[str, float]]:
    """Materialise the batch array and optionally reorder it according to the runtime strategy."""

    batch_array = backend.asarray(batch_points, dtype=backend.default_float)
    batch_array = backend.device_put(batch_array)
    num_points = int(batch_array.shape[0]) if batch_array.shape else 0
    if (
        not apply_batch_order
        or runtime.batch_order_strategy == "natural"
        or num_points <= 1
    ):
        return batch_array, None, {}
    points_np = np.asarray(backend.to_numpy(batch_array), dtype=np.float64)
    order_result = compute_batch_order(
        points_np,
        strategy=runtime.batch_order_strategy,
        seed=runtime.batch_order_seed if runtime.batch_order_seed is not None else runtime.mis_seed,
    )
    permutation = order_result.permutation
    metrics = dict(order_result.metrics)
    if permutation is None:
        return batch_array, None, metrics
    perm_backend = backend.asarray(permutation, dtype=backend.default_int)
    perm_backend = backend.device_put(perm_backend)
    ordered = backend.xp.take(batch_array, perm_backend, axis=0)
    ordered = backend.device_put(ordered)
    return ordered, permutation, metrics


def choose_prefix_factor(runtime: cx_config.RuntimeConfig, domination_ratio: float) -> float:
    if domination_ratio >= runtime.prefix_density_high:
        return runtime.prefix_growth_small
    if domination_ratio <= runtime.prefix_density_low:
        return runtime.prefix_growth_large
    return runtime.prefix_growth_mid


def prefix_slices(length: int) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    size = 1
    start = 0
    while start < length:
        end = min(start + size, length)
        slices.append((start, end))
        start = end
        remaining = length - start
        if remaining <= 0:
            break
        size = min(size * 2, remaining)
    return slices
