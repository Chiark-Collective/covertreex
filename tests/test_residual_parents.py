from __future__ import annotations

import numpy as np

from covertreex.algo.traverse.strategies import residual as residual_strategy
from covertreex.metrics.residual import ResidualCorrHostData, ResidualDistanceTelemetry


def _build_host_backend() -> ResidualCorrHostData:
    v_matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p_diag = np.array([0.9, 1.1, 1.2], dtype=np.float64)
    kernel_full = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.6],
            [0.2, 0.6, 1.0],
        ],
        dtype=np.float64,
    )
    kernel_diag = np.diag(kernel_full).copy()

    def kernel_provider(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        row_idx = np.asarray(rows, dtype=np.int64)
        col_idx = np.asarray(cols, dtype=np.int64)
        return kernel_full[np.ix_(row_idx, col_idx)]

    backend = ResidualCorrHostData(
        v_matrix=v_matrix,
        p_diag=p_diag,
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        chunk_size=2,
        gate1_enabled=False,
    )
    return backend


def test_residual_parent_search_skips_whitened_path_when_gate_disabled() -> None:
    backend = _build_host_backend()
    telemetry = ResidualDistanceTelemetry()
    query_indices = np.array([0, 1], dtype=np.int64)
    tree_indices = np.array([0, 1, 2], dtype=np.int64)

    parents, distances = residual_strategy._residual_find_parents(
        host_backend=backend,
        query_indices=query_indices,
        tree_indices=tree_indices,
        telemetry=telemetry,
    )

    assert telemetry.whitened_calls == 0
    assert telemetry.kernel_calls > 0
    assert parents.shape == query_indices.shape
    assert np.all(parents >= 0)
    assert np.all(np.isfinite(distances))
