from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from types import SimpleNamespace

from covertreex.algo.traverse.strategies import residual as residual_strategy
from covertreex.metrics.residual import (
    ResidualCorrHostData,
    ResidualDistanceTelemetry,
    ResidualWorkspace,
)


def _build_host_backend(chunk_size: int = 2) -> ResidualCorrHostData:
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
        chunk_size=int(chunk_size),
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


def _dummy_tree(num_points: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        next_cache=np.full(num_points, -1, dtype=np.int64),
        top_levels=np.zeros(num_points, dtype=np.int64),
    )


def test_scope_streaming_respects_force_whitened_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_host_backend()
    query_indices = np.array([0], dtype=np.int64)
    tree_indices = np.array([0, 1, 2], dtype=np.int64)
    parent_positions = np.array([0], dtype=np.int64)
    radii = np.array([1.0], dtype=np.float64)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=2)
    telemetry = ResidualDistanceTelemetry()
    tree = _dummy_tree(num_points=tree_indices.size)

    flags: list[bool] = []

    def _fake_compute(**kwargs):
        flags.append(bool(kwargs.get("force_whitened", False)))
        chunk = kwargs["chunk_indices"]
        size = int(np.asarray(chunk).size)
        return np.zeros(size, dtype=np.float64), np.ones(size, dtype=np.uint8)

    monkeypatch.setattr(
        residual_strategy,
        "compute_residual_distances_with_radius",
        _fake_compute,
    )

    residual_strategy._collect_residual_scopes_streaming_serial(
        tree=tree,
        host_backend=backend,
        query_indices=query_indices,
        tree_indices=tree_indices,
        parent_positions=parent_positions,
        radii=radii,
        workspace=workspace,
        telemetry=telemetry,
        force_whitened=False,
    )
    assert flags and not any(flags)

    flags.clear()
    residual_strategy._collect_residual_scopes_streaming_serial(
        tree=tree,
        host_backend=backend,
        query_indices=query_indices,
        tree_indices=tree_indices,
        parent_positions=parent_positions,
        radii=radii,
        workspace=workspace,
        telemetry=telemetry,
        force_whitened=True,
    )
    assert flags and all(flags)


def test_resolve_scope_limits_dense_fallback() -> None:
    runtime = SimpleNamespace(
        scope_chunk_target=0,
        residual_scope_member_limit=None,
    )
    limit, scan_cap = residual_strategy._resolve_scope_limits(runtime, gate_active=False)
    assert limit == residual_strategy._RESIDUAL_SCOPE_DENSE_FALLBACK_LIMIT
    assert scan_cap == 0


def test_resolve_scope_limits_respects_override() -> None:
    runtime = SimpleNamespace(
        scope_chunk_target=2048,
        residual_scope_member_limit=512,
    )
    limit, scan_cap = residual_strategy._resolve_scope_limits(runtime, gate_active=False)
    assert limit == 512
    assert scan_cap == 2048


def test_resolve_scope_limits_gate_on_skips_fallback() -> None:
    runtime = SimpleNamespace(
        scope_chunk_target=0,
        residual_scope_member_limit=None,
    )
    limit, scan_cap = residual_strategy._resolve_scope_limits(runtime, gate_active=True)
    assert limit == residual_strategy._RESIDUAL_SCOPE_DEFAULT_LIMIT
    assert scan_cap == 0


def test_parallel_streaming_honors_stream_tile_override() -> None:
    backend = _build_host_backend(chunk_size=64)
    base_kernel = backend.kernel_provider
    recorded: list[int] = []

    def tracking_kernel(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        recorded.append(int(np.asarray(cols, dtype=np.int64).size))
        return base_kernel(rows, cols)

    backend = replace(backend, kernel_provider=tracking_kernel)
    tree_indices = np.tile(np.array([0, 1, 2], dtype=np.int64), 32)
    tree = _dummy_tree(num_points=tree_indices.size)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=backend.chunk_size)
    telemetry = ResidualDistanceTelemetry()

    residual_strategy._collect_residual_scopes_streaming_parallel(
        tree=tree,
        host_backend=backend,
        query_indices=np.array([0], dtype=np.int64),
        tree_indices=tree_indices,
        parent_positions=np.array([0], dtype=np.int64),
        radii=np.array([1.0], dtype=np.float64),
        scope_limit=None,
        stream_tile=5,
        workspace=workspace,
        telemetry=telemetry,
    )

    assert recorded and max(recorded) <= 5


def test_parallel_streaming_tiles_with_scope_limit() -> None:
    backend = _build_host_backend(chunk_size=64)
    recorded: list[int] = []
    base_kernel = backend.kernel_provider

    def tracking_kernel(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        recorded.append(int(np.asarray(cols, dtype=np.int64).size))
        return base_kernel(rows, cols)

    backend = replace(backend, kernel_provider=tracking_kernel)
    tree_indices = np.tile(np.array([0, 1, 2], dtype=np.int64), 32)
    tree = _dummy_tree(num_points=tree_indices.size)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=backend.chunk_size)
    telemetry = ResidualDistanceTelemetry()

    residual_strategy._collect_residual_scopes_streaming_parallel(
        tree=tree,
        host_backend=backend,
        query_indices=np.array([0], dtype=np.int64),
        tree_indices=tree_indices,
        parent_positions=np.array([0], dtype=np.int64),
        radii=np.array([1.0], dtype=np.float64),
        scope_limit=3,
        stream_tile=None,
        workspace=workspace,
        telemetry=telemetry,
    )

    assert recorded and max(recorded) <= 3


def test_parallel_streaming_dynamic_tile_respects_budget() -> None:
    backend = _build_host_backend(chunk_size=32)
    base_kernel = backend.kernel_provider
    recorded: list[int] = []

    def tracking_kernel(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        recorded.append(int(np.asarray(cols, dtype=np.int64).size))
        return base_kernel(rows, cols)

    backend = replace(backend, kernel_provider=tracking_kernel)
    tree_indices = np.tile(np.array([0, 1, 2], dtype=np.int64), 16)
    tree = _dummy_tree(num_points=tree_indices.size)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=backend.chunk_size)
    telemetry = ResidualDistanceTelemetry()

    residual_strategy._collect_residual_scopes_streaming_parallel(
        tree=tree,
        host_backend=backend,
        query_indices=np.array([0], dtype=np.int64),
        tree_indices=tree_indices,
        parent_positions=np.array([0], dtype=np.int64),
        radii=np.array([1.0], dtype=np.float64),
        scope_limit=64,
        scope_budget_schedule=(4,),
        scope_budget_up_thresh=2.0,
        scope_budget_down_thresh=0.5,
        workspace=workspace,
        telemetry=telemetry,
    )

    assert recorded and max(recorded) <= 4


def test_parallel_streaming_uses_numba_scope_append(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_host_backend(chunk_size=4)
    tree = _dummy_tree(num_points=3)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=backend.chunk_size)
    telemetry = ResidualDistanceTelemetry()

    append_calls: list[int] = []
    reset_counts: list[int] = []

    def fake_append(
        flags: np.ndarray,
        positions: np.ndarray,
        buffer: np.ndarray,
        count: int,
        limit: int,
        *,
        respect_limit: bool = True,
    ) -> tuple[int, int, bool]:
        append_calls.append(int(np.asarray(positions, dtype=np.int64).size))
        new_count = int(count) + append_calls[-1]
        return new_count, 0, False

    def fake_reset(flags: np.ndarray, buffer: np.ndarray, count: int) -> None:
        reset_counts.append(int(count))

    monkeypatch.setattr(residual_strategy, "residual_scope_append", fake_append)
    monkeypatch.setattr(residual_strategy, "residual_scope_reset", fake_reset)

    residual_strategy._collect_residual_scopes_streaming_parallel(
        tree=tree,
        host_backend=backend,
        query_indices=np.array([0], dtype=np.int64),
        tree_indices=np.array([0, 1, 2], dtype=np.int64),
        parent_positions=np.array([0], dtype=np.int64),
        radii=np.array([1.0], dtype=np.float64),
        scope_limit=2,
        stream_tile=1,
        workspace=workspace,
        telemetry=telemetry,
    )

    assert append_calls, "expected the Numba append helper to be used"
    assert reset_counts and reset_counts[0] > 0


def test_parallel_streaming_masked_append_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _build_host_backend(chunk_size=4)
    tree = _dummy_tree(num_points=3)
    workspace = ResidualWorkspace(max_queries=1, max_chunk=backend.chunk_size)
    telemetry = ResidualDistanceTelemetry()

    calls = {"masked": 0}

    def fake_masked(**kwargs):  # type: ignore[no-untyped-def]
        calls["masked"] = calls.get("masked", 0) + 1
        scope_count = int(kwargs.get("scope_count", 0))
        return scope_count + 1, 0, False, 1, 0.25

    monkeypatch.setattr(residual_strategy, "_append_scope_positions_masked", fake_masked)

    residual_strategy._collect_residual_scopes_streaming_parallel(
        tree=tree,
        host_backend=backend,
        query_indices=np.array([0], dtype=np.int64),
        tree_indices=np.array([0, 1, 2], dtype=np.int64),
        parent_positions=np.array([0], dtype=np.int64),
        radii=np.array([1.0], dtype=np.float64),
        scope_limit=2,
        stream_tile=1,
        workspace=workspace,
        telemetry=telemetry,
        masked_scope_append=True,
    )
    assert calls["masked"] > 0

    calls["masked"] = 0
    residual_strategy._collect_residual_scopes_streaming_parallel(
        tree=tree,
        host_backend=backend,
        query_indices=np.array([0], dtype=np.int64),
        tree_indices=np.array([0, 1, 2], dtype=np.int64),
        parent_positions=np.array([0], dtype=np.int64),
        radii=np.array([1.0], dtype=np.float64),
        scope_limit=2,
        stream_tile=1,
        workspace=workspace,
        telemetry=telemetry,
        masked_scope_append=False,
    )
    assert calls["masked"] == 0


def test_compute_dynamic_tile_stride_delegates_to_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {}

    def fake_stride(*args, **kwargs):
        called["count"] = called.get("count", 0) + 1
        return 7

    monkeypatch.setattr(residual_strategy, "NUMBA_RESIDUAL_SCOPE_AVAILABLE", True)
    monkeypatch.setattr(residual_strategy, "residual_scope_dynamic_tile_stride", fake_stride)
    stride = residual_strategy._compute_dynamic_tile_stride(
        base_stride=16,
        active_idx=np.array([0], dtype=np.int64),
        block_idx_arr=np.array([0], dtype=np.int64),
        scope_counts=np.array([0], dtype=np.int64),
        limit_value=32,
        budget_enabled=True,
        budget_applied=np.array([True]),
        budget_limits=np.array([32], dtype=np.int64),
    )
    assert stride == 7
    assert called.get("count") == 1


def test_update_scope_budget_state_delegates_to_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {}

    def fake_update(*args, **kwargs):
        called["count"] = called.get("count", 0) + 1

    monkeypatch.setattr(residual_strategy, "NUMBA_RESIDUAL_SCOPE_AVAILABLE", True)
    monkeypatch.setattr(residual_strategy, "residual_scope_update_budget_state", fake_update)
    residual_strategy._update_scope_budget_state(
        qi=0,
        chunk_points=np.array([0], dtype=np.int64),
        scan_cap_value=10,
        budget_applied=np.array([True]),
        budget_up=2.0,
        budget_down=0.5,
        budget_schedule_arr=np.array([2, 4], dtype=np.int64),
        budget_indices=np.array([0], dtype=np.int64),
        budget_limits=np.array([2], dtype=np.int64),
        budget_final_limits=np.array([2], dtype=np.int64),
        budget_escalations=np.array([0], dtype=np.int64),
        budget_low_streak=np.array([0], dtype=np.int64),
        budget_survivors=np.array([0], dtype=np.int64),
        budget_early_flags=np.array([0], dtype=np.uint8),
        saturated=np.array([False]),
        saturated_flags=np.array([0], dtype=np.uint8),
    )
    assert called.get("count") == 1


def test_residual_scope_append_masked_appends_members() -> None:
    flags = np.zeros(8, dtype=np.uint8)
    buffer = np.zeros(4, dtype=np.int64)
    mask_row = np.array([1, 0, 1, 0], dtype=np.uint8)
    distances_row = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    tile_positions = np.array([0, 1, 2, 3], dtype=np.int64)

    new_count, dedupe, trimmed, added, observed = residual_strategy.residual_scope_append_masked(
        flags,
        buffer,
        mask_row,
        distances_row,
        tile_positions,
        count=0,
        limit=4,
    )

    assert new_count == 2
    assert added == 2
    assert dedupe == 0
    assert not trimmed
    assert observed == pytest.approx(0.3)


def test_append_scope_positions_masked_prefers_numba_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {}

    def fake_masked(*args, **kwargs):
        called["count"] = called.get("count", 0) + 1
        return 3, 1, True, 2, 0.5

    monkeypatch.setattr(residual_strategy, "residual_scope_append_masked", fake_masked)

    flags_row = np.zeros(6, dtype=np.uint8)
    buffer_row = np.zeros(4, dtype=np.int64)
    mask_row = np.array([1, 0, 1], dtype=np.uint8)
    distances_row = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    tile_positions = np.array([0, 1, 2], dtype=np.int64)

    result = residual_strategy._append_scope_positions_masked(
        flags_row=flags_row,
        bitset_row=None,
        mask_row=mask_row,
        distances_row=distances_row,
        tile_positions=tile_positions,
        limit_value=4,
        scope_count=0,
        buffer_row=buffer_row,
    )

    assert called.get("count") == 1
    assert result[0] == 3
    assert result[1] == 1
    assert result[2] is True
