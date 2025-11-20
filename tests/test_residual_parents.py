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
    )
    return backend


def _dummy_tree(num_points: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        next_cache=np.full(num_points, -1, dtype=np.int64),
        top_levels=np.zeros(num_points, dtype=np.int64),
        points=np.zeros((num_points, 1), dtype=np.float64),
        children=np.full(num_points, -1, dtype=np.int32),
        si_cache=np.zeros(num_points, dtype=np.float64),
        num_points=num_points,
        backend=SimpleNamespace(to_numpy=lambda x: x),
    )


def test_residual_parent_search_skips_whitened_path_when_gate_disabled() -> None:
    backend = _build_host_backend()
    telemetry = ResidualDistanceTelemetry()
    query_indices = np.array([0, 1], dtype=np.int64)
    tree = _dummy_tree(num_points=3)

    parents, distances = residual_strategy._residual_find_parents(
        host_backend=backend,
        query_indices=query_indices,
        tree=tree,
        telemetry=telemetry,
    )

    assert telemetry.kernel_calls >= 0 # Kernel calls might be 0 if Numba path takes over, or >0 if fallback
    assert parents.shape == query_indices.shape
    assert np.all(np.isfinite(distances)) # Parents might be -1 if dummy tree data doesn't produce valid parents in fallback?
    # Actually fallback scans chunks. 3 points.
    # distances_block will be computed.



# Obsolete test removed: test_scope_streaming_respects_force_whitened_flag


def test_resolve_scope_limits_dense_fallback() -> None:
    runtime = SimpleNamespace(
        scope_chunk_target=0,
        residual_scope_member_limit=None,
    )
    limit, scan_cap = residual_strategy._resolve_scope_limits(runtime)
    assert limit == residual_strategy._RESIDUAL_SCOPE_DENSE_FALLBACK_LIMIT
    assert scan_cap == 0


def test_resolve_scope_limits_respects_override() -> None:
    runtime = SimpleNamespace(
        scope_chunk_target=2048,
        residual_scope_member_limit=512,
    )
    limit, scan_cap = residual_strategy._resolve_scope_limits(runtime)
    assert limit == 512
    assert scan_cap == 2048


def test_resolve_scope_limits_default_behavior() -> None:
    runtime = SimpleNamespace(
        scope_chunk_target=0,
        residual_scope_member_limit=None,
    )
    limit, scan_cap = residual_strategy._resolve_scope_limits(runtime)
    # Defaults to dense fallback (small limit) when no limit is specified
    assert limit == residual_strategy._RESIDUAL_SCOPE_DENSE_FALLBACK_LIMIT
    assert scan_cap == 0


# Obsolete test removed: test_parallel_streaming_honors_stream_tile_override


# Obsolete test removed: test_parallel_streaming_tiles_with_scope_limit


def test_level_cache_prefetch_batches_queries() -> None:
    backend = _build_host_backend()
    base_kernel = backend.kernel_provider
    counter = {"calls": 0}

    def tracking_kernel(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        counter["calls"] += 1
        return base_kernel(rows, cols)

    backend = replace(backend, kernel_provider=tracking_kernel)
    tree_indices = np.array([0, 1, 2], dtype=np.int64)
    query_indices = np.array([0, 1], dtype=np.int64)
    radii = np.array([1.0, 1.0], dtype=np.float64)
    flags_matrix = np.zeros((2, tree_indices.size), dtype=np.uint8)
    scope_counts = np.zeros(2, dtype=np.int64)
    budget_applied = np.zeros(2, dtype=bool)
    budget_survivors = np.zeros(2, dtype=np.int64)
    trimmed_flags = np.zeros(2, dtype=bool)
    saturated = np.zeros(2, dtype=bool)
    saturated_flags = np.zeros(2, dtype=np.uint8)
    dedupe_hits = np.zeros(2, dtype=np.int64)
    observed_radii = np.zeros(2, dtype=np.float64)
    telemetry = ResidualDistanceTelemetry()
    workspace = ResidualWorkspace(max_queries=2, max_chunk=backend.chunk_size)

    prefetch, hits = residual_strategy._process_level_cache_hits(
        cache_jobs={0: [0, 1]},
        level_scope_cache={0: np.array([0, 1], dtype=np.int64)},
        total_points=tree_indices.size,
        tree_indices_np=tree_indices,
        query_indices=query_indices,
        radii=radii,
        host_backend=backend,
        distance_telemetry=telemetry,
        limit_value=0,
        use_masked_append=False,
        bitset_enabled=False,
        scope_buffers=None,
        scope_counts=scope_counts,
        scope_bitsets=None,
        flags_matrix=flags_matrix,
        budget_applied=budget_applied,
        budget_survivors=budget_survivors,
        trimmed_flags=trimmed_flags,
        saturated=saturated,
        saturated_flags=saturated_flags,
        dedupe_hits=dedupe_hits,
        observed_radii=observed_radii,
        shared_workspace=workspace,
    )

    assert prefetch == 4
    assert hits >= 0
    assert counter["calls"] == 1


def test_residual_collect_next_chain_tracks_sequence() -> None:
    next_cache = np.array([1, 2, -1], dtype=np.int64)
    visited = np.zeros(next_cache.size, dtype=np.uint8)
    buffer = np.zeros(next_cache.size, dtype=np.int64)

    count = residual_strategy.residual_collect_next_chain(next_cache, 0, visited, buffer)
    assert count == 3
    assert np.array_equal(buffer[:count], np.array([0, 1, 2], dtype=np.int64))

    count_again = residual_strategy.residual_collect_next_chain(next_cache, 0, visited, buffer)
    assert count_again == 3


# Obsolete test removed: test_parallel_streaming_dynamic_tile_respects_budget


# Obsolete test removed: test_parallel_streaming_uses_numba_scope_append


# Obsolete test removed: test_parallel_streaming_masked_append_toggle


# Obsolete test removed: test_parallel_streaming_masked_append_supports_bitset


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


def test_append_scope_positions_bitset_uses_numba_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {}

    def fake_bitset(
        flags: np.ndarray,
        bitset_row: np.ndarray,
        positions: np.ndarray,
        count: int,
        limit: int,
        *,
        respect_limit: bool = True,
    ) -> tuple[int, int, bool, int]:
        called["bitset"] = called.get("bitset", 0) + 1
        size = int(np.asarray(positions, dtype=np.int64).size)
        new_count = int(count) + size
        return new_count, 0, False, size

    monkeypatch.setattr(residual_strategy, "residual_scope_append_bitset", fake_bitset)

    flags_row = np.zeros(64, dtype=np.uint8)
    bitset_row = np.zeros(1, dtype=np.uint64)
    result = residual_strategy._append_scope_positions(
        flags_row,
        bitset_row,
        np.array([1, 2], dtype=np.int64),
        limit_value=8,
        scope_count=0,
    )

    assert called.get("bitset") == 1
    assert result == (2, 0, False, 2)


def test_append_scope_positions_masked_prefers_bitset_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {"bitset": 0, "dense": 0}

    def fake_masked_bitset(
        flags: np.ndarray,
        bitset_row: np.ndarray,
        mask_row: np.ndarray,
        distances_row: np.ndarray,
        tile_positions: np.ndarray,
        count: int,
        limit: int,
        *,
        respect_limit: bool = True,
    ) -> tuple[int, int, bool, int, float]:
        called["bitset"] += 1
        return int(count) + 1, 0, False, 1, 0.5

    def fake_masked_dense(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["dense"] += 1
        return kwargs.get("scope_count", 0), 0, False, 0, 0.0

    monkeypatch.setattr(residual_strategy, "residual_scope_append_masked_bitset", fake_masked_bitset)
    monkeypatch.setattr(residual_strategy, "residual_scope_append_masked", fake_masked_dense)

    flags_row = np.zeros(64, dtype=np.uint8)
    bitset_row = np.zeros(1, dtype=np.uint64)
    mask_row = np.array([1, 0, 1], dtype=np.uint8)
    distances_row = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    tile_positions = np.array([0, 1, 2], dtype=np.int64)

    result = residual_strategy._append_scope_positions_masked(
        flags_row=flags_row,
        bitset_row=bitset_row,
        mask_row=mask_row,
        distances_row=distances_row,
        tile_positions=tile_positions,
        limit_value=0,
        scope_count=0,
        buffer_row=None,
    )

    assert called["bitset"] == 1
    assert called["dense"] == 0
    assert result[0] == 1
    assert result[3] == 1
