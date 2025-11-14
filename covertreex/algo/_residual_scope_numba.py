from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit  # type: ignore

    NUMBA_RESIDUAL_SCOPE_AVAILABLE = True
except Exception:  # pragma: no cover - when numba unavailable
    njit = None  # type: ignore
    NUMBA_RESIDUAL_SCOPE_AVAILABLE = False


if NUMBA_RESIDUAL_SCOPE_AVAILABLE:

    @njit(cache=True)
    def _append_positions_impl(
        flags: np.ndarray,
        positions: np.ndarray,
        buffer: np.ndarray,
        count: int,
        limit: int,
        respect_limit: bool,
    ) -> tuple[int, int, int]:
        dedupe = 0
        saturated = 0
        num_flags = flags.shape[0]
        capacity = buffer.shape[0]
        limit_enabled = respect_limit and limit > 0

        for idx in range(positions.shape[0]):
            pos = int(positions[idx])
            if pos < 0 or pos >= num_flags:
                continue
            if flags[pos] != 0:
                dedupe += 1
                continue
            flags[pos] = 1
            if count < capacity:
                buffer[count] = pos
            count += 1
            if capacity > 0 and count >= capacity:
                saturated = 1
                break
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated

    @njit(cache=True)
    def _append_positions_masked_impl(
        flags: np.ndarray,
        mask_row: np.ndarray,
        distances_row: np.ndarray,
        tile_positions: np.ndarray,
        buffer: np.ndarray,
        count: int,
        limit: int,
        respect_limit: bool,
    ) -> tuple[int, int, int, int, float]:
        dedupe = 0
        saturated = 0
        added = 0
        max_distance = 0.0
        num_flags = flags.shape[0]
        capacity = buffer.shape[0]
        limit_enabled = respect_limit and limit > 0
        current = count

        for col in range(mask_row.shape[0]):
            if mask_row[col] == 0:
                continue
            pos = int(tile_positions[col])
            if pos < 0 or pos >= num_flags:
                continue
            if flags[pos] != 0:
                dedupe += 1
                continue
            flags[pos] = 1
            if capacity > 0 and current < capacity:
                buffer[current] = pos
            current += 1
            added += 1
            dist_val = float(distances_row[col])
            if dist_val > max_distance:
                max_distance = dist_val
            if capacity > 0 and current >= capacity:
                saturated = 1
                break
            if limit_enabled and current >= limit:
                saturated = 1
                break

        return current, dedupe, saturated, added, max_distance

    @njit(cache=True)
    def _append_positions_bitset_impl(
        flags: np.ndarray,
        bitset_row: np.ndarray,
        positions: np.ndarray,
        count: int,
        limit: int,
        respect_limit: bool,
    ) -> tuple[int, int, int, int]:
        dedupe = 0
        saturated = 0
        added = 0
        num_flags = flags.shape[0]
        limit_enabled = respect_limit and limit > 0

        for idx in range(positions.shape[0]):
            pos = int(positions[idx])
            if pos < 0 or pos >= num_flags:
                continue
            word = pos >> 6
            bit_mask = np.uint64(1) << np.uint64(pos & 63)
            if (bitset_row[word] & bit_mask) != 0:
                dedupe += 1
                continue
            bitset_row[word] |= bit_mask
            flags[pos] = 1
            count += 1
            added += 1
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated, added

    @njit(cache=True)
    def _append_positions_masked_bitset_impl(
        flags: np.ndarray,
        bitset_row: np.ndarray,
        mask_row: np.ndarray,
        distances_row: np.ndarray,
        tile_positions: np.ndarray,
        count: int,
        limit: int,
        respect_limit: bool,
    ) -> tuple[int, int, int, int, float]:
        dedupe = 0
        saturated = 0
        added = 0
        max_distance = 0.0
        num_flags = flags.shape[0]
        limit_enabled = respect_limit and limit > 0

        for idx in range(mask_row.shape[0]):
            if mask_row[idx] == 0:
                continue
            pos = int(tile_positions[idx])
            if pos < 0 or pos >= num_flags:
                continue
            word = pos >> 6
            bit_mask = np.uint64(1) << np.uint64(pos & 63)
            if (bitset_row[word] & bit_mask) != 0:
                dedupe += 1
                continue
            bitset_row[word] |= bit_mask
            flags[pos] = 1
            count += 1
            added += 1
            dist_val = float(distances_row[idx])
            if dist_val > max_distance:
                max_distance = dist_val
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated, added, max_distance

    @njit(cache=True)
    def _reset_flags_impl(flags: np.ndarray, buffer: np.ndarray, count: int) -> None:
        num_flags = flags.shape[0]
        total = buffer.shape[0]
        limit = count if count < total else total
        for idx in range(limit):
            pos = int(buffer[idx])
            if 0 <= pos < num_flags:
                flags[pos] = 0

    @njit(cache=True)
    def _dynamic_tile_stride_impl(
        base_stride: int,
        active_idx: np.ndarray,
        block_idx_arr: np.ndarray,
        scope_counts: np.ndarray,
        limit_value: int,
        budget_enabled: bool,
        budget_applied: np.ndarray,
        budget_limits: np.ndarray,
    ) -> int:
        stride = base_stride if base_stride > 0 else 1
        if active_idx.size == 0 or (limit_value <= 0 and not budget_enabled):
            return stride
        max_remaining = 0
        for slot in active_idx:
            qi = int(block_idx_arr[int(slot)])
            cap = limit_value if limit_value > 0 else 0
            if budget_enabled and budget_applied[qi]:
                budget_cap = int(budget_limits[qi])
                if budget_cap > 0:
                    if cap <= 0 or budget_cap < cap:
                        cap = budget_cap
            if cap <= 0:
                continue
            remaining = cap - int(scope_counts[qi])
            if remaining > max_remaining:
                max_remaining = remaining
                if max_remaining >= stride:
                    return stride
        if max_remaining <= 0:
            return stride
        if max_remaining < stride:
            stride = max_remaining
        if stride <= 0:
            return 1
        return stride

    @njit(cache=True)
    def _update_budget_state_impl(
        qi: int,
        chunk_points: np.ndarray,
        scan_cap_value: int,
        budget_applied: np.ndarray,
        budget_up: float,
        budget_down: float,
        budget_schedule: np.ndarray,
        budget_indices: np.ndarray,
        budget_limits: np.ndarray,
        budget_final_limits: np.ndarray,
        budget_escalations: np.ndarray,
        budget_low_streak: np.ndarray,
        budget_survivors: np.ndarray,
        budget_early_flags: np.ndarray,
        saturated: np.ndarray,
        saturated_flags: np.ndarray,
    ) -> None:
        cap_value = scan_cap_value if scan_cap_value > 0 else -1
        if cap_value > 0 and chunk_points[qi] >= cap_value:
            saturated[qi] = True
            saturated_flags[qi] = 1
            return
        if budget_applied[qi] and (not saturated[qi]) and chunk_points[qi] > 0:
            ratio = budget_survivors[qi] / float(chunk_points[qi])
            schedule_size = budget_schedule.shape[0]
            if (
                ratio >= budget_up
                and schedule_size > 0
                and budget_indices[qi] + 1 < schedule_size
            ):
                next_limit = int(budget_schedule[budget_indices[qi] + 1])
                if cap_value > 0 and next_limit > cap_value:
                    next_limit = cap_value
                if next_limit > budget_limits[qi]:
                    budget_indices[qi] += 1
                    budget_limits[qi] = next_limit
                    budget_final_limits[qi] = next_limit
                    budget_escalations[qi] += 1
                    budget_low_streak[qi] = 0
            elif ratio < budget_down:
                budget_low_streak[qi] += 1
                if budget_low_streak[qi] >= 2:
                    budget_early_flags[qi] = 1
                    saturated[qi] = True
                    saturated_flags[qi] = 1
                    return
            else:
                budget_low_streak[qi] = 0
        if (
            budget_applied[qi]
            and not saturated[qi]
            and budget_limits[qi] > 0
            and budget_survivors[qi] >= budget_limits[qi]
        ):
            saturated[qi] = True
            saturated_flags[qi] = 1

else:  # pragma: no cover - executed when numba missing

    def _append_positions_impl(flags, positions, buffer, count, limit, respect_limit):
        dedupe = 0
        saturated = 0
        num_flags = flags.shape[0]
        capacity = buffer.shape[0]
        limit_enabled = respect_limit and limit > 0

        for pos in positions:
            pos_int = int(pos)
            if pos_int < 0 or pos_int >= num_flags:
                continue
            if flags[pos_int] != 0:
                dedupe += 1
                continue
            flags[pos_int] = 1
            if count < capacity:
                buffer[count] = pos_int
            count += 1
            if capacity > 0 and count >= capacity:
                saturated = 1
                break
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated

    def _append_positions_masked_impl(
        flags,
        mask_row,
        distances_row,
        tile_positions,
        buffer,
        count,
        limit,
        respect_limit,
    ):
        dedupe = 0
        saturated = 0
        added = 0
        max_distance = 0.0
        num_flags = flags.shape[0]
        capacity = buffer.shape[0]
        limit_enabled = respect_limit and limit > 0
        current = count

        for col in range(len(mask_row)):
            if mask_row[col] == 0:
                continue
            pos = int(tile_positions[col])
            if pos < 0 or pos >= num_flags:
                continue
            if flags[pos] != 0:
                dedupe += 1
                continue
            flags[pos] = 1
            if capacity > 0 and current < capacity:
                buffer[current] = pos
            current += 1
            added += 1
            dist_val = float(distances_row[col])
            if dist_val > max_distance:
                max_distance = dist_val
            if capacity > 0 and current >= capacity:
                saturated = 1
                break
            if limit_enabled and current >= limit:
                saturated = 1
                break

        return current, dedupe, saturated, added, max_distance

    def _append_positions_bitset_impl(flags, bitset_row, positions, count, limit, respect_limit):
        dedupe = 0
        saturated = 0
        added = 0
        num_flags = flags.shape[0]
        limit_enabled = respect_limit and limit > 0

        for pos in positions:
            pos_int = int(pos)
            if pos_int < 0 or pos_int >= num_flags:
                continue
            word = pos_int >> 6
            bit_mask = np.uint64(1) << np.uint64(pos_int & 63)
            if (bitset_row[word] & bit_mask) != 0:
                dedupe += 1
                continue
            bitset_row[word] |= bit_mask
            flags[pos_int] = 1
            count += 1
            added += 1
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated, added

    def _append_positions_masked_bitset_impl(
        flags,
        bitset_row,
        mask_row,
        distances_row,
        tile_positions,
        count,
        limit,
        respect_limit,
    ):
        dedupe = 0
        saturated = 0
        added = 0
        max_distance = 0.0
        num_flags = flags.shape[0]
        limit_enabled = respect_limit and limit > 0

        for idx, mask_val in enumerate(mask_row):
            if mask_val == 0:
                continue
            pos = int(tile_positions[idx])
            if pos < 0 or pos >= num_flags:
                continue
            word = pos >> 6
            bit_mask = np.uint64(1) << np.uint64(pos & 63)
            if (bitset_row[word] & bit_mask) != 0:
                dedupe += 1
                continue
            bitset_row[word] |= bit_mask
            flags[pos] = 1
            count += 1
            added += 1
            dist_val = float(distances_row[idx])
            if dist_val > max_distance:
                max_distance = dist_val
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated, added, max_distance

    def _reset_flags_impl(flags, buffer, count):
        num_flags = flags.shape[0]
        total = buffer.shape[0]
        limit = count if count < total else total
        for idx in range(limit):
            pos = int(buffer[idx])
            if 0 <= pos < num_flags:
                flags[pos] = 0

    def _dynamic_tile_stride_impl(
        base_stride,
        active_idx,
        block_idx_arr,
        scope_counts,
        limit_value,
        budget_enabled,
        budget_applied,
        budget_limits,
    ):
        stride = base_stride if base_stride > 0 else 1
        if len(active_idx) == 0 or (limit_value <= 0 and not budget_enabled):
            return stride
        max_remaining = 0
        for slot in active_idx:
            qi = int(block_idx_arr[int(slot)])
            cap = limit_value if limit_value > 0 else 0
            if budget_enabled and budget_applied[qi]:
                budget_cap = int(budget_limits[qi])
                if budget_cap > 0:
                    if cap <= 0 or budget_cap < cap:
                        cap = budget_cap
            if cap <= 0:
                continue
            remaining = cap - int(scope_counts[qi])
            if remaining > max_remaining:
                max_remaining = remaining
                if max_remaining >= stride:
                    return stride
        if max_remaining <= 0:
            return stride
        stride = max(1, min(stride, max_remaining))
        return stride

    def _update_budget_state_impl(
        qi,
        chunk_points,
        scan_cap_value,
        budget_applied,
        budget_up,
        budget_down,
        budget_schedule,
        budget_indices,
        budget_limits,
        budget_final_limits,
        budget_escalations,
        budget_low_streak,
        budget_survivors,
        budget_early_flags,
        saturated,
        saturated_flags,
    ):
        cap_value = scan_cap_value if scan_cap_value and scan_cap_value > 0 else -1
        if cap_value > 0 and chunk_points[qi] >= cap_value:
            saturated[qi] = True
            saturated_flags[qi] = 1
            return
        if budget_applied[qi] and not saturated[qi] and chunk_points[qi] > 0:
            ratio = budget_survivors[qi] / float(chunk_points[qi])
            if (
                ratio >= budget_up
                and budget_schedule.size > 0
                and budget_indices[qi] + 1 < budget_schedule.size
            ):
                next_limit = int(budget_schedule[budget_indices[qi] + 1])
                if cap_value > 0 and next_limit > cap_value:
                    next_limit = cap_value
                if next_limit > budget_limits[qi]:
                    budget_indices[qi] += 1
                    budget_limits[qi] = next_limit
                    budget_final_limits[qi] = next_limit
                    budget_escalations[qi] += 1
                    budget_low_streak[qi] = 0
            elif ratio < budget_down:
                budget_low_streak[qi] += 1
                if budget_low_streak[qi] >= 2:
                    budget_early_flags[qi] = 1
                    saturated[qi] = True
                    saturated_flags[qi] = 1
                    return
            else:
                budget_low_streak[qi] = 0
        if (
            budget_applied[qi]
            and not saturated[qi]
            and budget_limits[qi] > 0
            and budget_survivors[qi] >= budget_limits[qi]
        ):
            saturated[qi] = True
            saturated_flags[qi] = 1


def residual_scope_append(
    flags: np.ndarray,
    positions: np.ndarray,
    buffer: np.ndarray,
    count: int,
    limit: int,
    *,
    respect_limit: bool = True,
) -> tuple[int, int, bool]:
    """Append unique tree positions into the per-query buffer.

    Returns (new_count, dedupe_hits, hit_limit).
    """

    new_count, dedupe_hits, saturated = _append_positions_impl(
        flags,
        positions,
        buffer,
        int(count),
        int(limit),
        bool(respect_limit),
    )
    return int(new_count), int(dedupe_hits), bool(saturated)


def residual_scope_append_bitset(
    flags: np.ndarray,
    bitset_row: np.ndarray,
    positions: np.ndarray,
    count: int,
    limit: int,
    *,
    respect_limit: bool = True,
) -> tuple[int, int, bool, int]:
    """Append scope members using the bitset representation."""

    new_count, dedupe_hits, saturated, added = _append_positions_bitset_impl(
        flags,
        bitset_row,
        positions,
        int(count),
        int(limit),
        bool(respect_limit),
    )
    return int(new_count), int(dedupe_hits), bool(saturated), int(added)


def residual_scope_reset(flags: np.ndarray, buffer: np.ndarray, count: int) -> None:
    """Clear flag entries for the members stored in the buffer."""

    _reset_flags_impl(flags, buffer, int(count))


def residual_scope_append_masked(
    flags: np.ndarray,
    buffer: np.ndarray,
    mask_row: np.ndarray,
    distances_row: np.ndarray,
    tile_positions: np.ndarray,
    count: int,
    limit: int,
    *,
    respect_limit: bool = True,
) -> tuple[int, int, bool, int, float]:
    """Append members selected by a boolean mask without copying positions."""

    new_count, dedupe_hits, saturated, added, observed = _append_positions_masked_impl(
        flags,
        mask_row,
        distances_row,
        tile_positions,
        buffer,
        int(count),
        int(limit),
        bool(respect_limit),
    )
    return (
        int(new_count),
        int(dedupe_hits),
        bool(saturated),
        int(added),
        float(observed),
    )


def residual_scope_append_masked_bitset(
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
    """Append masked members when scopes are tracked via bitsets."""

    new_count, dedupe_hits, saturated, added, observed = _append_positions_masked_bitset_impl(
        flags,
        bitset_row,
        mask_row,
        distances_row,
        tile_positions,
        int(count),
        int(limit),
        bool(respect_limit),
    )
    return (
        int(new_count),
        int(dedupe_hits),
        bool(saturated),
        int(added),
        float(observed),
    )


def residual_scope_dynamic_tile_stride(
    base_stride: int,
    active_idx: np.ndarray,
    block_idx_arr: np.ndarray,
    scope_counts: np.ndarray,
    limit_value: int,
    budget_enabled: bool,
    budget_applied: np.ndarray,
    budget_limits: np.ndarray,
) -> int:
    """Return the stride to use for the next kernel tile."""

    return int(
        _dynamic_tile_stride_impl(
            int(base_stride),
            active_idx,
            block_idx_arr,
            scope_counts,
            int(limit_value),
            bool(budget_enabled),
            budget_applied,
            budget_limits,
        )
    )


def residual_scope_update_budget_state(
    qi: int,
    chunk_points: np.ndarray,
    scan_cap_value: int,
    budget_applied: np.ndarray,
    budget_up: float,
    budget_down: float,
    budget_schedule: np.ndarray,
    budget_indices: np.ndarray,
    budget_limits: np.ndarray,
    budget_final_limits: np.ndarray,
    budget_escalations: np.ndarray,
    budget_low_streak: np.ndarray,
    budget_survivors: np.ndarray,
    budget_early_flags: np.ndarray,
    saturated: np.ndarray,
    saturated_flags: np.ndarray,
) -> None:
    """Update budget escalation/early-stop state for a single query."""

    _update_budget_state_impl(
        int(qi),
        chunk_points,
        int(scan_cap_value),
        budget_applied,
        float(budget_up),
        float(budget_down),
        budget_schedule,
        budget_indices,
        budget_limits,
        budget_final_limits,
        budget_escalations,
        budget_low_streak,
        budget_survivors,
        budget_early_flags,
        saturated,
        saturated_flags,
    )


if NUMBA_RESIDUAL_SCOPE_AVAILABLE:

    @njit(cache=True)
    def _collect_next_chain_impl(
        next_cache: np.ndarray,
        start: int,
        visited: np.ndarray,
        buffer: np.ndarray,
    ) -> int:
        count = 0
        total = next_cache.shape[0]
        current = int(start)
        while 0 <= current < total:
            if visited[current] != 0:
                break
            visited[current] = 1
            buffer[count] = current
            count += 1
            nxt = int(next_cache[current])
            if nxt < 0:
                break
            current = nxt
        for idx in range(count):
            node = int(buffer[idx])
            if 0 <= node < total:
                visited[node] = 0
        return count


def residual_collect_next_chain(
    next_cache: np.ndarray,
    start: int,
    visited: np.ndarray,
    buffer: np.ndarray,
) -> int:
    """Collect parent→child chains using a reusable scratch buffer."""

    if start < 0 or start >= next_cache.shape[0]:
        return 0

    if NUMBA_RESIDUAL_SCOPE_AVAILABLE:
        return int(_collect_next_chain_impl(next_cache, int(start), visited, buffer))

    total = next_cache.shape[0]
    count = 0
    current = int(start)
    while 0 <= current < total:
        if visited[current] != 0:
            break
        visited[current] = 1
        buffer[count] = current
        count += 1
        nxt = int(next_cache[current])
        if nxt < 0:
            break
        current = nxt
    for idx in range(count):
        node = int(buffer[idx])
        if 0 <= node < total:
            visited[node] = 0
    return count


__all__ = [
    "NUMBA_RESIDUAL_SCOPE_AVAILABLE",
    "residual_scope_append",
    "residual_scope_append_bitset",
    "residual_scope_reset",
    "residual_scope_append_masked",
    "residual_scope_append_masked_bitset",
    "residual_scope_dynamic_tile_stride",
    "residual_scope_update_budget_state",
    "residual_collect_next_chain",
]
