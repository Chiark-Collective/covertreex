from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba as nb

    NUMBA_GRID_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nb = None  # type: ignore
    NUMBA_GRID_AVAILABLE = False

_UINT64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
_INV_TWO64 = np.float64(5.421010862427522e-20)


if NUMBA_GRID_AVAILABLE:  # pragma: no branch - compiled path

    @nb.njit(cache=True, parallel=True)
    def _prepare_level_arrays(levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = levels.shape[0]
        levels_clamped = np.empty(batch_size, dtype=np.int64)
        widths = np.empty(batch_size, dtype=np.float64)
        base_radius = np.empty(batch_size, dtype=np.float64)
        for idx in nb.prange(batch_size):
            level = levels[idx]
            if level < 0:
                level = 0
            levels_clamped[idx] = level
            width_val = math.ldexp(1.0, int(level))
            if width_val < 1e-6:
                width_val = 1e-6
            widths[idx] = width_val
            base_radius[idx] = math.ldexp(1.0, int(level) + 1)
        return levels_clamped, widths, base_radius


    @nb.njit(cache=True, parallel=True)
    def _fill_coords(points: np.ndarray, widths: np.ndarray, shift_vec: np.ndarray, coords: np.ndarray) -> None:
        dim = points.shape[1]
        for point_idx in nb.prange(points.shape[0]):
            inv_width = 1.0 / widths[point_idx]
            for axis in range(dim):
                value = points[point_idx, axis] * inv_width + shift_vec[axis]
                coords[point_idx, axis] = math.floor(value)


    @nb.njit(cache=True, parallel=True)
    def _fill_priorities(levels_clamped: np.ndarray, seed: int, priorities: np.ndarray) -> None:
        shift_seed = np.uint64(seed)
        for point_idx in nb.prange(levels_clamped.shape[0]):
            level_component = np.uint64(levels_clamped[point_idx]) << np.uint64(33)
            base_val = np.uint64(point_idx)
            mix_input = base_val ^ level_component ^ shift_seed
            priorities[point_idx] = _mix_uint64_scalar(mix_input)


    @nb.njit(cache=True)
    def _mix_uint64_scalar(value: np.uint64) -> np.uint64:
        x = np.uint64(value)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return x


    @nb.njit(cache=True)
    def _shift_vector(dim: int, seed: int) -> np.ndarray:
        vec = np.empty(dim, dtype=np.float64)
        base = np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
        for axis in range(dim):
            base = _mix_uint64_scalar(base ^ np.uint64(axis + 1))
            vec[axis] = float(base) * _INV_TWO64 - 0.5
        return vec


    @nb.njit(cache=True)
    def _mark_leaders_for_shift(
        coords: np.ndarray,
        priorities: np.ndarray,
        leader_mask: np.ndarray,
        leader_priorities: np.ndarray,
    ) -> int:
        n_points = coords.shape[0]
        if n_points == 0:
            return 0
        order = np.arange(n_points, dtype=np.int64)
        pri_sorted = np.argsort(priorities[order], kind="mergesort")
        order = order[pri_sorted]
        dims = coords.shape[1]
        for axis in range(dims - 1, -1, -1):
            axis_vals = np.empty(n_points, dtype=np.int64)
            for pos in range(n_points):
                axis_vals[pos] = coords[order[pos], axis]
            idx_sorted = np.argsort(axis_vals, kind="mergesort")
            order = order[idx_sorted]
        unique_cells = 0
        last_coords = np.empty(dims, dtype=np.int64)
        has_last = False
        for pos in range(order.size):
            idx = order[pos]
            if not has_last:
                for axis in range(dims):
                    last_coords[axis] = coords[idx, axis]
                leader_mask[idx] = 1
                if priorities[idx] < leader_priorities[idx]:
                    leader_priorities[idx] = priorities[idx]
                unique_cells += 1
                has_last = True
                continue
            same = True
            for axis in range(dims):
                if coords[idx, axis] != last_coords[axis]:
                    same = False
                    break
            if not same:
                for axis in range(dims):
                    last_coords[axis] = coords[idx, axis]
                leader_mask[idx] = 1
                if priorities[idx] < leader_priorities[idx]:
                    leader_priorities[idx] = priorities[idx]
                unique_cells += 1
        return unique_cells


    @nb.njit(cache=True)
    def _gather_indices(mask: np.ndarray) -> np.ndarray:
        count = 0
        for value in mask:
            if value != 0:
                count += 1
        indices = np.empty(count, dtype=np.int64)
        out_idx = 0
        for idx, value in enumerate(mask):
            if value != 0:
                indices[out_idx] = idx
                out_idx += 1
        return indices


    @nb.njit(cache=True)
    def _compute_leader_priorities(
        leader_indices: np.ndarray,
        leader_priorities: np.ndarray,
    ) -> np.ndarray:
        subset = np.empty(leader_indices.size, dtype=np.uint64)
        for i in range(leader_indices.size):
            subset[i] = leader_priorities[leader_indices[i]]
        return subset


    @nb.njit(cache=True)
    def _micro_mis(
        points: np.ndarray,
        base_radius: np.ndarray,
        leader_indices: np.ndarray,
        leader_priorities_subset: np.ndarray,
    ) -> Tuple[np.ndarray, int, int]:
        num_leaders = leader_indices.size
        if num_leaders == 0:
            return np.empty(0, dtype=np.int64), 0, 0
        order = np.argsort(leader_priorities_subset, kind="mergesort")
        accepted = np.empty(num_leaders, dtype=np.int64)
        accepted_count = 0
        local_edges = 0
        dims = points.shape[1]
        for pos in range(num_leaders):
            candidate = leader_indices[order[pos]]
            keep = True
            candidate_radius = base_radius[candidate]
            for existing_idx in range(accepted_count):
                other = accepted[existing_idx]
                local_edges += 1
                dist_sq = 0.0
                for axis in range(dims):
                    delta = points[candidate, axis] - points[other, axis]
                    dist_sq += delta * delta
                cutoff = candidate_radius
                other_radius = base_radius[other]
                if other_radius < cutoff:
                    cutoff = other_radius
                if dist_sq <= cutoff * cutoff:
                    keep = False
                    break
            if keep:
                accepted[accepted_count] = candidate
                accepted_count += 1
        if accepted_count == 0:
            accepted[0] = leader_indices[order[0]]
            accepted_count = 1
        return accepted, accepted_count, local_edges


    @nb.njit(cache=True)
    def grid_select_leaders_numba(
        points: np.ndarray,
        levels: np.ndarray,
        *,
        seed: int,
        num_shifts: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = points.shape[0]
        forced_selected = np.zeros(batch_size, dtype=np.uint8)
        forced_dominated = np.ones(batch_size, dtype=np.uint8)
        stats = np.zeros(4, dtype=np.int64)
        if batch_size == 0:
            return forced_selected, forced_dominated, stats
        dims = points.shape[1]
        levels_clamped, widths, base_radius = _prepare_level_arrays(levels)
        leader_mask = np.zeros(batch_size, dtype=np.uint8)
        leader_priorities = np.empty(batch_size, dtype=np.uint64)
        for idx in range(batch_size):
            leader_priorities[idx] = _UINT64_MAX
        coords = np.empty((batch_size, dims), dtype=np.int64)
        shift_count = num_shifts if num_shifts > 0 else 1
        total_cells = 0
        for shift in range(shift_count):
            shift_vec = _shift_vector(dims, seed + shift + 1)
            _fill_coords(points, widths, shift_vec, coords)
            priorities = np.empty(batch_size, dtype=np.uint64)
            _fill_priorities(levels_clamped, seed + shift + 1, priorities)
            total_cells += _mark_leaders_for_shift(
                coords,
                priorities,
                leader_mask,
                leader_priorities,
            )
        leader_indices = _gather_indices(leader_mask)
        stats[0] = total_cells
        stats[1] = leader_indices.size
        if leader_indices.size == 0:
            for idx in range(batch_size):
                forced_selected[idx] = 1
                forced_dominated[idx] = 0
            stats[2] = batch_size
            stats[3] = 0
            return forced_selected, forced_dominated, stats
        leader_subset = _compute_leader_priorities(leader_indices, leader_priorities)
        accepted, accepted_count, local_edges = _micro_mis(
            points,
            base_radius,
            leader_indices,
            leader_subset,
        )
        for idx in range(accepted_count):
            selected = accepted[idx]
            forced_selected[selected] = 1
            forced_dominated[selected] = 0
        stats[2] = accepted_count
        stats[3] = local_edges
        return forced_selected, forced_dominated, stats

else:  # pragma: no cover - executed when numba missing

    def grid_select_leaders_numba(
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise RuntimeError("Numba grid builder requested but numba is not available.")


__all__ = ["NUMBA_GRID_AVAILABLE", "grid_select_leaders_numba"]
