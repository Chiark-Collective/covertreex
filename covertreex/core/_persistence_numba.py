from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba as nb

    NUMBA_PERSISTENCE_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nb = None  # type: ignore
    NUMBA_PERSISTENCE_AVAILABLE = False

I32 = np.int32


def _require_numba() -> None:
    if not NUMBA_PERSISTENCE_AVAILABLE:  # pragma: no cover - defensive
        raise RuntimeError(
            "Numba persistence helpers requested but `numba` is not available. "
            "Install the '[numba]' extra or disable the feature via "
            "COVERTREEX_ENABLE_NUMBA=0."
        )


if NUMBA_PERSISTENCE_AVAILABLE:

    @nb.njit(cache=True)
    def _apply_journal_cow_impl(
        parents_in: np.ndarray,
        levels_in: np.ndarray,
        children_in: np.ndarray,
        next_in: np.ndarray,
        inserted_parents: np.ndarray,
        inserted_levels: np.ndarray,
        head_parents: np.ndarray,
        head_values: np.ndarray,
        next_nodes: np.ndarray,
        next_values: np.ndarray,
        parents_out: np.ndarray,
        levels_out: np.ndarray,
        children_out: np.ndarray,
        next_out: np.ndarray,
        base_length: int,
    ) -> None:
        total_length = parents_out.shape[0]

        for idx in range(base_length):
            parents_out[idx] = parents_in[idx]
            levels_out[idx] = levels_in[idx]
            children_out[idx] = children_in[idx]
            next_out[idx] = next_in[idx]

        for idx in range(base_length, total_length):
            children_out[idx] = I32(-1)
            next_out[idx] = I32(-1)

        inserted_count = inserted_parents.shape[0]
        for offset in range(inserted_count):
            target = base_length + offset
            parents_out[target] = inserted_parents[offset]
            levels_out[target] = inserted_levels[offset]

        head_updates = head_parents.shape[0]
        for offset in range(head_updates):
            parent = head_parents[offset]
            if 0 <= parent < total_length:
                children_out[parent] = head_values[offset]

        next_updates = next_nodes.shape[0]
        for offset in range(next_updates):
            node = next_nodes[offset]
            if 0 <= node < total_length:
                next_out[node] = next_values[offset]


def apply_journal_cow(
    parents_in: np.ndarray,
    levels_in: np.ndarray,
    children_in: np.ndarray,
    next_in: np.ndarray,
    inserted_parents: np.ndarray,
    inserted_levels: np.ndarray,
    head_parents: np.ndarray,
    head_values: np.ndarray,
    next_nodes: np.ndarray,
    next_values: np.ndarray,
    parents_out: np.ndarray,
    levels_out: np.ndarray,
    children_out: np.ndarray,
    next_out: np.ndarray,
    base_length: int,
) -> None:
    """Copy tree arrays once and realise journal updates using a Numba kernel."""

    _require_numba()
    _apply_journal_cow_impl(
        parents_in,
        levels_in,
        children_in,
        next_in,
        inserted_parents,
        inserted_levels,
        head_parents,
        head_values,
        next_nodes,
        next_values,
        parents_out,
        levels_out,
        children_out,
        next_out,
        base_length,
    )


__all__ = ["NUMBA_PERSISTENCE_AVAILABLE", "apply_journal_cow"]

