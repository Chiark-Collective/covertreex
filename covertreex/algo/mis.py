from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

from covertreex import config as cx_config
from covertreex.algo.conflict import ConflictGraph
from covertreex.core.tree import TreeBackend
from covertreex.algo._mis_numba import NUMBA_AVAILABLE, run_mis_numba

DEGREE_THRESHOLD = 8
_HASH_MULTIPLIER = np.uint32(0x9E3779B9)
_HASH_SHIFT_1 = np.uint32(13)
_HASH_SHIFT_2 = np.uint32(17)
_HASH_SHIFT_3 = np.uint32(5)


def _resolve_runtime_config(runtime: cx_config.RuntimeConfig | None) -> cx_config.RuntimeConfig:
    if runtime is not None:
        return runtime
    active = cx_config.current_runtime_context()
    if active is not None:
        return active.config
    return cx_config.RuntimeConfig.from_env()


def _block_until_ready(value: Any) -> None:
    """Synchronise on backends that execute asynchronously (e.g. JAX)."""

    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


@dataclass(frozen=True)
class MISResult:
    independent_set: Any
    iterations: int


def _repeat_nodes(indptr: jnp.ndarray, dtype) -> jnp.ndarray:
    degrees = indptr[1:] - indptr[:-1]
    return jnp.repeat(jnp.arange(degrees.shape[0], dtype=dtype), degrees)


def batch_mis_seeds(
    count: int,
    *,
    seed: int | None = None,
    runtime: cx_config.RuntimeConfig | None = None,
) -> Tuple[int, ...]:
    """Return a deterministic batch of MIS seeds derived from `seed` or runtime config."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return tuple()

    runtime_seed = seed
    if runtime_seed is None:
        cfg = _resolve_runtime_config(runtime)
        runtime_seed = cfg.seeds.resolved("mis")

    seed_sequence = np.random.SeedSequence(runtime_seed)
    spawned = seed_sequence.spawn(count)
    return tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in spawned)


def run_mis(
    backend: TreeBackend,
    graph: ConflictGraph,
    *,
    seed: int | None = None,
    runtime: cx_config.RuntimeConfig | None = None,
) -> MISResult:
    """Luby-style MIS using JAX primitives."""

    num_nodes = graph.num_nodes
    if num_nodes == 0:
        empty = backend.asarray([], dtype=backend.default_int)
        empty = backend.device_put(empty)
        _block_until_ready(empty)
        return MISResult(independent_set=empty, iterations=0)

    cfg = _resolve_runtime_config(runtime)
    runtime_seed = seed
    if runtime_seed is None:
        runtime_seed = cfg.seeds.resolved("mis")

    if cfg.enable_numba and NUMBA_AVAILABLE:
        indptr_np = np.asarray(
            backend.to_numpy(graph.indptr), dtype=np.int64
        )
        indices_np = np.asarray(
            backend.to_numpy(graph.indices), dtype=np.int64
        )
        indicator_np, iterations = run_mis_numba(indptr_np, indices_np, int(runtime_seed))
        indicator = backend.asarray(indicator_np, dtype=backend.default_int)
        indicator = backend.device_put(indicator)
        _block_until_ready(indicator)
        return MISResult(
            independent_set=indicator,
            iterations=int(iterations),
        )

    indptr = graph.indptr
    degrees = indptr[1:] - indptr[:-1]
    degrees_np = np.asarray(backend.to_numpy(degrees), dtype=np.int64)
    max_degree = int(degrees_np.max()) if degrees_np.size else 0

    if max_degree <= DEGREE_THRESHOLD:
        indptr_np = np.asarray(backend.to_numpy(graph.indptr), dtype=np.int64)
        indices_np = np.asarray(backend.to_numpy(graph.indices), dtype=np.int64)
        active = np.ones(num_nodes, dtype=np.uint8)
        selected_mask = np.zeros(num_nodes, dtype=np.uint8)
        for node in range(num_nodes):
            if not active[node]:
                continue
            selected_mask[node] = 1
            start = indptr_np[node]
            end = indptr_np[node + 1]
            neighbors = indices_np[start:end]
            for neighbor in neighbors:
                active[neighbor] = 0
            active[node] = 0
        indicator_np = selected_mask.astype(np.int8)
        indicator = backend.asarray(indicator_np, dtype=backend.default_int)
        indicator = backend.device_put(indicator)
        _block_until_ready(indicator)
        return MISResult(independent_set=indicator, iterations=1)

    indptr = graph.indptr
    indices = graph.indices
    dtype_int = backend.default_int
    dtype_float = backend.default_float

    sources = _repeat_nodes(indptr, dtype_int)

    nodes_uint32 = jnp.arange(num_nodes, dtype=jnp.uint32)
    seed_u32 = jnp.uint32(runtime_seed & 0xFFFFFFFF)
    hash_multiplier = jnp.uint32(_HASH_MULTIPLIER)
    shift1 = jnp.uint32(_HASH_SHIFT_1)
    shift2 = jnp.uint32(_HASH_SHIFT_2)
    shift3 = jnp.uint32(_HASH_SHIFT_3)
    denom = jnp.array(2**32, dtype=dtype_float)
    one = jnp.array(1.0, dtype=dtype_float)

    def cond_fn(state):
        active, _, _ = state
        return jnp.any(active)

    def body_fn(state):
        active, selected, iterations = state
        next_iterations = iterations + jnp.int32(1)
        iter_u32 = jnp.uint32(next_iterations)
        x = nodes_uint32 ^ seed_u32
        x = x ^ (iter_u32 * hash_multiplier)
        x = x ^ jnp.left_shift(x, shift1)
        x = x ^ jnp.right_shift(x, shift2)
        x = x ^ jnp.left_shift(x, shift3)
        priorities = x.astype(dtype_float)
        priorities = (priorities + one) / denom

        neighbor_priorities = priorities[indices]
        neighbor_active = active[indices]
        neg_inf = jnp.full_like(neighbor_priorities, -jnp.inf)
        neighbor_priorities = jnp.where(neighbor_active, neighbor_priorities, neg_inf)

        max_neighbor = jnp.full((num_nodes,), -jnp.inf, dtype=dtype_float)
        max_neighbor = max_neighbor.at[sources].max(neighbor_priorities)

        winner_mask = jnp.logical_and(active, priorities >= max_neighbor)
        no_winner = jnp.logical_not(jnp.any(winner_mask))
        candidate_indices = jnp.where(active, jnp.arange(num_nodes, dtype=dtype_int), jnp.full((num_nodes,), num_nodes, dtype=dtype_int))
        fallback_idx = jnp.min(candidate_indices)
        fallback_mask = jnp.arange(num_nodes, dtype=dtype_int) == fallback_idx
        winner_mask = jnp.logical_or(
            winner_mask, jnp.logical_and(no_winner, fallback_mask)
        )

        selected = jnp.logical_or(selected, winner_mask)

        winner_sources = winner_mask[sources]
        dominated = jnp.zeros_like(active)
        dominated = dominated.at[indices].max(winner_sources)

        active = jnp.logical_and(active, jnp.logical_not(winner_mask))
        active = jnp.logical_and(active, jnp.logical_not(dominated))

        return active, selected, next_iterations

    init_state = (
        jnp.ones((num_nodes,), dtype=bool),
        jnp.zeros((num_nodes,), dtype=bool),
        jnp.int32(0),
    )
    active_final, selected, iterations = jax.lax.while_loop(cond_fn, body_fn, init_state)

    indicator = selected.astype(dtype_int)
    indicator = backend.device_put(indicator)
    _block_until_ready(indicator)
    iterations_int = int(iterations)

    return MISResult(independent_set=indicator, iterations=iterations_int)
