from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from covertreex import config as cx_config
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.diagnostics import log_operation
from covertreex.logging import get_logger
from covertreex.queries._knn_numba import (
    NUMBA_QUERY_AVAILABLE,
    materialise_tree_view_cached,
    knn_numba as _knn_numba,
)
from covertreex.queries.residual_knn import residual_knn_query
from covertreex.queries.utils import ChildChainCache, to_numpy_array


LOGGER = get_logger("queries.knn")


def _fallback_bruteforce(
    query: np.ndarray, points: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-NN via dense distances as a safety net."""

    diff = points - query[None, :]
    dists = np.linalg.norm(diff, axis=1)
    order = np.argsort(dists)[:k]
    return order.astype(np.int64), dists[order].astype(np.float64)


def _single_query_knn(
    query: np.ndarray,
    *,
    points: np.ndarray,
    si_cache: np.ndarray,
    child_cache: ChildChainCache,
    root_indices: Sequence[int],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a cover-tree walk to collect the k nearest neighbours for one query."""

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("Cannot query an empty tree.")

    visited: set[int] = set()
    enqueued: set[int] = set()
    best_heap: List[Tuple[float, int]] = []  # max-heap of (-distance, index)
    candidate_heap: List[Tuple[float, int, int, float]] = []  # (lower, order, idx, dist)
    counter = 0
    distance_cache: dict[int, float] = {}

    def _push_candidate(idx: int) -> None:
        nonlocal counter
        if idx < 0 or idx >= num_points:
            return
        if idx in visited or idx in enqueued:
            return

        dist = distance_cache.get(idx)
        if dist is None:
            point = points[idx]
            dist = float(np.linalg.norm(query - point))
            distance_cache[idx] = dist
        radius = float(si_cache[idx]) if si_cache.size > idx else 0.0
        if math.isinf(radius):
            lower_bound = 0.0
        else:
            lower_bound = max(dist - radius, 0.0)

        current_bound = -best_heap[0][0] if len(best_heap) >= k else math.inf
        if lower_bound > current_bound:
            return

        heapq.heappush(candidate_heap, (lower_bound, counter, idx, dist))
        enqueued.add(idx)
        counter += 1

    for root_idx in root_indices:
        _push_candidate(int(root_idx))

    while candidate_heap:
        lower_bound, _, node_idx, node_dist = heapq.heappop(candidate_heap)
        enqueued.discard(node_idx)
        if node_idx in visited:
            continue

        current_bound = -best_heap[0][0] if len(best_heap) >= k else math.inf
        if len(best_heap) >= k and lower_bound > current_bound:
            break

        visited.add(node_idx)

        if len(best_heap) < k:
            heapq.heappush(best_heap, (-node_dist, node_idx))
        else:
            worst_dist, worst_idx = best_heap[0]
            worst_dist = -worst_dist
            if node_dist < worst_dist or (
                math.isclose(node_dist, worst_dist)
                and node_idx < worst_idx
            ):
                heapq.heapreplace(best_heap, (-node_dist, node_idx))

        for child in child_cache.get(node_idx):
            _push_candidate(int(child))

    if len(best_heap) < k:
        return _fallback_bruteforce(query, points, k)

    ordered = sorted((-dist, idx) for dist, idx in best_heap)
    indices = np.asarray([idx for _, idx in ordered], dtype=np.int64)
    distances = np.asarray([dist for dist, _ in ordered], dtype=np.float64)
    return indices, distances


def knn(
    tree: PCCTree,
    query_points: Any,
    *,
    k: int,
    return_distances: bool = False,
    backend: TreeBackend | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> Tuple[Any, Any] | Any:
    if "CoverTree" not in globals():
        try:
            from covertreex.engine import CoverTree as _CoverTree  # type: ignore

            globals()["CoverTree"] = _CoverTree
        except ImportError:
            _CoverTree = None
    else:
        _CoverTree = globals().get("CoverTree")
    if _CoverTree is not None and isinstance(tree, _CoverTree):
        return tree.knn(
            query_points,
            k=k,
            return_distances=return_distances,
            context=context,
        )

    backend = backend or tree.backend
    resolved_context = context or cx_config.current_runtime_context()
    if resolved_context is None:
        resolved_context = cx_config.runtime_context()
    with log_operation(LOGGER, "knn_query", context=resolved_context) as op_log:
        return _knn_impl(
            op_log,
            tree,
            query_points,
            k=k,
            return_distances=return_distances,
            backend=backend,
            context=resolved_context,
        )


def _knn_impl(
    op_log: Any,
    tree: PCCTree,
    query_points: Any,
    *,
    k: int,
    return_distances: bool,
    backend: TreeBackend,
    context: cx_config.RuntimeContext | None,
) -> Tuple[Any, Any] | Any:
    if tree.is_empty():
        raise ValueError("Cannot query an empty tree.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > tree.num_points:
        raise ValueError("k cannot exceed the number of points in the tree.")

    context = context or cx_config.runtime_context()
    runtime = context.config
    query_array = np.asarray(query_points)
    use_index_payload = (
        (runtime.metric == "residual_correlation" or runtime.residual_use_static_euclidean_tree)
        and query_array.dtype.kind in {"i", "u"}
    )
    batch_dtype = backend.default_int if use_index_payload else backend.default_float
    batch = backend.asarray(query_points, dtype=batch_dtype)
    if batch.ndim == 1:
        batch = batch.reshape(-1, 1) if use_index_payload else batch[None, :]
    
    if runtime.enable_rust:
        try:
            return _rust_knn_query(
                tree,
                batch,
                k=k,
                return_distances=return_distances,
                backend=backend,
                context=context,
                op_log=op_log,
            )
        except ImportError:
            pass
    
    if runtime.residual_use_static_euclidean_tree:
        return residual_knn_query(
            tree,
            batch,
            k=k,
            return_distances=return_distances,
            backend=backend,
            context=context,
        )

    use_numba = runtime.enable_numba and NUMBA_QUERY_AVAILABLE
    
    batch_np = to_numpy_array(backend, batch, dtype=np.float64)
    num_queries = batch_np.shape[0]

    if use_numba:
        # ... existing numba logic ...
        view = materialise_tree_view_cached(tree)
        numba_indices, numba_distances = _knn_numba(
            view,
            batch_np,
            k=int(k),
            return_distances=True,
        )
        indices_arr = (
            numba_indices if numba_indices.ndim > 1 else numba_indices[None, :]
        )
        distances_arr = (
            numba_distances if numba_distances.ndim > 1 else numba_distances[None, :]
        )
    else:
        # ... existing python logic ...
        points_np = to_numpy_array(backend, tree.points, dtype=np.float64)
        parents_np = to_numpy_array(backend, tree.parents, dtype=np.int64)
        children_np = to_numpy_array(backend, tree.children, dtype=np.int64)
        next_cache_np = to_numpy_array(backend, tree.next_cache, dtype=np.int64)
        si_cache_np = to_numpy_array(backend, tree.si_cache, dtype=np.float64)
        child_cache = ChildChainCache(children_np, next_cache_np)

        root_candidates = np.where(parents_np < 0)[0]
        if root_candidates.size == 0:
            root_candidates = np.asarray([0], dtype=np.int64)

        results_indices: List[np.ndarray] = []
        results_distances: List[np.ndarray] = []

        for query in batch_np:
            indices, distances = _single_query_knn(
                query,
                points=points_np,
                si_cache=si_cache_np,
                child_cache=child_cache,
                root_indices=root_candidates,
                k=int(k),
            )
            results_indices.append(indices)
            results_distances.append(distances)

        indices_arr = np.stack(results_indices, axis=0)
        distances_arr = np.stack(results_distances, axis=0)

    sorted_indices = backend.asarray(indices_arr, dtype=backend.default_int)

    if op_log is not None:
        op_log.add_metadata(
            queries=num_queries,
            k=k,
            return_distances=bool(return_distances),
        )

    if not return_distances:
        if sorted_indices.shape[0] == 1:
            squeezed = sorted_indices[0]
            return squeezed if squeezed.shape[0] > 1 else squeezed[0]
        return sorted_indices

    sorted_distances = backend.asarray(distances_arr, dtype=backend.default_float)
    if sorted_indices.shape[0] == 1:
        squeezed_idx = sorted_indices[0]
        squeezed_dist = sorted_distances[0]
        if squeezed_idx.shape[0] == 1:
            return squeezed_idx[0], squeezed_dist[0]
        return squeezed_idx, squeezed_dist
    return sorted_indices, sorted_distances


def _rust_knn_query(
    tree: PCCTree,
    batch: Any,
    *,
    k: int,
    return_distances: bool,
    backend: TreeBackend,
    context: cx_config.RuntimeContext,
    op_log: Any | None = None,
) -> Any:
    import covertreex_backend
    from covertreex.metrics.residual.core import get_residual_backend, decode_indices
    
    points_np = to_numpy_array(backend, tree.points, dtype=np.float32)
    parents_np = to_numpy_array(backend, tree.parents, dtype=np.int64)
    children_np = to_numpy_array(backend, tree.children, dtype=np.int64)
    next_np = to_numpy_array(backend, tree.next_cache, dtype=np.int64)
    levels_np = to_numpy_array(backend, tree.top_levels, dtype=np.int32)
    
    min_level = int(tree.min_scale) if tree.min_scale is not None else -100
    max_level = int(tree.max_scale) if tree.max_scale is not None else 100
    
    wrapper = covertreex_backend.CoverTreeWrapper(
        points_np, parents_np, children_np, next_np, levels_np, min_level, max_level
    )
    
    queries_np = to_numpy_array(backend, batch, dtype=np.float32)
    
    is_residual = (
        context.config.metric == "residual_correlation" 
        or context.config.residual_use_static_euclidean_tree
    )
    
    if is_residual:
        host_backend = get_residual_backend()
        
        # For residual query, queries_np must be decodable to indices
        query_indices = decode_indices(host_backend, queries_np)
        
        # For residual query on Static Tree, we need node_to_dataset mapping.
        # Try decode tree points
        try:
            tree_indices = decode_indices(host_backend, points_np)
        except ValueError:
            # Fallback: Identity
            if points_np.shape[0] == host_backend.num_points:
                tree_indices = np.arange(host_backend.num_points, dtype=np.int64)
            else:
                raise RuntimeError("Cannot map tree nodes to dataset indices for residual query.")
        
        node_to_dataset = tree_indices.tolist()
        
        v_matrix = host_backend.v_matrix
        p_diag = host_backend.p_diag
        coords = host_backend.kernel_points_f32
        rbf_var = float(getattr(host_backend, "rbf_variance", 1.0))
        rbf_ls = getattr(host_backend, "rbf_lengthscale", 1.0)
        if np.ndim(rbf_ls) == 0:
            dim = coords.shape[1]
            rbf_ls = np.full(dim, float(rbf_ls), dtype=np.float32)
        else:
            rbf_ls = np.asarray(rbf_ls, dtype=np.float32)
            
        indices, dists = wrapper.knn_query_residual(
            query_indices,
            node_to_dataset,
            v_matrix,
            p_diag,
            coords,
            rbf_var,
            rbf_ls,
            k
        )
    else:
        indices, dists = wrapper.knn_query(queries_np, k)

    rust_telemetry = None
    if hasattr(wrapper, "last_query_telemetry"):
        try:
            rust_telemetry = wrapper.last_query_telemetry()
        except Exception:
            rust_telemetry = None
    if op_log is not None and rust_telemetry is not None:
        op_log.add_metadata(rust_query_telemetry=rust_telemetry)
        # clear to avoid accidental reuse
        if hasattr(wrapper, "clear_last_query_telemetry"):
            try:
                wrapper.clear_last_query_telemetry()
            except Exception:
                pass

    sorted_indices = backend.asarray(indices, dtype=backend.default_int)
    sorted_distances = backend.asarray(dists, dtype=backend.default_float)
    
    if not return_distances:
        if sorted_indices.shape[0] == 1:
            squeezed = sorted_indices[0]
            return squeezed if squeezed.shape[0] > 1 else squeezed[0]
        return sorted_indices

    if sorted_indices.shape[0] == 1:
        squeezed_idx = sorted_indices[0]
        squeezed_dist = sorted_distances[0]
        if squeezed_idx.shape[0] == 1:
            return squeezed_idx[0], squeezed_dist[0]
        return squeezed_idx, squeezed_dist
    return sorted_indices, sorted_distances


def nearest_neighbor(
    tree: PCCTree,
    query_points: Any,
    *,
    return_distances: bool = False,
    backend: TreeBackend | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> Tuple[Any, Any] | Any:
    return knn(
        tree,
        query_points,
        k=1,
        return_distances=return_distances,
        backend=backend,
        context=context,
    )
