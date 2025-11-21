from __future__ import annotations

import heapq
import math
from typing import Any, List, Tuple, Optional

import numpy as np

from covertreex import config as cx_config
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics.residual.core import (
    get_residual_backend,
    decode_indices,
    compute_residual_distances,
    ResidualCorrHostData,
)
from covertreex.queries.utils import to_numpy_array, ChildChainCache
from covertreex.queries._residual_knn_numba import residual_knn_search_numba

def _batch_residual_distances(
    backend: ResidualCorrHostData,
    query_idx_in_dataset: int,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Compute residual distances for a single query against multiple candidates."""
    q_arr = np.array([query_idx_in_dataset], dtype=np.int64)
    # candidate_indices are already dataset indices (if the tree stores dataset indices)
    # BUT check if decoding is needed.
    # The PCCTree points are usually whatever was passed to build.
    # If built with Euclidean, points are usually coordinates or indices.
    # The Residual Backend expects dataset indices.
    # We need to decode.
    
    # Actually, if we are in "Static Tree" mode, the tree was built with Euclidean.
    # If the input to build() was the raw coordinates, then `tree.points` has coordinates.
    # If the input was indices, `tree.points` has indices.
    # `get_residual_backend()` usually implies we have `v_matrix` etc aligned with the DATASET.
    # So we need to map `tree.points[node_idx]` to a dataset index.
    # If `tree.points` are coordinates, we CANNOT map back easily unless we have a map.
    # Usually for VIF, we pass INDICES to the tree builder if we want to use them later?
    # Or we pass coordinates and the tree stores them.
    
    # `covertreex.algo.traverse.strategies.residual._ResidualTraversal` does:
    # query_indices = decode_indices(host_backend, queries_np)
    # tree_indices = decode_indices(host_backend, tree_points_np)
    
    # So `decode_indices` handles it.
    return compute_residual_distances(backend, q_arr, candidate_indices).flatten()


def residual_knn_query(
    tree: PCCTree,
    query_points: Any,
    *,
    k: int,
    return_distances: bool = False,
    backend: TreeBackend | None = None,
    context: cx_config.RuntimeContext | None = None,
) -> Any:
    """
    Perform k-NN search using the Static Euclidean Tree structure but Dynamic Residual Metric.
    
    This strategy assumes:
    1. The tree was built with Euclidean metric (so children/radii are Euclidean).
    2. The Residual Backend is configured and active.
    3. There exists a correlation between Euclidean proximity and Residual correlation
       that allows us to use the tree for ordering (and potentially pruning).
    """
    backend = backend or tree.backend
    context = context or cx_config.runtime_context()
    
    host_backend = get_residual_backend()
    
    # Decode queries to dataset indices
    # Note: This implies queries must be present in the dataset or we must be able to
    # compute residual distance to new points.
    # `compute_residual_distances` usually expects indices into the cached V-matrix.
    # If query_points are coordinates, we need `compute_residual_distances` to support coords?
    # Currently `ResidualCorrHostData` is index-based.
    # So `query_points` MUST be indices or decodable to indices.
    
    batch = backend.asarray(query_points, dtype=backend.default_float)
    if batch.ndim == 1:
        batch = batch[None, :]
    
    queries_np = to_numpy_array(backend, batch, dtype=np.float64)
    # Attempt to decode queries. If they are floats but represent indices, decode_indices handles it.
    query_indices = decode_indices(host_backend, queries_np)
    
    points_np = to_numpy_array(backend, tree.points, dtype=np.float64)
    try:
        tree_indices = decode_indices(host_backend, points_np)
    except ValueError:
        # Fallback for Static Tree mode:
        # If the tree was built on Coordinates (floats), decode_indices fails.
        # If the tree size matches the backend dataset size, assume 1:1 mapping (Identity).
        if (
            context.config.residual_use_static_euclidean_tree
            and points_np.shape[0] == host_backend.num_points
        ):
            tree_indices = np.arange(host_backend.num_points, dtype=np.int64)
        else:
            raise
    
    parents_np = to_numpy_array(backend, tree.parents, dtype=np.int64)
    children_np = to_numpy_array(backend, tree.children, dtype=np.int64)
    next_cache_np = to_numpy_array(backend, tree.next_cache, dtype=np.int64)
    si_cache_np = to_numpy_array(backend, tree.si_cache, dtype=np.float64)
    
    child_cache = ChildChainCache(children_np, next_cache_np)
    
    root_candidates = np.where(parents_np < 0)[0]
    if root_candidates.size == 0:
        root_candidates = np.asarray([0], dtype=np.int64)
        
    results_indices = []
    results_distances = []
    
    # For each query, perform search
    
    # Check for Numba RBF Fast Path
    # We look for 'rbf_variance' and 'rbf_lengthscale' on the host backend
    # This is a convention for the VIF application to enable the fast path.
    rbf_var = getattr(host_backend, "rbf_variance", None)
    rbf_ls = getattr(host_backend, "rbf_lengthscale", None)
    kernel_coords = host_backend.kernel_points_f32
    
    use_numba = (
        context.config.enable_numba 
        and rbf_var is not None 
        and rbf_ls is not None 
        and kernel_coords is not None
    )
    
    if use_numba:
        # Pre-allocate Numba Scratchpads
        # Max heap size: Number of nodes in tree? Or dynamic?
        # Our heap implementation uses fixed size array?
        # _push_min_heap takes an array. We need it large enough.
        # For a Cover Tree, the queue size can be large but bounded by O(N).
        # Let's allocate reasonable scratchpad.
        max_heap_size = 1024 * 64 # 64k nodes queue?
        heap_keys = np.empty(max_heap_size, dtype=np.float64)
        heap_vals = np.empty(max_heap_size, dtype=np.int64)
        heap_extras = np.empty(max_heap_size, dtype=np.int64)
        
        knn_keys = np.empty(k, dtype=np.float64)
        knn_indices = np.empty(k, dtype=np.int64)
        
        # Bitset for visited
        n_nodes = parents_np.shape[0]
        n_words = (n_nodes + 63) // 64
        visited_bitset = np.empty(n_words, dtype=np.int64)
        
        # Map Node Index -> Dataset Index (Identity or Mapping)
        # tree_indices is the mapping.
        node_to_dataset = tree_indices
        
        # Prepare Residual Arrays (ensure float32/64 compat)
        # The Numba kernel expects float64?
        # _compute_residual_dist_rbf_batch uses `v_matrix[q_idx]` which is float32.
        # It promotes to float64 in calc.
        
        v_matrix = host_backend.v_matrix
        p_diag = host_backend.p_diag
        v_norm_sq = host_backend.v_norm_sq
        
        # Handle Lengthscales (Scalar or ARD)
        dim = kernel_coords.shape[1]
        if np.ndim(rbf_ls) == 0:
            ls_scalar = float(rbf_ls)
            ls_sq_arr = np.full(dim, ls_scalar**2, dtype=np.float64)
        else:
            ls_arr = np.asarray(rbf_ls, dtype=np.float64)
            # Flatten to ensure 1D check
            if ls_arr.ndim != 1: ls_arr = ls_arr.flatten()
            
            if ls_arr.size == 1:
                # Broadcast
                ls_sq_arr = np.full(dim, ls_arr.item()**2, dtype=np.float64)
            elif ls_arr.size == dim:
                # ARD
                ls_sq_arr = ls_arr**2
            else:
                raise ValueError(f"RBF lengthscale array size {ls_arr.size} does not match dimension {dim}.")
        
        for q_idx, q_dataset_idx in enumerate(query_indices):
            # Numba Call
            indices, dists = residual_knn_search_numba(
                children_np, next_cache_np, parents_np,
                node_to_dataset, v_matrix, p_diag, v_norm_sq,
                kernel_coords, float(rbf_var), ls_sq_arr,
                int(q_dataset_idx), int(k),
                root_candidates,
                heap_keys, heap_vals, heap_extras,
                knn_keys, knn_indices, visited_bitset
            )
            results_indices.append(indices.copy())
            
    else:
        # Python Fallback
        for q_idx, q_dataset_idx in enumerate(query_indices):
            indices, dists = _single_query_residual_knn(
                q_dataset_idx,
                tree_indices=tree_indices,
                si_cache=si_cache_np,
                child_cache=child_cache,
                root_indices=root_candidates,
                k=k,
                host_backend=host_backend,
            )
            results_indices.append(indices)
            results_distances.append(dists)
        
    indices_arr = np.stack(results_indices, axis=0)
    distances_arr = np.stack(results_distances, axis=0)
    
    sorted_indices = backend.asarray(indices_arr, dtype=backend.default_int)
    sorted_distances = backend.asarray(distances_arr, dtype=backend.default_float)
    
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


def _single_query_residual_knn(
    q_dataset_idx: int,
    *,
    tree_indices: np.ndarray,
    si_cache: np.ndarray,
    child_cache: ChildChainCache,
    root_indices: Sequence[int],
    k: int,
    host_backend: ResidualCorrHostData,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-query k-NN using Best-First Search on Euclidean tree with Residual Metric.
    
    Note: Without a rigorous Euc->Res bound, we cannot safely prune branches based on
    Euclidean radius. We rely on the heuristic that visiting nodes in Euclidean order
    is efficient. We basically perform a global search ordered by Euclidean logic,
    but we can stop if we exhaust the queue?
    No, if we don't prune, we visit everyone.
    
    To make this "Opt-In" viable, we MUST assume some pruning capability or use
    a very generous heuristic.
    
    For this implementation, we will use a **Loose Pruning** strategy:
    If Euclidean Lower Bound > X, we assume Residual Lower Bound > Y.
    Currently, since we don't have the function, we will NOT prune, but we will
    use the Euclidean structure to prioritize the search.
    To prevent O(N), we might implement a max_visited cap or beam width?
    
    Actually, let's implement **Batched Evaluation** of children to speed it up.
    """
    
    # Max-heap for k-NN (stores -distance, index)
    # We want to keep the k SMALLEST distances.
    best_heap: List[Tuple[float, int]] = []
    
    # Min-heap for candidates (stores lower_bound, counter, node_idx, euc_dist)
    # Ordered by Euclidean lower bound? Or Estimated Residual Lower Bound?
    # Since we only have Euclidean structure, we order by Euclidean properties.
    candidate_heap: List[Tuple[float, int, int]] = [] # (priority, counter, node_idx)
    
    counter = 0
    visited = set()
    
    # 1. Initialize with roots
    # We need to compute Residual Distances to roots to start?
    # Or just Euclidean priority?
    # Roots cover the whole space.
    for root in root_indices:
        heapq.heappush(candidate_heap, (0.0, counter, int(root)))
        counter += 1
        
    # We need to track "Exact Residual Distance" for visited nodes to update best_heap.
    
    # Optimization: Evaluate points in batches.
    # While queue is not empty:
    #   Pop a batch of nodes (e.g. 32) from candidate_heap.
    #   Compute Residual Distances for them.
    #   Update best_heap.
    #   Expand their children.
    #   Compute Euclidean Priorities for children (approximate).
    #   Push children to candidate_heap.
    
    BATCH_SIZE = 32
    
    while candidate_heap:
        batch_nodes = []
        while candidate_heap and len(batch_nodes) < BATCH_SIZE:
            _, _, node_idx = heapq.heappop(candidate_heap)
            if node_idx not in visited:
                visited.add(node_idx)
                batch_nodes.append(node_idx)
        
        if not batch_nodes:
            break
            
        # Compute Residual Distances for the batch
        batch_indices = np.array(batch_nodes, dtype=np.int64)
        # Map to dataset indices
        batch_dataset_indices = tree_indices[batch_indices]
        
        # Compute actual residual distances
        dists = _batch_residual_distances(host_backend, q_dataset_idx, batch_dataset_indices)
        
        # Update k-NN heap
        for i, node_idx in enumerate(batch_nodes):
            d = float(dists[i])
            
            # Add to best_heap
            if len(best_heap) < k:
                heapq.heappush(best_heap, (-d, node_idx))
            else:
                worst_dist = -best_heap[0][0]
                if d < worst_dist:
                    heapq.heapreplace(best_heap, (-d, node_idx))
            
            # Expand children
            children = child_cache.get(node_idx)
            if children.size > 0:
                # Here we should compute priorities for children.
                # Since we don't want to compute Exact Residual for all children immediately,
                # we need a heuristic.
                # Heuristic: Parent's Residual Distance?
                # Or assume spatial locality: closer in tree structure -> check first.
                # Cover Tree invariant: children are "close" to parent.
                # So we push children with the priority = Parent's Residual Distance (minus something?)
                # Or just use the Parent's Residual Distance as the priority.
                # This assumes triangular inequality: d(q, child) >= d(q, parent) - d(parent, child)
                # d(parent, child) <= Radius(parent).
                # So LowerBound(child) = max(0, d(q, parent) - Radius(parent)).
                # We know d(q, parent) = d (residual).
                # We DON'T know Radius(parent) in Residual Metric.
                # But we can assume it's small.
                # So priority = d.
                priority = d
                
                # PRUNING OPPORTUNITY (Experimental):
                # If priority (Parent Dist) is HUGE compared to current k-th best,
                # and we trust the triangle inequality holds somewhat, maybe prune?
                # For now, NO PRUNING to ensure correctness/strict opt-in behavior.
                
                for child in children:
                    if child not in visited:
                        heapq.heappush(candidate_heap, (priority, counter, int(child)))
                        counter += 1

    # Extract results
    ordered = sorted((-dist, idx) for dist, idx in best_heap)
    indices = np.asarray([idx for _, idx in ordered], dtype=np.int64)
    distances = np.asarray([dist for dist, _ in ordered], dtype=np.float64)
    return indices, distances
