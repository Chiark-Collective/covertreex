from __future__ import annotations

import numpy as np
from numba import njit, objmode
from typing import Tuple

# ... (Heap Utils unchanged) ...

@njit(fastmath=True)
def _push_min_heap(
    heap_keys: np.ndarray,
    heap_vals: np.ndarray,
    heap_extras: np.ndarray,
    size: int,
    key: float,
    val: int,
    extra: int
) -> int:
    i = size
    size += 1
    while i > 0:
        p = (i - 1) >> 1
        if heap_keys[p] <= key:
            break
        heap_keys[i] = heap_keys[p]
        heap_vals[i] = heap_vals[p]
        heap_extras[i] = heap_extras[p]
        i = p
    heap_keys[i] = key
    heap_vals[i] = val
    heap_extras[i] = extra
    return size

@njit(fastmath=True)
def _pop_min_heap(
    heap_keys: np.ndarray,
    heap_vals: np.ndarray,
    heap_extras: np.ndarray,
    size: int,
) -> Tuple[float, int, int, int]:
    if size <= 0:
        return 0.0, -1, -1, 0
    ret_key = heap_keys[0]
    ret_val = heap_vals[0]
    ret_extra = heap_extras[0]
    size -= 1
    last_key = heap_keys[size]
    last_val = heap_vals[size]
    last_extra = heap_extras[size]
    i = 0
    while (i << 1) + 1 < size:
        child = (i << 1) + 1
        if child + 1 < size and heap_keys[child + 1] < heap_keys[child]:
            child += 1
        if last_key <= heap_keys[child]:
            break
        heap_keys[i] = heap_keys[child]
        heap_vals[i] = heap_vals[child]
        heap_extras[i] = heap_extras[child]
        i = child
    heap_keys[i] = last_key
    heap_vals[i] = last_val
    heap_extras[i] = last_extra
    return ret_key, ret_val, ret_extra, size

@njit(fastmath=True)
def _update_knn_sorted(
    keys: np.ndarray,
    indices: np.ndarray,
    k: int,
    current_size: int,
    new_key: float,
    new_idx: int,
) -> int:
    if current_size < k:
        pos = current_size
        while pos > 0 and keys[pos - 1] > new_key:
            keys[pos] = keys[pos - 1]
            indices[pos] = indices[pos - 1]
            pos -= 1
        keys[pos] = new_key
        indices[pos] = new_idx
        return current_size + 1
    if new_key >= keys[k - 1]:
        return k
    pos = k - 1
    while pos > 0 and keys[pos - 1] > new_key:
        keys[pos] = keys[pos - 1]
        indices[pos] = indices[pos - 1]
        pos -= 1
    keys[pos] = new_key
    indices[pos] = new_idx
    return k

@njit(fastmath=True)
def _get_children(
    node_idx: int,
    children_arr: np.ndarray,
    next_arr: np.ndarray,
    out_buffer: np.ndarray
) -> int:
    count = 0
    if node_idx < 0 or node_idx >= children_arr.shape[0]:
        return 0
    child = children_arr[node_idx]
    while child >= 0:
        out_buffer[count] = child
        count += 1
        child = next_arr[child]
        if child == children_arr[node_idx]:
            break
    return count

@njit(fastmath=True)
def _compute_residual_dist_rbf_batch(
    q_idx: int,
    candidates: np.ndarray,
    count: int,
    v_matrix: np.ndarray,
    p_diag: np.ndarray,
    v_norm_sq: np.ndarray,
    coords: np.ndarray,
    var: float,
    ls_sq_arr: np.ndarray,
    out_dists: np.ndarray
) -> None:
    """Compute residual distance for RBF kernel (Scalar or ARD) fully in Numba."""
    vq = v_matrix[q_idx]
    pq = p_diag[q_idx]
    nq = v_norm_sq[q_idx]
    xq = coords[q_idx]
    
    # Pre-fetch dimension to avoid bound checks in loop
    dim = xq.shape[0]
    
    for i in range(count):
        c_idx = candidates[i]
        xc = coords[c_idx]
        d2 = 0.0
        for d in range(dim):
            diff = xq[d] - xc[d]
            # ARD Logic: Scale by 1/ls^2 for each dimension.
            # We expect ls_sq_arr to be shape (dim,)
            d2 += (diff * diff) / ls_sq_arr[d]
            
        k_val = var * np.exp(-0.5 * d2)
        
        vc = v_matrix[c_idx]
        pc = p_diag[c_idx]
        dot = 0.0
        for r in range(vq.shape[0]):
            dot += vq[r] * vc[r]
            
        denom = np.sqrt(pq * pc)
        if denom < 1e-9:
            out_dists[i] = 1.0
        else:
            rho = (k_val - dot) / denom
            if rho > 1.0: rho = 1.0
            if rho < -1.0: rho = -1.0
            out_dists[i] = np.sqrt(1.0 - abs(rho))

@njit(fastmath=True)
def residual_knn_search_numba(
    children: np.ndarray,
    next_node: np.ndarray,
    parents: np.ndarray,
    si_cache: np.ndarray,
    node_to_dataset: np.ndarray,
    v_matrix: np.ndarray,
    p_diag: np.ndarray,
    v_norm_sq: np.ndarray,
    coords: np.ndarray,
    var: float,
    lengthscales_sq: np.ndarray,
    q_dataset_idx: int,
    k: int,
    radius_floor: float,
    root_indices: np.ndarray,
    heap_keys: np.ndarray,
    heap_vals: np.ndarray,
    heap_extras: np.ndarray,
    knn_keys: np.ndarray,
    knn_indices: np.ndarray,
    visited_bitset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    
    heap_size = 0
    # Push all roots
    for r in range(root_indices.shape[0]):
        root = root_indices[r]
        heap_size = _push_min_heap(heap_keys, heap_vals, heap_extras, heap_size, 0.0, root, 0)
        
    knn_size = 0
    knn_keys[:] = 1e30
    knn_indices[:] = -1
    visited_bitset[:] = 0
    
    child_buf = np.empty(1024, dtype=np.int64)
    batch_nodes = np.empty(32, dtype=np.int64)
    batch_dists = np.empty(32, dtype=np.float64)
    dataset_indices = np.empty(32, dtype=np.int64)
    
    # ls_sq is passed as array
    
    while heap_size > 0:
        batch_count = 0
        while heap_size > 0 and batch_count < 32:
            prio, node_idx, _, heap_size = _pop_min_heap(heap_keys, heap_vals, heap_extras, heap_size)
            word_idx = node_idx >> 6
            bit_idx = node_idx & 63
            if not (visited_bitset[word_idx] & (1 << bit_idx)):
                visited_bitset[word_idx] |= (1 << bit_idx)
                batch_nodes[batch_count] = node_idx
                batch_count += 1
        
        if batch_count == 0:
            break
            
        for i in range(batch_count):
            dataset_indices[i] = node_to_dataset[batch_nodes[i]]
            
        _compute_residual_dist_rbf_batch(
            q_dataset_idx,
            dataset_indices,
            batch_count,
            v_matrix, p_diag, v_norm_sq,
            coords, var, lengthscales_sq,
            batch_dists
        )
        
        for i in range(batch_count):
            dist = batch_dists[i]
            node_idx = batch_nodes[i]
            knn_size = _update_knn_sorted(knn_keys, knn_indices, k, knn_size, dist, node_idx)
            n_children = _get_children(node_idx, children, next_node, child_buf)
            # Lower bound pruning (triangle inequality heuristic)
            parent_radius = 0.0
            if node_idx < si_cache.shape[0]:
                parent_radius = si_cache[node_idx]
            if parent_radius < radius_floor:
                parent_radius = radius_floor
            lb = dist - parent_radius
            kth = knn_keys[k - 1] if knn_size == k else 1e30
            for c in range(n_children):
                child_idx = child_buf[c]
                word_idx = child_idx >> 6
                bit_idx = child_idx & 63
                if visited_bitset[word_idx] & (1 << bit_idx):
                    continue
                if lb > kth:
                    # Cannot beat current kth; skip enqueuing this branch.
                    continue
                heap_size = _push_min_heap(heap_keys, heap_vals, heap_extras, heap_size, dist, child_idx, 0)

    return knn_indices[:k], knn_keys[:k]
