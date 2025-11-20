from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    prange = None
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _compute_dist_sq(
        x: np.ndarray,
        y: np.ndarray,
        variance: float,
        lengthscale: float,
    ) -> float:
        # RBF Kernel: k(x,y) = var * exp(-0.5 * |x-y|^2 / l^2)
        # Residual dist d(x,y) = sqrt(1 - rho)
        # rho = K(x,y) / sqrt(K(x,x)*K(y,y))
        # For stationary RBF, K(x,x) = variance.
        # So rho = exp(...)
        # d(x,y) = sqrt(1 - exp(-0.5 * |x-y|^2 / l^2))
        
        d2 = 0.0
        for i in range(x.shape[0]):
            diff = x[i] - y[i]
            d2 += diff * diff
            
        scaled = -0.5 * d2 / (lengthscale * lengthscale)
        rho = math.exp(scaled)
        # Clip rho for numerical stability
        if rho > 1.0: rho = 1.0
        elif rho < -1.0: rho = -1.0
        
        return math.sqrt(1.0 - rho)

    @njit(cache=True, fastmath=True, parallel=True)
    def find_parents_numba(
        points: np.ndarray,
        queries: np.ndarray,
        children: np.ndarray,
        next_nodes: np.ndarray,
        si_cache: np.ndarray,
        variance: float,
        lengthscale: float,
        root_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_queries = queries.shape[0]
        # Arrays to store results
        best_idx = np.full(num_queries, -1, dtype=np.int64)
        best_dist = np.full(num_queries, np.inf, dtype=np.float64)
        
        # Stack for traversal (max depth 64 is usually enough for cover trees, 
        # but let's be safe with a larger fixed buffer or dynamic if needed.
        # Numba requires fixed size arrays or lists. 
        # We can use a list for the stack.
        
        for qi in prange(num_queries):
            q_point = queries[qi]
            
            # Stack: node_idx
            stack = [root_idx]
            
            while len(stack) > 0:
                node = stack.pop()
                if node < 0: continue
                
                # Compute distance
                dist = _compute_dist_sq(q_point, points[node], variance, lengthscale)
                
                # Update best
                if dist < best_dist[qi]:
                    best_dist[qi] = dist
                    best_idx[qi] = node
                    
                # Pruning
                # Lower bound to any descendant = max(0, dist - max_descendant_dist)
                # si_cache stores the covering radius (max dist to any descendant)
                lb = dist - si_cache[node]
                if lb >= best_dist[qi]:
                    continue
                    
                # Push children
                child = children[node]
                while child != -1:
                    stack.append(child)
                    child = next_nodes[child]
                    
        return best_idx, best_dist
else:
    def find_parents_numba(*args, **kwargs):
        raise NotImplementedError("Numba is not installed")