***

# VIF Neighbor Selection: Analysis of the Cover Tree Approach

This document analyzes the specialized neighbor selection strategy for Vecchia-Inducing-Points (VIF) approximations, focusing on the use of a **Cover Tree**. The analysis is based on the paper "Vecchia-Inducing-Points Full-Scale Approximations for Gaussian Processes" and its reference implementation in the C++ library GPBoost. The final section outlines a plan for implementing this strategy in the `survi` JAX codebase.

## 1. The "Why": Motivation for a Specialized Neighbor Search

Standard Vecchia approximations and k-Nearest Neighbors (k-NN) algorithms typically rely on Euclidean distance in the input space to find neighbors. The VIF approximation, however, introduces a crucial complexity that makes this naive approach suboptimal and, in some cases, incorrect.

### The VIF Residual Process

The core idea of VIF is to decompose the Gaussian Process `b(s)` into a global low-rank component `b_l(s)` (modeled by inducing points) and a local residual process `b_s(s)`.

`b(s) + ε(s) = b_l(s) + b_s(s)`

The Vecchia approximation is then applied to the *residual process* `b_s(s)`. The challenge is that the covariance of this residual process is no longer stationary and cannot be described by a simple distance-based kernel. Its covariance is:

`Cov(b_s(s_i), b_s(s_j)) = K(s_i, s_j) - k(s_i, u) K(u, u)⁻¹ k(u, s_j)`

Where `u` are the inducing points.

### The Right Metric: Residual Correlation

For a Vecchia approximation to be accurate, the conditioning set (the "neighbors") for a point `s_i` should contain the points that are most strongly correlated with `s_i`. For the VIF residual process, this means we must find neighbors based on the **residual correlation**, not Euclidean distance.

As the paper states in Section 6:
> In our case, though, simply using the Euclidean metric in a transformed space is not applicable as we compute a Vecchia approximation for the residual process with covariance `Σ – Σᵀₘₙ Σₘ⁻¹ Σₘₙ`, which corresponds to a non-stationary covariance function... we present a computationally efficient method for the correlation-based selection of Vecchia neighbors for VIF approximations.

The paper defines the metric `d_c` based on this residual correlation `ρ_c`:

`d_c(s_i, s_j) = sqrt(1 - |ρ_c(s_i, s_j)|)`

where `ρ_c(s_i, s_j) = Cov(b_s(s_i), b_s(s_j)) / sqrt(Var(b_s(s_i)) * Var(b_s(s_j)))`

### Why Standard k-NN Fails

Fast k-NN data structures like k-d trees or ball trees rely on the geometric properties of the metric space (typically L_p norms). They work by pruning large parts of the search space based on geometric bounds.

The residual correlation metric `d_c` is an **arbitrary metric**. It depends on the inducing points and the kernel hyperparameters, and it does not correspond to a simple geometric distance. Therefore, k-d trees cannot be used. A brute-force search, which computes `n(n-1)/2` pairwise distances, is computationally prohibitive, scaling as `O(n²)`.

This is precisely the problem my current `JAXBatchedCorrelationNeighborSelector` faces. While it uses GPU batching, its fundamental complexity is still quadratic in nature, as it must compute correlations between each point and all of its predecessors.

## 2. The "What": Introducing the Cover Tree

To overcome the `O(n²)` complexity of brute-force search with an arbitrary metric, the paper proposes using a **Cover Tree**.

A cover tree is a hierarchical data structure for indexing points in a metric space. Its key advantage is that its construction and query algorithms **only require the metric to satisfy the triangle inequality**, which the correlation distance `d_c` does. It does not need any other geometric properties.

This allows for a provably efficient nearest neighbor search, with query times that are significantly better than brute-force. The paper notes:
> ...we utilize cover trees [Beygelzimer et al., 2006] which... enable m_v-nearest neighbor search for a set of n points with a complexity of `O(C_d · n · log(m_v) · (m_v + log(n)))`

This is a near-linear complexity in `n`, a massive improvement over `O(n²)`.

## 3. The "How": GPBoost's Cover Tree Implementation

The GPBoost C++ code provides a concrete implementation of this strategy. Let's break down the key components in `GPBoost/src/GPBoost/Vecchia_utils.cpp`.

### A. The Metric (`distances_funct`)

This function is the heart of the matter; it implements the VIF residual correlation distance.

```cpp
// GPBoost/src/GPBoost/Vecchia_utils.cpp

void distances_funct(const int& coord_ind_i,
	const std::vector<int>& coords_ind_j,
	/* ... other args ... */
	vec_t& distances,
	/* ... */ ) {

	if (dist_function == "residual_correlation_FSA") {
		// pp_node = k(u, s_j)ᵀ K(u,u)⁻¹ k(u, s_i)
		vec_t pp_node(coords_ind_j.size());
		vec_t chol_ip_cross_cov_sample = chol_ip_cross_cov.col(coord_ind_i);
#pragma omp parallel for schedule(static)
		for (int j = 0; j < pp_node.size(); j++) {
			pp_node[j] = chol_ip_cross_cov.col(coords_ind_j[j]).dot(chol_ip_cross_cov_sample);
		}
		// ...
		// corr_mat contains K(s_i, s_j)
		re_comps_vecchia_cluster_i[0]->CalcSigmaAndSigmaGradVecchia(/*...args...*/);
		
		double corr_diag_sample = corr_diag(coord_ind_i); // Var(b_s(s_i))
#pragma omp parallel for schedule(static)
		for (int j = 0; j < (int)coords_ind_j.size(); j++) {
			// distances[j] = sqrt(1 - | (K(si,sj) - pp_node[j]) / sqrt(Var_s(i)*Var_s(j)) |)
			distances[j] = std::sqrt((1. - std::abs((corr_mat.data()[j] - pp_node[j]) /
				std::sqrt(corr_diag_sample * corr_diag[coords_ind_j[j]]))));
		}
	}
}
```
- **`chol_ip_cross_cov`**: This is `L⁻¹ K(u, s)`, where `L` is the Cholesky factor of `K(u, u)`. The dot product `chol_ip_cross_cov.col(j).dot(chol_ip_cross_cov.col(i))` correctly computes `k(u, s_j)ᵀ K(u,u)⁻¹ k(u, s_i)`.
- **`corr_diag`**: This vector holds the diagonal of the residual covariance matrix, `Var(b_s(s_i)) = K(s_i, s_i) - k(s_i, u) K(u, u)⁻¹ k(u, s_i)`.
- **`corr_mat`**: This holds the prior covariance `K(s_i, s_j)`.
- The final calculation perfectly implements the `d_c` metric from the paper.

### B. Tree Construction (`CoverTree_kNN`)

This function builds the multi-level tree structure.

```cpp
// GPBoost/src/GPBoost/Vecchia_utils.cpp

void CoverTree_kNN(/*...args...*/
	std::map<int, std::vector<int>>& cover_tree,
	/*...args...*/) {
	// ... initialization ...
	int root = start;
	cover_tree.insert({ -1, { root } });
	// ...
	while ((int)(cover_tree.size() - 1) != (int)(coords.rows())) {
		level += 1;
		R_l = R_max / std::pow(base, level);
		std::map<int, std::vector<int>> covert_points;
		// ... loop over parent nodes ...
		for (const auto& key_covert_points_old_i : covert_points_old) {
			// ...
			while (not_all_covered) {
				// Systematic selection of new node from remaining points
				int sample_ind = covert_points_old_i[0]; 
				cover_tree[key + start].push_back(sample_ind + start);
				// ...
                // new covered points per node, using distances_funct
                // ...
				// Remove covered points from the parent's set
				covert_points_old_i.erase(covert_points_old_i.begin());
				// ...
			}
		}
		covert_points_old = covert_points;
	}
}
```
- The data structure is a `std::map<int, std::vector<int>>`, mapping a parent node index to a vector of its children's indices. This is a classic tree representation that is difficult to replicate in JAX.
- The construction is iterative, level by level (`while` loop). The radius `R_l` shrinks at each level.
- **Key Modification**: The paper mentions a modification: "instead of randomly selecting knots from the remaining data, we systematically insert the point with the smallest index into the cover tree." The line `int sample_ind = covert_points_old_i[0];` confirms this. It simply takes the first available point (which, given the ordered nature of the process, is the one with the smallest index) as the center for a new child node.

### C. Neighbor Search (`find_kNN_CoverTree`)

This function uses the constructed tree to find the `k` nearest neighbors for a query point `i`.

```cpp
// GPBoost/src/GPBoost/Vecchia_utils.cpp

void find_kNN_CoverTree(const int i, const int k, /*...args...*/
	std::vector<int>& neighbors_i, /*...args...*/) {
	// ... initialization ...
	std::vector<int> Q; // Candidate set of neighbors
	// ...
	double dist_k_Q_cor = max_dist; // Distance to the current k-th farthest neighbor

	for (int ii = 1; ii < levels; ii++) {
		// ... expand Q by adding children of current candidates ...
		for (int j : diff_rev) { // diff_rev are newly added candidates
			for (int jj : cover_tree[j]) {
				if (jj < i) { // ENFORCE PREDECESSOR CONSTRAINT
					Q.push_back(jj);
					// ...
				} else {
					break; // Assumes children are ordered by index
				}
			}
		}

		// ... update dist_k_Q_cor (distance to k-th best candidate in Q) ...
		dist_k_Q_cor += 1 / std::pow(base, ii - 1); // Crucial bound from cover tree theory

		// PRUNING STEP
		if (dist_k_Q_cor < max_dist) {
			// ... create new Q_interim by keeping only candidates 'c' where d(i, c) <= dist_k_Q_cor ...
		}
		// ...
	}
	// ... Final brute-force search within the refined candidate set Q ...
}
```
- **Top-Down Search**: The search proceeds from the root down through the levels of the tree.
- **Candidate Set `Q`**: It maintains a set `Q` of potential neighbors.
- **Predecessor Constraint**: The line `if (jj < i)` is critical. It ensures that only points that appear earlier in the ordering are considered as neighbors, which is fundamental to the Vecchia approximation's likelihood construction.
- **Pruning**: The core of the efficiency gain comes from pruning. At each level, the algorithm has a bound on the distance to the true k-th neighbor. It can discard entire branches of the tree (nodes and all their children) if the center of that branch is too far from the query point `i`. This is captured by the check `if ((double)*xi <= dist_k_Q_cor)`.

## 4. Implementation Plan for `survi` (JAX)

Directly porting the GPBoost C++ implementation to JAX is not feasible or idiomatic. JAX is designed for computations on dense, rectangular arrays with static shapes, and it heavily penalizes or forbids dynamic data structures (like maps of vectors) and data-dependent loop terminations within JIT-compiled code.

The most practical approach is a **hybrid CPU/JAX strategy**.

### Strategy: Pre-computation on CPU

The neighbor selection depends on the kernel hyperparameters and inducing point locations. During training, these change at each optimization step. However, the paper's strategy is to refresh the neighbors less frequently: "we re-determine both the inducing points and the Vecchia neighbors in every optimization iteration that is a power of two." My `survi` implementation has a similar callback-based refresh mechanism.

This suggests the following workflow:
1.  **Trigger Refresh**: Inside the training loop (or via a callback), when a neighbor refresh is needed, pause the JAX-based optimization.
2.  **Extract Data**: Pull the current `X_train`, `inducing_points`, and `kernel_params` from JAX device memory to the host (CPU).
3.  **CPU Computation**: Use a pure Python/NumPy/SciPy implementation of the cover tree to build the tree and perform the neighbor search for all `n` points. This will be single-threaded or multi-threaded on the CPU but avoids the `O(n²)` scaling issue.
4.  **Push Result to JAX**: The result of this computation is a dense `(n, k)` integer array of neighbor indices. This static, rectangular array is perfectly suited for JAX. Push it back to the device.
5.  **Resume JAX Optimization**: Continue the JIT-compiled training steps using this new, fixed neighbor array until the next refresh is triggered.

### Implementation Steps

1.  **Create a Pure Python Cover Tree Class:**
    -   It should be a generic implementation that accepts a callable distance function.
    -   It will need a `build(points)` method and a `query(point, k, predecessors_only=True)` method.
    -   The internal representation can use Python dictionaries and lists, similar to the C++ version.

2.  **Create `CoverTreeCorrelationNeighborSelector`:**
    -   This new class will implement the `NeighborSelector` protocol.
    -   Its `select` method will be the orchestrator for the hybrid strategy.
    -   It will not be JIT-compatible (`@jit`).

    ```python
    # In survi/selectors/
    from .base import NeighborSelector
    # from ..sfc.covertree import PyCoverTree # The new Python implementation

    @dataclass(frozen=True)
    class CoverTreeCorrelationNeighborSelector(NeighborSelector):
        num_neighbors: int
        # ... other config ...

        def select(self, X, inducing_points, kernel_strategy, unconstrained_kernel_params):
            # 1. This method runs on the CPU. Convert JAX arrays to NumPy.
            X_np = np.asarray(X)
            inducing_points_np = np.asarray(inducing_points)
            
            # 2. Define the distance function closure.
            #    This function will be called repeatedly by the cover tree.
            #    It needs access to kernel params and inducing points.
            #    It can use JAX for the kernel evals for speed, or be pure NumPy.
            
            memo = {} # Simple memoization for distance calculations
            def metric_fn(i, j):
                if (i, j) in memo: return memo[(i,j)]
                # ... logic from distances_funct ...
                # ... compute residual correlation between X[i] and X[j] ...
                dist = np.sqrt(1.0 - np.abs(rho_ij))
                memo[(i,j)] = dist
                return dist

            # 3. Build and query the tree.
            tree = PyCoverTree(metric_fn)
            tree.build(X_np) # Builds the tree structure
            
            all_neighbors = np.full((X_np.shape[0], self.num_neighbors), -1, dtype=np.int32)
            for i in range(X_np.shape[0]):
                 # The query function must respect the predecessor constraint
                neighbors_i = tree.query(i, k=self.num_neighbors, max_index=i-1) 
                all_neighbors[i, :len(neighbors_i)] = neighbors_i

            # 4. Return a JAX array for consumption by downstream ops.
            return jnp.array(all_neighbors)

    ```

3.  **Integrate into `VIFGP`:**
    -   The `VIFGP` model can be instantiated with this new `CoverTreeCorrelationNeighborSelector`.
    -   The existing `ops.update_state` logic, which handles periodic refreshing of neighbors, will now call this new CPU-bound selector. The performance hit will be localized to the refresh steps, while the majority of the training (gradient steps) will remain fast on the GPU.

This approach balances the strengths of both environments: Python's flexibility for complex, dynamic algorithms and JAX's speed for numerical linear algebra on dense arrays. It correctly implements the paper's sophisticated neighbor selection strategy, moving beyond the limitations of the current brute-force batched approach in `survi`.
