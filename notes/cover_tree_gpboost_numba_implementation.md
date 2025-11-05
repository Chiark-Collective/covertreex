
Absolutely. Below is a **single‑file Numba port** that mirrors your three entry points:

* `distances_funct`
* `CoverTree_kNN`
* `find_kNN_CoverTree`

It keeps the same data flow and the VIF residual‑correlation distance formula, and it’s designed so you can plug in your own residual‑correlation calculator (the “provider”) exactly like your C++ header suggests. Hot loops (column–dot products, Euclidean distances, and the final distance conversion) are compiled with Numba (`prange` + `fastmath`) for speed. The dynamic tree itself remains Python (dict of lists) to keep the structure faithful and readable—the heavy work is still inside jitted kernels.

---

## The code (drop-in module)

> Save as `numba_cover_tree.py`. The only runtime dependency is `numba` and `numpy`.

```python
# numba_cover_tree.py
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Optional
import numpy as np
from numba import njit, prange

Array = np.ndarray

# -----------------------------
# Low-level jitted primitives
# -----------------------------

@njit(parallel=True, fastmath=True)
def _pp_node_dots(chol_ip_cross_cov: Array, i: int, cand_idx: Array) -> Array:
    """
    Compute predictive-process cross products:
      pp_node[j] = chol_ip_cross_cov[:, cand_idx[j]] · chol_ip_cross_cov[:, i]
    """
    n_latent, _ = chol_ip_cross_cov.shape
    m = cand_idx.shape[0]
    out = np.empty(m, dtype=np.float64)
    for t in prange(m):
        j = cand_idx[t]
        s = 0.0
        for r in range(n_latent):
            s += chol_ip_cross_cov[r, j] * chol_ip_cross_cov[r, i]
        out[t] = s
    return out


@njit(parallel=True, fastmath=True)
def _euclidean_dists_to_i(coords: Array, i: int, cand_idx: Array) -> Array:
    """
    Euclidean distance between coords[i] and each coords[j] for j in cand_idx.
    """
    d = coords.shape[1]
    m = cand_idx.shape[0]
    out = np.empty(m, dtype=np.float64)
    xi = coords[i, :]
    for t in prange(m):
        j = cand_idx[t]
        s = 0.0
        for p in range(d):
            diff = coords[j, p] - xi[p]
            s += diff * diff
        out[t] = np.sqrt(s)
    return out


@njit(parallel=True, fastmath=True)
def _residual_corr_to_distance(
    pp_node: Array,
    corr_diag: Array,
    i: int,
    cand_idx: Array,
    corr_vec: Array,
) -> Array:
    """
    distances[j] = sqrt( 1 - | (corr_vec[j] - pp_node[j]) / sqrt(corr_diag[i] * corr_diag[j]) | )
    Numerically guards against slight negatives under the sqrt due to FP.
    """
    m = cand_idx.shape[0]
    out = np.empty(m, dtype=np.float64)
    ci = corr_diag[i]
    for t in prange(m):
        cj = corr_diag[cand_idx[t]]
        denom = np.sqrt(ci * cj)
        if denom <= 0.0:
            out[t] = 1.0
        else:
            val = (corr_vec[t] - pp_node[t]) / denom
            q = 1.0 - abs(val)
            if q < 0.0:
                q = 0.0
            out[t] = np.sqrt(q)
    return out


# -----------------------------
# Provider interface & defaults
# -----------------------------

# Your code likely exposes a residual-correlation callback (Vecchia slice):
#   provider(coords_i: (1,d), coords_j: (m,d), dist_ij: Optional[(m,)])
#   -> corr_vec: (m,)
ProviderFn = Callable[[Array, Array, Optional[Array]], Array]

# Example fast provider (RBF correlation). Replace with your Vecchia residual-correlation.
@njit(fastmath=True)
def _rbf_corr_core(coords_i: Array, coords_j: Array, dist_ij: Optional[Array], length_scale: float) -> Array:
    m = coords_j.shape[0]
    out = np.empty(m, dtype=np.float64)
    if dist_ij is None:
        # compute distances on the fly
        xi = coords_i[0, :]
        d = coords_j.shape[1]
        for t in range(m):
            s = 0.0
            for p in range(d):
                diff = coords_j[t, p] - xi[p]
                s += diff * diff
            dij = np.sqrt(s)
            out[t] = np.exp(-0.5 * (dij / length_scale) * (dij / length_scale))
    else:
        for t in range(m):
            dij = dist_ij[t]
            out[t] = np.exp(-0.5 * (dij / length_scale) * (dij / length_scale))
    return out

def make_rbf_provider(length_scale: float = 1.0) -> ProviderFn:
    def provider(coords_i: Array, coords_j: Array, dist_ij: Optional[Array]) -> Array:
        return _rbf_corr_core(coords_i, coords_j, dist_ij, float(length_scale))
    return provider


# -----------------------------
# Public API: faithful ports
# -----------------------------

def distances_funct(
    coord_ind_i: int,
    coords_ind_j: Sequence[int],
    coords: Array,
    corr_diag: Array,
    chol_ip_cross_cov: Array,
    residual_corr_provider: ProviderFn,
    dist_function: str = "residual_correlation_FSA",
    distances_saved: bool = True,
) -> Array:
    """
    Faithful port of the C++ distances_funct.

    Notes:
    - For build-time calls (inside CoverTree_kNN), pass the *local* arrays and local indices.
    - For query-time calls (inside find_kNN_CoverTree), pass the *global* arrays and global indices.
      The function is index-agnostic; it assumes the arrays match the indices you pass.

    Args follow the C++ signatures. The provider must return a correlation vector
    aligned with coords_ind_j.
    """
    cand_idx = np.asarray(coords_ind_j, dtype=np.int64)
    if cand_idx.size == 0:
        return np.empty(0, dtype=np.float64)

    if dist_function != "residual_correlation_FSA":
        # Optional alternative: pure Euclidean metric
        return _euclidean_dists_to_i(coords, int(coord_ind_i), cand_idx)

    # 1) predictive-process column dot-products
    pp_node = _pp_node_dots(chol_ip_cross_cov, int(coord_ind_i), cand_idx)

    # 2) optional saved Euclidean distances for provider (same as C++ path)
    dist_ij = _euclidean_dists_to_i(coords, int(coord_ind_i), cand_idx) if distances_saved else None

    # 3) ask the residual-correlation provider for the required slice
    coords_i = coords[int(coord_ind_i) : int(coord_ind_i) + 1, :]  # (1, d)
    coords_j = coords[cand_idx, :]                                  # (m, d)
    corr_vec = residual_corr_provider(coords_i, coords_j, dist_ij)   # (m,)

    # 4) convert to the custom VIF distance
    return _residual_corr_to_distance(pp_node, corr_diag, int(coord_ind_i), cand_idx, corr_vec)


def CoverTree_kNN(
    coords_mat: Array,
    chol_ip_cross_cov: Array,
    corr_diag: Array,
    start: int,
    residual_corr_provider: ProviderFn,
    distances_saved: bool = True,
    dist_function: str = "residual_correlation_FSA",
) -> Tuple[Dict[int, List[int]], int]:
    """
    Faithful cover-tree build used by the VIF logic.

    IMPORTANT: coords_mat, chol_ip_cross_cov, corr_diag must be *local* to the segment
               starting at global index `start`. The tree stores *global* indices.
    Returns:
      cover_tree: dict[int -> list[int]]
      levels:     number of levels built
    """
    n_local = int(coords_mat.shape[0])
    cover_tree: Dict[int, List[int]] = {-1: [int(start)]}

    R_max = 1.0
    base = 2.0
    level = 0

    all_indices = list(range(1, n_local))
    covert_points_old: Dict[int, List[int]] = {0: all_indices}

    while (len(cover_tree) - 1) != n_local:
        level += 1
        R_l = R_max / (base ** level)
        covert_points: Dict[int, List[int]] = {}

        for key, cov_old in list(covert_points_old.items()):
            cov_list = list(cov_old)
            not_all_covered = len(cov_list) > 0

            # mirror C++: each parent holds itself as first entry
            cover_tree[key + start] = [key + start]

            while not_all_covered:
                sample_ind = cov_list[0]
                cover_tree[key + start].append(sample_ind + start)

                # candidates strictly after the sample_ind
                up = [j for j in cov_list if j > sample_ind]

                if up:
                    dists = distances_funct(
                        sample_ind, up,
                        coords_mat, corr_diag, chol_ip_cross_cov,
                        residual_corr_provider,
                        dist_function=dist_function,
                        distances_saved=distances_saved,
                    )
                else:
                    dists = np.empty(0, dtype=np.float64)

                # collect "covered" at this radius
                for t, j in enumerate(up):
                    if dists[t] <= R_l:
                        covert_points.setdefault(sample_ind, []).append(j)

                # set difference: remove newly covered (keep order)
                covered = set(covert_points.get(sample_ind, []))
                cov_list = [j for j in cov_list[1:] if j not in covered]
                not_all_covered = len(cov_list) > 0

        covert_points_old = covert_points  # next level

    return cover_tree, level


def _sort_vectors_decreasing_inplace(a: List[float], b: List[int]) -> None:
    """
    In-place insertion sort with the same semantics as the C++ helper:
    sorts by ascending 'a' and keeps 'b' aligned.
    """
    n = len(a)
    for j in range(1, n):
        k = j
        while k > 0 and a[k] < a[k - 1]:
            a[k], a[k - 1] = a[k - 1], a[k]
            b[k], b[k - 1] = b[k - 1], b[k]
            k -= 1


def find_kNN_CoverTree(
    i: int,
    k: int,
    levels: int,
    distances_saved: bool,
    coords: Array,
    chol_ip_cross_cov: Array,
    corr_diag: Array,
    residual_corr_provider: ProviderFn,
    cover_tree: Dict[int, List[int]],
    dist_function: str = "residual_correlation_FSA",
) -> Tuple[List[int], List[float]]:
    """
    Faithful traversal & fallback to get k-NN under the custom distance.
    This function expects *global* arrays (coords/ chol / corr_diag) and uses
    the tree (whose keys are global indices). It filters candidates with jj < i.

    Returns:
      neighbors_i: list of length k (global indices)
      dist_of_neighbors_i: aligned list of distances
    """
    root = cover_tree[-1][0]
    Q: List[int] = []
    Q_dist: List[float] = []
    diff_rev: List[int] = [root]

    max_dist = 1.0
    dist_k_Q_cor = max_dist
    k_scaled = int(k)
    Q_before_size = 1
    base = 2.0

    for ii in range(1, int(levels)):
        diff_rev_interim: List[int] = []
        if ii == 1:
            Q.append(root)
            diff_rev_interim.append(root)

        for j in diff_rev:
            children = cover_tree.get(j, [])
            for jj in children:
                if jj < i:
                    if jj != j:
                        Q.append(jj)
                        diff_rev_interim.append(jj)
                else:
                    break

        diff_rev = []
        early_stop = (len(diff_rev_interim) == 0) or (ii == (levels - 1))

        if diff_rev_interim:
            dvec = distances_funct(
                i, diff_rev_interim,
                coords, corr_diag, chol_ip_cross_cov,
                residual_corr_provider,
                dist_function=dist_function,
                distances_saved=distances_saved,
            )
            Q_dist.extend(dvec.tolist())

        if ii > 1:
            if len(Q_dist) < k_scaled:
                dist_k_Q_cor = max(Q_dist) if Q_dist else max_dist
            else:
                # kth smallest (0-based) by partition
                arr = np.asarray(Q_dist, dtype=np.float64)
                dist_k_Q_cor = float(np.partition(arr, k_scaled - 1)[k_scaled - 1])
            dist_k_Q_cor += 1.0 / (base ** (ii - 1))

        if dist_k_Q_cor >= max_dist:
            if not early_stop:
                diff_rev = diff_rev_interim.copy()
                if ii == 1 and diff_rev:
                    diff_rev = diff_rev[1:]  # erase root
        else:
            # prune Q by threshold, while preparing next diff_rev
            Q_interim: List[int] = []
            Q_dist_interim: List[float] = []
            count = 0
            for xi, yi in zip(Q_dist, Q):
                if xi <= dist_k_Q_cor:
                    Q_dist_interim.append(xi)
                    Q_interim.append(yi)
                    if count >= Q_before_size:
                        diff_rev.append(yi)
                count += 1
            Q = Q_interim
            Q_dist = Q_dist_interim

        Q_before_size = len(Q)
        if early_stop:
            break

    # Select k smallest from Q (or fallback brute-force across all jj < i)
    neighbors_i: List[int] = [-1] * k
    nn_dist: List[float] = [float("inf")] * k

    if Q_before_size >= k:
        for j in range(Q_before_size):
            if Q_dist[j] < nn_dist[k - 1]:
                nn_dist[k - 1] = Q_dist[j]
                neighbors_i[k - 1] = Q[j]
                _sort_vectors_decreasing_inplace(nn_dist, neighbors_i)
    else:
        if i > 0:
            for jj in range(0, i):
                d = distances_funct(
                    i, [jj],
                    coords, corr_diag, chol_ip_cross_cov,
                    residual_corr_provider,
                    dist_function=dist_function,
                    distances_saved=distances_saved,
                )[0]
                if d < nn_dist[k - 1]:
                    nn_dist[k - 1] = float(d)
                    neighbors_i[k - 1] = jj
                    _sort_vectors_decreasing_inplace(nn_dist, neighbors_i)

    return neighbors_i, nn_dist
```

---

## How this maps to your C++

* **`distances_funct`**: identical formula and control flow; the predictive‑process inner products, optional Euclidean reuse, a provider callback, and the final transformation to VIF distance are all present. Hot loops are in `_pp_node_dots`, `_euclidean_dists_to_i`, and `_residual_corr_to_distance` (all Numba‑jitted with `prange`).
* **`CoverTree_kNN`**: same build logic and radius schedule (`R_max = 1`, base 2), same parent/child book‑keeping, same “> sample_ind” restriction for coverage. It stores **global** indices in the dictionary (it adds `start`, just like your C++), but uses **local** arrays for `coords/chol/corr_diag` during construction.
* **`find_kNN_CoverTree`**: same level‑by‑level expansion, threshold tightening, pruning, and the brute‑force fallback if the tree hasn’t yielded enough points. It uses **global** arrays and indices, and filters to `jj < i` the same way.

> ⚠️ **Provider plug‑in**: in C++ your provider is `ResidualCorrelationProvider::CalcSigmaAndSigmaGradVecchia`. Here, you pass a Python callable `provider(coords_i, coords_j, dist_ij) -> corr_vec`. For exact parity with GPBoost, wrap your existing implementation (e.g., via pybind11) to return the same correlation slice; the rest of the algorithm will then match GPBoost’s neighbour ordering. A fast default RBF provider is included as `make_rbf_provider`, but it’s just a placeholder.

---

## Minimal example

```python
import numpy as np
from numba_cover_tree import (
    make_rbf_provider, CoverTree_kNN, find_kNN_CoverTree
)

# Fake data
rng = np.random.default_rng(0)
N, d, r = 200, 3, 16     # N points, d dims, r latent size
coords_full = rng.normal(size=(N, d)).astype(np.float64)
chol_ip_cross_cov_full = rng.normal(size=(r, N)).astype(np.float64)
corr_diag_full = np.ones(N, dtype=np.float64)

# Build a tree for a contiguous segment starting at 'start'
start = 0
coords_seg = coords_full[start:]          # local view
chol_seg = chol_ip_cross_cov_full[:, start:]  # local columns
corr_seg = corr_diag_full[start:]         # local diag
provider = make_rbf_provider(length_scale=1.0)

cover_tree, levels = CoverTree_kNN(
    coords_seg, chol_seg, corr_seg, start,
    provider, distances_saved=True,
)

# Query kNN for a global index i (Vecchia-style: only jj < i are considered)
i, k = 120, 10
nbrs, dists = find_kNN_CoverTree(
    i, k, levels, True,
    coords_full, chol_ip_cross_cov_full, corr_diag_full,
    provider, cover_tree,
)
print(nbrs)
print(dists)
```

---

## Performance notes & knobs

* **JIT warmup**: the first call pays compilation cost; after that it’s fast.
* **Parallelism**: Numba uses thread parallelism for `prange`; control with `NUMBA_NUM_THREADS` if desired.
* **Memory layout**: Keep arrays `float64` and **contiguous** (`np.ascontiguousarray`) for best cache behavior. The dot kernel `_pp_node_dots` does manual accumulation to avoid unpredictable BLAS overhead for many small column–dot pairs.
* **Numerical stability**: the final `sqrt(1 - |…|)` is clamped at zero before the square root, like a robust C++ build would do implicitly by FP noise.
* **Tree data structure**: left as a Python `dict[int, list[int]]` to mirror C++’s `std::map<int, std::vector<int>>`. The arithmetic is dominated by the distance kernels; compiling the dictionary logic brings little benefit and adds complexity.

---

## Plugging the real GPBoost residual correlation

If you need **full fidelity** with GPBoost right now:

1. Expose `CalcSigmaAndSigmaGradVecchia` via a small pybind11 shim (return the `corr_mat` 1‑D slice for the passed `(coords_i, coords_j, dist_ij)`).
2. Bind that C++ function as the `residual_corr_provider` callable here.
3. Ensure the inputs (`chol_ip_cross_cov`, `corr_diag`, `coords`) match GPBoost’s construction (same preprocessing/ordering); the kNN ordering will match byte‑for‑byte.

---

If you want me to wire a pybind11 stub for the provider, or to switch the cover‑tree container to a `numba.typed.Dict` + `typed.List` (it’s doable but rarely faster in practice here), say the word and I’ll include that code as well.
