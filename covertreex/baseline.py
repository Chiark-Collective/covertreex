from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if not hasattr(np, "float"):  # pragma: no cover - compatibility for external baseline
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):  # pragma: no cover - compatibility for external baseline
    np.int = int  # type: ignore[attr-defined]

try:  # pragma: no cover - optional dependency
    from covertree.covertree import CoverTree as _ExternalCoverTree

    _HAS_EXTERNAL_COVERTREE = True
except Exception:  # pragma: no cover - missing optional dependency
    _ExternalCoverTree = Any  # type: ignore[assignment]
    _HAS_EXTERNAL_COVERTREE = False

try:  # pragma: no cover - optional dependency gate for GPBoost baseline
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is not installed
    njit = prange = None  # type: ignore
    _HAS_NUMBA = False

try:  # pragma: no cover - optional dependency gate for mlpack baseline
    import mlpack  # type: ignore[import]

    _HAS_MLPACK = True
except Exception:  # pragma: no cover - mlpack is not installed
    mlpack = Any  # type: ignore[assignment]
    _HAS_MLPACK = False

if _HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _gpboost_euclidean_distances(coords: np.ndarray, i: int, cand_idx: np.ndarray) -> np.ndarray:
        d = coords.shape[1]
        m = cand_idx.shape[0]
        out = np.empty(m, dtype=np.float64)
        xi = coords[i, :]
        for t in prange(m):
            j = cand_idx[t]
            acc = 0.0
            for p in range(d):
                diff = coords[j, p] - xi[p]
                acc += diff * diff
            out[t] = np.sqrt(acc)
        return out


    def _gpboost_distances_funct(
        coord_ind_i: int,
        coords_ind_j: Sequence[int],
        coords: np.ndarray,
    ) -> np.ndarray:
        cand_idx = np.asarray(coords_ind_j, dtype=np.int64)
        if cand_idx.size == 0:
            return np.empty(0, dtype=np.float64)
        return _gpboost_euclidean_distances(coords, int(coord_ind_i), cand_idx)


    def _gpboost_cover_tree_knn(
        coords_mat: np.ndarray,
        *,
        start: int = 0,
        max_radius: float,
    ) -> Tuple[Dict[int, List[int]], int]:
        n_local = int(coords_mat.shape[0])
        cover_tree: Dict[int, List[int]] = {-1: [int(start)]}
        if n_local == 0:
            return cover_tree, 0

        R_max = max(1.0, float(max_radius))
        base = 2.0
        level = 0

        all_indices = list(range(1, n_local))
        covert_points_old: Dict[int, List[int]] = {0: all_indices}

        while (len(cover_tree) - 1) != n_local:
            level += 1
            if base == 2.0:
                R_l = math.ldexp(R_max, -level)
            else:
                R_l = R_max / (base ** level)
            covert_points: Dict[int, List[int]] = {}

            for key, cov_old in list(covert_points_old.items()):
                cov_list = list(cov_old)
                not_all_covered = len(cov_list) > 0

                cover_tree[key + start] = [key + start]

                while not_all_covered:
                    sample_ind = cov_list[0]
                    cover_tree[key + start].append(sample_ind + start)

                    up = [j for j in cov_list if j > sample_ind]

                    if up:
                        dists = _gpboost_distances_funct(sample_ind, up, coords_mat)
                    else:
                        dists = np.empty(0, dtype=np.float64)

                    covered = {up[idx] for idx, value in enumerate(dists) if value <= R_l}

                    cov_list = [j for j in cov_list[1:] if j not in covered]
                    not_all_covered = len(cov_list) > 0

                    if covered:
                        covert_points.setdefault(sample_ind, []).extend(sorted(covered))

            if not covert_points:
                parent_key = start
                if parent_key not in cover_tree:
                    cover_tree[parent_key] = [parent_key]
                existing = {node for node in cover_tree if node >= start}
                for idx in range(n_local):
                    node_id = idx + start
                    if node_id not in existing:
                        cover_tree.setdefault(node_id, [node_id])
                        if node_id not in cover_tree[parent_key]:
                            cover_tree[parent_key].append(node_id)
                break

            covert_points_old = covert_points

        return cover_tree, level


    def _gpboost_find_knn(
        *,
        query_index: int,
        k: int,
        levels: int,
        coords: np.ndarray,
        cover_tree: Dict[int, List[int]],
    ) -> Tuple[List[int], List[float]]:
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
                    if jj != j:
                        Q.append(jj)
                        diff_rev_interim.append(jj)

            diff_rev = []
            early_stop = (len(diff_rev_interim) == 0) or (ii == (levels - 1))

            if diff_rev_interim:
                dvec = _gpboost_distances_funct(query_index, diff_rev_interim, coords)
                Q_dist.extend(dvec.tolist())

            if ii > 1:
                if len(Q_dist) < k_scaled:
                    dist_k_Q_cor = max(Q_dist) if Q_dist else max_dist
                else:
                    arr = np.asarray(Q_dist, dtype=np.float64)
                    dist_k_Q_cor = float(np.partition(arr, k_scaled - 1)[k_scaled - 1])
                dist_k_Q_cor += 1.0 / (base ** (ii - 1))

            if dist_k_Q_cor >= max_dist:
                if not early_stop:
                    diff_rev = diff_rev_interim.copy()
                    if ii == 1 and diff_rev:
                        diff_rev = diff_rev[1:]
            else:
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

        neighbors_i: List[int] = [-1] * k
        nn_dist: List[float] = [float("inf")] * k

        if Q_before_size >= k:
            for j in range(Q_before_size):
                if Q_dist[j] < nn_dist[k - 1]:
                    nn_dist[k - 1] = Q_dist[j]
                    neighbors_i[k - 1] = Q[j]
                    _sort_vectors_decreasing_inplace(nn_dist, neighbors_i)
        else:
            num_points = query_index
            for jj in range(0, num_points):
                d = _gpboost_distances_funct(query_index, [jj], coords)[0]
                if d < nn_dist[k - 1]:
                    nn_dist[k - 1] = float(d)
                    neighbors_i[k - 1] = jj
                    _sort_vectors_decreasing_inplace(nn_dist, neighbors_i)

        return neighbors_i, nn_dist


else:

    def _gpboost_cover_tree_knn(*_: Any, **__: Any) -> Tuple[Dict[int, List[int]], int]:
        raise ImportError("numba is required for the GPBoost baseline.")


    def _gpboost_find_knn(*_: Any, **__: Any) -> Tuple[List[int], List[float]]:
        raise ImportError("numba is required for the GPBoost baseline.")


def _log2_ceil(value: float) -> int:
    if value <= 1.0:
        return 0
    return int(math.ceil(math.log(value, 2)))


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _pairwise_distances(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    diff = points - query[None, :]
    return np.linalg.norm(diff, axis=1)


def _sort_vectors_decreasing_inplace(a: List[float], b: List[int]) -> None:
    n = len(a)
    for j in range(1, n):
        k_idx = j
        while k_idx > 0 and a[k_idx] < a[k_idx - 1]:
            a[k_idx], a[k_idx - 1] = a[k_idx - 1], a[k_idx]
            b[k_idx], b[k_idx - 1] = b[k_idx - 1], b[k_idx]
            k_idx -= 1


@dataclass
class BaselineNode:
    point_index: int
    level: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)


@dataclass
class BaselineCoverTree:
    points: np.ndarray
    nodes: List[BaselineNode] = field(default_factory=list)
    levels: Dict[int, List[int]] = field(default_factory=dict)
    root_id: Optional[int] = None
    distance_count: int = 0

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> "BaselineCoverTree":
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        tree = cls(points=pts.copy())
        if pts.size == 0:
            return tree
        root_idx = 0
        max_dist = float(np.max(np.linalg.norm(pts - pts[root_idx], axis=1)))
        height = _log2_ceil(max(1.0, max_dist))
        node_id = len(tree.nodes)
        tree.nodes.append(BaselineNode(point_index=root_idx, level=height, parent=None))
        tree.levels.setdefault(height, []).append(node_id)
        tree.root_id = node_id
        for idx in range(1, pts.shape[0]):
            tree.insert_point(idx)
        return tree

    def _children_at_level(self, level: int) -> List[int]:
        out: List[int] = []
        for node_id in self.levels.get(level, []):
            out.extend(self.nodes[node_id].children)
        return out

    def insert_point(self, point_index: int) -> None:
        if self.root_id is None:
            raise ValueError("Cannot insert into an empty tree without initialisation.")

        point = self.points[point_index]
        current_levels = list(self.levels.keys())
        if not current_levels:
            level = 0
        else:
            level = max(current_levels)
        active: List[int] = [self.root_id]

        for k in range(level, -1, -1):
            children = self._children_at_level(k)
            within: List[int] = []
            for node_id in children:
                pivot = self.points[self.nodes[node_id].point_index]
                dist = _euclidean(pivot, point)
                self.distance_count += 1
                if dist <= (2 ** k):
                    within.append(node_id)
            if within:
                active = within
                continue

            parent_candidates = self.levels.get(k, [])
            parent_id: Optional[int] = None
            for node_id in parent_candidates:
                pivot = self.points[self.nodes[node_id].point_index]
                dist = _euclidean(pivot, point)
                self.distance_count += 1
                if dist <= (2 ** (k + 1)):
                    parent_id = node_id
                    break

            if parent_id is None and k == level:
                parent_id = self.root_id

            new_id = len(self.nodes)
            self.nodes.append(
                BaselineNode(point_index=point_index, level=k, parent=parent_id)
            )
            self.levels.setdefault(k, []).append(new_id)
            if parent_id is not None:
                self.nodes[parent_id].children.append(new_id)
            break

    def nearest(self, query: Sequence[float]) -> Tuple[int, float]:
        if self.root_id is None or not self.nodes:
            raise ValueError("Cannot query an empty tree.")

        query_vec = np.asarray(query, dtype=float)
        dists = np.linalg.norm(self.points - query_vec, axis=1)
        self.distance_count += int(dists.shape[0])
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def knn(
        self,
        queries: Sequence[Sequence[float]] | Sequence[float],
        k: int,
        *,
        return_distances: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        if self.points.size == 0:
            raise ValueError("Cannot query an empty tree.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if k > self.points.shape[0]:
            raise ValueError("k cannot exceed the number of points.")

        query_arr = np.asarray(queries, dtype=float)
        squeeze = False
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
            squeeze = True

        indices_out: List[np.ndarray] = []
        distances_out: List[np.ndarray] = []
        for query in query_arr:
            dists = _pairwise_distances(self.points, query)
            self.distance_count += int(dists.shape[0])
            order = np.argsort(dists)[:k]
            indices_out.append(order.astype(np.int64))
            distances_out.append(dists[order].astype(np.float64))

        indices_arr = np.stack(indices_out, axis=0)
        distances_arr = np.stack(distances_out, axis=0)

        if squeeze:
            indices_arr = indices_arr[0]
            distances_arr = distances_arr[0]
            if not return_distances:
                return indices_arr
            return indices_arr, distances_arr

        if not return_distances:
            return indices_arr
        return indices_arr, distances_arr


class ExternalCoverTreeBaseline:
    """Adapter around the optional `covertree` package for parity checks."""

    def __init__(self, tree: "_ExternalCoverTree", points: np.ndarray) -> None:
        if not _HAS_EXTERNAL_COVERTREE:
            raise ImportError("covertree package is not available.")
        self._tree = tree
        self.points = points

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        *,
        leafsize: int = 10,
        base: int = 2,
    ) -> "ExternalCoverTreeBaseline":
        if not _HAS_EXTERNAL_COVERTREE:
            raise ImportError("covertree package is not available.")
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        def _metric(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        tree = _ExternalCoverTree(arr, distance=_metric, leafsize=leafsize, base=base)
        return cls(tree=tree, points=arr)

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def nearest(self, query: Sequence[float]) -> Tuple[int, float]:
        indices, distances = self.knn(query, k=1, return_distances=True)
        idx = int(np.asarray(indices).reshape(-1)[0])
        dist = float(np.asarray(distances).reshape(-1)[0])
        return idx, dist

    def knn(
        self,
        queries: Sequence[Sequence[float]] | Sequence[float],
        *,
        k: int,
        return_distances: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        if not _HAS_EXTERNAL_COVERTREE:
            raise ImportError("covertree package is not available.")

        query_arr = np.asarray(queries, dtype=float)
        squeeze = False
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
            squeeze = True

        distances, indices = self._tree.query(query_arr, k=k)
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float64)

        if squeeze:
            indices = indices[0]
            distances = distances[0]
            if k == 1:
                indices = int(indices)
                distances = float(distances)

        if not return_distances:
            return indices
        return indices, distances


class MlpackCoverTreeBaseline:
    """Adapter around mlpack's cover-tree powered k-NN bindings."""

    def __init__(self, *, points: np.ndarray, model: Any, leaf_size: int = 20) -> None:
        if not _HAS_MLPACK:
            raise ImportError("mlpack python bindings are not available.")
        self.points = points
        self.leaf_size = int(max(1, leaf_size))
        self._model = model

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        *,
        leaf_size: int = 20,
    ) -> "MlpackCoverTreeBaseline":
        if not _HAS_MLPACK:
            raise ImportError("mlpack python bindings are not available.")
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] == 0:
            raise ValueError("mlpack cover tree baseline requires at least one point.")
        arr = np.ascontiguousarray(arr)
        result = mlpack.knn(
            reference=arr,
            k=1,
            tree_type="cover",
            leaf_size=int(leaf_size),
            copy_all_inputs=True,
        )
        model = result.get("output_model")
        if model is None:
            raise RuntimeError("mlpack.knn did not return an output model.")
        return cls(points=arr, model=model, leaf_size=int(leaf_size))

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def nearest(self, query: Sequence[float]) -> Tuple[int, float]:
        indices, distances = self.knn(query, k=1, return_distances=True)
        idx_arr = np.asarray(indices).reshape(-1)
        dist_arr = np.asarray(distances).reshape(-1)
        return int(idx_arr[0]), float(dist_arr[0])

    def knn(
        self,
        queries: Sequence[Sequence[float]] | Sequence[float],
        *,
        k: int,
        return_distances: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        if not _HAS_MLPACK:
            raise ImportError("mlpack python bindings are not available.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if k > self.num_points:
            raise ValueError("k cannot exceed the number of reference points.")

        query_arr = np.asarray(queries, dtype=np.float64)
        squeeze = False
        if query_arr.ndim == 1:
            query_arr = query_arr.reshape(1, -1)
            squeeze = True
        query_arr = np.ascontiguousarray(query_arr)
        single_query = query_arr.shape[0] == 1
        if single_query:
            query_arr = np.vstack([query_arr, query_arr])

        result = mlpack.knn(
            input_model=self._model,
            query=query_arr,
            k=int(k),
            tree_type="cover",
            copy_all_inputs=True,
        )
        neighbors = np.asarray(result["neighbors"], dtype=np.int64).copy()
        distances = np.asarray(result["distances"], dtype=np.float64).copy()
        model = result.get("output_model")
        if model is None:
            raise RuntimeError("mlpack.knn did not return an updated model.")
        self._model = model
        if single_query:
            neighbors = neighbors[:1]
            distances = distances[:1]

        if squeeze:
            neighbors = neighbors[0]
            distances = distances[0]
            if not return_distances:
                return neighbors
            return neighbors, distances

        if not return_distances:
            return neighbors
        return neighbors, distances


def has_external_cover_tree() -> bool:
    return _HAS_EXTERNAL_COVERTREE


class GPBoostCoverTreeBaseline:
    """Numba-backed cover tree mirroring the GPBoost implementation."""

    def __init__(
        self,
        *,
        cover_tree: Dict[int, List[int]],
        levels: int,
        points: np.ndarray,
    ) -> None:
        if not _HAS_NUMBA:  # pragma: no cover - guarded during import
            raise ImportError("numba is required for GPBoostCoverTreeBaseline.")
        self._cover_tree = cover_tree
        self._levels = int(levels)
        self.points = points

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> "GPBoostCoverTreeBaseline":
        if not _HAS_NUMBA:
            raise ImportError("numba is required for GPBoostCoverTreeBaseline.")
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        pts = np.ascontiguousarray(pts)
        if pts.size == 0:
            cover_tree, levels = {-1: []}, 0
        else:
            root = pts[0]
            max_radius = float(np.max(np.linalg.norm(pts - root, axis=1)))
            cover_tree, levels = _gpboost_cover_tree_knn(
                pts,
                start=0,
                max_radius=max_radius,
            )
        return cls(cover_tree=cover_tree, levels=levels, points=pts)

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    def nearest(self, query: Sequence[float]) -> Tuple[int, float]:
        indices, distances = self.knn(query, k=1, return_distances=True)
        return int(indices), float(distances)

    def knn(
        self,
        queries: Sequence[Sequence[float]] | Sequence[float],
        *,
        k: int,
        return_distances: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        if not _HAS_NUMBA:
            raise ImportError("numba is required for GPBoostCoverTreeBaseline.")

        query_arr = np.asarray(queries, dtype=float)
        squeeze = False
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
            squeeze = True

        num_points = self.points.shape[0]
        dim = self.points.shape[1] if self.points.ndim == 2 else 1

        work_coords = np.empty((num_points + 1, dim), dtype=np.float64)
        work_coords[:num_points] = self.points

        indices_out: List[np.ndarray] = []
        distances_out: List[np.ndarray] = []

        for query in query_arr:
            work_coords[num_points] = query
            neighbors, dists = _gpboost_find_knn(
                query_index=num_points,
                k=k,
                levels=self._levels,
                coords=work_coords,
                cover_tree=self._cover_tree,
            )
            indices_out.append(np.asarray(neighbors, dtype=np.int64))
            distances_out.append(np.asarray(dists, dtype=np.float64))

        indices_arr = np.stack(indices_out, axis=0) if indices_out else np.empty((0, k), dtype=np.int64)
        distances_arr = (
            np.stack(distances_out, axis=0)
            if distances_out
            else np.empty((0, k), dtype=np.float64)
        )

        if squeeze:
            indices_arr = indices_arr[0]
            distances_arr = distances_arr[0]
            if not return_distances:
                return indices_arr
            return indices_arr, distances_arr

        if not return_distances:
            return indices_arr
        return indices_arr, distances_arr


def has_gpboost_cover_tree() -> bool:
    return _HAS_NUMBA


def has_mlpack_cover_tree() -> bool:
    return _HAS_MLPACK


__all__ = [
    "BaselineCoverTree",
    "BaselineNode",
    "ExternalCoverTreeBaseline",
    "GPBoostCoverTreeBaseline",
    "MlpackCoverTreeBaseline",
    "has_external_cover_tree",
    "has_gpboost_cover_tree",
    "has_mlpack_cover_tree",
]
