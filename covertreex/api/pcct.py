from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

from covertreex.algo.batch import BatchInsertPlan, batch_insert
from covertreex.algo.batch_delete import BatchDeletePlan, batch_delete
from covertreex.api.runtime import Runtime
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.queries.knn import knn as knn_query


def _ensure_points(backend: TreeBackend, value: Any) -> Any:
    arr = backend.asarray(value, dtype=backend.default_float)
    if arr.ndim == 0:
        arr = backend.xp.reshape(arr, (1, 1))
    elif arr.ndim == 1:
        length = int(arr.shape[0])
        if length == 0:
            arr = backend.xp.reshape(arr, (0, 0))
        else:
            arr = backend.xp.reshape(arr, (1, length))
    return backend.device_put(arr)


def _ensure_indices(backend: TreeBackend, value: Any) -> Any:
    arr = backend.asarray(value, dtype=backend.default_int)
    return backend.device_put(arr)


def _convert_tree(tree: PCCTree, backend: TreeBackend) -> PCCTree:
    if tree.backend is backend:
        return tree
    same_backend = (
        tree.backend.name == backend.name
        and tree.backend.default_float == backend.default_float
        and tree.backend.default_int == backend.default_int
    )
    if same_backend:
        return tree
    return tree.to_backend(backend)


@dataclass(frozen=True)
class PCCT:
    """Thin façade around batch insert/delete + query helpers."""

    runtime: Runtime = field(default_factory=Runtime)
    tree: PCCTree | None = None

    def fit(
        self,
        points: Any,
        *,
        apply_batch_order: bool = True,
        mis_seed: int | None = None,
        return_plan: bool = False,
    ) -> PCCTree | Tuple[PCCTree, BatchInsertPlan]:
        context = self.runtime.activate()
        backend = context.get_backend()
        batch = _ensure_points(backend, points)
        dimension = int(batch.shape[1])
        base_tree = PCCTree.empty(dimension=dimension, backend=backend)
        new_tree, plan = batch_insert(
            base_tree,
            batch,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
        )
        return (new_tree, plan) if return_plan else new_tree

    def insert(
        self,
        batch_points: Any,
        *,
        mis_seed: int | None = None,
        apply_batch_order: bool | None = None,
        return_plan: bool = False,
    ) -> PCCTree | Tuple[PCCTree, BatchInsertPlan]:
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        batch = _ensure_points(backend, batch_points)
        tree_backend = _convert_tree(tree, backend)
        new_tree, plan = batch_insert(
            tree_backend,
            batch,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
        )
        return (new_tree, plan) if return_plan else new_tree

    def delete(
        self,
        indices: Any,
        *,
        return_plan: bool = False,
    ) -> PCCTree | Tuple[PCCTree, BatchDeletePlan]:
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        remove = _ensure_indices(backend, indices)
        tree_backend = _convert_tree(tree, backend)
        new_tree, plan = batch_delete(tree_backend, remove, backend=backend)
        return (new_tree, plan) if return_plan else new_tree

    def knn(
        self,
        query_points: Any,
        *,
        k: int,
        return_distances: bool = False,
    ) -> Any:
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        tree_backend = _convert_tree(tree, backend)
        queries = _ensure_points(tree_backend.backend, query_points)
        return knn_query(
            tree_backend,
            queries,
            k=k,
            return_distances=return_distances,
            backend=tree_backend.backend,
        )

    def nearest(self, query_points: Any, *, return_distances: bool = False) -> Any:
        return self.knn(query_points, k=1, return_distances=return_distances)

    def _require_tree(self) -> PCCTree:
        if self.tree is None:
            raise ValueError("PCCT requires an existing tree; call fit() first.")
        return self.tree
