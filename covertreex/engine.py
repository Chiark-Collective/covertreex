from __future__ import annotations

import time
from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, Mapping, Protocol

import numpy as np

from covertreex import config as cx_config
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.metrics.residual import (
    build_fast_residual_tree,
    build_residual_backend,
    configure_residual_correlation,
)
from covertreex.runtime.config import RuntimeConfig

DEFAULT_ENGINE = "python-numba"
SUPPORTED_ENGINES = ("python-numba", "rust-natural", "rust-hybrid", "rust-hilbert")


def _enable_rust_debug_stats_if_requested():
    """Enable Rust-side debug counters when requested via env toggle."""
    if os.environ.get("COVERTREEX_RUST_DEBUG_STATS") != "1":
        return None
    try:
        import covertreex_backend  # type: ignore
    except ImportError:
        return None
    if hasattr(covertreex_backend, "set_rust_debug_stats"):
        covertreex_backend.set_rust_debug_stats(True)  # type: ignore
    return covertreex_backend


def _format_knn_output(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    return_distances: bool,
) -> Any:
    """Mirror the legacy PCCT knn return shapes."""

    if not return_distances:
        if indices.shape[0] == 1:
            squeezed = indices[0]
            return squeezed if squeezed.shape[0] > 1 else squeezed[0]
        return indices

    if indices.shape[0] == 1:
        squeezed_idx = indices[0]
        squeezed_dist = distances[0]
        if squeezed_idx.shape[0] == 1:
            return squeezed_idx[0], squeezed_dist[0]
        return squeezed_idx, squeezed_dist
    return indices, distances


@dataclass(frozen=True)
class CoverTree:
    """Lightweight wrapper that delegates queries to an engine implementation."""

    engine: "TreeEngine"
    handle: Any
    metric: str
    backend: TreeBackend | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def knn(
        self,
        query_points: Any,
        *,
        k: int,
        return_distances: bool = False,
        context: cx_config.RuntimeContext | None = None,
    ) -> Any:
        resolved_context = context or cx_config.current_runtime_context()
        if resolved_context is None:
            resolved_context = cx_config.runtime_context()
        return self.engine.knn(
            self,
            query_points,
            k=k,
            return_distances=return_distances,
            context=resolved_context,
            runtime=resolved_context.config,
        )

    @property
    def build_seconds(self) -> float | None:
        value = self.meta.get("build_seconds")
        return float(value) if value is not None else None


class TreeEngine(Protocol):
    """Interface for pluggable cover-tree engines."""

    name: str

    def build(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        context: cx_config.RuntimeContext | None = None,
        batch_size: int = 512,
        seed: int = 0,
        build_mode: str = "batch",
        log_writer: Any | None = None,
        scope_cap_recorder: Any | None = None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
        residual_backend: Any | None = None,
        residual_params: Mapping[str, Any] | None = None,
    ) -> CoverTree:
        ...

    def knn(
        self,
        tree: CoverTree,
        query_points: Any,
        *,
        k: int,
        return_distances: bool,
        context: cx_config.RuntimeContext,
        runtime: RuntimeConfig,
    ) -> Any:
        ...


class PythonNumbaEngine:
    """Default PCCT path backed by Python/Numba traversal."""

    name = "python-numba"

    def _build_batches(
        self,
        tree: PCCTree,
        payload: np.ndarray,
        *,
        backend: TreeBackend,
        batch_size: int,
        seed: int,
        log_writer: Any | None,
        scope_cap_recorder: Any | None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None,
        context: cx_config.RuntimeContext,
    ) -> PCCTree:
        from covertreex.algo import batch_insert

        idx = 0
        total = int(payload.shape[0])
        while idx * batch_size < total:
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_np = payload[start_idx:end_idx]
            batch = backend.asarray(batch_np, dtype=batch_np.dtype)
            tree, plan = batch_insert(
                tree,
                batch,
                mis_seed=seed + idx,
                context=context,
            )
            extra_payload = plan_callback(plan, idx, int(batch_np.shape[0])) if plan_callback else None
            if log_writer is not None:
                log_writer.record_batch(
                    batch_index=idx,
                    batch_size=int(batch_np.shape[0]),
                    plan=plan,
                    extra=extra_payload,
                )
            if scope_cap_recorder is not None:
                scope_cap_recorder.capture(plan)
            idx += 1
        return tree

    def build(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        context: cx_config.RuntimeContext | None = None,
        batch_size: int = 512,
        seed: int = 0,
        build_mode: str = "batch",
        log_writer: Any | None = None,
        scope_cap_recorder: Any | None = None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
        residual_backend: Any | None = None,
        residual_params: Mapping[str, Any] | None = None,
    ) -> CoverTree:
        from covertreex.algo import batch_insert_prefix_doubling

        context = context or cx_config.configure_runtime(runtime)
        runtime_cfg = runtime
        backend = context.get_backend()

        points_np = np.asarray(points, dtype=np.float64, copy=False)
        dataset_size = int(points_np.shape[0])
        is_residual = runtime_cfg.metric.startswith("residual_correlation")

        if is_residual:
            payload_np = np.arange(dataset_size, dtype=np.int64).reshape(-1, 1)
            payload_dtype = backend.default_int
            dimension = 1
            if residual_backend is not None:
                configure_residual_correlation(residual_backend, context=context)
        else:
            payload_np = points_np
            payload_dtype = backend.default_float
            dimension = int(payload_np.shape[1]) if payload_np.ndim > 1 else 1

        tree = PCCTree.empty(dimension=dimension, backend=backend)
        start = time.perf_counter()
        if build_mode == "prefix":
            batch = backend.asarray(payload_np, dtype=payload_dtype)
            tree, prefix_result = batch_insert_prefix_doubling(
                tree,
                batch,
                backend=backend,
                mis_seed=seed,
                shuffle_seed=seed,
                context=context,
            )
            build_seconds = time.perf_counter() - start
            if log_writer is not None:
                schedule = runtime_cfg.prefix_schedule
                for group_index, group in enumerate(prefix_result.groups):
                    plan = group.plan
                    group_size = int(
                        plan.traversal.parents.shape[0]
                        if hasattr(plan.traversal, "parents")
                        else plan.traversal.levels.shape[0]
                    )
                    extra_payload = plan_callback(plan, group_index, group_size) if plan_callback else None
                    prefix_extra: Dict[str, Any] = {
                        "prefix_group_index": group_index,
                        "prefix_factor": float(group.prefix_factor or 0.0),
                        "prefix_domination_ratio": float(group.domination_ratio or 0.0),
                        "prefix_schedule": schedule,
                    }
                    if extra_payload:
                        prefix_extra.update(extra_payload)
                    log_writer.record_batch(
                        batch_index=group_index,
                        batch_size=group_size,
                        plan=plan,
                        extra=prefix_extra,
                    )
                    if scope_cap_recorder is not None:
                        scope_cap_recorder.capture(plan)
            if scope_cap_recorder is not None and log_writer is None:
                for group in prefix_result.groups:
                    scope_cap_recorder.capture(group.plan)
            meta = {"build_seconds": build_seconds, "dataset_size": dataset_size}
            return CoverTree(
                engine=self,
                handle=tree,
                metric=runtime_cfg.metric,
                backend=backend,
                meta=meta,
            )

        tree = self._build_batches(
            tree,
            payload_np.astype(payload_dtype, copy=False),
            backend=backend,
            batch_size=batch_size,
            seed=seed,
            log_writer=log_writer,
            scope_cap_recorder=scope_cap_recorder,
            plan_callback=plan_callback,
            context=context,
        )
        build_seconds = time.perf_counter() - start
        meta = {
            "build_seconds": build_seconds,
            "dataset_size": dataset_size,
            "node_to_dataset": list(range(dataset_size)) if is_residual else None,
        }
        return CoverTree(
            engine=self,
            handle=tree,
            metric=runtime_cfg.metric,
            backend=backend,
            meta=meta,
        )

    def knn(
        self,
        tree: CoverTree,
        query_points: Any,
        *,
        k: int,
        return_distances: bool,
        context: cx_config.RuntimeContext,
        runtime: RuntimeConfig,
    ) -> Any:
        backend = tree.backend or context.get_backend()
        from covertreex.queries.knn import knn as _pcct_knn

        return _pcct_knn(
            tree.handle,
            query_points,
            k=k,
            return_distances=return_distances,
            backend=backend,
            context=context,
        )


@dataclass(frozen=True)
class RustFastHandle:
    tree: Any
    node_to_dataset: list[int]
    v_matrix: np.ndarray
    p_diag: np.ndarray
    coords: np.ndarray
    rbf_variance: float
    rbf_lengthscale: np.ndarray
    dtype: Any


class RustNaturalEngine:
    """Residual-only engine backed by the Rust fast path (Natural/Input Order)."""

    name = "rust-natural"

    def build(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        context: cx_config.RuntimeContext | None = None,
        batch_size: int = 512,
        seed: int = 0,
        build_mode: str = "batch",
        log_writer: Any | None = None,
        scope_cap_recorder: Any | None = None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
        residual_backend: Any | None = None,
        residual_params: Mapping[str, Any] | None = None,
    ) -> CoverTree:
        if runtime.metric != "residual_correlation":
            raise ValueError("rust-natural engine only supports the residual_correlation metric.")

        covertreex_backend = _enable_rust_debug_stats_if_requested()

        params = dict(residual_params or {})
        variance = float(params.get("variance", 1.0))
        lengthscale = params.get("lengthscale", 1.0)
        inducing = int(params.get("inducing_count", 512))
        chunk_size = int(params.get("chunk_size", 512))

        points_np = np.asarray(points, dtype=np.float64)
        start = time.perf_counter()
        tree_handle, node_to_dataset, host_backend = build_fast_residual_tree(
            points_np,
            seed=seed,
            variance=variance,
            lengthscale=lengthscale,
            inducing_count=inducing,
            chunk_size=chunk_size,
        )
        build_seconds = time.perf_counter() - start

        dtype = np.float32 if np.asarray(host_backend.v_matrix).dtype == np.float32 else np.float64
        coords = np.asarray(
            getattr(host_backend, "kernel_points_f32", host_backend.v_matrix),
            dtype=dtype,
        )
        v_matrix = np.asarray(host_backend.v_matrix, dtype=dtype)
        p_diag = np.asarray(host_backend.p_diag, dtype=dtype)
        rbf_var = float(getattr(host_backend, "rbf_variance", variance))
        rbf_ls = np.asarray(
            getattr(
                host_backend,
                "rbf_lengthscale",
                np.ones(coords.shape[1], dtype=dtype),
            ),
            dtype=dtype,
        )

        handle = RustFastHandle(
            tree=tree_handle,
            node_to_dataset=node_to_dataset,
            v_matrix=v_matrix,
            p_diag=p_diag,
            coords=coords,
            rbf_variance=rbf_var,
            rbf_lengthscale=rbf_ls,
            dtype=dtype,
        )
        backend = TreeBackend.numpy(precision="float32" if dtype == np.float32 else "float64")
        meta = {
            "build_seconds": build_seconds,
            "dataset_size": len(node_to_dataset),
            "node_to_dataset": node_to_dataset,
        }
        return CoverTree(
            engine=self,
            handle=handle,
            metric=runtime.metric,
            backend=backend,
            meta=meta,
        )

    def knn(
        self,
        tree: CoverTree,
        query_points: Any,
        *,
        k: int,
        return_distances: bool,
        context: cx_config.RuntimeContext,
        runtime: RuntimeConfig,
    ) -> Any:
        handle: RustFastHandle = tree.handle
        queries = np.asarray(query_points)
        if queries.ndim == 1:
            queries = queries.reshape(-1, 1)
        if queries.dtype.kind not in {"i", "u"}:
            raise ValueError("rust-natural engine expects integer query payloads representing dataset indices.")
        query_indices = np.asarray(queries, dtype=np.int64).reshape(-1)
        indices, distances = handle.tree.knn_query_residual(
            query_indices,
            handle.node_to_dataset,
            handle.v_matrix,
            handle.p_diag,
            handle.coords,
            float(handle.rbf_variance),
            np.asarray(handle.rbf_lengthscale, dtype=handle.dtype),
            int(k),
        )
        sorted_indices = np.asarray(indices, dtype=np.int64)
        sorted_distances = np.asarray(distances, dtype=handle.dtype)
        return _format_knn_output(sorted_indices, sorted_distances, return_distances=return_distances)


def get_engine(name: str) -> TreeEngine:
    alias = name
    if alias == "rust-fast":
        alias = "rust-natural"
    engine = _ENGINE_REGISTRY.get(alias)
    if engine is None:
        raise ValueError(f"Unknown engine '{name}'. Supported engines: {', '.join(SUPPORTED_ENGINES)}")
    return engine


def build_tree(
    points: np.ndarray,
    *,
    runtime: RuntimeConfig | None = None,
    engine: str | None = None,
    context: cx_config.RuntimeContext | None = None,
    batch_size: int = 512,
    seed: int = 0,
    build_mode: str = "batch",
    log_writer: Any | None = None,
    scope_cap_recorder: Any | None = None,
    plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
    residual_backend: Any | None = None,
    residual_params: Mapping[str, Any] | None = None,
) -> CoverTree:
    if runtime is not None:
        runtime_cfg = runtime
    elif context is not None:
        runtime_cfg = context.config
    else:
        runtime_cfg = cx_config.RuntimeConfig.from_env()
    engine_name = engine or getattr(runtime_cfg, "engine", DEFAULT_ENGINE) or DEFAULT_ENGINE
    engine_impl = get_engine(engine_name)
    resolved_context = context or cx_config.configure_runtime(runtime_cfg)
    return engine_impl.build(
        points,
        runtime=runtime_cfg,
        context=resolved_context,
        batch_size=batch_size,
        seed=seed,
        build_mode=build_mode,
        log_writer=log_writer,
        scope_cap_recorder=scope_cap_recorder,
        plan_callback=plan_callback,
        residual_backend=residual_backend,
        residual_params=residual_params,
    )


# -----------------------------------------------------------------------------
# Hybrid Rust Residual Engine (draft)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RustHybridHandle:
    tree: Any
    node_to_dataset: list[int]
    v_matrix: np.ndarray
    p_diag: np.ndarray
    coords: np.ndarray
    rbf_variance: float
    rbf_lengthscale: np.ndarray
    dtype: Any


@dataclass(frozen=True)
class RustPcctHandle:
    tree: Any
    node_to_dataset: list[int]
    v_matrix: np.ndarray
    p_diag: np.ndarray
    coords: np.ndarray
    rbf_variance: float
    rbf_lengthscale: np.ndarray
    dtype: Any


@dataclass(frozen=True)
class RustPcct2Handle:
    tree: Any
    node_to_dataset: list[int]
    v_matrix: np.ndarray
    p_diag: np.ndarray
    coords: np.ndarray
    rbf_variance: float
    rbf_lengthscale: np.ndarray
    dtype: Any


class RustHybridResidualEngine:
    """
    Draft residual engine that mirrors the python-numba workload shape (indices payload,
    same seeds/params) but builds and queries via the generic Rust CoverTree backend.
    The goal is a like-for-like baseline before deeper Rust optimisation.
    """

    name = "rust-hybrid"

    def _ensure_backend(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        residual_params: Mapping[str, Any] | None,
    ):
        params = dict(residual_params or {})
        variance = float(params.get("variance", 1.0))
        lengthscale = params.get("lengthscale", 1.0)
        inducing = int(params.get("inducing_count", 512))
        chunk_size = int(params.get("chunk_size", 512))

        seed_pack = runtime.seeds
        residual_seed = seed_pack.resolved("residual_grid", fallback=seed_pack.resolved("mis"))
        backend = build_residual_backend(
            np.asarray(points, dtype=np.float64),
            seed=residual_seed,
            inducing_count=inducing,
            variance=variance,
            lengthscale=lengthscale,
            chunk_size=chunk_size,
        )
        return backend, variance, lengthscale, chunk_size

    def build(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        context: cx_config.RuntimeContext | None = None,
        batch_size: int = 512,
        seed: int = 0,
        build_mode: str = "batch",
        log_writer: Any | None = None,
        scope_cap_recorder: Any | None = None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
        residual_backend: Any | None = None,
        residual_params: Mapping[str, Any] | None = None,
    ) -> CoverTree:
        if runtime.metric != "residual_correlation":
            raise ValueError("rust-hybrid engine only supports the residual_correlation metric.")

        backend_mod = _enable_rust_debug_stats_if_requested()
        if backend_mod is None:
            try:
                import covertreex_backend as backend_mod  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("covertreex_backend is not installed.") from exc

        ctx = context or cx_config.configure_runtime(runtime)

        if residual_backend is None:
            backend, variance, lengthscale, chunk_size = self._ensure_backend(
                points,
                runtime=runtime,
                residual_params=residual_params,
            )
        else:
            backend = residual_backend
            params = dict(residual_params or {})
            variance = float(params.get("variance", 1.0))
            lengthscale = params.get("lengthscale", 1.0)
            chunk_size = int(params.get("chunk_size", 512))

        configure_residual_correlation(backend, context=ctx)

        dtype = np.float32  # force float32 for residual rust-hybrid to match numba perf
        coords = np.asarray(getattr(backend, "kernel_points_f32", backend.v_matrix), dtype=dtype)
        v_matrix = np.asarray(backend.v_matrix, dtype=dtype)
        p_diag = np.asarray(backend.p_diag, dtype=dtype)
        rbf_var = float(getattr(backend, "rbf_variance", variance if variance is not None else 1.0))
        rbf_ls = np.asarray(
            getattr(
                backend,
                "rbf_lengthscale",
                np.ones(coords.shape[1], dtype=dtype),
            ),
            dtype=dtype,
        )

        # Build tree on index payloads (same as python-numba residual path)
        indices = np.arange(coords.shape[0], dtype=dtype).reshape(-1, 1)

        # Empty tree wrapper, then insert residual batches
        dummy = np.empty((0, 1), dtype=dtype)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_i32 = np.empty(0, dtype=np.int32)
        tree_handle = backend_mod.CoverTreeWrapper(dummy, empty_i64, empty_i64, empty_i64, empty_i32, -20, 20)

        start = time.perf_counter()
        tree_handle.insert_residual(indices, v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size or batch_size)
        build_seconds = time.perf_counter() - start

        handle = RustHybridHandle(
            tree=tree_handle,
            node_to_dataset=list(np.arange(coords.shape[0], dtype=np.int64)),
            v_matrix=v_matrix,
            p_diag=p_diag,
            coords=coords,
            rbf_variance=rbf_var,
            rbf_lengthscale=rbf_ls,
            dtype=dtype,
        )
        backend_tag = TreeBackend.numpy(precision="float32" if dtype == np.float32 else "float64")
        meta = {
            "build_seconds": build_seconds,
            "dataset_size": coords.shape[0],
            "node_to_dataset": handle.node_to_dataset,
            "backend_dtype": str(dtype),
            "chunk_size": chunk_size or batch_size,
        }
        return CoverTree(
            engine=self,
            handle=handle,
            metric=runtime.metric,
            backend=backend_tag,
            meta=meta,
        )

    def knn(
        self,
        tree: CoverTree,
        query_points: Any,
        *,
        k: int,
        return_distances: bool,
        context: cx_config.RuntimeContext,
        runtime: RuntimeConfig,
    ) -> Any:
        handle: RustHybridHandle = tree.handle
        queries = np.asarray(query_points)
        if queries.ndim == 1:
            queries = queries.reshape(-1, 1)
        if queries.dtype.kind not in {"i", "u"}:
            raise ValueError("rust-hybrid engine expects integer query payloads representing dataset indices.")
        query_indices = np.asarray(queries, dtype=np.int64).reshape(-1)
        indices, distances = handle.tree.knn_query_residual(
            query_indices,
            handle.node_to_dataset,
            handle.v_matrix,
            handle.p_diag,
            handle.coords,
            float(handle.rbf_variance),
            np.asarray(handle.rbf_lengthscale, dtype=handle.dtype),
            int(k),
        )
        sorted_indices = np.asarray(indices, dtype=np.int64)
        sorted_distances = np.asarray(distances, dtype=handle.dtype)
        return _format_knn_output(sorted_indices, sorted_distances, return_distances=return_distances)


# -----------------------------------------------------------------------------
# PCCT-ish Rust Engine (Hilbert order, residual metric)
# -----------------------------------------------------------------------------


class RustPcctEngine:
    """
    Residual-only engine that builds the Rust cover tree using a Hilbert-like
    insertion order to mirror the PCCT batch ordering. It keeps the same index
    payload semantics as python-numba and reuses the Rust residual metric.
    """

    name = "rust-pcct"

    def _ensure_backend(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        residual_params: Mapping[str, Any] | None,
    ):
        params = dict(residual_params or {})
        variance = float(params.get("variance", 1.0))
        lengthscale = params.get("lengthscale", 1.0)
        inducing = int(params.get("inducing_count", 512))
        chunk_size = int(params.get("chunk_size", 512))

        seed_pack = runtime.seeds
        residual_seed = seed_pack.resolved("residual_grid", fallback=seed_pack.resolved("mis"))
        backend = build_residual_backend(
            np.asarray(points, dtype=np.float64),
            seed=residual_seed,
            inducing_count=inducing,
            variance=variance,
            lengthscale=lengthscale,
            chunk_size=chunk_size,
        )
        return backend, variance, lengthscale, chunk_size

    def build(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        context: cx_config.RuntimeContext | None = None,
        batch_size: int = 512,
        seed: int = 0,
        build_mode: str = "batch",
        log_writer: Any | None = None,
        scope_cap_recorder: Any | None = None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
        residual_backend: Any | None = None,
        residual_params: Mapping[str, Any] | None = None,
    ) -> CoverTree:
        if runtime.metric != "residual_correlation":
            raise ValueError("rust-pcct engine only supports the residual_correlation metric.")

        backend_mod = _enable_rust_debug_stats_if_requested()
        if backend_mod is None:
            try:
                import covertreex_backend as backend_mod  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("covertreex_backend is not installed.") from exc

        ctx = context or cx_config.configure_runtime(runtime)

        if residual_backend is None:
            backend, variance, lengthscale, chunk_size = self._ensure_backend(
                points,
                runtime=runtime,
                residual_params=residual_params,
            )
        else:
            backend = residual_backend
            params = dict(residual_params or {})
            variance = float(params.get("variance", 1.0))
            lengthscale = params.get("lengthscale", 1.0)
            chunk_size = int(params.get("chunk_size", 512))

        configure_residual_correlation(backend, context=ctx)

        dtype = np.float32
        coords = np.asarray(getattr(backend, "kernel_points_f32", backend.v_matrix), dtype=dtype)
        v_matrix = np.asarray(backend.v_matrix, dtype=dtype)
        p_diag = np.asarray(backend.p_diag, dtype=dtype)
        rbf_var = float(getattr(backend, "rbf_variance", variance if variance is not None else 1.0))
        rbf_ls = np.asarray(
            getattr(
                backend,
                "rbf_lengthscale",
                np.ones(coords.shape[1], dtype=dtype),
            ),
            dtype=dtype,
        )

        start = time.perf_counter()
        tree_handle, node_to_dataset = backend_mod.build_pcct_residual_tree(  # type: ignore
            v_matrix,
            p_diag,
            coords,
            rbf_var,
            rbf_ls,
            chunk_size or batch_size,
            "hilbert",
        )
        build_seconds = time.perf_counter() - start

        handle = RustPcctHandle(
            tree=tree_handle,
            node_to_dataset=list(map(int, node_to_dataset)),
            v_matrix=v_matrix,
            p_diag=p_diag,
            coords=coords,
            rbf_variance=rbf_var,
            rbf_lengthscale=rbf_ls,
            dtype=dtype,
        )
        meta = {
            "build_seconds": build_seconds,
            "dataset_size": coords.shape[0],
            "node_to_dataset": handle.node_to_dataset,
            "backend_dtype": str(dtype),
            "chunk_size": chunk_size or batch_size,
            "batch_order": "hilbert-morton",
        }
        backend_tag = TreeBackend.numpy(precision="float32")
        return CoverTree(
            engine=self,
            handle=handle,
            metric=runtime.metric,
            backend=backend_tag,
            meta=meta,
        )

    def knn(
        self,
        tree: CoverTree,
        query_points: Any,
        *,
        k: int,
        return_distances: bool,
        context: cx_config.RuntimeContext,
        runtime: RuntimeConfig,
    ) -> Any:
        handle: RustPcctHandle = tree.handle
        queries = np.asarray(query_points)
        if queries.ndim == 1:
            queries = queries.reshape(-1, 1)
        if queries.dtype.kind not in {"i", "u"}:
            raise ValueError("rust-pcct engine expects integer query payloads representing dataset indices.")
        query_indices = np.asarray(queries, dtype=np.int64).reshape(-1)
        # Block-scanned residual query; bypass cover tree traversal for now.
        indices, distances = handle.tree.knn_query_residual_block(
            query_indices,
            handle.node_to_dataset,
            handle.v_matrix,
            handle.p_diag,
            handle.coords,
            float(handle.rbf_variance),
            np.asarray(handle.rbf_lengthscale, dtype=handle.dtype),
            int(k),
        )
        sorted_indices = np.asarray(indices, dtype=np.int64)
        sorted_distances = np.asarray(distances, dtype=handle.dtype)
        return _format_knn_output(sorted_indices, sorted_distances, return_distances=return_distances)


class RustHilbertEngine:
    """
    Opt-in residual engine that mirrors PCCT semantics (telemetry, Hilbert order)
    but runs the build/query path in Rust. Preserves index-payload surface.
    """

    name = "rust-hilbert"

    def _ensure_backend(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        residual_params: Mapping[str, Any] | None,
    ):
        params = dict(residual_params or {})
        variance = float(params.get("variance", 1.0))
        lengthscale = params.get("lengthscale", 1.0)
        inducing = int(params.get("inducing_count", 512))
        chunk_size = int(params.get("chunk_size", 512))

        seed_pack = runtime.seeds
        residual_seed = seed_pack.resolved("residual_grid", fallback=seed_pack.resolved("mis"))
        backend = build_residual_backend(
            np.asarray(points, dtype=np.float64),
            seed=residual_seed,
            inducing_count=inducing,
            variance=variance,
            lengthscale=lengthscale,
            chunk_size=chunk_size,
        )
        return backend, variance, lengthscale, chunk_size

    def build(
        self,
        points: np.ndarray,
        *,
        runtime: RuntimeConfig,
        context: cx_config.RuntimeContext | None = None,
        batch_size: int = 512,
        seed: int = 0,  # seed unused but kept for parity
        build_mode: str = "batch",
        log_writer: Any | None = None,
        scope_cap_recorder: Any | None = None,
        plan_callback: Callable[[Any, int, int], Mapping[str, Any] | None] | None = None,
        residual_backend: Any | None = None,
        residual_params: Mapping[str, Any] | None = None,
    ) -> CoverTree:
        if runtime.metric != "residual_correlation":
            raise ValueError("rust-hilbert engine only supports the residual_correlation metric.")

        backend_mod = _enable_rust_debug_stats_if_requested()
        if backend_mod is None:
            try:
                import covertreex_backend as backend_mod  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("covertreex_backend is not installed.") from exc

        ctx = context or cx_config.configure_runtime(runtime)

        if residual_backend is None:
            backend, variance, lengthscale, chunk_size = self._ensure_backend(
                points,
                runtime=runtime,
                residual_params=residual_params,
            )
        else:
            backend = residual_backend
            params = dict(residual_params or {})
            variance = float(params.get("variance", 1.0))
            lengthscale = params.get("lengthscale", 1.0)
            chunk_size = int(params.get("chunk_size", 512))

        configure_residual_correlation(backend, context=ctx)

        dtype = np.float32
        coords = np.asarray(getattr(backend, "kernel_points_f32", backend.v_matrix), dtype=dtype)
        v_matrix = np.asarray(backend.v_matrix, dtype=dtype)
        p_diag = np.asarray(backend.p_diag, dtype=dtype)
        rbf_var = float(getattr(backend, "rbf_variance", variance if variance is not None else 1.0))
        rbf_ls = np.asarray(
            getattr(
                backend,
                "rbf_lengthscale",
                np.ones(coords.shape[1], dtype=dtype),
            ),
            dtype=dtype,
        )

        start = time.perf_counter()
        batch_order = str(getattr(runtime, "batch_order_strategy", "hilbert") or "hilbert")
        tree_handle, node_to_dataset = backend_mod.build_pcct2_residual_tree(  # type: ignore
            v_matrix,
            p_diag,
            coords,
            rbf_var,
            rbf_ls,
            chunk_size or batch_size,
            batch_order,
            log_writer,
            float(getattr(runtime, "residual_grid_whiten_scale", 1.0)),
            int(runtime.scope_chunk_target) if getattr(runtime, "scope_chunk_target", 0) and runtime.scope_chunk_target > 0 else None,
            int(runtime.conflict_degree_cap) if getattr(runtime, "conflict_degree_cap", 0) and runtime.conflict_degree_cap > 0 else None,
            list(runtime.scope_budget_schedule) if getattr(runtime, "scope_budget_schedule", ()) else None,
            float(runtime.scope_budget_up_thresh) if getattr(runtime, "scope_budget_schedule", ()) else None,
            float(runtime.scope_budget_down_thresh) if getattr(runtime, "scope_budget_schedule", ()) else None,
            bool(runtime.residual_masked_scope_append),
        )
        build_seconds = time.perf_counter() - start

        handle = RustPcct2Handle(
            tree=tree_handle,
            node_to_dataset=list(map(int, node_to_dataset)),
            v_matrix=v_matrix,
            p_diag=p_diag,
            coords=coords,
            rbf_variance=rbf_var,
            rbf_lengthscale=rbf_ls,
            dtype=dtype,
        )
        meta = {
            "build_seconds": build_seconds,
            "dataset_size": coords.shape[0],
            "node_to_dataset": handle.node_to_dataset,
            "backend_dtype": str(dtype),
            "chunk_size": chunk_size or batch_size,
            "batch_order": "hilbert-morton",
        }
        backend_tag = TreeBackend.numpy(precision="float32")
        return CoverTree(
            engine=self,
            handle=handle,
            metric=runtime.metric,
            backend=backend_tag,
            meta=meta,
        )

    def knn(
        self,
        tree: CoverTree,
        query_points: Any,
        *,
        k: int,
        return_distances: bool,
        context: cx_config.RuntimeContext,
        runtime: RuntimeConfig,
    ) -> Any:
        handle: RustPcct2Handle = tree.handle  # type: ignore
        queries = np.asarray(query_points)
        if queries.ndim == 1:
            queries = queries.reshape(-1, 1)
        if queries.dtype.kind not in {"i", "u"}:
            raise ValueError("rust-hilbert engine expects integer query payloads representing dataset indices.")
        query_indices = np.asarray(queries, dtype=np.int64).reshape(-1)

        indices, distances = handle.tree.knn_query_residual(
            query_indices,
            handle.node_to_dataset,
            handle.v_matrix,
            handle.p_diag,
            handle.coords,
            float(handle.rbf_variance),
            np.asarray(handle.rbf_lengthscale, dtype=handle.dtype),
            int(k),
        )

        # Pull telemetry when enabled so it lands in op logs.
        try:
            telem = handle.tree.last_query_telemetry()
        except Exception:
            telem = None
        if telem is not None:
            try:
                # Attach to the active op log if present
                op_log = getattr(context, "op_log", None)
                if op_log is not None and hasattr(op_log, "add_metadata"):
                    op_log.add_metadata(rust_query_telemetry=telem)
            except Exception:
                pass
            try:
                handle.tree.clear_last_query_telemetry()
            except Exception:
                pass

        sorted_indices = np.asarray(indices, dtype=np.int64)
        sorted_distances = np.asarray(distances, dtype=handle.dtype)
        return _format_knn_output(sorted_indices, sorted_distances, return_distances=return_distances)

# Instantiate engines and registry
_PYTHON_ENGINE = PythonNumbaEngine()
_RUST_FAST_ENGINE = RustNaturalEngine()
_RUST_HYBRID_ENGINE = RustHybridResidualEngine()
_RUST_PCCT2_ENGINE = RustHilbertEngine()
_ENGINE_REGISTRY: Dict[str, TreeEngine] = {
    _PYTHON_ENGINE.name: _PYTHON_ENGINE,
    _RUST_FAST_ENGINE.name: _RUST_FAST_ENGINE,
    _RUST_HYBRID_ENGINE.name: _RUST_HYBRID_ENGINE,
    _RUST_PCCT2_ENGINE.name: _RUST_PCCT2_ENGINE,
}

__all__ = [
    "CoverTree",
    "TreeEngine",
    "DEFAULT_ENGINE",
    "SUPPORTED_ENGINES",
    "build_tree",
    "get_engine",
    "PythonNumbaEngine",
    "RustNaturalEngine",
    "RustHybridResidualEngine",
    "RustHilbertEngine",
]
