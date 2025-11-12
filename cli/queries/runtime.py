from __future__ import annotations

import os
from typing import Any, Dict

from covertreex import config as cx_config
from covertreex.core.tree import TreeBackend
from covertreex.telemetry import resolve_artifact_path


def _gate_active_for_backend(host_backend: Any | None) -> bool:
    if host_backend is None:
        return False
    if not bool(getattr(host_backend, "gate1_enabled", False)):
        return False
    radius_cap = getattr(host_backend, "gate1_radius_cap", None)
    if radius_cap is not None and radius_cap <= 0.0:
        return False
    gate_vectors = getattr(host_backend, "gate_v32", None)
    lookup = getattr(host_backend, "gate_lookup", None)
    return gate_vectors is not None or lookup is not None


def _ensure_thread_env_defaults() -> Dict[str, str]:
    cores = max(1, os.cpu_count() or 1)
    defaults = {
        "MKL_NUM_THREADS": str(cores),
        "OPENBLAS_NUM_THREADS": str(cores),
        "OMP_NUM_THREADS": str(cores),
        "NUMBA_NUM_THREADS": str(cores),
    }
    applied: Dict[str, str] = {}
    for key, value in defaults.items():
        current = os.environ.get(key)
        if current and current.strip():
            applied[key] = current
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def _thread_env_snapshot() -> Dict[str, str]:
    def _value(*names: str) -> str:
        for name in names:
            val = os.environ.get(name)
            if val:
                return val
        return "auto"

    return {
        "blas_threads": _value("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"),
        "numba_threads": os.environ.get("NUMBA_NUM_THREADS", "auto"),
    }


def _emit_engine_banner(engine: str, gate_active: bool, threads: Dict[str, str]) -> None:
    gate_state = "on" if gate_active else "off"
    blas_val = threads.get("blas_threads", "auto")
    numba_val = threads.get("numba_threads", "auto")
    print(
        f"[queries] engine={engine} gate={gate_state} "
        f"blas_threads={blas_val} numba_threads={numba_val}"
    )


def _resolve_backend() -> TreeBackend:
    runtime = cx_config.runtime_config()
    if runtime.backend == "jax":
        return TreeBackend.jax(precision=runtime.precision)
    if runtime.backend == "numpy":
        return TreeBackend.numpy(precision=runtime.precision)
    raise NotImplementedError(f"Backend '{runtime.backend}' is not supported yet.")


def _resolve_artifact_arg(path: str | None, *, category: str = "benchmarks") -> str | None:
    if not path:
        return None
    resolved = resolve_artifact_path(path, category=category)
    return str(resolved)


__all__ = [
    "_gate_active_for_backend",
    "_ensure_thread_env_defaults",
    "_thread_env_snapshot",
    "_emit_engine_banner",
    "_resolve_backend",
    "_resolve_artifact_arg",
]
