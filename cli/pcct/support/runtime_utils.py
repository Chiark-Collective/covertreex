from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from covertreex import config as cx_config
from covertreex.core.tree import TreeBackend
from covertreex.telemetry import artifact_root, resolve_artifact_path


def ensure_thread_env_defaults() -> Dict[str, str]:
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


def thread_env_snapshot() -> Dict[str, str]:
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


def emit_engine_banner(engine: str, threads: Dict[str, str]) -> None:
    blas_val = threads.get("blas_threads", "auto")
    numba_val = threads.get("numba_threads", "auto")
    print(
        f"[queries] engine={engine} "
        f"blas_threads={blas_val} numba_threads={numba_val}"
    )


def resolve_runtime_config(
    *, 
    context: cx_config.RuntimeContext | None = None,
    runtime: cx_config.RuntimeConfig | None = None,
) -> cx_config.RuntimeConfig:
    if runtime is not None:
        return runtime
    if context is not None:
        return context.config
    active = cx_config.current_runtime_context()
    if active is not None:
        return active.config
    return cx_config.RuntimeConfig.from_env()


def resolve_backend(
    *, 
    context: cx_config.RuntimeContext | None = None,
    runtime: cx_config.RuntimeConfig | None = None,
) -> TreeBackend:
    cfg = resolve_runtime_config(context=context, runtime=runtime)
    if cfg.backend == "jax":
        return TreeBackend.jax(precision=cfg.precision)
    if cfg.backend == "numpy":
        return TreeBackend.numpy(precision=cfg.precision)
    raise NotImplementedError(f"Backend '{cfg.backend}' is not supported yet.")


def resolve_artifact_arg(path: str | None, *, category: str = "benchmarks") -> str | None:
    if not path:
        return None
    raw = Path(path).expanduser()
    if raw.is_absolute():
        raw.parent.mkdir(parents=True, exist_ok=True)
        return str(raw)

    root = artifact_root()
    parts = raw.parts
    root_aliases = {root.name, "artifacts"}
    if parts and parts[0] in root_aliases:
        remainder = Path(*parts[1:]) if len(parts) > 1 else Path()
        resolved = root if remainder == Path() else root / remainder
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    resolved = resolve_artifact_path(raw, category=category)
    return str(resolved)


def validate_residual_runtime(snapshot: Dict[str, Any]) -> None:
    errors = []
    backend = snapshot.get("backend")
    if backend != "numpy":
        errors.append(f"expected backend 'numpy' but runtime selected {backend!r}")
    if not snapshot.get("enable_numba"):
        errors.append(
            "Numba acceleration is required for residual traversal; enable it by leaving "
            "COVERTREEX_ENABLE_NUMBA unset or passing --enable-numba=1."
        )
    if errors:
        joined = "\n - ".join(errors)
        raise RuntimeError(
            "Residual metric requires the residual traversal engine; unable to satisfy the"
            f" prerequisites:\n - {joined}"
        )
