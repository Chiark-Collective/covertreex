from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import math
import os
import resource
import subprocess
import time
from typing import Dict, Iterator, Optional

from covertreex import config as cx_config


@dataclass(frozen=True)
class _ProcessSnapshot:
    timestamp: float
    wall: float
    cpu_user: Optional[float]
    cpu_system: Optional[float]
    max_rss_bytes: Optional[int]
    rss_bytes: Optional[int]
    gpu_used_bytes: Optional[int]
    gpu_total_bytes: Optional[int]


def _now_snapshot(*, with_resources: bool) -> _ProcessSnapshot:
    wall = time.perf_counter()
    timestamp = time.time()
    if not with_resources:
        return _ProcessSnapshot(
            timestamp=timestamp,
            wall=wall,
            cpu_user=None,
            cpu_system=None,
            max_rss_bytes=None,
            rss_bytes=None,
            gpu_used_bytes=None,
            gpu_total_bytes=None,
        )

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_bytes = int(usage.ru_maxrss * 1024) if usage.ru_maxrss else 0
    rss_bytes = _read_rss_bytes()
    gpu_used, gpu_total = _query_gpu_memory()
    return _ProcessSnapshot(
        timestamp=timestamp,
        wall=wall,
        cpu_user=float(usage.ru_utime),
        cpu_system=float(usage.ru_stime),
        max_rss_bytes=max_rss_bytes,
        rss_bytes=rss_bytes,
        gpu_used_bytes=gpu_used,
        gpu_total_bytes=gpu_total,
    )


def _read_rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            contents = handle.readline().strip().split()
        if len(contents) < 2:
            return None
        rss_pages = int(contents[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(rss_pages * page_size)
    except (OSError, ValueError):
        return None


_GPU_FAILURE = False


def _query_gpu_memory() -> tuple[Optional[int], Optional[int]]:
    if _GPU_FAILURE:
        return None, None

    runtime = cx_config.runtime_config()
    device = next((dev for dev in runtime.devices if dev.startswith("gpu:")), None)
    if device is None:
        return None, None

    try:
        _, index_str = device.split(":", 1)
        index = index_str.strip()
    except ValueError:
        index = "0"

    cmd = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        _mark_gpu_failure()
        return None, None

    if proc.returncode != 0 or not proc.stdout.strip():
        _mark_gpu_failure()
        return None, None

    try:
        used_str, total_str = proc.stdout.strip().split(",")
        used_bytes = int(float(used_str.strip())) * 1024 * 1024
        total_bytes = int(float(total_str.strip())) * 1024 * 1024
        return used_bytes, total_bytes
    except (ValueError, IndexError):
        _mark_gpu_failure()
        return None, None


def _mark_gpu_failure() -> None:
    global _GPU_FAILURE
    _GPU_FAILURE = True


def _format_bytes(value: Optional[int]) -> str:
    if value is None or value < 0:
        return "NA"
    if value == 0:
        return "0"
    units = ["B", "KB", "MB", "GB", "TB"]
    log_index = min(int(math.log(value, 1024)), len(units) - 1)
    scaled = value / (1024 ** log_index)
    return f"{scaled:.2f}{units[log_index]}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _diff_optional(end: Optional[float], start: Optional[float], *, scale: float = 1.0) -> Optional[float]:
    if end is None or start is None:
        return None
    return (end - start) * scale


def _max_optional(*values: Optional[int]) -> Optional[int]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


@dataclass
class OperationMetrics:
    label: str
    logger: logging.Logger
    level: int
    metadata: Dict[str, object] = field(default_factory=dict)
    _start: _ProcessSnapshot = field(init=False)
    _end: Optional[_ProcessSnapshot] = field(default=None, init=False)
    _error: Optional[BaseException] = field(default=None, init=False)
    collect_resources: bool = field(default=True, init=False)

    def __enter__(self) -> "OperationMetrics":
        self.collect_resources = bool(cx_config.runtime_config().enable_diagnostics)
        self._start = _now_snapshot(with_resources=self.collect_resources)
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        self._end = _now_snapshot(with_resources=self.collect_resources)
        if exc is not None:
            self._error = exc
        self._emit()

    def add_metadata(self, **kwargs: object) -> None:
        self.metadata.update(kwargs)

    def _emit(self) -> None:
        metrics = self._compute_metrics()
        status = "error" if self._error else "ok"
        fields = {"op": self.label, "status": status, **metrics, **self.metadata}
        message = " ".join(f"{key}={_stringify(value)}" for key, value in fields.items())
        self.logger.log(self.level, message)

    def _compute_metrics(self) -> Dict[str, object]:
        end = self._end or _now_snapshot(with_resources=self.collect_resources)
        start = self._start
        wall_ms = (end.wall - start.wall) * 1000.0
        metrics: Dict[str, object] = {"wall_ms": _format_float(wall_ms)}

        if not self.collect_resources:
            metrics.update(
                {
                    "cpu_user_ms": "NA",
                    "cpu_system_ms": "NA",
                    "rss_delta": "NA",
                    "max_rss": "NA",
                    "gpu_used": "NA",
                    "gpu_delta": "NA",
                }
            )
            return metrics

        cpu_user_ms = _diff_optional(end.cpu_user, start.cpu_user, scale=1000.0)
        cpu_system_ms = _diff_optional(end.cpu_system, start.cpu_system, scale=1000.0)
        max_rss = _max_optional(end.max_rss_bytes, start.max_rss_bytes)
        rss_delta = _diff_optional(end.rss_bytes, start.rss_bytes)
        gpu_used = end.gpu_used_bytes
        gpu_delta = _diff_optional(end.gpu_used_bytes, start.gpu_used_bytes)

        metrics.update(
            {
                "cpu_user_ms": _format_float(cpu_user_ms) if cpu_user_ms is not None else "NA",
                "cpu_system_ms": _format_float(cpu_system_ms) if cpu_system_ms is not None else "NA",
                "rss_delta": _format_bytes(int(rss_delta)) if rss_delta is not None else "NA",
                "max_rss": _format_bytes(max_rss),
                "gpu_used": _format_bytes(gpu_used),
                "gpu_delta": _format_bytes(int(gpu_delta)) if gpu_delta is not None else "NA",
            }
        )
        return metrics


def _stringify(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, (int, str)):
        return str(value)
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    return repr(value)


@contextmanager
def log_operation(
    logger: logging.Logger,
    label: str,
    *,
    level: int = logging.INFO,
) -> Iterator[OperationMetrics]:
    context = OperationMetrics(label=label, logger=logger, level=level)
    with context:
        yield context


__all__ = ["log_operation"]
