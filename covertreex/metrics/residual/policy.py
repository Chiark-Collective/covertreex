from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from covertreex import config as cx_config
from ._gate_profile_numba import update_quantile_reservoir

RESIDUAL_EPS = 1e-9
_DEFAULT_PROFILE_SAMPLE_CAP = 2048
_DEFAULT_PROFILE_QUANTILES = (95.0, 99.0, 99.9)


def _normalise_percentiles(values: Sequence[float] | None) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    # Accept either fractions (0..1) or percentages (0..100).
    if np.any(arr > 1.0):
        arr = arr / 100.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.unique(arr)
    return arr


def _format_percentile(value: float) -> str:
    percent = value * 100.0
    if percent.is_integer():
        return f"{int(percent)}"
    return f"{percent:.3f}".rstrip("0").rstrip(".")


@dataclass
class ResidualGateTelemetry:
    candidates: int = 0
    kept: int = 0
    pruned: int = 0
    seconds: float = 0.0

    def snapshot(self) -> Tuple[int, int, int, float]:
        return (self.candidates, self.kept, self.pruned, self.seconds)

    def delta(self, snapshot: Tuple[int, int, int, float]) -> "ResidualGateTelemetry":
        cand0, kept0, pruned0, seconds0 = snapshot
        return ResidualGateTelemetry(
            candidates=self.candidates - cand0,
            kept=self.kept - kept0,
            pruned=self.pruned - pruned0,
            seconds=self.seconds - seconds0,
        )


@dataclass
class ResidualGateProfile:
    bin_edges: np.ndarray
    max_whitened: np.ndarray
    max_ratio: np.ndarray
    counts: np.ndarray
    samples_total: int = 0
    false_negative_samples: int = 0
    radius_eps: float = RESIDUAL_EPS
    path: Path | None = None
    dirty: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    quantile_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    quantile_labels: Tuple[str, ...] = ()
    quantile_capacity: int = 0
    quantile_samples: np.ndarray | None = None
    quantile_sample_counts: np.ndarray | None = None
    quantile_total_counts: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        *,
        bins: int,
        radius_max: float,
        path: str | Path | None,
        radius_eps: float = RESIDUAL_EPS,
        quantile_percentiles: Sequence[float] | None = None,
        quantile_sample_cap: int = _DEFAULT_PROFILE_SAMPLE_CAP,
    ) -> "ResidualGateProfile":
        edges = np.linspace(0.0, radius_max, bins + 1, dtype=np.float64)
        size = max(edges.size - 1, 1)
        target = Path(path).expanduser() if path else None
        targets = quantile_percentiles if quantile_percentiles is not None else _DEFAULT_PROFILE_QUANTILES
        percentiles = _normalise_percentiles(targets)
        capacity = max(int(quantile_sample_cap), 0)
        labels = tuple(_format_percentile(p) for p in percentiles)
        if percentiles.size == 0 or capacity == 0:
            samples = None
            sample_counts = None
            total_counts = None
        else:
            samples = np.full((size, capacity), np.nan, dtype=np.float32)
            sample_counts = np.zeros(size, dtype=np.int64)
            total_counts = np.zeros(size, dtype=np.int64)
        return cls(
            bin_edges=edges,
            max_whitened=np.zeros(size, dtype=np.float64),
            max_ratio=np.zeros(size, dtype=np.float64),
            counts=np.zeros(size, dtype=np.int64),
            radius_eps=radius_eps,
            path=target,
            quantile_percentiles=percentiles,
            quantile_labels=labels,
            quantile_capacity=capacity,
            quantile_samples=samples,
            quantile_sample_counts=sample_counts,
            quantile_total_counts=total_counts,
        )

    def _bin_indices(self, values: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self.bin_edges, values, side="right") - 1
        return np.clip(indices, 0, self.max_whitened.size - 1)

    def _record_quantiles(self, idx: np.ndarray, whitened: np.ndarray) -> None:
        if (
            self.quantile_samples is None
            or self.quantile_sample_counts is None
            or self.quantile_total_counts is None
        ):
            return
        if idx.size == 0 or whitened.size == 0:
            return
        bin_idx = np.asarray(idx, dtype=np.int64)
        values = np.asarray(whitened, dtype=np.float32)
        update_quantile_reservoir(
            bin_idx,
            values,
            self.quantile_samples,
            self.quantile_sample_counts,
            self.quantile_total_counts,
        )
        self.dirty = True

    def record_chunk(
        self,
        *,
        residual_distances: np.ndarray,
        whitened_distances: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> None:
        if whitened_distances is None or residual_distances.size == 0:
            return
        mask = inclusion_mask.astype(bool, copy=False)
        if not np.any(mask):
            return
        residual = np.asarray(residual_distances[mask], dtype=np.float64)
        whitened = np.asarray(whitened_distances[mask], dtype=np.float64)
        residual = np.clip(residual, self.radius_eps, None)
        idx = self._bin_indices(residual)
        np.maximum.at(self.max_whitened, idx, whitened)
        np.maximum.at(self.max_ratio, idx, np.divide(whitened, residual))
        np.add.at(self.counts, idx, 1)
        self._record_quantiles(idx, whitened)
        self.samples_total += whitened.size
        self.dirty = True

    def record_false_negatives(
        self,
        *,
        residual_distances: np.ndarray,
        whitened_distances: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> None:
        self.record_chunk(
            residual_distances=residual_distances,
            whitened_distances=whitened_distances,
            inclusion_mask=inclusion_mask,
        )
        self.false_negative_samples += int(np.count_nonzero(inclusion_mask))

    def annotate_metadata(self, **metadata: Any) -> None:
        for key, value in metadata.items():
            if value is None:
                continue
            self.metadata[str(key)] = value
        self.dirty = True

    def _cumulative_whitened(self) -> np.ndarray:
        return np.maximum.accumulate(self.max_whitened)

    def _cumulative_ratio(self) -> np.ndarray:
        return np.maximum.accumulate(self.max_ratio)

    def _quantile_payload(self) -> Dict[str, Any] | None:
        if (
            self.quantile_samples is None
            or self.quantile_sample_counts is None
            or self.quantile_total_counts is None
            or self.quantile_percentiles.size == 0
        ):
            return None
        payload: Dict[str, Any] = {}
        for pct, label in zip(self.quantile_percentiles, self.quantile_labels):
            values = np.zeros_like(self.max_whitened, dtype=np.float64)
            for idx in range(values.size):
                count = int(self.quantile_sample_counts[idx])
                if count <= 0:
                    values[idx] = 0.0
                    continue
                samples = self.quantile_samples[idx, :count].astype(np.float64, copy=False)
                values[idx] = float(np.quantile(samples, pct))
            payload[label] = np.maximum.accumulate(values).tolist()
        return payload

    def to_dict(self) -> dict:
        payload = {
            "schema": 2,
            "radius_bin_edges": self.bin_edges.tolist(),
            "max_whitened": self._cumulative_whitened().tolist(),
            "max_ratio": self._cumulative_ratio().tolist(),
            "counts": self.counts.tolist(),
            "samples_total": int(self.samples_total),
            "false_negative_samples": int(self.false_negative_samples),
            "radius_eps": float(self.radius_eps),
            "metadata": dict(self.metadata),
        }
        quantiles = self._quantile_payload()
        if quantiles is not None:
            payload["quantiles"] = quantiles
            payload["quantile_percentiles"] = (self.quantile_percentiles * 100.0).tolist()
            payload["quantile_capacity"] = int(self.quantile_capacity)
            payload["quantile_counts"] = (
                self.quantile_sample_counts.astype(np.int64).tolist()
                if self.quantile_sample_counts is not None
                else []
            )
            payload["quantile_totals"] = (
                self.quantile_total_counts.astype(np.int64).tolist()
                if self.quantile_total_counts is not None
                else []
            )
        return payload

    def dump(self, path: str | Path | None = None, *, force: bool = False) -> None:
        target = Path(path).expanduser() if path else self.path
        if target is None or (not self.dirty and not force):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["generated_at"] = float(time.time())
        payload["bins"] = int(self.max_whitened.size)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.path = target
        self.dirty = False


@dataclass
class ResidualGateLookup:
    radius_bins: np.ndarray
    keep_thresholds: np.ndarray
    prune_thresholds: np.ndarray
    margin: float

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        *,
        margin: float,
        keep_pct: float,
        prune_pct: float,
    ) -> "ResidualGateLookup":
        edges = np.asarray(payload.get("radius_bin_edges", []), dtype=np.float64)
        maxima = np.asarray(payload.get("max_whitened", []), dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Residual gate lookup file is missing 'radius_bin_edges'.")
        bins = edges[1:]
        if maxima.size != bins.size:
            raise ValueError("Residual gate lookup file has mismatched 'max_whitened' length.")
        quantiles = payload.get("quantiles", {}) or {}
        quantile_map: Dict[float, np.ndarray] = {}
        for label, values in quantiles.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size != bins.size:
                continue
            try:
                percent = float(label)
            except (TypeError, ValueError):
                continue
            quantile_map[percent] = arr

        def _resolve(percent: float, default: np.ndarray) -> np.ndarray:
            key = round(percent, 6)
            if key in quantile_map:
                return np.maximum.accumulate(quantile_map[key].astype(np.float64, copy=False))
            return default

        base_keep = np.zeros_like(maxima, dtype=np.float64)
        base_prune = np.maximum.accumulate(maxima.astype(np.float64, copy=False))
        keep_values = _resolve(keep_pct, base_keep)
        prune_values = _resolve(prune_pct, base_prune)
        return cls(
            radius_bins=bins,
            keep_thresholds=keep_values,
            prune_thresholds=np.maximum(prune_values, keep_values),
            margin=float(margin),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        margin: float,
        keep_pct: float,
        prune_pct: float,
    ) -> "ResidualGateLookup":
        target = Path(path).expanduser()
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_payload(payload, margin=margin, keep_pct=keep_pct, prune_pct=prune_pct)

    def thresholds(self, radius: float) -> tuple[float, float]:
        if self.radius_bins.size == 0:
            return 0.0, 0.0
        clipped = np.clip(radius, 0.0, float(self.radius_bins[-1]))
        idx = np.searchsorted(self.radius_bins, clipped, side="right")
        if idx <= 0:
            idx = 0
        else:
            idx = min(idx - 1, self.keep_thresholds.size - 1)
        keep = float(self.keep_thresholds[idx])
        prune = float(self.prune_thresholds[idx])
        return keep, prune


@dataclass(frozen=True)
class ResidualPolicy:
    gate1_enabled: bool
    gate1_alpha: float
    gate1_margin: float
    gate1_eps: float
    gate1_audit: bool
    gate1_radius_cap: float
    gate1_band_eps: float
    gate1_keep_pct: float
    gate1_prune_pct: float
    gate1_profile_path: str | None
    gate1_profile_bins: int
    gate1_lookup_path: str | None
    gate1_lookup_margin: float
    radius_floor: float
    scope_cap_path: str | None
    scope_cap_default: float
    prefilter_enabled: bool
    prefilter_lookup_path: str | None
    prefilter_margin: float
    prefilter_radius_cap: float
    prefilter_audit: bool

    @classmethod
    def from_runtime(cls, runtime: cx_config.RuntimeConfig) -> "ResidualPolicy":
        return cls(
            gate1_enabled=runtime.residual_gate1_enabled,
            gate1_alpha=runtime.residual_gate1_alpha,
            gate1_margin=runtime.residual_gate1_margin,
            gate1_eps=runtime.residual_gate1_eps,
            gate1_audit=runtime.residual_gate1_audit,
            gate1_radius_cap=runtime.residual_gate1_radius_cap,
            gate1_band_eps=runtime.residual_gate1_band_eps,
            gate1_keep_pct=runtime.residual_gate1_keep_pct,
            gate1_prune_pct=runtime.residual_gate1_prune_pct,
            gate1_profile_path=runtime.residual_gate1_profile_path,
            gate1_profile_bins=runtime.residual_gate1_profile_bins,
            gate1_lookup_path=runtime.residual_gate1_lookup_path,
            gate1_lookup_margin=runtime.residual_gate1_lookup_margin,
            radius_floor=runtime.residual_radius_floor,
            scope_cap_path=runtime.residual_scope_cap_path,
            scope_cap_default=runtime.residual_scope_cap_default,
            prefilter_enabled=runtime.residual_prefilter_enabled,
            prefilter_lookup_path=runtime.residual_prefilter_lookup_path,
            prefilter_margin=runtime.residual_prefilter_margin,
            prefilter_radius_cap=runtime.residual_prefilter_radius_cap,
            prefilter_audit=runtime.residual_prefilter_audit,
        )


def get_residual_policy(
    runtime: cx_config.RuntimeConfig | None = None,
) -> ResidualPolicy:
    ctx = runtime or cx_config.runtime_config()
    return ResidualPolicy.from_runtime(ctx)


__all__ = [
    "RESIDUAL_EPS",
    "ResidualGateTelemetry",
    "ResidualGateProfile",
    "ResidualGateLookup",
    "ResidualPolicy",
    "get_residual_policy",
]
