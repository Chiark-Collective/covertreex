from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

from covertreex import config as cx_config

RESIDUAL_EPS = 1e-9


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

    @classmethod
    def create(
        cls,
        *,
        bins: int,
        radius_max: float,
        path: str | Path | None,
        radius_eps: float = RESIDUAL_EPS,
    ) -> "ResidualGateProfile":
        edges = np.linspace(0.0, radius_max, bins + 1, dtype=np.float64)
        size = max(edges.size - 1, 1)
        target = Path(path).expanduser() if path else None
        return cls(
            bin_edges=edges,
            max_whitened=np.zeros(size, dtype=np.float64),
            max_ratio=np.zeros(size, dtype=np.float64),
            counts=np.zeros(size, dtype=np.int64),
            radius_eps=radius_eps,
            path=target,
        )

    def _bin_indices(self, values: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self.bin_edges, values, side="right") - 1
        return np.clip(indices, 0, self.max_whitened.size - 1)

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
        residual = np.asarray(residual_distances[mask], dtype=np.float64, copy=False)
        whitened = np.asarray(whitened_distances[mask], dtype=np.float64, copy=False)
        residual = np.clip(residual, self.radius_eps, None)
        idx = self._bin_indices(residual)
        np.maximum.at(self.max_whitened, idx, whitened)
        np.maximum.at(self.max_ratio, idx, np.divide(whitened, residual))
        np.add.at(self.counts, idx, 1)
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

    def _cumulative_whitened(self) -> np.ndarray:
        return np.maximum.accumulate(self.max_whitened)

    def _cumulative_ratio(self) -> np.ndarray:
        return np.maximum.accumulate(self.max_ratio)

    def to_dict(self) -> dict:
        payload = {
            "schema": 1,
            "radius_bin_edges": self.bin_edges.tolist(),
            "max_whitened": self._cumulative_whitened().tolist(),
            "max_ratio": self._cumulative_ratio().tolist(),
            "counts": self.counts.tolist(),
            "samples_total": int(self.samples_total),
            "false_negative_samples": int(self.false_negative_samples),
            "radius_eps": float(self.radius_eps),
        }
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
    thresholds: np.ndarray
    margin: float

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        *,
        margin: float,
    ) -> "ResidualGateLookup":
        edges = np.asarray(payload.get("radius_bin_edges", []), dtype=np.float64)
        maxima = np.asarray(payload.get("max_whitened", []), dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Residual gate lookup file is missing 'radius_bin_edges'.")
        bins = edges[1:]
        if maxima.size != bins.size:
            raise ValueError("Residual gate lookup file has mismatched 'max_whitened' length.")
        thresholds = np.maximum.accumulate(maxima.astype(np.float64, copy=False))
        return cls(radius_bins=bins, thresholds=thresholds, margin=float(margin))

    @classmethod
    def load(cls, path: str | Path, *, margin: float) -> "ResidualGateLookup":
        target = Path(path).expanduser()
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_payload(payload, margin=margin)

    def threshold(self, radius: float) -> float:
        if self.radius_bins.size == 0:
            return float(self.margin)
        clipped = np.clip(radius, 0.0, float(self.radius_bins[-1]))
        idx = np.searchsorted(self.radius_bins, clipped, side="right")
        if idx <= 0:
            return float(self.margin)
        idx = min(idx - 1, self.thresholds.size - 1)
        return float(self.thresholds[idx] + self.margin)


@dataclass(frozen=True)
class ResidualPolicy:
    gate1_enabled: bool
    gate1_alpha: float
    gate1_margin: float
    gate1_eps: float
    gate1_audit: bool
    gate1_radius_cap: float
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
