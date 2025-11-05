# Neighbor Graph Caching & Async Rebuild Plan (2025-10-30)

> **Consolidated summary:** key points are incorporated into `notes/vif_runtime_optimisation_2025-10.md` §4.5. This document retains the full design sketch, code snippets, and configuration proposals.

## Problem Statement
- **Symptom:** Each LBFGS epoch triggers a full cover-tree rebuild and neighbor selection whenever the length-scale drift exceeds the gin tolerance (`neighbor_refresh_tol_log_lengthscale`). The rebuild is synchronous, blocking the host for 1–4 s at 8 k–32 k samples.
- **Impact:** The “other” bucket in the LBFGS timing breakdown grows super-linearly (1.0 s → 4.9 s between 2 048 and 8 192 points), dominating the steady-state epoch cost and keeping overall runtime CPU-bound.

## Evidence
1. **Microbenchmark timings (`artifacts/lbfgs_microbench/lbfgs_microbench_20251030-171943.json`):**
   - `sample_limit=4 096`: `other ≈ 1.80 s` per epoch.
   - `sample_limit=8 192`: `other ≈ 4.88 s` per epoch.
2. **Structured logs (`tools/lbfgs_microbenchmark.py` run, 2025-10-30 17:18 UTC):**
   - `survi.models.vif.ops:maybe_refresh_state` emits `neighbor_refresh.trigger` every epoch.
   - `survi.models.selectors.cover_tree:build` reports `CoverTree.build: levels=7, time=0.862s` and `CT select: numba query … 0.119s`. Both events appear back-to-back with `trainer.epoch.finish`, confirming synchronous rebuild.
3. **Scaling benchmark (`tools/vif_scaling_benchmark.py`, noted in `notes/vif_runtime_status_2025-10-30.md`):**
   - CPU utilisation peaks >3 000 % at n=4 096 despite low GPU usage, aligning with heavy cover-tree work.

## Current Implementation Snapshot
1. **Refresh trigger (`survi/models/vif/ops.py:1201-1237`):**
   - Every `neighbor_refresh_cooldown_epochs` (default 1) the code compares current vs stored log-lengthscale (`_last_log_ell`). If `delta >= tol`, it logs `neighbor_refresh.trigger` and calls `neighbor_selector.select(...)`.
   - The refresh updates `state.neighbor_indices` and optionally recomputes distance caches. All work happens inline before returning to the trainer.
2. **Cover tree selector (`survi/models/selectors/cover_tree.py:262-438`):**
   - `CoverTree.build` constructs the hierarchical tree level by level on the host, using Python loops and NumPy distances. Complexity is roughly O(n log n) but with large constants.
   - `CoverTree.select` runs immediately after `build`, calling a Numba kernel to extract k-nearest neighbors (`select:807` logs runtime).
3. **State storage (`survi/models/vif/model.py:67-117`):**
   - `VIFGP._init_state` stores `neighbor_indices`, but there is no caching of the cover tree or previous neighbor sets beyond the raw indices.
4. **Configuration knobs (`configs/main.gin:120-134`):**
   - `neighbor_refresh_tol_log_lengthscale=0.05`, `neighbor_refresh_cooldown_epochs=1`, `cache_neighbor_dists=False`—so every epoch rebuilds after even modest length-scale movement.

## Proposed Solution: Cache & Asynchronous Rebuilds

### Goals
1. Avoid redundant neighbor recomputation when length-scale changes are small.
2. When a rebuild is required, overlap it with ongoing LBFGS computation to hide latency.
3. Preserve determinism and numerical safety; ensure fallbacks exist if async work lags behind.

### Design Overview
#### 1. Persistent Neighbor Cache
- Extend `GPState` to hold a cached neighbor graph (indices + optional metadata) that survives across epochs.
- Modify `maybe_refresh_state` to:
  - Check both `delta` and a new `max_epochs_between_refresh` (guard against staleness).
  - If cache is valid and refresh not required, return existing state without recomputation.
  - Record diagnostics (`neighbor_refresh.skip_cached`) so we can monitor how often caching triggers.

#### 2. Asynchronous Rebuild Worker
- Introduce a lightweight worker (thread or process) that runs `neighbor_selector.select` independent of the training loop:
  1. At the end of epoch `k`, if a refresh is needed, enqueue `(params, state, metadata)` to the worker.
  2. Begin epoch `k+1` immediately with the cached neighbors; training continues.
  3. Before epoch `k+1` completes (or at epoch start), check worker status:
     - If the worker finished, atomically swap in the new neighbor indices for next epoch.
     - If still running, optionally pause before the next epoch to wait (configurable timeout), otherwise fall back to synchronous rebuild to maintain correctness.
- Use Python `concurrent.futures.ThreadPoolExecutor` (since cover tree uses NumPy and Numba, no GIL issues in the heavy sections). Ensure all JAX interactions remain on the main thread.

#### 3. Distance Cache Reuse
- When caching is enabled (`cache_neighbor_dists`), store distance matrices alongside indices and reuse them unless a rebuild occurs.
- This prevents redundant `compute_neighbor_distance_caches` recalculations, lowering the overhead of deterministic diagnostics that rely on distances.

### Constraints & Risks
- **Memory Footprint:** Storing two neighbor graphs (current + in-flight rebuild) scales with `n * k`. For n=32 k, k=64 implies ~8 MB per graph (int32). With distance caches, footprint is higher (~16 bytes per pair). Need to budget memory (monitor RSS via existing benchmarks).
- **Determinism:** Async rebuild must preserve ordering. Use immutable seeds and avoid randomness in the worker; cover tree already uses deterministic pool ordering (sorted indices).
- **Race Conditions:** Ensure worker completion is synchronised via futures. Do not swap neighbors mid-epoch; update only between epochs.
- **Fallback:** If async computation throws, log `neighbor_refresh.async_error` and revert to synchronous rebuild next epoch.
- **Configuration:** Add gin/CLI knobs:
  - `neighbor_cache_enabled` (default True once stable),
  - `neighbor_async_refresh` (default False during rollout),
  - `neighbor_async_timeout_s` (grace period at epoch boundary).

### Implementation Steps
1. **State Extensions:**
   - Update `survi/models/common/state.py` `GPState` dataclass to carry `neighbor_cache_version`, `pending_neighbors`, `async_job_id`.
   - Extend serialization/`replace` calls to propagate new fields.
2. **Async Controller Module:**
   - Create `survi/models/vif/neighbor_refresh.py` encapsulating the executor and helper functions (`schedule_refresh`, `poll_refresh`, `cancel_refresh`).
   - Initialise controller in `VIFGP.__post_init__` or lazily inside `maybe_refresh_state`.
3. **Trainer Integration:**
   - At epoch start (`trainer.train`, around `line 210`), poll for completed refresh; if available, swap state via `replace`.
   - At epoch finish, trigger async refresh when needed instead of rebuilding inline.
   - Maintain backwards-compatible behaviour by falling back to current synchronous path when async disabled.
4. **Logging & Metrics:**
   - Emit structured logs: `neighbor_refresh.schedule`, `neighbor_refresh.swap`, `neighbor_refresh.timeout`.
   - Instrument timers with `SURVI_DIAG_TIMING` (add new events under `vif.neighbors.*`) so the microbench captures improvements.
5. **Testing:**
   - Unit tests for caching logic (no refresh when delta < tol).
   - Async tests using small datasets to ensure future completes and state swaps correctly.
   - Regression test verifying determinism (neighbors identical to synchronous path) and time budgets (e.g., asynchronous path shortens epoch “other” bucket).

### Validation Plan
1. Run `tools/lbfgs_microbenchmark.py` with async caching on/off at n=8 192; expect “other” bucket to drop from ~4.9 s to ≤2 s.
2. Re-execute the full `alpine_ridge_long` CLI run (20 epochs) to confirm total runtime drops by ≥20 min.
3. Monitor MLflow timings and structured logs for `neighbor_refresh.*` events to ensure we do not starve rebuilds.
4. Stress-test with `neighbor_refresh_tol_log_lengthscale=0` (forces rebuild every epoch) to validate async fallback.

### Follow-on Opportunities
- Experiment with length-scale change prediction (use gradient history to anticipate tol breaches and prefetch neighbors earlier).
- Consider caching the entire cover tree object (`CoverTree` instance) to skip rebuilds entirely when `k` or dataset geometry stays stable.
- Integrate with future GPU-based neighbor selection once available; the async scaffold can evolve into a device-offloaded job.

# FEEDBACK


Awesome write‑up. I went through it line‑by‑line and turned your plan into a concrete, drop‑in draft implementation with:

* **A persistent neighbor cache**
* **An asynchronous rebuild controller** using a `ThreadPoolExecutor`
* **Trainer/ops integration** that avoids blocking epochs
* **Config knobs, structured logs, and basic tests**

Below are focused diffs and new files you can paste into the repo. I’ve tried to keep dependencies and call sites conservative so this compiles cleanly and is easy to revert. Where I had to infer an interface, I’ve noted it with a short comment.

---

## 0) TL;DR—What changes at runtime?

* If `neighbor_async_refresh=True`, epochs no longer block while the cover‑tree + selection runs.
* We **reuse** the previous `neighbor_indices`/`neighbor_dists` until the async job finishes; then we **swap** at the next epoch boundary.
* We **skip recomputation** when `|Δ log ℓ| < neighbor_refresh_tol_log_lengthscale` **and** we haven’t exceeded `max_epochs_between_refresh`.
* Structured logs: `neighbor_refresh.skip_cached`, `.schedule`, `.complete`, `.swap`, `.timeout`, `.async_error`, plus the existing `.trigger`.

---

## 1) Config knobs

**`configs/main.gin`**

```diff
@@
 # Neighbor refresh behavior
-neighbor_refresh_tol_log_lengthscale = 0.05
-neighbor_refresh_cooldown_epochs = 1
-cache_neighbor_dists = False
+neighbor_refresh_tol_log_lengthscale = 0.05
+neighbor_refresh_cooldown_epochs = 1
+cache_neighbor_dists = False
+
+# NEW: caching + async controls
+neighbor_cache_enabled = True
+max_epochs_between_refresh = 5
+
+# NEW: async refresh rollout toggles
+neighbor_async_refresh = False
+neighbor_async_timeout_s = 0.0    # wait at epoch boundary for up to this long to swap
+neighbor_async_max_workers = 1    # 1 recommended; cover-tree build is CPU-heavy
+neighbor_async_coalesce = True    # don't queue multiple overlapping jobs
```

---

## 2) State extensions

**`survi/models/common/state.py`** (or wherever `GPState` lives)

```diff
@@
-from dataclasses import dataclass
+from dataclasses import dataclass, replace
 from typing import Optional
 import numpy as np

 @dataclass
 class GPState:
     # existing fields...
     neighbor_indices: Optional[np.ndarray] = None
-    # no caching fields previously
+    neighbor_dists: Optional[np.ndarray] = None  # optional distance cache
+
+    # NEW: cache & bookkeeping
+    neighbor_cache_version: int = 0
+    neighbor_cache_epoch: Optional[int] = None
+    neighbor_cache_log_ell: Optional[np.ndarray] = None
+    pending_neighbor_job_id: Optional[int] = None  # job id tracked outside JAX pytrees
+
+    # Helper used below in ops to atomically update neighbors
+    def with_neighbors(self,
+                       indices: np.ndarray,
+                       dists: Optional[np.ndarray],
+                       log_ell: np.ndarray,
+                       epoch: int) -> "GPState":
+        return replace(self,
+                       neighbor_indices=indices,
+                       neighbor_dists=dists,
+                       neighbor_cache_version=self.neighbor_cache_version + 1,
+                       neighbor_cache_epoch=epoch,
+                       neighbor_cache_log_ell=np.array(log_ell, copy=True),
+                       pending_neighbor_job_id=None)
```

> **Why not keep Futures in state?** `GPState` is likely a JAX PyTree; keeping a `Future` there is brittle. We only store a **job id** and let a controller map ids → futures/results.

---

## 3) Async controller (new)

**`survi/models/vif/neighbor_refresh.py`** (new file)

```python
from __future__ import annotations

import itertools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeighborResult:
    indices: np.ndarray            # [n, k], int32
    dists: Optional[np.ndarray]    # [n, k], float32
    n: int
    k: int
    log_ell: np.ndarray            # shape [d]
    epoch: int
    build_wall_s: float = 0.0
    select_wall_s: float = 0.0


@dataclass
class _Job:
    id: int
    fn: Callable[[], NeighborResult]
    meta: Dict[str, object]
    future: Future


class NeighborRefreshController:
    """
    Manages background neighbor rebuilds. One controller per model/trainer.
    Thread-based to avoid pickling data into a process; heavy math releases GIL.
    """
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="neighbors")
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._jobs: Dict[int, _Job] = {}
        self._latest_completed: Optional[Tuple[int, NeighborResult]] = None

    def schedule(self, fn: Callable[[], NeighborResult], meta: Dict[str, object]) -> int:
        job_id = next(self._seq)
        with self._lock:
            fut = self._executor.submit(self._run, job_id, fn, meta)
            self._jobs[job_id] = _Job(job_id, fn, meta, fut)
        logger.info("neighbor_refresh.schedule", extra={"job_id": job_id, **meta})
        return job_id

    def _run(self, job_id: int, fn: Callable[[], NeighborResult], meta: Dict[str, object]) -> NeighborResult:
        try:
            t0 = time.perf_counter()
            res = fn()
            dt = time.perf_counter() - t0
            logger.info("neighbor_refresh.complete", extra={"job_id": job_id, "wall_s": dt, **meta})
            with self._lock:
                self._latest_completed = (job_id, res)
            return res
        except Exception:
            logger.exception("neighbor_refresh.async_error", extra={"job_id": job_id, **meta})
            raise

    def poll_latest(self) -> Optional[Tuple[int, NeighborResult]]:
        # Opportunistically harvest completed jobs
        with self._lock:
            for jid, job in list(self._jobs.items()):
                if job.future.done():
                    _ = job.future.result()  # re-raises if failed
                    del self._jobs[jid]
            return self._latest_completed

    def result_for(self, job_id: int, timeout_s: Optional[float] = None) -> Optional[NeighborResult]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            latest = self.poll_latest()
            if latest and latest[0] == job_id:
                return latest[1]
            return None
        try:
            res = job.future.result(timeout=timeout_s)
        except Exception:
            raise
        with self._lock:
            self._jobs.pop(job_id, None)
            self._latest_completed = (job_id, res)
        return res

    def shutdown(self, wait: bool = True):
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.future.cancel()
        self._executor.shutdown(wait=wait)


def make_async_select_fn(*,  # captures the EXACT sync call you already use
                         select_call: Callable[[], Tuple[np.ndarray, Optional[np.ndarray]]],
                         n: int,
                         k: int,
                         log_ell: np.ndarray,
                         epoch: int) -> Callable[[], NeighborResult]:
    """
    Wraps your existing synchronous neighbor selection call in a callable the controller can run.
    `select_call` must return (indices, dists_or_None).
    """
    def _fn() -> NeighborResult:
        t0 = time.perf_counter()
        indices, dists = select_call()
        dt = time.perf_counter() - t0
        return NeighborResult(indices=indices, dists=dists, n=n, k=k, log_ell=np.array(log_ell, copy=True),
                              epoch=epoch, build_wall_s=dt, select_wall_s=0.0)
    return _fn
```

> **Key idea:** We don’t guess your selector’s full signature. We **capture** the *exact synchronous call* you already make as a zero‑arg `select_call`. This ensures async and sync paths are bit‑for‑bit identical.

---

## 4) `ops.py` integration

**`survi/models/vif/ops.py`**
(adding caching logic + async scheduling & polling; keep existing imports and logging scheme)

```diff
@@
 import logging
 import numpy as np
+from dataclasses import replace
+from typing import Optional, Tuple, Callable
+
+try:
+    import gin
+except Exception:  # gin is optional at import time for tests
+    gin = None
+
+from .neighbor_refresh import (
+    NeighborRefreshController,
+    make_async_select_fn,
+)

 logger = logging.getLogger(__name__)
 
+# Controller is hung off the module and/or model to avoid PyTree issues.
+_GLOBAL_NEIGHBOR_CONTROLLER: Optional[NeighborRefreshController] = None
+
+def _get_controller(max_workers: int) -> NeighborRefreshController:
+    global _GLOBAL_NEIGHBOR_CONTROLLER
+    if _GLOBAL_NEIGHBOR_CONTROLLER is None:
+        _GLOBAL_NEIGHBOR_CONTROLLER = NeighborRefreshController(max_workers=max_workers)
+    return _GLOBAL_NEIGHBOR_CONTROLLER
+
+
+def _max_abs(x: np.ndarray) -> float:
+    return float(np.max(np.abs(x))) if x is not None else float("inf")
+
+
+def _should_refresh(cur_log_ell: np.ndarray,
+                    last_log_ell: Optional[np.ndarray],
+                    epoch: int,
+                    last_epoch: Optional[int],
+                    tol: float,
+                    cooldown_epochs: int,
+                    max_epochs_between_refresh: int) -> Tuple[bool, float, int]:
+    delta = _max_abs(cur_log_ell - last_log_ell) if last_log_ell is not None else float("inf")
+    epochs_since = (epoch - (last_epoch or -10**9))
+    cooldown_ok = (epochs_since >= cooldown_epochs)
+    staleness_ok = (epochs_since >= max_epochs_between_refresh)
+    refresh = (cooldown_ok and (delta >= tol or staleness_ok))
+    return refresh, delta, epochs_since
+
+
+def _same_geometry(state, n: int, k: int) -> bool:
+    inds = state.neighbor_indices
+    if inds is None:
+        return False
+    if inds.shape[0] != n:
+        return False
+    if inds.shape[1] != k:
+        return False
+    return True
+
@@
-# In maybe_refresh_state we used to synchronously rebuild the neighbor graph.
-# Keep the same entrypoint but extend with caching + async behavior.
+# In maybe_refresh_state we used to synchronously rebuild the neighbor graph.
+# Keep the same entrypoint but extend with caching + async behavior.
- def maybe_refresh_state(...):
+@gin.configurable
+def maybe_refresh_state(
+    *,
+    model,
+    state,
+    epoch: int,
+    cur_log_ell: np.ndarray,
+    k_neighbors: int,
+    select_call_factory: Callable[[bool], Callable[[], Tuple[np.ndarray, Optional[np.ndarray]]]],
+    neighbor_refresh_tol_log_lengthscale: float = 0.05,
+    neighbor_refresh_cooldown_epochs: int = 1,
+    neighbor_cache_enabled: bool = True,
+    max_epochs_between_refresh: int = 5,
+    cache_neighbor_dists: bool = False,
+    neighbor_async_refresh: bool = False,
+    neighbor_async_timeout_s: float = 0.0,
+    neighbor_async_max_workers: int = 1,
+    neighbor_async_coalesce: bool = True,
+):
+    """
+    Arguments
+    ---------
+    - model, state: your usual objects.
+    - epoch: current epoch number (int).
+    - cur_log_ell: np.ndarray of shape [d].
+    - k_neighbors: int (current k).
+    - select_call_factory: callable that takes (return_dists: bool) and returns a zero-arg function
+      that performs the *exact* synchronous selection you currently run and returns (indices[, dists]).
+      This avoids duplicating selector arguments here and keeps sync/async parity.
+    """
+    n_points = int(getattr(model, "n_points", getattr(state, "n_points", 0)) or 0)
+    # If we can't infer n, try from cached indices
+    if n_points <= 0 and state.neighbor_indices is not None:
+        n_points = state.neighbor_indices.shape[0]
+
+    # 1) If caching is enabled and geometry matches and Δlogℓ is small, skip recompute.
+    if neighbor_cache_enabled and _same_geometry(state, n_points, k_neighbors):
+        refresh_needed, delta, epochs_since = _should_refresh(
+            cur_log_ell,
+            state.neighbor_cache_log_ell,
+            epoch,
+            state.neighbor_cache_epoch,
+            tol=neighbor_refresh_tol_log_lengthscale,
+            cooldown_epochs=neighbor_refresh_cooldown_epochs,
+            max_epochs_between_refresh=max_epochs_between_refresh,
+        )
+        if not refresh_needed:
+            logger.info("neighbor_refresh.skip_cached",
+                        extra={"delta": delta, "epochs_since": epochs_since, "epoch": epoch})
+            return state  # keep using cached neighbors
+    else:
+        # cache missing or geometry changed → force refresh
+        refresh_needed, delta, epochs_since = True, float("inf"), (
+            epoch - (state.neighbor_cache_epoch or -10**9)
+        )
+
+    # 2) If async disabled → run synchronously like before.
+    if not neighbor_async_refresh:
+        logger.info("neighbor_refresh.trigger",
+                    extra={"delta": float(delta), "epoch": epoch, "mode": "sync"})
+        select_call = select_call_factory(cache_neighbor_dists)
+        indices, dists = select_call()
+        new_state = state.with_neighbors(indices=indices, dists=dists, log_ell=cur_log_ell, epoch=epoch)
+        return new_state
+
+    # 3) Async path: queue a job unless one is already in-flight (coalesce).
+    ctrl = _get_controller(neighbor_async_max_workers)
+    if neighbor_async_coalesce and state.pending_neighbor_job_id is not None:
+        # A job is already in-flight; keep training and let poller handle swap.
+        logger.info("neighbor_refresh.trigger",
+                    extra={"delta": float(delta), "epoch": epoch, "mode": "async_coalesced"})
+        return state
+
+    logger.info("neighbor_refresh.trigger",
+                extra={"delta": float(delta), "epoch": epoch, "mode": "async"})
+    select_call = select_call_factory(cache_neighbor_dists)
+    async_fn = make_async_select_fn(
+        select_call=select_call, n=n_points, k=k_neighbors, log_ell=cur_log_ell, epoch=epoch
+    )
+    job_id = ctrl.schedule(
+        fn=async_fn,
+        meta={"epoch": epoch, "n": n_points, "k": k_neighbors},
+    )
+    return replace(state, pending_neighbor_job_id=job_id)
+
+
+@gin.configurable
+def poll_and_maybe_swap_neighbors(
+    *,
+    state,
+    epoch: int,
+    neighbor_async_timeout_s: float = 0.0,
+    neighbor_async_max_workers: int = 1,
+):
+    """
+    Call this once per epoch (start or end) to harvest async results.
+    Will optionally block up to `neighbor_async_timeout_s` to get the latest job's result.
+    """
+    if state.pending_neighbor_job_id is None:
+        # Opportunistic swap if any completed job exists
+        ctrl = _get_controller(neighbor_async_max_workers)
+        latest = ctrl.poll_latest()
+        if latest is None:
+            return state
+        job_id, res = latest
+    else:
+        ctrl = _get_controller(neighbor_async_max_workers)
+        try:
+            res = ctrl.result_for(state.pending_neighbor_job_id,
+                                  timeout_s=neighbor_async_timeout_s if neighbor_async_timeout_s > 0 else None)
+        except Exception:
+            logger.exception("neighbor_refresh.async_error",
+                             extra={"job_id": state.pending_neighbor_job_id, "epoch": epoch})
+            # Clear pending job id; we'll fall back to sync next time.
+            return replace(state, pending_neighbor_job_id=None)
+        if res is None:
+            # Timed out or not ready yet
+            if neighbor_async_timeout_s > 0:
+                logger.info("neighbor_refresh.timeout",
+                            extra={"job_id": state.pending_neighbor_job_id, "epoch": epoch,
+                                   "timeout_s": neighbor_async_timeout_s})
+            return state
+        job_id = state.pending_neighbor_job_id
+
+    # Swap-in atomically between epochs
+    logger.info("neighbor_refresh.swap",
+                extra={"job_id": job_id, "epoch": epoch, "n": res.n, "k": res.k})
+    return state.with_neighbors(indices=res.indices, dists=res.dists, log_ell=res.log_ell, epoch=epoch)
```

### About `select_call_factory`

To keep this non-invasive, you pass a factory that builds the *exact* synchronous selector call you already make. Example (in your current `maybe_refresh_state` call site):

```python
def _select_call_factory(return_dists: bool):
    # This is the call you do today, verbatim — just wrap it.
    # Example arguments shown; use the real ones from your code.
    def _call():
        indices, dists = neighbor_selector.select(
            X=points_np,
            k=k_neighbors,
            log_lengthscale=cur_log_ell,
            return_dists=return_dists,
            # ...any other current kwargs...
        )
        return indices, (dists if return_dists else None)
    return _call
```

Then invoke:

```python
state = maybe_refresh_state(
    model=model,
    state=state,
    epoch=epoch,
    cur_log_ell=cur_log_ell,
    k_neighbors=k_neighbors,
    select_call_factory=_select_call_factory,
    # the rest comes from gin
)
```

> This avoids hard‑coding selector internals inside `ops.py` and preserves determinism (async job runs the *exact* same code path).

---

## 5) Trainer integration (poll at epoch boundary)

**`survi/trainer.py`** (or your training loop module)

At **start of each epoch**, opportunistically swap in any finished result (optionally wait for `neighbor_async_timeout_s`):

```diff
@@ def train(...):
-    # existing
+    from survi.models.vif.ops import poll_and_maybe_swap_neighbors
+    state = poll_and_maybe_swap_neighbors(state=state, epoch=epoch)
```

At **end of each epoch**, instead of forcing a synchronous rebuild, call the new `maybe_refresh_state` (it will either skip, queue async, or do sync depending on gin):

```diff
@@ end_of_epoch():
-    state = maybe_refresh_state(...)  # old path did sync rebuild
+    state = maybe_refresh_state(
+        model=model,
+        state=state,
+        epoch=epoch,
+        cur_log_ell=cur_log_ell,
+        k_neighbors=k_neighbors,
+        select_call_factory=_select_call_factory,
+    )
```

> If you prefer to **always swap at the *start*** of the *next* epoch, pass `neighbor_async_timeout_s=0.0` (default) so the trainer never stalls.

---

## 6) Optional: cover‑tree distance cache reuse

If you already compute distances for diagnostics, set `cache_neighbor_dists=True` (gin). The async job will compute and store them alongside indices, and the ops path will propagate them in `state.neighbor_dists`. Your diagnostics can short‑circuit distance recomputation by reading that field when non‑`None`.

---

## 7) Structured logging & metrics

New log events emitted:

* `neighbor_refresh.skip_cached` — cached graph reused (log delta + epochs since)
* `neighbor_refresh.trigger` — computed whenever a rebuild is requested (mode: `sync|async|async_coalesced`)
* `neighbor_refresh.schedule` — queued a background job (job_id, n, k)
* `neighbor_refresh.complete` — background job finished (job_id, wall_s)
* `neighbor_refresh.swap` — atomic swap at epoch boundary (job_id, n, k)
* `neighbor_refresh.timeout` — epoch boundary chose not to wait longer
* `neighbor_refresh.async_error` — worker raised

Your microbench will pick these up automatically if it already tails the logger.

---

## 8) Tests (draft)

**`tests/test_neighbor_async.py`**

```python
import time
import unittest
import numpy as np
from dataclasses import replace

from survi.models.vif.neighbor_refresh import NeighborRefreshController, make_async_select_fn
from survi.models.common.state import GPState
from survi.models.vif.ops import maybe_refresh_state, poll_and_maybe_swap_neighbors

class FakeModel:
    n_points = 128

def naive_knn(X, k, log_lengthscale, return_dists):
    # Very slow O(n^2) naive KNN, but deterministic and pure NumPy for tests.
    Xs = X / np.exp(log_lengthscale)[None, :]
    D = ((Xs[:, None, :] - Xs[None, :, :])**2).sum(-1)
    idx = np.argsort(D, axis=1)[:, 1:k+1]  # skip self
    dists = np.take_along_axis(D, idx, axis=1) if return_dists else None
    time.sleep(0.05)  # simulate work
    return idx.astype(np.int32), (dists.astype(np.float32) if dists is not None else None)

class NeighborAsyncTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((128, 8)).astype(np.float32)
        self.log_ell = np.log(np.ones((8,), dtype=np.float32))
        self.k = 8
        self.model = FakeModel()
        self.state = GPState()

    def _factory(self, return_dists: bool):
        return lambda: naive_knn(self.X, self.k, self.log_ell, return_dists)

    def test_skip_cached_when_delta_small(self):
        idx, dst = naive_knn(self.X, self.k, self.log_ell, True)
        st = self.state.with_neighbors(idx, dst, self.log_ell, epoch=0)
        st2 = maybe_refresh_state(model=self.model, state=st, epoch=1, cur_log_ell=self.log_ell + 1e-3,
                                  k_neighbors=self.k, select_call_factory=self._factory,
                                  neighbor_refresh_tol_log_lengthscale=0.05,
                                  neighbor_cache_enabled=True, neighbor_async_refresh=False)
        self.assertIs(st2, st)  # exact object reuse -> skipped

    def test_async_schedule_and_swap(self):
        st = maybe_refresh_state(model=self.model, state=self.state, epoch=0, cur_log_ell=self.log_ell,
                                 k_neighbors=self.k, select_call_factory=self._factory,
                                 neighbor_async_refresh=True, neighbor_cache_enabled=False)
        self.assertIsNotNone(st.pending_neighbor_job_id)
        # Poll without blocking → may still be running
        st2 = poll_and_maybe_swap_neighbors(state=st, epoch=0, neighbor_async_timeout_s=0.0)
        # Now wait and swap
        st3 = poll_and_maybe_swap_neighbors(state=st2, epoch=1, neighbor_async_timeout_s=5.0)
        self.assertIsNotNone(st3.neighbor_indices)
        self.assertEqual(st3.neighbor_indices.shape[1], self.k)

if __name__ == "__main__":
    unittest.main()
```

> The tests do not require JAX and validate: cache skip, async schedule, and eventual swap.

---

## 9) Notes on determinism & safety

* **Determinism:** We reuse the *exact* sync selector call in async (`select_call_factory → make_async_select_fn`), so results should match bit‑for‑bit. Ensure any randomness (if any) is seeded outside and captured in the factory closure.
* **Race‑free swap:** We never swap mid‑epoch; trainer calls `poll_and_maybe_swap_neighbors()` only at epoch boundaries.
* **Geometry changes:** If `n` or `k` changes (e.g., new sampling), cache is invalidated by `_same_geometry()`.
* **Memory:** Double‑buffering the neighbor graph (current + in‑flight result) means peak memory ≈ 2× `n*k*(4B + optional 4B)`. For `n=32k`, `k=64`: ~8 MB (indices) + ~8 MB (dists) * 2 = ~32 MB worst case; acceptable on typical hosts.
* **Threading vs. processes:** Using threads keeps things zero‑copy. The heavy work uses NumPy/Numba and should release the GIL for inner loops; if your build path is still GIL‑bound, you can swap the controller to a `ProcessPoolExecutor`, but that will copy large arrays—use only if necessary.

---

## 10) How to wire it in your code today

1. **Add the new file** `survi/models/vif/neighbor_refresh.py`.
2. **Patch** `GPState` as shown.
3. **Replace** the current inline neighbor rebuild in `ops.maybe_refresh_state` with the code above, and provide a tiny `_select_call_factory` wrapper at the call site that mirrors your current synchronous `neighbor_selector.select(...)` call.
4. **Trainer:** call `poll_and_maybe_swap_neighbors(state=state, epoch=epoch)` at the start of each epoch (optionally also right before logging `trainer.epoch.finish`).
5. **Enable via gin:**

   * First test synchronous caching only:

     ```
     neighbor_cache_enabled = True
     neighbor_async_refresh = False
     ```
   * Then flip to async:

     ```
     neighbor_async_refresh = True
     neighbor_async_timeout_s = 0.0
     neighbor_async_max_workers = 1
     ```

---

## 11) Validation checklist (mirrors your plan)

* `tools/lbfgs_microbenchmark.py` with `neighbor_async_refresh=True` @ n=8192 should show “other” dropping substantially (no synchronous cover‑tree build).
* Logs should include `neighbor_refresh.skip_cached` most epochs once ℓ stabilizes; when tol breached, expect `.trigger` then `.schedule` → (next epoch) `.swap`.
* Stress with `neighbor_refresh_tol_log_lengthscale=0.0` to force rebuilds and validate that training proceeds without blocking (only swap at epoch boundaries).

---

## 12) Small accuracy nits in the doc (fixed above)

* Don’t store `Future`/controller handles in `GPState` (likely a PyTree) — keep an integer `pending_neighbor_job_id` and map it via the controller.
* Coalescing multiple refreshes while one is in flight reduces churn; I added `neighbor_async_coalesce`.
* The cache validity must also check `(n, k)` geometry; I added `_same_geometry`.

---

If you want, I can also draft a tight `select_call_factory` for your current selector call once you paste the exact line (args) you use today in `ops.py:1201–1237`.
