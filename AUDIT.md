Below is a focused, **DX/UX refactor audit** mapped to your tree, with concrete changes, small code samples, and a **sequenced plan** that turns the current “powerful but dense” system into something contributors and users find *obvious*, *discoverable*, and *hard to misuse*.

---

## Executive summary

**North-star DX/UX**

* *Single mental model*: “Build a tree, run queries, record telemetry.”
* *Fewer foot‑guns*: No invisible global state; reproducible, testable runs.
* *Discoverable configuration*: Fewer knobs at once; curated presets; clear override rules.
* *Slick CLI*: Human-friendly help, profiles, progress bars, and `doctor` checks.
* *Extensible internals*: Pluggable strategies/metrics via first-class plugin registry (not import side effects).
* *Actionable telemetry*: One schema, one formatter, quick comparisons.

**Top 10 refactors (prioritized)**

1. **Runtime/global state → explicit contexts** (covertreex/runtime/*, api/runtime.py)
2. **CLI surface → profiles + subcommands with Typer** (cli/queries/app.py → cli/pcct/*)
3. **Plugin registries** for traversal/conflict/metrics (algo/*/strategies, core/metrics.py)
4. **Split residual traversal monolith** (algo/traverse/strategies/residual.py)
5. **Typed config with validation** (pydantic) + one place to describe every knob (runtime/config.py)
6. **Preflight `doctor` & environment guardrails** (CLI + tools)
7. **Telemetry schema consolidation + formatters** (covertreex/telemetry/*, cli/queries/telemetry.py)
8. **Public API surface hardening** (covertreex/api/* and **init** exports)
9. **Determinism & seeds policy** (algo/mis.py, CLI, telemetry)
10. **Contributor ergonomics**: pre-commit (ruff/mypy), docs, examples, presets.

I annotate each with *benefit* and *effort* (S/M/L), then give code sketches you can paste in.

---

## 1) Runtime/global state → explicit, testable contexts

**Symptoms**

* `runtime/config.py` (781 LOC) holds caches + singletons (`_RUNTIME_CONTEXT`, `_RUNTIME_CONFIG_CACHE`), and import order matters across modules (registries, metrics).
* Hard to reason about reproducibility; library consumers can’t easily “isolate” runs.

**Refactor**

* Introduce an explicit, frozen `RuntimeConfig` + `RuntimeContext` that are **constructed and passed**, not pulled from hidden globals.
* Provide a **context manager** for activation, but keep pure functions that accept context explicitly.

**Sketch**

```python
# covertreex/runtime/model.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Sequence

class RuntimeConfig(BaseModel, frozen=True):
    backend: Literal["numpy", "jax", "gpu"] = "numpy"
    precision: Literal["fp32", "fp64"] = "fp32"
    devices: Sequence[str] = Field(default_factory=tuple)
    enable_numba: bool = True
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    # Residual policy knobs, grouped:
    residual: "ResidualConfig" = Field(default_factory=lambda: ResidualConfig())

class ResidualConfig(BaseModel, frozen=True):
    lengthscale: float = 1.0
    variance: float = 1.0
    # …all the residual_* fields now live here, with validation & docs

# covertreex/runtime/context.py
from contextlib import contextmanager

class RuntimeContext:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.backend = _resolve_backend(cfg)  # pure

    @contextmanager
    def activate(self):
        prev = _push_context(self)     # thread-local, not global module
        try:
            yield self
        finally:
            _pop_context(prev)
```

**Migration**

* `covertreex.core.tree.get_runtime_backend()` reads thread‑local, not module global.
* `PCCT.fit/insert/knn` accept optional `context: RuntimeContext | None` (and default to current thread-local).

**Benefit**: reproducible tests, fewer spooky interactions, easier embedding.
**Effort**: M.

---

## 2) CLI surface → *profiles* + subcommands (Typer), not 70 flags on one command

**Symptoms**

* `cli/queries/app.py` exposes ~70 flags; users must know internal defaults; discovery is difficult.
* Multiple responsibilities (parse → validate → run → telemetry).

**Refactor**

* Replace argparse with **Typer** (or Click) for:

  * **Subcommands**: `pcct build`, `pcct query`, `pcct benchmark`, `pcct doctor`, `pcct profile`.
  * **Profiles**: YAML/JSON preset bundles (e.g. `--profile residual-fast`, `--profile gpu-latency`).
  * **Overrides**: `--set residual.lengthscale=0.8 --set k=50` (dot-path).
  * **Nice help** with sections, validation, and examples.

**Sketch**

```python
# cli/pcct/main.py
import typer
from covertreex.api import PCCT, Runtime
from covertreex.runtime.model import RuntimeConfig
from .profiles import load_profile, apply_overrides

app = typer.Typer(help="PCCT engine CLI")

@app.command()
def build(dataset: str, profile: str = "default", set: list[str] = typer.Option(None)):
    cfg = load_profile(profile)  # RuntimeConfig
    cfg = apply_overrides(cfg, set)  # dot-notation -> new validated model
    ctx = Runtime(cfg).activate()
    with ctx:
        # … build and persist tree artifact

@app.command()
def query(artifact: str, k: int = 10, profile: str = "default", set: list[str] = None):
    # … loads tree, runs queries, writes telemetry

@app.command()
def doctor():
    # checks numba, jax, CPU features, memory, version pins; pretty output

if __name__ == "__main__":
    app()
```

**Profiles**

* `cli/profiles/` ships curated presets (residual fast/accurate, cpu vs gpu).
* `pcct profile list/show` displays values with provenance (preset vs override).

**Benefit**: simpler UX, copy‑pastable recipes, fewer support questions.
**Effort**: M.

---

## 3) Strategy/metric registries → first-class plugins (no import side effects)

**Symptoms**

* `algo/traverse/strategies/registry.py`, `algo/conflict/strategies.py`, `core/metrics.py` rely on **import-time registration**.
* Tests and users can be bitten by import order.

**Refactor**

* Introduce `covertreex.plugins` with **explicit registration** and **entry points** (setuptools) or a simple plugin loader.
* Provide **reset/isolation** helpers for tests.

**Sketch**

```python
# covertreex/plugins/registry.py
from typing import Callable, Protocol

class Strategy(Protocol):
    name: str
    predicate: Callable[[RuntimeConfig, object], bool]
    factory: Callable[[RuntimeConfig, object], object]

_REG: dict[str, Strategy] = {}

def register(strategy: Strategy, *, overwrite: bool = False) -> None:
    if strategy.name in _REG and not overwrite:
        raise KeyError(f"Strategy exists: {strategy.name}")
    _REG[strategy.name] = strategy

def select(kind: str, cfg: RuntimeConfig, backend) -> Strategy:
    # filter by kind + predicate; deterministic tie-break
    ...

def reset(): _REG.clear()
```

In each strategy module:

```python
def plugin() -> Strategy: ...
# Setup entry point in pyproject:
# [project.entry-points."covertreex.traversal"]
# residual = "covertreex.algo.traverse.strategies.residual:plugin"
```

**Benefit**: deterministic, testable, extensible by downstream users without patching imports.
**Effort**: M.

---

## 4) Split the residual traversal monolith (1,839 LOC) into cohesive modules

**Symptoms**

* `algo/traverse/strategies/residual.py` mixes: cache handling, budget state machines, streaming, append logic, bitsets, gating, telemetry.

**Refactor plan (files)**

* `residual/orchestrator.py` — high-level `collect()` pipeline (public strategy).
* `residual/cache.py` — level cache hits/misses and reuse policy.
* `residual/budget.py` — `_update_scope_budget_state`, schedules, escalation logic.
* `residual/append.py` — append paths (dense/bitset/masked/numba) and limit logic.
* `residual/streaming.py` — serial vs parallel streamers + dynamic tile stride.
* `residual/gate.py` — whitened/gate thresholds, margin math, policy lookups.
* `residual/telemetry.py` — strongly typed “events” consumed by the CLI layer.

**Guidelines**

* Each module: < 400 LOC, one responsibility, unit tests per module.
* Replace “big dicts” with small frozen dataclasses.
* All pure logic functions accept `{arrays, limits, cfg}` explicitly.

**Benefit**: comprehension skyrockets, PR review scope shrinks, safer change velocity.
**Effort**: L (mechanical but high LOC).

---

## 5) Typed config & validation with Pydantic

**Symptoms**

* Many CLI/runtime values (e.g., percentiles, caps, margins) have implicit contracts.
* Errors are discovered late.

**Refactor**

* Move all knob definitions into **Pydantic models** with validation and doc strings; render help from model metadata.
* Provide `to_env()` and `from_env()` bridges for compatibility, but favor explicit config objects.

**Sketch**

```python
class GateConfig(BaseModel, frozen=True):
    margin: float = Field(0.0, ge=0.0, le=1.0, description="Gate margin in [0, 1]")
    keep_pct: float = Field(0.95, gt=0, le=1)
    prune_pct: float = Field(0.50, gt=0, le=1)
```

**Benefit**: instant, friendly validation messages; self-documenting CLI.
**Effort**: M.

---

## 6) Preflight `doctor` & environment guardrails

**Symptoms**

* Numba/gpu/JAX failures can occur *during* long runs; fallbacks are partial.

**Refactor**

* Add `pcct doctor` that checks:

  * numba availability & SIMD features, JAX devices & precision, GPU memory, BLAS/OpenMP threads, Python version, env flags.
* Fail fast or suggest flags (`--enable-numba=false`, `--precision=fp64`, etc.).

**Benefit**: fewer wasted runs, crisp UX for new users.
**Effort**: S.

---

## 7) Telemetry schema consolidation + rich formatting

**Symptoms**

* Telemetry spans `covertreex/telemetry`, `cli/queries/telemetry.py`, tools; formats exist but require synchrony.

**Refactor**

* One **Typed schema** module (pydantic) exported for both CLI and tools.
* Standard **record envelope**: run_id, cfg hash, git sha, dataset fingerprint, hostname, seed.
* Renderers:

  * `pcct telemetry summarize path.jsonl` → neat table (median, p95), ratios, min/median/max.
  * `—format md|csv|json` for downstream tools.
* Incorporate **units** consistently (ms, MB) and ensure integer formatting only at the edge.

**Benefit**: easier comparisons, stable tool API.
**Effort**: M.

---

## 8) Public API surface hardening

**Symptoms**

* Re-exports are broad; implicit guarantees unclear.

**Refactor**

* Establish a **tiny, stable** public API:

```python
from covertreex import PCCT, Runtime, ResidualPolicy
```

* Everything else becomes **internal** (`_` modules) or behind `covertreex.plugins`.
* Add `from __future__ import annotations` and `typing_extensions` `Protocol`/`TypeAlias` for clarity.
* Freeze dataclasses returned to users; no mutation foot-guns.

**Benefit**: semantic versioning you can honor; fewer breaking changes.
**Effort**: S/M.

---

## 9) Determinism & seeds policy

**Symptoms**

* Seeds appear in many places (MIS, ordering, grid leaders).

**Refactor**

* A single `SeedPack` in `RuntimeConfig`, disseminated to all stochastic subsystems via context.
* Every stochastic function **must accept a seed** argument; default from context.
* Telemetry records the **effective seeds** and per-stage sub-seeds.

**Benefit**: reproducible results, fewer “works on my machine” issues.
**Effort**: S/M.

---

## 10) Contributor ergonomics

**Refactor**

* `pyproject.toml` with Ruff, mypy (strict optional), pytest plugins, numba “doctest skip” markers.
* `pre-commit` with ruff, mypy, isort, black.
* `docs/` via MkDocs Material: “Add a traversal strategy”, “Write a residual backend”, “Run benchmarks”.
* Example notebooks → **scripts with `pcct` CLI** that emit artifacts, so they’re CI-friendly.

**Benefit**: lower ramp time, consistent style & hints.
**Effort**: S.

---

## UX polish (micro but delightful)

* **Progress & sections**: show “Build → Traverse → Conflict → MIS → Persist” with elapsed/ETA and batch counts.
* **Human errors**: when validation fails, print *exact* flag path & accepted range, plus the nearest profile name.
* **Presets & “what changed”**: After parsing, print a diff of overrides vs profile defaults.
* **Artifacts**: Standard tree bundle layout with a single `metadata.json` (schema versioned) that includes runtime config, seeds, counts, radii stats.
* **`--explain plan`**: print traversal strategy chosen, conflict strategy selected, scope chunking params, with short reasons (“residual gate active + dense scopes → residual-grid”).
* **`pcct replay <log>`**: re-materialize a run from telemetry to re-run one stage (e.g., conflict graph only).

---

## File-by-file actionable suggestions

### `covertreex/runtime/config.py`

* Extract **env parsing** into `runtime/env.py`.
* Move models to `runtime/model.py` (pydantic).
* Replace globals with **thread-local** context stack.
* Add `describe(self)` that renders nicely for CLI and logs.

**Effort**: M.

### `cli/queries/app.py` (961 LOC)

* Split into `cli/pcct/build.py`, `cli/pcct/query.py`, `cli/pcct/benchmark.py`, `cli/pcct/doctor.py`.
* Replace `QueryCLIOptions` with `RuntimeConfig` + small command-specific options.
* Validation → pydantic errors (already pretty).
* Keep `run_queries()` as a thin orchestrator with injected services (no module-level reads).

**Effort**: M.

### `algo/traverse/strategies/residual.py` (1839 LOC)

* Apply the module split in §4.
* Introduce **pure adapters** around numba kernels; isolate side-effects.
* Add **unit tests** per module; keep integration tests for orchestrator.

**Effort**: L.

### `covertreex/core/metrics.py` & `metrics/residual/*`

* Move registration to `covertreex.plugins`.
* Keep `Metric` Protocol stable; avoid mutable globals; expose `reset()` only in test helpers.

**Effort**: M.

### `cli/queries/telemetry.py`

* Replace ad‑hoc formatting with a typed schema + **renderers** (md/csv/json).
* Ensure *units* and *scales* consistent; remove logic like `_format_int` from core areas.

**Effort**: S/M.

### `runtime_breakdown.py` & `tools/*`

* Align on the same telemetry schema.
* `tools/*` become thin wrappers around `pcct` subcommands where possible.

**Effort**: S/M.

---

## Safer numba/JAX story

* Add *feature probes* at import-time behind try-excepts, but **do not** choose silently; surface in `pcct doctor`.
* Each numba path has a **documented fallback** (with perf warning and hint).
* Cache warmups gated behind explicit `RuntimeConfig.warmup=True`.

---

## Public API ergonomics

**Before**

```python
from covertreex.api import PCCT, Runtime
pcct = PCCT(Runtime(...))
pcct.fit(points, apply_batch_order=True, mis_seed=0)
neighbors = pcct.knn(queries, k=10, return_distances=True)
```

**After (clearer, immutable configs, contextual)**

```python
from covertreex import PCCT, Runtime
from covertreex.runtime.model import RuntimeConfig

cfg = RuntimeConfig(backend="numpy", precision="fp64",
                    residual=ResidualConfig(lengthscale=0.7))
with Runtime(cfg).activate() as ctx:
    tree = PCCT().fit(points, apply_batch_order=True)  # ctx bound
    out = tree.knn(queries, k=10, return_distances=True)
```

* `PCCT.fit/insert/delete/knn` return **new tree / result** objects; no hidden mutation.
* Add `.describe()` on results for pretty CLI summaries.

---

## Testing plan (lean, high ROI)

* **Contract tests** for each public method (PCCT, Runtime).
* **Determinism tests**: same seeds → identical telemetry hashes.
* **Plugin isolation tests**: loading new strategies via entry points does not affect selection unless predicates match.
* **Budget/gate unit tests**: for each residual submodule with small synthetic matrices.
* **Golden telemetry**: small .jsonl snapshots verified with tolerant thresholds (±ε).
* **Environment matrix**: fp32/fp64, numba on/off, jax on/off (skips allowed).

---

## Migration notes (non-breaking path)

* Keep current argparse CLI as **compat layer** that shells out to `pcct` subcommands (deprecate with warnings).
* Maintain registry import side effects for **one minor version**, but emit a deprecation notice urging `covertreex.plugins`.
* Provide `runtime/config.py` shims mapping to `runtime/model.py` (types preserved).
* Document “How to migrate your script in 3 steps”.

---

## “Quick PRs” you can open immediately

1. **Thread-local runtime context & `with Runtime(...).activate()`** hook (S/M)
2. **`pcct doctor`** with Numba/JAX/threads check (S)
3. **Profiles**: add `profiles/default.yaml`, `profiles/residual-fast.yaml` + loader (S)
4. **Plugin registry scaffolding** (`covertreex.plugins`) + adapters for traversal/conflict (M)
5. **Telemetry envelope type** (S) and a `--format md|csv|json` renderer (S)

*(Then tackle the residual traversal split and the CLI migration.)*

---

## Appendix: tiny, drop‑in code helpers

**Dot‑path overrides** (for `--set residual.lengthscale=0.8`)

```python
def apply_overrides(cfg: BaseModel, overrides: list[str]) -> BaseModel:
    if not overrides: return cfg
    data = cfg.model_dump()
    for item in overrides:
        path, raw = item.split("=", 1)
        *parents, leaf = path.split(".")
        cursor = data
        for p in parents:
            cursor = cursor.setdefault(p, {})
        cursor[leaf] = _coerce(raw)  # int/float/bool/json
    return cfg.__class__(**data)
```

**Seed pack**

```python
@dataclass(frozen=True)
class SeedPack:
    global_seed: int = 0
    mis: int = 0
    ordering: int = 0
    grid: int = 0

# in RuntimeConfig
seeds: SeedPack = Field(default_factory=SeedPack)
```

**Pretty strategy “explain”**

```python
def explain_strategy_choice(kind: str, cfg: RuntimeConfig, backend) -> str:
    # Build a short sentence from predicates matched, e.g.:
    # "residual-grid: residual gate active, dense scopes, numba enabled"
```

---

### What this buys you

* **Users** get an obvious CLI (`pcct …`), curated profiles, sharp errors, and pretty summaries.
* **Contributors** get small, testable modules, explicit contexts, deterministic runs, and a plugin system.
* **You** get freedom to change internals without breaking users, faster reviews, and fewer “what does this flag do” issues.

If you want, I can turn any section above into a concrete PR diff scaffold next.
