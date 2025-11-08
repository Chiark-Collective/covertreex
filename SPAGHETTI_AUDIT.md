Below is an end‑to‑end “Spaghetti Audit” for **covertreex** with a concrete refactor plan aimed at: (1) ergonomic, intuitive user APIs; (2) clear separation of concerns; (3) extensibility without global state; (4) repeatable benchmarking & diagnostics; and (5) maintainable tests. I keep this practical—what to extract, where to move it, how to name things, how the public API should feel, and how to ship it incrementally without breaking your tests.

---

## Executive summary

**What’s strong now**

* The **PCCT** core types are already clean and immutable (`PCCTree`, `TreeBackend`, persistence journal, traversal & conflict graph dataclasses).
* Metrics and diagnostics are **present** (resource snapshots, residual gates, scope caps, JSONL writers).
* Benchmarks + tests are **rich**, exercise the full pipeline (insert/delete/traverse/conflict/MIS/knn), and already validate invariants.

**Main pain points**

1. **Orchestration mixed with types**: planner results (`BatchInsertPlan`, `PrefixBatch*`) live beside pipeline orchestration, making it hard to reuse types without importing runners.
2. **Runtime is ambient**: environment mutation is used to configure behavior (residual gates, prefix schedule), which leaks across runs and tests.
3. **Strategy selection is spread out**: conflict‑graph variants (dense/segmented/residual/grid) and traversal variants (dense/sparse/residual) are chosen by ad‑hoc logic instead of strategy objects.
4. **Bench CLI is too big**: `benchmarks/queries.py` interleaves CLI, env mutation, residual metric wiring, baselines, and telemetry in ~1k lines.
5. **Artefact sprawl**: dozens of `.jsonl`/`.log` files in repo root—difficult to scan and review; encourages accidental churn.
6. **Ergonomics**: the public “API” surfaces feel like internal building blocks. Users should be able to do `fit/insert/delete/knn` with a small, predictable set of calls.

---

## Public API – what end users should import

Target a tiny, stable surface under `covertreex.api` with fluent, discoverable calls. Keep everything else behind **explicit subpackages**.

```python
from covertreex.api import PCCT, Runtime, Residual

# 1) Configure once (no ambient env mutation)
rt = Runtime(
    backend="numpy",           # or "jax"
    precision="float64",
    metric="euclidean",        # or "residual"
    devices=("cpu",),          # future-proof for gpu
    diagnostics=True,
    conflict_graph="grid",     # "dense" | "segmented" | "residual" | "grid"
    batch_order="hilbert",     # "natural" | "hilbert" | "prefix"
    mis_seed=0,
    scope_chunk_target=8192,
    residual=Residual(         # Optional metric-specific policy
        gate1_enabled=True,
        gate1_alpha=2.0,
        gate1_margin=0.0,
        gate1_radius_cap=10.0,
        scope_cap_path="docs/data/residual_scope_caps_32768.json",
        profile_lookup_path="docs/data/residual_gate_profile_32768_caps.json",
    ),
)

# 2) Build a tree
tree = PCCT(rt).fit(points)              # batch builder
# 3) Update
tree = PCCT(rt, tree).insert(batch)      # returns a *new* tree
tree = PCCT(rt, tree).delete(indices)
# 4) Query
idx, dist = PCCT(rt, tree).nearest(query, return_distances=True)
knn_idx = PCCT(rt, tree).knn(queries, k=10)  # returns (idx, dist) if requested
```

**Design notes**

* `Runtime` (immutable dataclass) replaces ambient `os.environ`. It composes:

  * general exec knobs (backend, precision, devices),
  * algorithm knobs (conflict graph impl, traversal flavor, batch order),
  * metric‑specific policy (`Residual`) via typed subobject.
* `PCCT` is a thin façade over the orchestration pipeline, with clear methods: `fit`, `insert`, `delete`, `knn`, `nearest`.
* Existing lower‑level functions remain (for tests and power users), but the official docs recommend the `api` façade.

---

## Package layout – clear layers & seams

Refactor to these **packages** and **roles** (all code that exists today stays, but many symbols move):

```
covertreex/
├── api/                     # public, ergonomic façade (new)
│   ├── __init__.py          # exports PCCT, Runtime, Residual policies
│   ├── pcct.py              # class PCCT: fit/insert/delete/knn façade
│   └── runtime.py           # Runtime dataclass (+ validation) and adapters
├── core/                    # immutable domain types & persistence
│   ├── tree.py              # PCCTree, TreeBackend
│   ├── persistence.py       # journals, clone-on-write
│   ├── metrics.py           # MetricRegistry, euclidean/residual registration
│   └── ...
├── algo/                    # algorithms with no I/O or env effects
│   ├── traverse/            # strategy objects (dense/sparse/residual)
│   ├── conflict/            # strategy objects (dense/segmented/residual/grid)
│   ├── mis/                 # MIS runners (jax/numba)
│   ├── batch/               # planning & updates (insert/delete)
│   └── order/               # batch ordering (natural/hilbert/prefix)
├── metrics/                 # metric-specific helpers & policies
│   ├── residual/            # ResidualCorrHostData, gate1, scope caps, profiles
│   └── ...
├── runtime/                 # configuration, logging, diagnostics (no env writes)
│   ├── config.py            # RuntimeConfig parsing & merging (env/args/obj)
│   ├── logging.py           # get_logger
│   └── diagnostics.py       # OperationMetrics, resource snapshots
├── telemetry/               # logs, writers, schemas (new)
│   ├── logs.py              # BenchmarkLogWriter, residual cap recorders
│   └── schemas.py           # JSON schemas for JSONL events
├── cli/                     # small, testable CLI shims (new)
│   ├── queries.py           # parse args -> call runners (no env mutation)
│   ├── batch_ops.py
│   └── runtime_breakdown.py
└── tests/                   # unchanged in spirit; imports updated
```

**Why this works**

* **API**: thin, documented, versioned; everything else is “internal”.
* **algo/**: pure and deterministic; no logging, no env; only types in/out.
* **runtime/** & **telemetry/**: the only places with side effects (logging, resource reads).
* **metrics/**: sealed per‑metric policies (e.g., residual) isolated from orchestration.

---

## Concrete refactors (file‑by‑file)

Below I map your current files to the target structure with what to extract and how to name it.

### 1) `benchmarks/queries.py`  →  `cli/queries.py` + `telemetry/logs.py` + `api/runtime.py`

* **Problems**: argparse, global env mutation (`_apply_residual_gate_preset`), residual wiring, baselines, and JSONL logging are interleaved.
* **Refactor**:

  * `BenchmarkLogWriter`, `ResidualScopeCapRecorder` ⇒ **move** to `telemetry/logs.py`. Make them **pure** (no `os.environ` writes). Accept a typed `Runtime` snapshot and a run ID.
  * `_apply_residual_gate_preset` ⇒ **delete and replace** with construction of `Runtime(residual=Residual(...))`. If you still need presets, codify them as dataclasses under `api.runtime`:

    ```python
    @dataclass(frozen=True)
    class ResidualPreset:
        name: str
        gate1_alpha: float
        # ...

    def apply_preset(rt: Runtime, preset: ResidualPreset) -> Runtime: ...
    ```
  * Kernel wiring (`_build_residual_backend`) ⇒ **move** to `metrics/residual/host_backend.py` so both CLI and library users share it.
  * `_generate_points_*` ⇒ **put** into `tests/utils/datasets.py` (and import here) to ensure one source of randomness and shapes.
  * CLI `main()` ⇒ **only parses args**, constructs `Runtime`, then calls `api.PCCT(...).fit/knn`. JSONL is written via `telemetry.logs`.

### 2) `covertreex/algo/batch_insert.py`  →  split types from orchestration

* **Move “types”**: `BatchInsertPlan`, `LevelSummary`, `PrefixBatchGroup`, `PrefixBatchResult`, `BatchInsertTimings` ⇒ `algo/batch/types.py`.
* **Move batch ordering helpers**: `_prepare_batch_points`, `_choose_prefix_factor`, `_prefix_slices` ⇒ `algo/order/ops.py`.
* **Keep orchestration**: `plan_batch_insert`, `batch_insert`, `batch_insert_prefix_doubling` in `algo/batch/insert.py` which **only** sequences steps: traverse → conflict graph → MIS → persistence. Inject strategies and runtime via arguments (no env reads).

### 3) `covertreex/algo/conflict_graph*.py`  →  Strategy objects

* Introduce:

  ```python
  class ConflictGraphStrategy(Protocol):
      def build(self, tree: PCCTree, traversal: TraversalResult, batch_points: Any, *, backend: TreeBackend) -> ConflictGraph: ...
  ```
* Implement four strategies in separate modules:

  * `algo/conflict/dense.py`
  * `algo/conflict/segmented.py`
  * `algo/conflict/residual.py`
  * `algo/conflict/grid.py`
* `algo/conflict/factory.py` chooses the strategy from `Runtime.conflict_graph`. No `os.environ` peeking. The returned `ConflictGraph` still carries `ConflictGraphTimings` (including `grid_*` counters).
* Keep Numba helpers (`_grid_numba.py`) under `algo/conflict/_numba/` to be explicit.

### 4) `covertreex/algo/traverse*.py`  →  Strategy objects

* Mirror the approach used for conflict graphs:

  ```python
  class TraversalStrategy(Protocol):
      def collect(self, tree: PCCTree, batch: Any, *, backend: TreeBackend, runtime: Runtime) -> TraversalResult: ...
  ```
* Implement `EuclideanDenseTraversal`, `EuclideanSparseTraversal`, `ResidualTraversal` in `algo/traverse/`.
* Dispatch only in `algo/traverse/factory.py`.

### 5) `covertreex/config.py`  →  `runtime/config.py` & `api/runtime.py`

* **Stop writing** to env. Replace with a pure `Runtime` dataclass and **layered parsing**:

  1. Defaults
  2. Environment (optional)
  3. CLI args (optional)
  4. Programmatic overrides (preferred)
* `runtime/config.py` keeps env parsing helpers for backward compatibility; `api/runtime.py` exposes the dataclasses used by library users.
* Add `.validate()` method to `Runtime` to catch misconfigurations early (e.g., unknown metric/strategy values).

### 6) `covertreex/metrics/residual*.py`  →  `metrics/residual/`

* Group residual pieces under `metrics/residual/`:

  * `host_backend.py` (currently `ResidualCorrHostData` + gate/profile/lookup wiring)
  * `gate1.py` (gate‑1 threshold logic, FP/FN audit, `ResidualGateTelemetry`)
  * `scope_caps.py` (cap tables & cache)
  * `profile.py` (profile model + persistence)
  * `pairwise.py` (distance/bounds helpers)
* Expose a single **policy** object that implements:

  ```python
  @dataclass(frozen=True)
  class Residual:
      gate1_enabled: bool = False
      gate1_alpha: float = 2.0
      gate1_margin: float = 0.0
      gate1_radius_cap: float = float("inf")
      scope_cap_path: str | None = None
      profile_lookup_path: str | None = None
      # ...

      def build_host_backend(self, points: np.ndarray, *, seed: int, inducing: int, ...) -> ResidualCorrHostData: ...
  ```
* Metrics registration (`configure_residual_correlation`) becomes a pure function taking a `ResidualCorrHostData` and installing kernels via the **existing** `MetricRegistry` (still located in `core/metrics.py`).

### 7) `benchmarks/runtime_breakdown.py`  →  `cli/runtime_breakdown.py` + `telemetry/schemas.py`

* Keep the resource snapshot logic in `runtime/diagnostics.py` (already there).
* Move CSV emit schema and segment naming to `telemetry/schemas.py` so future dashboards can rely on stable column names.
* Ensure `_plot_results` is optional and headless‑safe; plotting imports remain local.

### 8) `covertreex/queries/knn.py` & `_knn_numba.py`

* Keep both implementations, but expose a single **selector** that uses `Runtime.enable_numba` and/or an explicit `backend="numba"` knob instead of implicit flags.
* Strengthen return type shape guarantees (document `(#queries, k)` for indices and distances, `(#queries,)` for nearest).
* Provide a `queries/api.py` with:

  ```python
  def knn(tree: PCCTree, queries: ArrayLike, *, k: int, return_distances: bool, runtime: Runtime) -> ...
  ```

---

## Ergonomics: consistent naming & typing

* Prefer **explicit types** over `Any` in dataclasses. For host/device arrays, adopt a single alias:

  ```python
  from numpy.typing import ArrayLike as NPArrayLike
  ArrayLike = Any  # keep, but annotate fields that are definitely np.ndarray
  ```
* Public dataclasses should avoid `Any` except for backend‑dependent arrays. Internally, use `np.ndarray` where accurate (e.g., in Numba views).
* Follow a consistent **snake_case** for telemetry keys (already good); keep them stable in `telemetry/schemas.py`.

---

## Configuration model (no global mutation)

* **Now**: some helpers write to `os.environ` mid‑run (e.g., `_apply_residual_gate_preset`).
* **Goal**: all behavior controlled by an **immutable `Runtime`** instance, passed down the call chain.
* Add:

  ```python
  @dataclass(frozen=True)
  class Runtime:
      backend: Literal["numpy", "jax"] = "numpy"
      precision: Literal["float32", "float64"] = "float64"
      devices: Tuple[str, ...] = ("cpu",)
      metric: Literal["euclidean", "residual"] = "euclidean"
      conflict_graph: Literal["dense", "segmented", "residual", "grid"] = "dense"
      batch_order: Literal["natural", "hilbert", "prefix"] = "natural"
      mis_seed: int | None = None
      diagnostics: bool = True
      scope_chunk_target: int = 8192
      # metric policy:
      residual: Residual | None = None
      # toggles:
      enable_numba: bool = False
      enable_sparse_traversal: bool = False

      def validate(self) -> None: ...
  ```
* Provide shims in `runtime/config.py`:

  * `Runtime.from_env()` (parity with current `from_env`).
  * `describe()`—a typed, serialisable view (replacing `describe_runtime()`).

---

## Telemetry & artefacts

* Create `artifacts/` directory with subfolders:

  ```
  artifacts/
  ├── benchmarks/   # jsonl, csv
  ├── profiles/     # residual gate profiles/lookups
  └── logs/         # raw run logs
  ```
* Put a conservative `.gitignore` in `artifacts/` and **adjust CLI defaults** to write there.
* Define minimal JSON Schemas (Python dicts in `telemetry/schemas.py`) for:

  * `benchmark_batch` event
  * `residual_scope_cap` event
  * `runtime_breakdown` event
* `BenchmarkLogWriter` accepts a `path` or a `BaseWriter` interface so users can stream to stdout or a file.

---

## Tests: keep coverage, improve isolation

* **Stop env leaks**: with the `Runtime` object, tests pass explicit runtimes and avoid `monkeypatch` except for true env‑parsing tests.
* Add **property-based** checks for:

  * `batch_order.compute` invariants (permutation validity, monotonicity of Hilbert code).
  * MIS: independent set validity vs. conflict graph edges for random seeds.
* Move dataset builders into `tests/utils/datasets.py` to avoid duplicated RNG logic across tests.

---

## Extensibility hooks

* **Registries** (thin, typed, optional):

  * `metrics.registry` already exists. Mirror it for:

    * `conflict_graph.registry` (name → `ConflictGraphStrategy`)
    * `traversal.registry` (name → `TraversalStrategy`)
  * This allows adding new strategies without touching orchestrators.

* **Backends**:

  * Keep `TreeBackend.numpy/jax` factories.
  * Add `TreeBackend.gpu` **placeholder** that raises with a helpful message unless enabled—matching your GPU‑gated plan in `PARALLEL_COMPRESSED_PLAN.md`.

---

## Migration plan (safe, incremental)

**Phase 1 (PRs 1–3): build the seams, no behavior changes**

1. Introduce `api.Runtime` + adapters in `runtime/config.py` (keep old functions; mark as internal).
2. Extract `telemetry/logs.py` and redirect `benchmarks/queries.py` to use it; **stop env mutation** in the CLI by building a `Runtime`.
3. Split `batch_insert.py` types into `algo/batch/types.py`; import paths updated; tests unchanged.

**Phase 2 (PRs 4–7): strategies & factories**

4. Add traversal strategy classes + factory; change orchestrator to accept `TraversalStrategy` (default from runtime).
5. Add conflict graph strategy classes + factory; orchestrator selects via runtime.
6. Move residual metric bits into `metrics/residual/*`; provide a single `Residual` policy used by `Runtime` and by the CLI.
7. Stabilise telemetry schemas and redirect outputs to `artifacts/`.

**Phase 3 (PRs 8–10): public façade & docs**

8. Add `covertreex.api.PCCT` façade; refactor examples and smoke tests to use it.
9. Write **API docs** (README section + `docs/CORE_IMPLEMENTATIONS.md` update + a new `docs/API.md`).
10. Sweep for `os.environ` writes; remove or confine to `runtime/config.py` env‑parse only.

---

## Risk register & mitigations

* **Import churn**: moving symbols breaks imports.

  * **Mitigation**: add re‑export shims and `DeprecationWarning`s for one release (e.g., `covertreex.algo.batch_insert.BatchInsertPlan` re‑exports from `algo.batch.types`).
* **Hidden env dependencies**: some tests may assume ambient flags.

  * **Mitigation**: update tests to pass a `Runtime`; keep env parsing for CLI tests only.
* **Performance regressions** due to extra indirection.

  * **Mitigation**: strategies are zero‑cost at runtime (one function pointer); keep data in NumPy/JAX arrays unchanged.

---

## Concrete API sketches (drop‑in building blocks)

### Strategy factories

```python
# algo/traverse/factory.py
def from_runtime(rt: Runtime) -> TraversalStrategy:
    if rt.metric == "residual":
        return ResidualTraversal()
    if rt.enable_sparse_traversal:
        return EuclideanSparseTraversal()
    return EuclideanDenseTraversal()

# algo/conflict/factory.py
def from_runtime(rt: Runtime) -> ConflictGraphStrategy:
    return {
        "dense": DenseConflictStrategy(),
        "segmented": SegmentedConflictStrategy(),
        "residual": ResidualConflictStrategy(),
        "grid": GridConflictStrategy(),
    }[rt.conflict_graph]
```

### Batch order module

```python
# algo/order/api.py
@dataclass(frozen=True)
class BatchOrderResult:
    permutation: np.ndarray | None
    metrics: dict[str, float]

def compute(points: np.ndarray, *, strategy: str, seed: int | None) -> BatchOrderResult: ...
```

### Residual policy

```python
# metrics/residual/policy.py
@dataclass(frozen=True)
class Residual:
    gate1_enabled: bool = False
    gate1_alpha: float = 2.0
    gate1_margin: float = 0.0
    gate1_radius_cap: float = float("inf")
    scope_cap_path: str | None = None
    profile_lookup_path: str | None = None

    def build_host_backend(... ) -> ResidualCorrHostData: ...
```

---

## Developer ergonomics & CI

* Add `py.typed` to package → downstream mypy users get types.
* Adopt **ruff** + **mypy** + **pytest -q** in CI.
* Add `pre-commit` with black/isort/ruff hooks.
* Keep `NUMBA` optional but exercise both code paths in CI matrix:

  * `ENABLE_NUMBA=0/1`, `BACKEND=numpy/jax`, `METRIC=euclidean/residual`.

---

## Documentation updates

* `README.md`: new “Getting started” with `covertreex.api.PCCT`.
* `docs/CORE_IMPLEMENTATIONS.md`: update architecture diagram to show strategies and runtime.
* `PARALLEL_COMPRESSED_PLAN.md`: reference the `Runtime` knobs that will eventually flip GPU pathways on.

---

## What to cut or rename (quick wins)

* Rename `benchmarks/queries.py` → `cli/queries.py`; keep an entry‑point script `covertreex-queries`.
* Rename `covertreex/baseline.py` classes to `SequentialCoverTreeBaseline`, `GPBoostCoverTreeBaseline` (already mostly clear).
* Replace trailing `...` stubs with `raise NotImplementedError("TODO: audit stub")` in modules users might mistakenly import—this avoids silent failures.

---

## Acceptance criteria (done = we can ship)

* ✅ Public façade `covertreex.api` has `Runtime`, `PCCT`, `Residual` with examples.
* ✅ No module mutates `os.environ` during execution; env parsing only occurs in CLI and `Runtime.from_env()`.
* ✅ Conflict/traversal builders chosen via strategy factories; tests pass across strategy permutations.
* ✅ Benchmarks write to `artifacts/` with stable JSONL schemas.
* ✅ Existing tests pass (with import shims where needed) and new API has basic smoke tests.

---

## Final notes tailored to your codebase

* Your **dataclasses** are already the right abstraction: keep them, but relocate them out of orchestration modules so they can be re‑used and documented independently.
* The **residual gating** work is excellent; turning it into a first‑class *policy* object will pay off (easy to A/B, serialize, and log without env leaks).
* The **grid** conflict builder belongs as a distinct strategy with its own counters; you’ve prepped timings and counters—moving to strategies will also make the MIS experiments cleaner.

If you want, I can draft the exact `covertreex/api/pcct.py` and `api/runtime.py` files and a compatibility shim mapping old imports to new modules so your test suite continues to pass while you migrate incrementally.
