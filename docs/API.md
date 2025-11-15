# Covertreex API Overview

The `covertreex.api` module exposes a small, typed façade that keeps experiments out of the internal
modules. Treat it as the authoritative way to configure runtimes, stage residual policies, and drive
the public PCCT interface.

## Runtime configuration

`Runtime` is a thin dataclass that feeds the frozen `RuntimeModel` (a Pydantic schema backed by the
values in `covertreex.runtime.model`). Calling `.activate()` materialises a `RuntimeContext`: logging
levels are applied, backend state is initialised, and the context can be used as a context manager so
runs are lexically scoped.

```python
from covertreex.api import Runtime, Residual
from covertreex.algo.traverse import traverse_collect_scopes

runtime = Runtime(
    metric="euclidean",
    backend="numpy",
    precision="float64",
    enable_numba=True,
    diagnostics=False,
)

with runtime.activate() as context:
    traversal = traverse_collect_scopes(tree, batch_points, context=context)
    # low-level helpers always accept `context` so calling code never reaches for globals
```

Residual settings are aggregated through the `Residual` helper so callers can declaratively override
gate/lookup behaviour:

```python
residual_runtime = Runtime(
    metric="residual",
    backend="numpy",
    residual=Residual(
        gate1_enabled=True,
        lookup_path="docs/data/residual_gate_profile_diag0.json",
        scope_cap_path="docs/data/residual_scope_caps_32768.json",
    ),
)

with residual_runtime.activate() as context:
    # run residual traversal/conflict code inside this explicit context
    ...
```

### Profiles & overrides

Profiles in `profiles/*.yaml` provide curated baselines for common workloads. Build a runtime straight
from those definitions and layer overrides using dot-path syntax:

```python
runtime = Runtime.from_profile(
    "residual-fast",
    overrides=[
        "diagnostics.enabled=true",
        "residual.scope_member_limit=32768",
    ],
)
with runtime.activate():
    ...
```

Call `profiles.loader.available_profiles()` or inspect the YAML files directly to understand which
knobs each profile sets. The migration checklist in `docs/migrations/runtime_v_next.md` maps legacy
environment variables and flags to these dot-path overrides so older scripts can upgrade incrementally.

For advanced scenarios you can also inspect/clone the validated model:

```python
config_model = runtime.to_model()
assert config_model.metric == "euclidean"
patched = Runtime.from_config(config_model.to_runtime_config())
```

## PCCT façade

The `covertreex.api.PCCT` façade wraps the immutable tree plus helper operations. Typical usage:

```python
from covertreex.api import PCCT, Runtime
import numpy as np

runtime = Runtime(metric="euclidean", enable_numba=True)

points = np.random.default_rng(0).normal(size=(2048, 8))
queries = np.random.default_rng(1).normal(size=(512, 8))

pcct = PCCT(runtime)
tree = pcct.fit(points)
tree = PCCT(runtime, tree).insert(points * 0.5)
indices, distances = PCCT(runtime, tree).knn(queries, k=8)
```

Batch inserts accept both NumPy and JAX arrays depending on the runtime backend. Conflicts,
traversal, and MIS selection now run through strategy registries (`covertreex.algo.traverse` and
`covertreex.algo.conflict`) so custom strategies can be registered by calling
`register_traversal_strategy(...)` or `register_conflict_strategy(...)` during startup. Third-party
packages can also expose plugins via setuptools entry points (`covertreex.traversal`,
`covertreex.conflict`, `covertreex.metrics`); the CLI’s `pcct plugins` command reports which
plugins loaded plus their origin modules. When you step outside the `PCCT` façade, always reuse the
explicit `RuntimeContext` returned by `Runtime.activate()`—low-level helpers (`batch_insert`,
`traverse_collect_scopes`, conflict graph builders, telemetry, etc.) all accept a `context` keyword
and never consult global state.

## Command-line entrypoints

All benchmark commands now live under `cli/` with compatibility shims in `benchmarks/`:

- `python -m cli.pcct query …` replaces `benchmarks.queries`. The Typer app handles dataset
  generation, runtime activation, telemetry, and (optionally) baseline comparisons. The legacy
  `python -m cli.queries` runner now prints a warning before dispatching to the same command.
- `python -m cli.pcct build …` constructs trees (batch or prefix mode) and can export them as `.npz`
  artifacts for downstream consumption.
- `python -m cli.pcct benchmark …` repeats query runs to gather aggregate latency/build summaries.
- `python -m cli.pcct profile …` lists and describes runtime presets so scripts can adopt Stage 2
  profiles without hand-parsing YAML.
- `python -m cli.pcct doctor …` performs environment checks (Numba/JAX availability, artifact root
  accessibility, thread settings) for the selected profile and can gate CI with `--fail-on-warning`.
- `python -m cli.runtime_breakdown …` captures per-phase timings, CSV summaries, and plots.

You can continue invoking `python -m benchmarks.queries` while downstream tooling migrates, but the
`cli.*` modules are the supported entrypoints going forward.

## Examples & further reading

- `docs/examples/profile_workflows.md` — concrete CLI + API recipes built around Stage 2 profiles and
  the Stage 6 seed/determinism policy.
- `docs/migrations/runtime_v_next.md` — migration guide for legacy scripts (`cli.queries`,
  `COVERTREEX_*` env vars) moving to the profile-first CLI plus telemetry renderers.
