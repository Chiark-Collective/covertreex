# Runtime & CLI Refactor Plan

This is the living work plan for the “DX/UX refactor” described in `AUDIT.md`. The audit’s north star is “single mental model: build a tree, run queries, record telemetry” while keeping historical artefacts untouched. Each stage below explains **why** the change is needed (linking back to audit findings) and lists the concrete steps to implement it. Update the checkboxes and add PR links as we execute.

---

## Stage 0 – Baseline & Guardrails

**Why (AUDIT §Executive summary):** We must treat the repo as an audit surface. Before touching runtime internals we need reproducible artefacts and safety rails to prove we didn’t regress performance or behaviour.

- [ ] Snapshot representative runs (CLI + API) under current defaults; archive JSONL + configs under `artifacts/benchmarks/` for regression checks and link them here.
- [ ] Verify CI always exercises unit suite, telemetry tests, and at least one CLI smoke command (`python -m cli.queries ... --no-log-file`) so we detect CLI breakages immediately.
- [ ] Freeze current documentation of env knobs (`docs/CLI.md`, `docs/API.md`) to have a before/after diff once new profiles/CLI land.

## Stage 1 – Explicit Runtime Contexts

**Why (AUDIT §1 “Runtime/global state → explicit, testable contexts”):** Global singletons (`covertreex.runtime.config._CONTEXT_CACHE`) make runs order-dependent and prevent embedding covertreex in larger apps. We need immutable configs, explicit activation, and thread-local contexts to satisfy determinism and reproducibility goals.

1. **Model & Validation**
   - [x] Introduce `covertreex/runtime/model.py` with immutable `RuntimeConfig` plus nested models (`ResidualConfig`, `SeedPack`, `DiagnosticsConfig`). Use Pydantic to centralise validation instead of ad-hoc env parsing. *(Implemented 2025-01-05; env parsing now lives in `RuntimeModel`.)*
   - [x] Implement conversion helpers:
     - `RuntimeModel.from_env()` to parse `COVERTREEX_*` vars.
     - `RuntimeModel.from_legacy_config(legacy: RuntimeConfig)` to keep compatibility.

2. **Context Lifecycle**
   - [x] Implement new `RuntimeContext` exposing `.activate()` and `@contextmanager` interface so users can bind configs lexically (`with Runtime(cfg).activate(): ...`).
   - [x] Replace `_CONTEXT_CACHE` with a thread-local stack to avoid cross-thread contamination; keep shims in `covertreex.config` for backwards compatibility but route them through the new stack.
   - [x] Update `covertreex.api.Runtime` to wrap the Pydantic model internally but preserve the ergonomic dataclass interface (legacy keyword args feed into the model, `.activate()` returns the new context). *(Runtime now builds `RuntimeModel` instances, and API `PCCT` threads the resulting `RuntimeContext` down to batch-insert + k-NN helpers.)*

3. **Call-Site Migration**
   - [x] Audit all `cx_config.runtime_config()` usages and, where practical, thread an explicit `RuntimeContext` argument (e.g. `PCCT.fit`, traversal builders, conflict graph).
     - [x] PCCT façade + `batch_insert`/`plan_batch_insert`/`knn` now accept optional contexts so API consumers no longer touch globals during fit/query calls.
     - [x] Conflict graph, traversal, MIS helpers, persistence, and batch delete now accept explicit contexts/runtime configs so algorithm pipelines stay inside the activated `Runtime`.
     - [x] Guard dog: `tests/test_runtime_callsite_audit.py` fails the suite if new call sites appear.
   - [x] Update tests (`tests/test_api_pcct.py`, traversal/conflict suites) to create contexts explicitly to guarantee isolation.
    - [x] Added `tests/test_traverse.py::test_traversal_prefers_explicit_context` and the conflict-graph twin to prove explicit contexts beat ambient globals.
   - [x] Document the “old vs new” activation flow in `docs/API.md` including migration guidelines for library consumers (per AUDIT’s “document how to migrate your script in 3 steps”).
     - [x] API runtime section now calls out `RuntimeModel`, context managers, and updated PCCT snippets; `docs/CLI.md` explains how the CLI activates and threads contexts.

## Stage 2 – Config Profiles & Overrides

**Why (AUDIT §2 “Discoverable configuration” + §Quick PRs #3):** Users currently face 70+ CLI flags with no presets. Profiles provide a single source of truth for supported knobs and feed both CLI and API ergonomics.

- [x] Create `profiles/` with curated YAML: `default`, `residual-fast`, `residual-audit`, `cpu-debug`, etc. Include metadata (description, intended workload) per file.
- [x] Implement `profiles.loader.load_profile(name: str) -> RuntimeModel` with caching and validation errors that cite the offending field.
- [x] Add dot-path overrides utility (based on AUDIT helper) to support `--set residual.lengthscale=0.8` semantics across CLI/API.
- [x] Extend `Runtime` (or add `Runtime.from_profile()`) so scripts can use profiles without touching CLI internals.
- [x] Update docs to list available profiles and show how overrides merge; CLI now exposes `--profile` + `--set` to drive the new flow.

## Stage 3 – CLI Restructure (`pcct` Typer app)

**Why (AUDIT §2 “CLI surface → profiles + subcommands with Typer”):** The current `cli.queries` command is monolithic, hard to discover, and hides profile intent. A structured CLI with subcommands + nice help is central to the DX refactor.

1. **Scaffold**
   - [ ] Create `cli/pcct/__init__.py` and `cli/pcct/main.py` hosting the Typer root (`pcct`). Subcommands (subject to refinement):
     - `pcct build` (tree construction / persistence)
     - `pcct query` (knn runs, baseline comparisons)
     - `pcct benchmark` (batch sweeps, telemetry on/off)
     - `pcct profile` (list/describe/apply configs)
     - `pcct doctor` (environment guardrails)
   - [ ] Refactor logic currently in `cli/queries/app.py` into reusable modules so subcommands share the same runtime/profiles/telemetry glue.

2. **Compatibility Layer**
   - [ ] Keep `python -m cli.queries` operational for at least one release: parse legacy flags, translate them to profile + overrides, and invoke the equivalent `pcct` subcommand (emit warning linking to migration doc).
   - [ ] Update tests, README snippets, and docs to showcase `pcct` usage; add new CLI tests verifying subcommand help output.

3. **Runtime Visualization**
   - [ ] Implement `pcct profile list` and `pcct profile describe NAME --format yaml|json` to increase discoverability (per “discoverable configuration” audit goal).
   - [ ] Ensure every subcommand accepts `--profile` + `--set key=value` overrides wired through Stage 2 loader.

4. **Doctor Command**
   - [ ] Build `pcct doctor` that:
     - Checks numba/JAX availability vs runtime config.
     - Confirms artifact root is writable and logs warnings if telemetry is disabled (audit emphasises reproducible logs).
     - Reports CPU features, thread settings, and environment conflicts.
   - [ ] Provide exit codes suitable for CI preflight.

## Stage 4 – Plugin Registries

**Why (AUDIT §3 “Plugin registries”):** Current strategy selection hinges on import side effects. External teams can’t plug in new traversals/conflict builders without editing core modules, and tests can’t isolate strategies.

- [ ] Promote `algo/traverse/strategies/registry` to a first-class plugin interface: expose `register_traversal_strategy` publicly and add `covertreex.plugins.traversal` module as the sanctioned import path.
- [ ] Repeat for conflict graph strategies and metrics registry. Provide entry-point based auto-discovery (setuptools) so downstream packages can register without monkeypatching.
- [ ] Add CLI support (`pcct plugins list`) to show active strategies, their predicates, and source module (useful for audits).
- [ ] Write tests that:
  - Register mock strategies via entry points.
  - Ensure predicates throwing exceptions do not break selection (current behaviour is logging + fallback).
  - Confirm deregistration works to keep tests isolated.

## Stage 5 – Telemetry Schema & Formatters

**Why (AUDIT §7 “Telemetry schema consolidation”):** Today telemetry is JSONL without schema guarantees, and there’s no easy way to render results. The audit wants “one schema, one formatter, quick comparisons.”

- [ ] Define a canonical schema (versioned) that covers batch-level metrics, runtime snapshot, seeds, and residual annotations. Provide JSON Schema + Python dataclass for validation.
- [ ] Update `BenchmarkLogWriter` to emit `schema_id`, `schema_version`, and a deterministic `run_hash` computed from config + `SeedPack`. Include compatibility path for old logs.
- [ ] Build `pcct telemetry render LOG.jsonl --format md|csv|json --show fields` to pretty-print telemetry without spreadsheets.
- [ ] Refactor residual traversal telemetry to append structured records to the same schema (replace ad-hoc printouts once `render` exists).
- [ ] Document the schema in `docs/telemetry.md` and provide sample output for each format.

## Stage 6 – Determinism & Seed Packs

**Why (AUDIT §9 “Determinism & seeds policy” + testing plan):** Without structured seed handling, small code changes can shift random sequences. We need a unified seed policy covering MIS, batch ordering, residual gates, and telemetry hashing.

- [ ] Extend the runtime model with `SeedPack` (fields: `global_seed`, `mis`, `batch_order`, `residual_grid`, etc.) and propagate them through profiles + CLI.
- [ ] Refactor `algo/mis.batch_mis_seeds`, `algo/order`, residual traversal caches, and telemetry to consume `SeedPack` instead of ad-hoc seeds/environment vars.
- [ ] Include the seed pack (and config digest) in telemetry headers; use it to compute deterministic `run_hash`.
- [ ] Add regression tests:
  - `tests/test_determinism_runtime.py` (new) verifying identical PCCTrees/logs when rerunning with the same config/seeds.
  - CLI smoke test that runs `pcct query ... --set seeds.global=123` twice and compares telemetry hashes.

## Stage 7 – Polish & Documentation

**Why:** Once the refactor lands, the new ergonomics must be obvious to contributors and users. The audit explicitly requests migration docs and contributor ergonomics.

- [ ] Refresh `README.md`, `docs/CLI.md`, `docs/API.md`, and `docs/examples/*.md` with:
  - Profile-driven workflows.
  - `pcct` CLI examples.
  - Guidance on `RuntimeContext` activation for library embeds.
- [ ] Publish a migration guide (`docs/migrations/runtime_v_next.md`) covering:
  - Replacing `python -m cli.queries` with `pcct`.
  - Mapping environment variables to profile overrides.
  - Using the new telemetry renderers.
- [ ] Summarise outstanding future work (e.g. GPU backend revival, plugin packaging) in `BACKLOG.md` so it’s clear what remains post-refactor.

---

**Working notes**

- When a stage spans multiple PRs, add indented sub-bullets linking to PR numbers and key decisions.
- Keep this file ASCII and version-controlled; treat updates as part of each refactor PR so reviewers see progress.
- If scope changes (e.g. reorder stages), capture rationale inline referencing the relevant audit paragraph to maintain traceability.

---

**Working notes**

- Keep this file updated as stages land; status ticks + short changelog snippets help reviewers see momentum.
- When a stage spans multiple PRs, add indented bullet(s) under the relevant checkbox with PR links once merged.
