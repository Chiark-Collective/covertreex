# Profile-Driven Workflows

Stage 7 of the DX/UX refactor standardises on curated runtime profiles (`profiles/*.yaml`) and Typer
subcommands so every benchmark, tree build, and telemetry artefact follows the same recipe. The
snippets below show how to combine CLI presets, Python APIs, and telemetry renderers without touching
legacy globals.

## CLI benchmark with overrides

```bash
# 1) Inspect the presets and pick one for your workload
python -m cli.pcct profile list
python -m cli.pcct profile describe residual-fast --format yaml

# 2) Run a benchmark with the chosen profile and dot-path overrides
python -m cli.pcct query \
  --profile residual-fast \
  --dimension 8 --tree-points 32768 --queries 1024 \
  --batch-size 512 --k 8 \
  --set diagnostics.enabled=false \
  --set residual.scope_member_limit=32768 \
  --set seeds.global=20250107 \
  --log-file artifacts/benchmarks/residual_fast_profile_run.jsonl

# 3) Render the telemetry for auditors
python -m cli.pcct telemetry render artifacts/benchmarks/residual_fast_profile_run.jsonl --format md
```

Key points:

- `--profile` selects one of the YAML presets documented in `profiles/`.
- `--set PATH=VALUE` mutates nested runtime fields using dot-path syntax; values follow YAML parsing
  rules, so booleans (`true/false`), numbers, and quoted strings behave as expected.
- Seeds now live inside the `SeedPack` section (`seeds.global`, `seeds.mis`, `seeds.batch_order`,
  `seeds.residual_grid`), making deterministic reruns straightforward.

## Embedding the same profile in Python

```python
from covertreex.api import PCCT, Runtime

runtime = Runtime.from_profile(
    "residual-fast",
    overrides=[
        "diagnostics.enabled=false",
        "seeds.global=20250107",
        "residual.scope_member_limit=32768",
    ],
)

with runtime.activate() as context:
    tree = PCCT(runtime).fit(tree_points)
    indices, distances = PCCT(runtime, tree).knn(query_points, k=8)

    # Low-level helpers take the same context explicitly
    # residual_scope_caps = collect_scope_caps(tree, context=context)
```

Tips:

- `Runtime.from_profile()` validates the YAML payload and produces the same configuration that the CLI
  would use.
- `Runtime.activate()` returns a `RuntimeContext` you can pass to traversal/conflict helpers without
  mutating globals.
- Call `runtime.to_model()` if you need to introspect the frozen `RuntimeModel` (e.g., to log the
  `SeedPack` or residual gate settings).

## Legacy migration cheat-sheet

- Replace `python -m cli.queries ...` with `python -m cli.pcct query --profile default ...`.
- Translate `COVERTREEX_*` environment variables to profile overrides (e.g.,
  `COVERTREEX_BACKEND=jax` → `--set backend=jax`, `COVERTREEX_SCOPE_CHUNK_TARGET=2048` →
  `--set scope_chunk_target=2048`).
- Use `python -m cli.pcct telemetry render ...` instead of ad-hoc scripts for JSONL summaries.

Full details, including environment-variable mappings and telemetry renderer flags, live in
`docs/migrations/runtime_v_next.md`.
