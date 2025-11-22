# Agent Operating Guidelines

## Core Philosophy: R&D and Performance
- **Goal**: We are in an R&D phase focused on high-performance implementation of Cover Trees. Our primary objective is to beat `mlpack` benchmarks.
- **Modernization**: actively modernize the codebase (starting with the CLI) and prune obsolete code. If a module or script is no longer serving the goal or has been superseded, delete it.
- **Additive Optimization**: When implementing new features or performance optimizations, prefer additive approaches (new modules, toggles) *initially* to ensure we can benchmark against previous iterations. Once a new approach is proven superior, previous versions can be removed.
- **Reproducibility**: Ensure that critical benchmarks remains reproducible. If you break a benchmark, you must have a good reason (e.g., the benchmark itself was flawed or the code path is being entirely replaced by a better one).

## Operational Rules
- **Audit Surface**: Treat the repository as a workspace for finding the optimal implementation.
- **Pruning**: Don't be afraid to delete dead code, but ensure you aren't deleting the *only* working implementation of a critical feature.
- **Communication**: If making a destructive change to a core workflow, briefly document why in the commit message or PR.

## IMPORTANT RULES

- **Reproducibility**: Whenever a new benchmark is run with noteworthy results, update all documentation that contains benchmarks and make sure that:
    - The benchmark is reproducible, with the exact command used to run it included in the docs next to the benchmark results themselves.
    - The commit message mentions a benchmark result update
- **Benchmark Hygiene**: Do **NOT** add ad-hoc benchmark scripts (e.g., `benchmarks/temp_test.py`) without explicit user consultation. Rely on `run_residual_gold_standard.sh` and `tools/run_reference_benchmarks.py` to ensure consistency. If a new scenario is needed, add it as a job to the reference suite.
