# Residual Dense Baseline Regression (2025-11-12)

The full investigative log now lives under [`docs/journal/2025-11-12_residual_dense_regression.md`](journal/2025-11-12_residual_dense_regression.md) so this document can stay focused on the current status and entry points back into the history.

## Current Snapshot — 2025-11-12

- Dense residual builds that previously clocked **≈83 s build / 0.030 s query** on commit `48334de` now exceed **2 hours** at `5030894`; the regression is still unresolved.
- Bisecting between `48334de` (good) and `5030894` (bad) remains the top action item before resuming Phase 5 tuning or sparse/gate experiments.
- The dense traversal currently forces per-query whitened chunks (even with Gate‑1 off), so the scope streamer rewrite + budget schedule fixes called out in the journal entry must land before rerunning the 32 k presets.

## How to Use This File

1. Read the archived journal entry for the detailed telemetry, commands, and mitigation attempts.
2. Update [`BACKLOG.md`](../BACKLOG.md#dense-residual-regression-bisect) when the bisect lands or when the dense baseline is healthy again.

### Quick Links

- [Historical journal entry](journal/2025-11-12_residual_dense_regression.md)
- [November 2025 journal index](journal/2025-11.md)
