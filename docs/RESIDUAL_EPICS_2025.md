# Residual Roadmap — 2025-11-24 (WIP)

We’re at a fork: finish correctness parity, then chase performance along two axes (Numba + Rust). This doc tracks three epics to keep scope clear and benchmarkable.

**Baseline correction (2025-11-25):** The previously cited ~40k q/s Python “gold” run was invalid (it measured Euclidean distance over 1D indices and returned sequential IDs). The current trustworthy baseline for *correct* residual queries is the Rust parity path at ~8k q/s on the 32k/1k/k=50 suite. Update future tables against this baseline until the Numba pruning patch lands.

---

## Epic 1 — Parity Lock & Cross-Engine Correctness
**Goal:** Full behavioral parity between Python/Numba “gold” and Rust residual traversal (parity mode), with deterministic equality tests and matching telemetry payloads.

- **Scope**
  - Add deterministic cross-engine fixture (1k pts, 64 queries, k=1..10, mixed dims/seeds) asserting identical neighbors/distances (f64 parity) and stable ordering.
  - Telemetry parity: ensure Rust `rust_query_telemetry` fields mirror Numba JSONL (frontier, prunes, evals, yields, block sizes).
  - Duplicate/visited semantics: explicit visited set in parity path; coord payload option if needed.
  - CI hook: parity fixture test gated in CI (fast) + optional long-run benchmark gate.
- **Deliverables**
  - Tests under `tests/` (python + rust) with fixed seeds.
  - Telemetry schema note and sample payload in docs.
  - Benchmark log pair showing parity run (gold vs rust-hilbert parity) with commands.
- **Checkpoints**
  - ✅ Parity env toggle in place; telemetry emitted for f32/f64.
  - 🔲 Equality fixture merged & green in CI.
  - 🔲 Telemetry field-by-field match documented.
  - 🔲 Parity benchmark rerun after fixes; audit updated.

## Epic 2 — Portable Heuristics Back to Numba
**Goal:** Lift gold baseline by porting proven Rust heuristics into Numba behind flags, keeping outputs unchanged.

- **Scope**
  - Dynamic block sizing tied to frontier/active set.
  - Survivor budget ladder + low-yield early-stop (with k-safe guard).
  - Optional child reordering (stable) to tighten kth sooner.
  - Keep masked dedupe/level-cache reuse; no cap default changes.
  - Benchmark via `run_residual_gold_standard.sh` with opt-in flags; document commands/results.
- **Deliverables**
  - Flags in Numba traversal + defaults off.
  - Bench table comparing gold baseline vs each heuristic combo.
  - Notes on output invariance (neighbor sets unchanged).
- **Checkpoints**
  - 🔲 Flags implemented and unit-tested for k-safety.
  - 🔲 Benchmark deltas recorded; winners proposed for default.

## Epic 3 — Rust Perf Mode After Parity
**Goal:** Recover and surpass prior Rust perf while keeping parity mode as correctness guard.

- **Scope**
  - Re-enable fast paths (SIMD/tiled, SGEMM) and pruning bounds; tune stream tiling and cap/budget ladders.
  - Child ordering heuristics, masked dedupe, dynamic tiles, survivor budgets: measure selectively.
  - Build-time targets: match/better best-known Rust build; Query targets: approach gold QPS.
  - Telemetry-driven profiling to cut distance evals/heap pushes.
- **Deliverables**
  - “perf” preset/toggle distinct from parity.
  - Benchmark logs (best-of-5) against gold, with commands.
  - Updated audit noting perf + correctness safeguards.
- **Checkpoints**
  - 🔲 Perf preset defined; parity left untouched.
  - 🔲 Distance evals/heap pushes reduced vs current parity run.
  - 🔲 Perf benchmark beats previous Rust best; documented.

---

## Operating Rules (for all epics)
- Keep `run_residual_gold_standard.sh` as the primary reproducible benchmark; include commands next to results.
- No ad-hoc benchmark scripts; add scenarios to reference suite if new coverage is needed.
- Changes that alter outputs must first be proven behind flags; only flip defaults after equality tests pass.
- Update benchmark docs when publishing new results; note command + env vars.
