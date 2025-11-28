# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.3] - 2025-11-28

### Fixed
- **Engine build defaults**: `compute_predecessor_bounds` now defaults to `True` in both `RustNaturalEngine.build()` and `RustHilbertEngine.build()`, ensuring predecessor_mode works correctly out of the box
- **`_rust_knn_query` subtree bounds**: When querying via the `CoverTree.knn()` API path with `predecessor_mode=True`, subtree bounds are now computed on-demand if not already available

### Added
- **Regression test**: Added `test_predecessor_mode_via_cover_tree_factory` to verify the recommended `cover_tree()` factory API works correctly with predecessor_mode

## [0.4.2] - 2025-11-28

### Added
- **CLI `--predecessor-mode` flag**: Added support for `--predecessor-mode` in the CLI query command and benchmark scripts
- **`hilbert_order()` Rust function**: Exposed Hilbert curve ordering via `covertreex_backend.hilbert_order(coords)` for pre-sorting datasets
- **Comprehensive predecessor mode tests**: Added integration tests covering both `rust-natural` and `rust-hilbert` engines at gold standard scale (N=32768), verifying zero predecessor violations

### Fixed
- **rust-hilbert predecessor_mode correctness**: Both `rust-hilbert` and `rust-natural` engines now correctly maintain the predecessor constraint (neighbor index j < query index i) through proper `node_to_dataset` mapping
- **Default `compute_predecessor_bounds=True`**: Engine builds now compute predecessor bounds by default, enabling subtree pruning for predecessor-constrained queries

### Changed
- **Benchmark script**: `run_residual_gold_standard.sh` now supports `PREDECESSOR_MODE=1` environment variable

## [0.4.1] - 2025-11-28

### Fixed
- **predecessor_mode k-fulfillment**: Search now continues until k valid predecessors are found, ensuring query i returns exactly min(k, i) neighbors. Previously, budget-based termination caused early exit with fewer neighbors than requested.
- **Subtree exploration**: When a node fails predecessor constraint but its subtree may contain valid predecessors, the search now explores the subtree instead of skipping it entirely.

### Changed
- **Default engine for factory**: `cover_tree()` now defaults to `rust-natural` engine instead of `rust-hilbert` for better predecessor_mode support. The rust-hilbert engine's sparse tree structure is not optimal for predecessor queries.

## [0.4.0] - 2025-11-28

### Added
- **Kernel-based API**: New `cover_tree()` factory function as the recommended entry point
  - Accepts kernel objects (`Matern52`, `RBF`) instead of raw parameters
  - Supports pre-computed V-matrices for integration with existing GP code
  - Unified interface for Euclidean and residual correlation metrics
- **Kernel classes**: `Matern52` and `RBF` in `covertreex.kernels` module
- **BuiltCoverTree wrapper**: Consistent `knn(k=...)` interface across all engines

### Deprecated
- `ResidualCoverTree` class (use `cover_tree(coords, kernel=...)` instead)

## [0.3.1] - 2025-11-28

### Added
- **ResidualCoverTree**: Simplified high-level API for residual correlation k-NN queries (Vecchia GP). Replaces complex setup ceremony with a single class.

### Fixed
- rust-hilbert predecessor mode now correctly enforces constraints (was comparing Hilbert indices vs dataset indices)

## [0.3.0] - 2025-11-28

Vecchia GP predecessor constraint release.

### Added
- **Predecessor constraint** for Vecchia GP neighbor selection: `tree.knn(indices, k=10, predecessor_mode=True)` ensures query `i` only returns neighbors with index `j < i`
- **Subtree index bounds** (opt-in): `compute_predecessor_bounds=True` on engine build enables aggressive subtree pruning for predecessor-constrained queries. Computes min/max dataset indices per subtree to skip entire branches when all descendants are invalid.
- New `predecessor_filtered` telemetry counter tracks nodes filtered by the predecessor constraint
- New `subtrees_pruned` telemetry counter tracks entire subtrees skipped by index bounds

## [0.2.1] - 2025-11-27

### Added
- Quick-start CLI guide: `python -m covertreex` prints library usage examples
- Links to GitHub and PyPI in quick-start output

### Fixed
- Test isolation issues with residual backend configuration
- Skip WIP tests that document unfinished features (JAX batch insert, grid conflict builder)

## [0.2.0] - 2025-11-27

API cleanup release with library-first focus.

### Changed
- **Renamed PCCT → CoverTree** as the primary class name (`PCCT` remains as alias)
- `fit()`, `insert()`, `delete()` now return `CoverTree` instances (enables method chaining)
- Added `num_points` and `dimension` properties to `CoverTree`
- Updated README and CLAUDE.md for library-first usage
- CLI help enhanced with examples and better option descriptions

### Removed
- Deprecated gate-related functionality from Residual class
- Removed tests for deprecated gate functionality and implementation-specific tie-breaking

## [0.1.0] - 2025-11-27

First stable release with production-ready performance.

### Highlights
- **44,000 queries/second** residual correlation k-NN on gold standard (N=32k, D=3, k=50)
- Sub-second tree builds with Hilbert curve ordering
- AVX2 SIMD optimized V-matrix dot products
- Comprehensive documentation and benchmarks

### Added
- Explicit AVX2 SIMD (f32x8) optimization for V-matrix dot product (~5% improvement)
- Improved README with performance numbers, features list, and usage examples
- License badge and proper Apache 2.0 attribution

### Changed
- V-matrix dot product uses hand-optimized SIMD with 2x loop unrolling

## [0.0.3] - 2025-11-27

### Added
- pytest-chronicle integration for historical test result tracking
- PyPI publishing infrastructure (twine, build)

### Fixed
- **Critical**: Matern 5/2 kernel now uses consistent V-matrix computation. Previously, the V-matrix and p_diag were always built with RBF kernel regardless of the specified kernel type, causing a 130x slowdown with rust-hilbert engine (133s → 1.5s for 32K points). The fix ensures mathematical consistency in the residual correlation formula by using the same kernel type throughout.

## [0.0.2] - 2025-11-26

### Added
- High-performance Rust backend (`covertreex_backend`) with 10-50x faster tree construction
- Three execution engines: `python-numba`, `rust-natural`, `rust-hilbert`
- Matern52 kernel support in Rust engine
- Profile-driven runtime configuration (`profiles/*.yaml`)
- Comprehensive CLI (`python -m cli.pcct`) with telemetry, baselines, and doctor commands
- `survi_adapter.py` for survi-v2 integration

### Changed
- Rust backend enabled by default when available
- Unified seed handling through `Runtime.seeds` / `SeedPack`

## [0.0.1] - 2025-11-01

### Added
- Initial PCCT implementation with Python/Numba backend
- Batch insert, MIS, conflict graph, and traversal algorithms
- k-NN query support with residual correlation metric
- Baseline comparisons against PyPI covertree and mlpack
