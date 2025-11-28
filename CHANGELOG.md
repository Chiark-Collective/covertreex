# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **170x faster** than GPBoost for residual correlation k-NN queries
- **47,000 queries/second** on the gold standard benchmark (N=32k, D=3, k=50)
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
