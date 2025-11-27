# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4] - 2025-11-27

### Added
- Explicit AVX2 SIMD (f32x8) optimization for V-matrix dot product in Rust backend (~5% query throughput improvement)

### Changed
- V-matrix dot product now uses hand-optimized SIMD with 2x loop unrolling instead of relying on auto-vectorization

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
