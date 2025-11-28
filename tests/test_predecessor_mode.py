"""Tests for predecessor constraint in k-NN queries.

The predecessor constraint is essential for Vecchia GP approximations:
for query point index i, only points with index j < i can be neighbors.
"""
from __future__ import annotations

import numpy as np
import pytest

from covertreex import config as cx_config, reset_residual_metric
from covertreex.api import CoverTree, Runtime
from covertreex.metrics import build_residual_backend, configure_residual_correlation
from covertreex.metrics.residual import set_residual_backend


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    cx_config.reset_runtime_context()
    yield
    cx_config.reset_runtime_context()
    reset_residual_metric()
    set_residual_backend(None)


def test_predecessor_mode_basic() -> None:
    """Each query i should only have neighbors j < i."""
    np.random.seed(42)
    n_points = 100
    points = np.random.randn(n_points, 2).astype(np.float32)

    # Build residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=128,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=64,
    )
    configure_residual_correlation(backend_state)

    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine="rust-natural",  # Use natural ordering to avoid Hilbert mapping issues
        residual_use_static_euclidean_tree=True,
    )
    tree = CoverTree(runtime).fit(points, mis_seed=7)

    # Query with all indices
    query_indices = np.arange(n_points, dtype=np.int64).reshape(-1, 1)
    neighbors = tree.knn(query_indices, k=10, predecessor_mode=True)
    neighbors = np.asarray(neighbors)

    # Verify constraint: for each query i, all neighbors should have index < i
    for i in range(n_points):
        valid = neighbors[i][neighbors[i] >= 0]
        assert all(j < i for j in valid), f"Query {i} has invalid neighbor >= {i}: {valid}"


def test_predecessor_mode_early_queries() -> None:
    """Early queries should have fewer valid neighbors."""
    np.random.seed(42)
    n_points = 50
    points = np.random.randn(n_points, 2).astype(np.float32)

    # Build residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=128,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=64,
    )
    configure_residual_correlation(backend_state)

    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine="rust-natural",  # Use natural ordering to avoid Hilbert mapping issues
        residual_use_static_euclidean_tree=True,
    )
    tree = CoverTree(runtime).fit(points, mis_seed=7)

    # Query early indices
    query_indices = np.arange(10, dtype=np.int64).reshape(-1, 1)
    neighbors = tree.knn(query_indices, k=10, predecessor_mode=True)
    neighbors = np.asarray(neighbors)

    # Query 0 has no predecessors - all should be -1 (padding)
    assert np.all(neighbors[0] == -1), f"Query 0 should have no neighbors, got: {neighbors[0]}"

    # Query 1 can only have neighbor 0
    valid_1 = neighbors[1][neighbors[1] >= 0]
    assert len(valid_1) <= 1, f"Query 1 can only have 1 neighbor, got: {valid_1}"
    if len(valid_1) == 1:
        assert valid_1[0] == 0, f"Query 1's only valid neighbor is 0, got: {valid_1[0]}"

    # Query i can have at most i neighbors
    for i in range(10):
        valid = neighbors[i][neighbors[i] >= 0]
        assert len(valid) <= i, f"Query {i} should have at most {i} neighbors, got {len(valid)}"


def test_predecessor_mode_off_by_default() -> None:
    """Without predecessor_mode, neighbors can have any index."""
    np.random.seed(42)
    n_points = 20
    points = np.random.randn(n_points, 2).astype(np.float32)

    # Build residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=128,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=64,
    )
    configure_residual_correlation(backend_state)

    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine="rust-natural",  # Use natural ordering to avoid Hilbert mapping issues
        residual_use_static_euclidean_tree=True,
    )
    tree = CoverTree(runtime).fit(points, mis_seed=7)

    # Query index 0 with k=5, no predecessor_mode
    query_indices = np.array([[0]], dtype=np.int64)
    neighbors = tree.knn(query_indices, k=5, predecessor_mode=False)
    neighbors = np.asarray(neighbors)

    # Without predecessor constraint, query 0 should have neighbors
    valid = neighbors[0][neighbors[0] >= 0]
    assert len(valid) > 0, "Query 0 should have neighbors when predecessor_mode=False"


def test_predecessor_mode_with_distances() -> None:
    """Test that distances are returned correctly with predecessor_mode."""
    np.random.seed(42)
    n_points = 30
    points = np.random.randn(n_points, 2).astype(np.float32)

    # Build residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=128,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=64,
    )
    configure_residual_correlation(backend_state)

    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine="rust-natural",  # Use natural ordering to avoid Hilbert mapping issues
        residual_use_static_euclidean_tree=True,
    )
    tree = CoverTree(runtime).fit(points, mis_seed=7)

    # Query with distances
    query_indices = np.arange(10, 20, dtype=np.int64).reshape(-1, 1)
    neighbors, distances = tree.knn(query_indices, k=5, predecessor_mode=True, return_distances=True)
    neighbors = np.asarray(neighbors)
    distances = np.asarray(distances)

    assert neighbors.shape[0] == 10, "Should have 10 query results"
    assert distances.shape[0] == 10, "Should have 10 distance results"
    assert neighbors.shape == distances.shape, "Neighbors and distances shapes should match"

    # For valid (non-padded) entries, distances should be non-negative and finite
    valid_mask = neighbors >= 0
    if np.any(valid_mask):
        valid_distances = distances[valid_mask]
        assert np.all(valid_distances >= 0), "Valid distances should be non-negative"
        assert np.all(np.isfinite(valid_distances)), "Valid distances should be finite"

    # Verify predecessor constraint is satisfied
    for i, query_idx in enumerate(range(10, 20)):
        valid = neighbors[i][neighbors[i] >= 0]
        assert all(j < query_idx for j in valid), f"Query {query_idx} has invalid neighbor >= {query_idx}"


def test_predecessor_mode_with_subtree_bounds() -> None:
    """Test that subtree bounds optimization works correctly with rust-hilbert engine."""
    np.random.seed(42)
    n_points = 100
    points = np.random.randn(n_points, 2).astype(np.float32)

    # Build residual backend
    backend_state = build_residual_backend(
        points,
        seed=99,
        inducing_count=128,
        variance=1.0,
        lengthscale=1.0,
        chunk_size=64,
    )
    configure_residual_correlation(backend_state)

    from covertreex.engine import RustHilbertEngine, CoverTree as EngineCoverTree

    engine = RustHilbertEngine()
    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine="rust-hilbert",
        residual_use_static_euclidean_tree=True,
    )

    # Build tree WITH subtree bounds (opt-in)
    tree = engine.build(
        points,
        runtime=runtime.to_config(),
        residual_backend=backend_state,
        residual_params={"variance": 1.0, "lengthscale": 1.0},
        compute_predecessor_bounds=True,
    )

    # Verify bounds were computed
    assert tree.meta.get("predecessor_bounds_computed") is True
    assert tree.handle.subtree_min_bounds is not None

    # Query with predecessor_mode
    query_indices = np.arange(n_points, dtype=np.int64).reshape(-1, 1)
    ctx = cx_config.runtime_context()
    neighbors = engine.knn(
        tree,
        query_indices,
        k=10,
        return_distances=False,
        predecessor_mode=True,
        context=ctx,
        runtime=runtime.to_config(),
    )
    neighbors = np.asarray(neighbors)

    # Verify constraint: for each query i, all neighbors should have index < i
    for i in range(n_points):
        valid = neighbors[i][neighbors[i] >= 0]
        assert all(j < i for j in valid), f"Query {i} has invalid neighbor >= {i}: {valid}"
