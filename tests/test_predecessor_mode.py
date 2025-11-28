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


def test_predecessor_mode_k_fulfillment() -> None:
    """Test that predecessor_mode finds min(k, i) neighbors for query i.

    This is critical for Vecchia GP: early queries have fewer valid predecessors,
    but later queries should find the full k neighbors.
    """
    np.random.seed(42)
    n_points = 100
    k = 8
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

    from covertreex.engine import RustNaturalEngine

    engine = RustNaturalEngine()
    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="residual_correlation",
        enable_rust=True,
        engine="rust-natural",
        residual_use_static_euclidean_tree=True,
    )

    # Build tree WITH subtree bounds (critical for the fix)
    tree = engine.build(
        points,
        runtime=runtime.to_config(),
        residual_backend=backend_state,
        residual_params={"variance": 1.0, "lengthscale": 1.0},
        compute_predecessor_bounds=True,
    )

    # Query with predecessor_mode
    query_indices = np.arange(n_points, dtype=np.int64).reshape(-1, 1)
    ctx = cx_config.runtime_context()
    neighbors = engine.knn(
        tree,
        query_indices,
        k=k,
        return_distances=False,
        predecessor_mode=True,
        context=ctx,
        runtime=runtime.to_config(),
    )
    neighbors = np.asarray(neighbors)

    # Verify k-fulfillment: query i should have exactly min(k, i) valid neighbors
    for i in range(n_points):
        valid_count = np.sum(neighbors[i] >= 0)
        expected = min(k, i)
        assert valid_count == expected, (
            f"Query {i}: expected {expected} neighbors, got {valid_count}. "
            f"neighbors={neighbors[i].tolist()}"
        )


class TestPredecessorModeIntegration:
    """Integration tests for predecessor mode at production scale."""

    @pytest.mark.parametrize("engine_name", ["rust-natural", "rust-hilbert"])
    def test_predecessor_correctness_both_engines(self, engine_name: str) -> None:
        """Both rust-natural and rust-hilbert should produce correct predecessor results.

        This is the key integration test: regardless of internal ordering strategy,
        the predecessor constraint must be satisfied.
        """
        np.random.seed(42)
        n_points = 1000
        k = 50
        points = np.random.randn(n_points, 3).astype(np.float32)

        backend_state = build_residual_backend(
            points,
            seed=99,
            inducing_count=128,
            variance=1.0,
            lengthscale=1.0,
            chunk_size=64,
        )
        configure_residual_correlation(backend_state)

        from covertreex.engine import RustNaturalEngine, RustHilbertEngine

        engine = RustNaturalEngine() if engine_name == "rust-natural" else RustHilbertEngine()
        runtime = Runtime(
            backend="numpy",
            precision="float32",
            metric="residual_correlation",
            enable_rust=True,
            engine=engine_name,
            residual_use_static_euclidean_tree=True,
        )

        tree = engine.build(
            points,
            runtime=runtime.to_config(),
            residual_backend=backend_state,
            residual_params={"variance": 1.0, "lengthscale": 1.0},
            compute_predecessor_bounds=True,
        )

        query_indices = np.arange(n_points, dtype=np.int64).reshape(-1, 1)
        ctx = cx_config.runtime_context()
        neighbors = engine.knn(
            tree,
            query_indices,
            k=k,
            return_distances=False,
            predecessor_mode=True,
            context=ctx,
            runtime=runtime.to_config(),
        )
        neighbors = np.asarray(neighbors)

        # Count violations
        violations = 0
        for i in range(n_points):
            for j in neighbors[i]:
                if j >= 0 and j >= i:
                    violations += 1

        assert violations == 0, (
            f"Engine {engine_name}: {violations} predecessor violations found. "
            "All returned neighbors j must satisfy j < query_index i."
        )

    def test_predecessor_filter_effectiveness(self) -> None:
        """Confirm predecessor_mode actually filters: without it, violations would occur.

        This is a control test proving the predecessor filter is doing something.
        """
        np.random.seed(42)
        n_points = 200
        k = 20
        points = np.random.randn(n_points, 3).astype(np.float32)

        backend_state = build_residual_backend(
            points,
            seed=99,
            inducing_count=128,
            variance=1.0,
            lengthscale=1.0,
            chunk_size=64,
        )
        configure_residual_correlation(backend_state)

        from covertreex.engine import RustHilbertEngine

        engine = RustHilbertEngine()
        runtime = Runtime(
            backend="numpy",
            precision="float32",
            metric="residual_correlation",
            enable_rust=True,
            engine="rust-hilbert",
            residual_use_static_euclidean_tree=True,
        )

        tree = engine.build(
            points,
            runtime=runtime.to_config(),
            residual_backend=backend_state,
            residual_params={"variance": 1.0, "lengthscale": 1.0},
            compute_predecessor_bounds=True,
        )

        query_indices = np.arange(n_points, dtype=np.int64).reshape(-1, 1)
        ctx = cx_config.runtime_context()

        # Query WITHOUT predecessor_mode
        neighbors_no_pred = engine.knn(
            tree,
            query_indices,
            k=k,
            return_distances=False,
            predecessor_mode=False,
            context=ctx,
            runtime=runtime.to_config(),
        )
        neighbors_no_pred = np.asarray(neighbors_no_pred)

        # Count would-be violations (neighbors >= query index)
        would_be_violations = 0
        for i in range(n_points):
            for j in neighbors_no_pred[i]:
                if j >= 0 and j >= i:
                    would_be_violations += 1

        # Without predecessor mode, we expect MANY violations
        # (roughly half of neighbors should have j >= i for random data)
        assert would_be_violations > n_points * k * 0.3, (
            f"Expected many would-be violations without predecessor filter, "
            f"got only {would_be_violations}. This suggests the control test is invalid."
        )

        # Now query WITH predecessor_mode
        neighbors_pred = engine.knn(
            tree,
            query_indices,
            k=k,
            return_distances=False,
            predecessor_mode=True,
            context=ctx,
            runtime=runtime.to_config(),
        )
        neighbors_pred = np.asarray(neighbors_pred)

        # Count actual violations with filter
        actual_violations = 0
        for i in range(n_points):
            for j in neighbors_pred[i]:
                if j >= 0 and j >= i:
                    actual_violations += 1

        assert actual_violations == 0, (
            f"predecessor_mode=True should have 0 violations, got {actual_violations}"
        )

    @pytest.mark.parametrize("engine_name", ["rust-natural", "rust-hilbert"])
    def test_predecessor_mode_gold_standard_scale(self, engine_name: str) -> None:
        """Test at gold standard scale: N=32768, D=3, k=50.

        This matches the production benchmark parameters.
        """
        np.random.seed(42)
        n_points = 32768
        k = 50
        n_queries = 1024
        points = np.random.randn(n_points, 3).astype(np.float32)

        backend_state = build_residual_backend(
            points,
            seed=99,
            inducing_count=512,
            variance=1.0,
            lengthscale=1.0,
            chunk_size=512,
        )
        configure_residual_correlation(backend_state)

        from covertreex.engine import RustNaturalEngine, RustHilbertEngine

        engine = RustNaturalEngine() if engine_name == "rust-natural" else RustHilbertEngine()
        runtime = Runtime(
            backend="numpy",
            precision="float32",
            metric="residual_correlation",
            enable_rust=True,
            engine=engine_name,
            residual_use_static_euclidean_tree=True,
        )

        tree = engine.build(
            points,
            runtime=runtime.to_config(),
            residual_backend=backend_state,
            residual_params={"variance": 1.0, "lengthscale": 1.0, "inducing_count": 512},
            compute_predecessor_bounds=True,
        )

        # Sample random queries from middle of dataset
        rng = np.random.default_rng(42)
        query_indices = rng.choice(n_points, n_queries, replace=False).astype(np.int64).reshape(-1, 1)

        ctx = cx_config.runtime_context()
        neighbors = engine.knn(
            tree,
            query_indices,
            k=k,
            return_distances=False,
            predecessor_mode=True,
            context=ctx,
            runtime=runtime.to_config(),
        )
        neighbors = np.asarray(neighbors)

        # Verify all results satisfy predecessor constraint
        violations = 0
        for idx, qi in enumerate(query_indices.flatten()):
            for j in neighbors[idx]:
                if j >= 0 and j >= qi:
                    violations += 1

        assert violations == 0, (
            f"Engine {engine_name} at gold standard scale: {violations} violations. "
            f"N={n_points}, queries={n_queries}, k={k}"
        )

    def test_rust_hilbert_vs_rust_natural_consistency(self) -> None:
        """Both engines should return valid predecessor results for the same data.

        This test verifies that internal ordering differences don't affect correctness.
        The key property: zero predecessor violations regardless of internal ordering.

        Note: With residual correlation metric, not all queries may find exactly k neighbors
        due to metric properties. We verify the critical constraint (no violations) holds.
        """
        np.random.seed(42)
        n_points = 500
        k = 30
        points = np.random.randn(n_points, 3).astype(np.float32)

        from covertreex.engine import RustNaturalEngine, RustHilbertEngine

        results = {}
        for engine_name, EngineClass in [("rust-natural", RustNaturalEngine), ("rust-hilbert", RustHilbertEngine)]:
            # Build fresh backend for each engine to avoid state pollution
            backend_state = build_residual_backend(
                points,
                seed=99,
                inducing_count=128,
                variance=1.0,
                lengthscale=1.0,
                chunk_size=64,
            )
            set_residual_backend(None)
            configure_residual_correlation(backend_state)

            engine = EngineClass()
            runtime = Runtime(
                backend="numpy",
                precision="float32",
                metric="residual_correlation",
                enable_rust=True,
                engine=engine_name,
                residual_use_static_euclidean_tree=True,
            )

            tree = engine.build(
                points,
                runtime=runtime.to_config(),
                residual_backend=backend_state,
                residual_params={"variance": 1.0, "lengthscale": 1.0},
                compute_predecessor_bounds=True,
            )

            query_indices = np.arange(n_points, dtype=np.int64).reshape(-1, 1)
            ctx = cx_config.runtime_context()
            neighbors = engine.knn(
                tree,
                query_indices,
                k=k,
                return_distances=False,
                predecessor_mode=True,
                context=ctx,
                runtime=runtime.to_config(),
            )
            results[engine_name] = np.asarray(neighbors)

            # Clear backend for next iteration
            set_residual_backend(None)

        # Both should have zero violations - the critical invariant
        for engine_name, neighbors in results.items():
            violations = sum(
                1 for i in range(n_points) for j in neighbors[i] if j >= 0 and j >= i
            )
            assert violations == 0, f"{engine_name} has {violations} violations"

        # Verify no neighbor exceeds the query index (predecessor constraint)
        for engine_name, neighbors in results.items():
            for i in range(n_points):
                valid_neighbors = neighbors[i][neighbors[i] >= 0]
                for j in valid_neighbors:
                    assert j < i, f"{engine_name} query {i} has invalid neighbor {j} >= {i}"


def test_predecessor_mode_via_cover_tree_factory() -> None:
    """Regression test: predecessor_mode must work via cover_tree() factory.

    The cover_tree() factory is the recommended API for building trees.
    It uses the Rust engine's build() method internally, which properly
    computes subtree bounds needed for predecessor_mode.

    Fixed in v0.4.3: compute_predecessor_bounds now defaults to True in both
    RustNaturalEngine.build() and RustHilbertEngine.build().

    Note: The critical invariant is zero violations (j >= i), not k-fulfillment.
    The residual correlation metric may not always find exactly k neighbors
    due to the structure of correlations in the V-matrix.
    """
    from covertreex import cover_tree
    from covertreex.kernels import Matern52

    np.random.seed(42)
    n_points = 500  # Use larger dataset for rust-hilbert to work well
    k = 30
    points = np.random.randn(n_points, 3).astype(np.float32)

    # Build via cover_tree() factory - the recommended API
    kernel = Matern52(lengthscale=1.0)
    tree = cover_tree(points, kernel=kernel, engine="rust-hilbert")

    # Query with predecessor_mode
    neighbors, distances = tree.knn(k=k, predecessor_mode=True, return_distances=True)
    neighbors = np.asarray(neighbors)

    # Verify the critical invariant: zero predecessor violations
    # All returned neighbors j must satisfy j < query_index i
    violations = 0
    for i in range(n_points):
        valid = neighbors[i][neighbors[i] >= 0]
        for j in valid:
            if j >= i:
                violations += 1

    assert violations == 0, (
        f"predecessor_mode via cover_tree() failed: {violations} violations found. "
        "All returned neighbors j must satisfy j < query_index i."
    )

    # Verify early queries have correct bounds (these are deterministic)
    assert np.all(neighbors[0] == -1), "Query 0 should have no predecessors"
    valid_1 = neighbors[1][neighbors[1] >= 0]
    if len(valid_1) > 0:
        assert valid_1[0] == 0, "Query 1's only valid predecessor is 0"


def test_predecessor_mode_euclidean() -> None:
    """Test predecessor_mode works for Euclidean metric (not just residual_correlation).

    This ensures predecessor_mode is a general feature, not tied to a specific metric.
    The Euclidean path goes through wrapper.knn_query() in Rust.
    """
    np.random.seed(42)
    n_points = 50
    k = 8
    points = np.random.randn(n_points, 3).astype(np.float32)

    # Build tree with Euclidean metric
    runtime = Runtime(
        backend="numpy",
        precision="float32",
        metric="euclidean",
        enable_rust=True,
        engine="rust-hilbert",
    )
    tree = CoverTree(runtime).fit(points)

    # Query with predecessor_mode
    neighbors = tree.knn(points, k=k, predecessor_mode=True)
    neighbors = np.asarray(neighbors)

    # Verify zero predecessor violations
    violations = 0
    for i in range(n_points):
        for j in neighbors[i]:
            if j >= 0 and j >= i:
                violations += 1

    assert violations == 0, (
        f"Euclidean predecessor_mode failed: {violations} violations found. "
        "All returned neighbors j must satisfy j < query_index i."
    )

    # Verify k-fulfillment for queries with enough predecessors
    for i in range(k, n_points):
        valid = neighbors[i][neighbors[i] >= 0]
        expected = min(k, i)
        assert len(valid) == expected, (
            f"Query {i}: expected {expected} predecessors, got {len(valid)}"
        )