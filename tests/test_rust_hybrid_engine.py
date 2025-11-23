import numpy as np
import pytest

from covertreex.api import Runtime
from covertreex.engine import build_tree

try:
    import covertreex_backend  # noqa: F401
except ImportError:  # pragma: no cover - optional backend
    covertreex_backend = None


@pytest.mark.skipif(covertreex_backend is None, reason="Rust backend not built")
def test_rust_hybrid_engine_build_and_query():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(64, 3)).astype(np.float32)
    queries = np.arange(8, dtype=np.int64).reshape(-1, 1)

    runtime = Runtime(
        metric="residual_correlation",
        engine="rust-hybrid",
        enable_rust=True,
        precision="float32",
    )
    context = runtime.activate()

    tree = build_tree(
        points,
        runtime=context.config,
        engine="rust-hybrid",
        context=context,
        residual_params={
            "inducing_count": 16,
            "variance": 1.0,
            "lengthscale": 1.0,
            "chunk_size": 64,
        },
    )

    indices, distances = tree.knn(queries, k=3, return_distances=True, context=context)

    assert indices.shape == (queries.shape[0], 3)
    assert distances.shape == (queries.shape[0], 3)
    assert indices.dtype == np.int64
    assert distances.dtype == np.float32
