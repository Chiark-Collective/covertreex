import logging
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex import config as cx_config
from covertreex.algo import batch_delete, batch_insert
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats
from covertreex.queries import knn


def _make_tree() -> PCCTree:
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2, 3], dtype=backend.default_int)
    si_cache = backend.asarray([4.0, 2.0, 1.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
    return PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        stats=TreeLogStats(num_batches=0),
        backend=backend,
    )


def test_batch_insert_emits_resource_log(caplog: pytest.LogCaptureFixture) -> None:
    cx_config.reset_runtime_config_cache()
    tree = _make_tree()
    batch = jnp.asarray([[2.5, 2.4], [0.2, 0.1]])
    caplog.set_level(logging.INFO, logger="covertreex.algo.batch_insert")

    batch_insert(tree, batch, mis_seed=0)

    records = [
        record for record in caplog.records if "op=batch_insert" in record.message
    ]
    assert records, "expected batch_insert operation log"
    message = records[-1].message
    assert "wall_ms=" in message
    assert "cpu_user_ms=" in message
    assert "rss_delta=" in message
    assert "candidates=" in message


def test_batch_delete_emits_resource_log(caplog: pytest.LogCaptureFixture) -> None:
    cx_config.reset_runtime_config_cache()
    tree = _make_tree()
    batch = jnp.asarray([[2.5, 2.4], [0.2, 0.1]])
    inserted, _ = batch_insert(tree, batch, mis_seed=0)
    caplog.set_level(logging.INFO, logger="covertreex.algo.batch_delete")

    batch_delete(inserted, [0])

    records = [
        record for record in caplog.records if "op=batch_delete" in record.message
    ]
    assert records, "expected batch_delete operation log"
    message = records[-1].message
    assert "removed=" in message
    assert "wall_ms=" in message


def test_knn_emits_resource_log(caplog: pytest.LogCaptureFixture) -> None:
    cx_config.reset_runtime_config_cache()
    tree = _make_tree()
    queries = jnp.asarray([[0.1, 0.1], [2.7, 2.8]])
    caplog.set_level(logging.INFO, logger="covertreex.queries.knn")

    indices, _ = knn(tree, queries, k=2, return_distances=True)

    assert np.asarray(indices).shape == (2, 2)
    records = [
        record for record in caplog.records if "op=knn_query" in record.message
    ]
    assert records, "expected knn operation log"
    message = records[-1].message
    assert "queries=2" in message
    assert "k=2" in message


def test_diagnostics_can_be_disabled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COVERTREEX_ENABLE_DIAGNOSTICS", "0")
    cx_config.reset_runtime_config_cache()

    tree = _make_tree()
    queries = jnp.asarray([[0.1, 0.1]])
    caplog.set_level(logging.INFO, logger="covertreex.queries.knn")

    knn(tree, queries, k=1)

    records = [
        record for record in caplog.records if "op=knn_query" in record.message
    ]
    assert records
    message = records[-1].message
    assert "cpu_user_ms=NA" in message
    assert "rss_delta=NA" in message

    monkeypatch.delenv("COVERTREEX_ENABLE_DIAGNOSTICS", raising=False)
    cx_config.reset_runtime_config_cache()
