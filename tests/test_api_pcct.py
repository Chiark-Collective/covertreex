import numpy as np
import pytest

from covertreex import config as cx_config
from covertreex.algo.batch_delete import BatchDeletePlan
from covertreex.algo.batch import BatchInsertPlan
from covertreex.api import PCCT, Runtime, Residual


@pytest.fixture(autouse=True)
def reset_runtime_context():
    cx_config.reset_runtime_context()
    yield
    cx_config.reset_runtime_context()


def _runtime() -> Runtime:
    return Runtime(
        backend="numpy",
        precision="float64",
        conflict_graph="dense",
        batch_order="natural",
        diagnostics=False,
        residual=Residual(gate1_enabled=False),
    )


def test_pcct_fit_insert_knn_roundtrip():
    runtime = _runtime()
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    tree, plan = PCCT(runtime).fit(points, return_plan=True)
    assert isinstance(plan, BatchInsertPlan)
    assert tree.num_points == points.shape[0]

    tree2 = PCCT(runtime, tree).insert([[3.0, 3.0]])
    assert tree2.num_points == points.shape[0] + 1

    indices, distances = PCCT(runtime, tree2).knn([[0.1, 0.1]], k=2, return_distances=True)
    assert indices.shape == (2,)
    assert distances.shape == (2,)

    nearest = PCCT(runtime, tree2).nearest([0.1, 0.1])
    assert np.isscalar(nearest)


def test_pcct_delete_returns_plan():
    runtime = _runtime()
    base_tree = PCCT(runtime).fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    new_tree, plan = PCCT(runtime, base_tree).delete([1], return_plan=True)
    assert isinstance(plan, BatchDeletePlan)
    assert new_tree.num_points == base_tree.num_points - 1
    assert int(plan.removed_indices.shape[0]) == 1
