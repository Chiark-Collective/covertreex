import json

import numpy as np
import pytest

from covertreex.metrics.residual.scope_caps import (
    ResidualScopeCapTable,
    get_scope_cap_table,
    reset_scope_cap_cache,
)


def test_scope_cap_table_lookup_applies_levels(tmp_path):
    payload = {
        "schema": 1,
        "default": 2.5,
        "levels": {
            "0": {"cap": 0.5},
            "2": 1.25,
        },
    }
    target = tmp_path / "caps.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    reset_scope_cap_cache()

    table = get_scope_cap_table(str(target))
    assert isinstance(table, ResidualScopeCapTable)

    levels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    caps = table.lookup(levels)
    assert caps.shape == levels.shape
    assert caps[0] == pytest.approx(0.5)
    assert np.isnan(caps[1])
    assert caps[2] == pytest.approx(1.25)
    assert np.isnan(caps[3])

    defaults = np.where(np.isfinite(caps), caps, payload["default"])
    assert defaults[1] == pytest.approx(2.5)


def test_scope_cap_cache_returns_same_instance(tmp_path):
    payload = {"schema": 1, "default": 1.0, "levels": {"0": 0.25}}
    target = tmp_path / "caps.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    reset_scope_cap_cache()

    first = get_scope_cap_table(str(target))
    again = get_scope_cap_table(str(target))
    assert first is again
