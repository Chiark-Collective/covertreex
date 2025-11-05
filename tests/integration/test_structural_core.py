import itertools
from collections import defaultdict

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from covertreex.algo import build_conflict_graph, traverse_collect_scopes
from covertreex.core.tree import DEFAULT_BACKEND, PCCTree, TreeLogStats


def _structural_tree() -> PCCTree:
    backend = DEFAULT_BACKEND
    points = backend.asarray(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [2.5, 2.4],
        ],
        dtype=backend.default_float,
    )
    top_levels = backend.asarray([2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 3, 3], dtype=backend.default_int)
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
        stats=TreeLogStats(num_batches=1),
        backend=backend,
    )


def _collect_chain(next_cache: np.ndarray, start: int) -> tuple[int, ...]:
    chain: list[int] = []
    seen: set[int] = set()
    current = start
    while 0 <= current < next_cache.shape[0] and current not in seen:
        chain.append(current)
        seen.add(current)
        nxt = int(next_cache[current])
        if nxt < 0:
            break
        current = nxt
    return tuple(chain)


def test_traversal_matches_naive_computation():
    tree = _structural_tree()
    batch = jnp.asarray([[2.6, 2.6], [0.5, 0.4], [3.2, 3.1]])

    result = traverse_collect_scopes(tree, batch)

    tree_points = np.asarray(tree.points)
    tree_levels = np.asarray(tree.top_levels)
    si_cache = np.asarray(tree.si_cache)
    next_cache = np.asarray(tree.next_cache)

    batch_np = np.asarray(batch)

    expected_parents: list[int] = []
    expected_levels: list[int] = []
    expected_scopes: list[tuple[int, ...]] = []

    for point in batch_np:
        dists = np.linalg.norm(tree_points - point, axis=1)
        parent_idx = int(np.argmin(dists))
        level = int(tree_levels[parent_idx])
        base_radius = 2.0 ** (level + 1)
        radius = max(base_radius, float(si_cache[parent_idx]))
        scope_nodes = {
            idx for idx, dist in enumerate(dists) if dist <= radius
        }
        scope_nodes.add(parent_idx)
        scope_nodes.update(_collect_chain(next_cache, parent_idx))
        sorted_scope = tuple(
            sorted(scope_nodes, key=lambda node: (-int(tree_levels[node]), node))
        )

        expected_parents.append(parent_idx)
        expected_levels.append(level)
        expected_scopes.append(sorted_scope)

    assert tuple(expected_parents) == tuple(result.parents.tolist())
    assert tuple(expected_levels) == tuple(result.levels.tolist())
    assert tuple(expected_scopes) == result.conflict_scopes

    indptr = result.scope_indptr.tolist()
    indices = result.scope_indices.tolist()
    reconstructed = [
        tuple(indices[indptr[i] : indptr[i + 1]]) for i in range(len(expected_scopes))
    ]
    assert reconstructed == expected_scopes


def test_conflict_graph_matches_bruteforce_edges():
    tree = _structural_tree()
    batch = jnp.asarray([[2.6, 2.6], [0.5, 0.4], [3.2, 3.1]])

    traversal = traverse_collect_scopes(tree, batch)
    graph = build_conflict_graph(tree, traversal, batch)

    batch_np = np.asarray(batch)
    pairwise = np.linalg.norm(batch_np[:, None, :] - batch_np[None, :, :], axis=-1)

    radii = []
    tree_si = np.asarray(tree.si_cache)
    for parent, level in zip(traversal.parents.tolist(), traversal.levels.tolist()):
        if parent < 0:
            radii.append(float("inf"))
            continue
        base = 2.0 ** (float(level) + 1.0)
        radii.append(max(base, float(tree_si[parent])))
    radii = np.asarray(radii)

    node_to_batch: defaultdict[int, set[int]] = defaultdict(set)
    for batch_idx, scope in enumerate(traversal.conflict_scopes):
        for node in scope:
            node_to_batch[int(node)].add(batch_idx)

    expected_edges: set[tuple[int, int]] = set()
    for batch_nodes in node_to_batch.values():
        for u, v in itertools.combinations(sorted(batch_nodes), 2):
            threshold = min(radii[u], radii[v])
            if pairwise[u, v] <= threshold:
                expected_edges.add((min(u, v), max(u, v)))

    adjacency = []
    indices = graph.indices.tolist()
    indptr = graph.indptr.tolist()
    for node in range(graph.num_nodes):
        neighbors = indices[indptr[node] : indptr[node + 1]]
        adjacency.append({int(nb) for nb in neighbors})

    actual_edges: set[tuple[int, int]] = set()
    for node, neighbors in enumerate(adjacency):
        for nb in neighbors:
            actual_edges.add((min(node, nb), max(node, nb)))

    assert actual_edges == expected_edges
    assert np.allclose(np.asarray(graph.radii), radii)
    assert graph.scope_indptr.tolist() == traversal.scope_indptr.tolist()
    assert graph.scope_indices.tolist() == traversal.scope_indices.tolist()
