# Bug Report: predecessor_mode k-Fulfillment Failure

**Date:** 2025-11-28
**Affected versions:** 0.4.2 (and likely earlier)
**Severity:** High - predecessor_mode is non-functional for most use cases

## Summary

When using `predecessor_mode=True`, the algorithm correctly enforces the `j < i` constraint but **fails to find k valid predecessors** - it terminates early, returning only 2-3 neighbors regardless of k value.

## Reproduction

```python
from covertreex import CoverTree, Runtime

# Build tree with residual_correlation metric
runtime = Runtime(metric="residual_correlation", engine="rust-hilbert")
tree = CoverTree(runtime).fit(point_indices)

# Query with predecessor_mode
neighbors = tree.knn(point_indices, k=8, predecessor_mode=True)

# Result: queries 3+ return only 2 valid predecessors instead of min(k, i)
```

**Expected output:**
```
Query 0: 0 predecessors []
Query 1: 1 predecessor [0]
Query 2: 2 predecessors [0, 1]
Query 3: 3 predecessors [0, 1, 2]
...
Query 8+: 8 predecessors [...]
```

**Actual output:**
```
Query 0: 0 predecessors []
Query 1: 1 predecessor [0]
Query 2: 2 predecessors [0, 1]
Query 3: 2 predecessors [1, 0]  # WRONG - should be 3
Query 4: 2 predecessors [1, 0]  # WRONG - should be 4
...
Query 8+: 2 predecessors [1, 0]  # WRONG - should be 8
```

## Root Cause Analysis

### Issue 1: `compute_predecessor_bounds` defaults to `False` in engine builds

The `subtree_min_bounds` array is critical for predecessor_mode to work correctly. Without it, the algorithm cannot determine which subtrees contain valid predecessors.

**File:** `covertreex/engine.py`

| Location | Default | Should Be |
|----------|---------|-----------|
| Line 348 (`RustNaturalEngine.build`) | `False` | `True` |
| Line 969 (`RustHilbertEngine.build`) | `False` | `True` |
| Line 497 (`build_residual_tree`) | `True` | ✓ Correct |

When users call `CoverTree(runtime).fit(points)`, it goes through the engine's `build()` method which defaults to `compute_predecessor_bounds=False`.

### Issue 2: Search terminates when tree structure exhausted

In `src/algo.rs`, when a node fails the predecessor constraint, the algorithm adds it to `next_frontier` for subtree exploration (lines 883-888):

```rust
if subtree_has_valid {
    let dist = metric.distance_idx(q_dataset_idx, child_idx);
    ctx.next_frontier.push((child_idx, dist));
}
```

With `subtree_min_bounds = None`, the check at lines 876-882 returns `true`:

```rust
let subtree_has_valid = match (max_neighbor_idx, subtree_min_bounds) {
    (Some(max_idx), Some(bounds)) if child_idx < bounds.len() => {
        (bounds[child_idx] as usize) < max_idx
    },
    (Some(_), None) => true, // No bounds, assume subtree might have valid nodes
    _ => false,
};
```

**The problem:** When the failed node becomes a parent in the next iteration, if it has no children (is a leaf), the search terminates because there's nowhere else to explore - even though k valid predecessors haven't been found.

With Hilbert-ordered trees, points with small dataset indices may be scattered throughout the tree structure. Without `subtree_min_bounds` to guide exploration, the algorithm blindly traverses the tree and terminates when it runs out of nodes, not when it finds k predecessors.

### Telemetry Evidence

```python
rust_query_telemetry = {
    'distance_evals': 8,           # Only 8 evals for 10 queries with k=5!
    'predecessor_filtered': 1,      # Only 1 node filtered
    'subtrees_pruned': 0,          # No pruning (no bounds available)
    'budget_early_terminate': 0,   # Not budget-related
}
```

The algorithm is exploring very few nodes because the tree structure (with ~10 points) is shallow, and most reachable nodes are leaves.

## Proposed Solutions

### Solution A: Change defaults (Recommended)

Change the default value of `compute_predecessor_bounds` to `True` in engine builds:

**File:** `covertreex/engine.py`

```python
# Line 348 - RustNaturalEngine.build
def build(
    ...
    compute_predecessor_bounds: bool = True,  # Changed from False
) -> CoverTree:

# Line 969 - RustHilbertEngine.build
def build(
    ...
    compute_predecessor_bounds: bool = True,  # Changed from False
) -> CoverTree:
```

**Pros:** Simple fix, ensures predecessor_mode works out of the box
**Cons:** Slightly increases build time for users who don't need predecessor_mode

### Solution B: Compute bounds lazily when predecessor_mode=True

Modify the query path to compute `subtree_min_bounds` on-the-fly if not already available when `predecessor_mode=True` is requested.

**File:** `covertreex/queries/knn.py` (around line 379)

```python
if predecessor_mode and not tree.has_predecessor_bounds:
    # Compute bounds before querying
    subtree_min_bounds = compute_subtree_bounds(tree)
    # Pass to Rust query
```

**Pros:** No overhead for users who don't use predecessor_mode
**Cons:** More complex, may cause unexpected latency on first predecessor_mode query

### Solution C: Fallback to brute-force when bounds unavailable

When `subtree_min_bounds = None` and `predecessor_mode = True` and `result_heap.len() < k`, fall back to checking ALL remaining nodes in the tree.

**File:** `src/algo.rs` (after main loop, around line 1169)

```rust
// If predecessor_mode and we haven't found k neighbors, do exhaustive search
if max_neighbor_idx.is_some() && result_heap.len() < k {
    // Iterate all nodes not yet visited
    for node_idx in 0..tree.len() {
        let ds_idx = node_to_dataset[node_idx] as usize;
        if ds_idx < max_neighbor_idx.unwrap() && !visited(node_idx) {
            // Compute distance and consider for result
        }
    }
}
```

**Pros:** Guarantees correctness without requiring bounds
**Cons:** Defeats purpose of tree-based search for predecessor queries without bounds

## Recommended Fix

**Solution A** is recommended because:
1. Simple one-line changes in two locations
2. The overhead of computing bounds is minimal for most use cases
3. Users who explicitly don't want bounds can pass `compute_predecessor_bounds=False`
4. Aligns with the documented behavior in CHANGELOG.md ("Default `compute_predecessor_bounds=True`")

## Test Case

Add to `tests/test_predecessor_mode.py`:

```python
def test_predecessor_mode_via_covertree_fit():
    """Ensure predecessor_mode works when using CoverTree.fit() API."""
    n, k = 50, 8
    points = np.random.randn(n, 2).astype(np.float32)

    # Build via CoverTree.fit() - the common user path
    runtime = Runtime(engine="rust-hilbert")
    tree = CoverTree(runtime).fit(points)

    neighbors = tree.knn(points, k=k, predecessor_mode=True)

    # Verify k-fulfillment
    for i in range(k, n):
        valid = neighbors[i][neighbors[i] >= 0]
        assert len(valid) == k, f"Query {i}: expected {k} neighbors, got {len(valid)}"
```

## Related Files

- `covertreex/engine.py` - Engine build methods with default parameters
- `src/algo.rs` - Rust implementation of predecessor constraint logic
- `src/lib.rs` - PyO3 bindings that pass `subtree_min_bounds` to Rust
- `covertreex/queries/knn.py` - Python query dispatch
- `docs/PREDECESSOR_MODE_STATUS_20251128.md` - Current status (needs update after fix)
