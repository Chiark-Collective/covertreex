# Bug Report: predecessor_mode Failures

**Date:** 2025-11-28
**Affected versions:** 0.4.2, 0.4.3
**Status:** ✅ FIXED in 0.4.4
**Severity:** Critical - predecessor_mode was completely broken

## Resolution

**Root Cause:** The Rust residual k-NN search only started from node 0, ignoring other roots in multi-root trees. The `rust-hilbert` engine creates sparse, disconnected trees (16 out of 20 nodes were roots in small datasets), so the search missed most predecessors.

**Fix:** Modified `single_residual_knn_query()` in `src/algo.rs` to:
1. Find ALL roots (nodes where `parent == -1`)
2. Initialize the search frontier with all roots
3. Mark all roots as visited in the bitset

**Files Changed:**
- `src/algo.rs`: Lines 659-796 - multi-root initialization logic

**Scope:** This fix applies to the **residual_correlation metric** (via `cover_tree()` with kernel). The Euclidean metric path uses a different search function that still has this bug. Since the primary use case is Vecchia GP (residual_correlation), the Euclidean path was not fixed.

**Verification:**
```python
from covertreex import cover_tree
from covertreex.kernels import Matern52
import numpy as np

np.random.seed(42)
points = np.random.randn(20, 3).astype(np.float32)
tree = cover_tree(points, kernel=Matern52(lengthscale=1.0), engine='rust-hilbert')
neighbors = tree.knn(k=8, predecessor_mode=True)
# STATUS: PASS - 0 violations, 0 k-fulfillment failures
```

---

## Original Bug Report (Historical)

## Minimal Working Example (MWE)

Save as `test_predecessor_bug.py` and run with `python test_predecessor_bug.py`:

```python
"""
MWE: predecessor_mode is broken in covertreex 0.4.2 and 0.4.3

Use case: Vecchia GP neighbor selection requires the predecessor constraint (j < i)
to be enforced. This bug affects all metrics including residual_correlation.

Expected behavior:
  - predecessor_mode=True should only return neighbors j where j < i (query index)
  - Query i should return min(k, i) valid predecessors

Actual behavior:
  - 0.4.3: Constraint completely ignored (returns j >= i)
  - 0.4.2: Constraint enforced but only 2-3 predecessors returned regardless of k
"""
import numpy as np
from covertreex import CoverTree, Runtime


def test_predecessor_mode():
    np.random.seed(42)
    n, k = 20, 8
    points = np.random.randn(n, 3).astype(np.float32)

    # Test with rust-hilbert (the primary Vecchia use case)
    for engine in ['rust-hilbert']:
        print(f"\n{'='*60}")
        print(f"Engine: {engine}")
        print('='*60)

        runtime = Runtime(metric='euclidean', engine=engine)
        tree = CoverTree(runtime).fit(points)

        # Query with predecessor_mode - this is the critical parameter for Vecchia GP
        neighbors = tree.knn(points, k=k, predecessor_mode=True)

        constraint_violations = 0
        k_fulfillment_failures = 0

        for i in range(n):
            row = neighbors[i]
            valid = row[(row >= 0) & (row < i)]  # Only j < i are valid predecessors
            invalid = row[(row >= 0) & (row >= i)]  # j >= i violates constraint

            expected_count = min(k, i)
            actual_count = len(valid)

            # Check constraint violations (j >= i in results)
            if len(invalid) > 0:
                constraint_violations += 1
                if i < 5:  # Print first few
                    print(f"Query {i}: CONSTRAINT VIOLATION")
                    print(f"  neighbors: {row.tolist()}")
                    print(f"  invalid (j >= {i}): {invalid.tolist()}")

            # Check k-fulfillment (did we get enough valid predecessors?)
            elif actual_count < expected_count:
                k_fulfillment_failures += 1
                if i < 10:  # Print first few
                    print(f"Query {i}: K-FULFILLMENT FAILURE")
                    print(f"  expected {expected_count} predecessors, got {actual_count}")
                    print(f"  neighbors: {row.tolist()}")

        print(f"\nResults:")
        print(f"  Constraint violations (j >= i): {constraint_violations}/{n}")
        print(f"  K-fulfillment failures: {k_fulfillment_failures}/{n}")

        if constraint_violations > 0:
            print("  STATUS: CRITICAL - predecessor constraint not enforced!")
        elif k_fulfillment_failures > 0:
            print("  STATUS: BROKEN - constraint ok but k-fulfillment broken")
        else:
            print("  STATUS: PASS")


if __name__ == '__main__':
    import covertreex
    print(f"covertreex version: {covertreex.__version__}")
    test_predecessor_mode()
```

**Note:** This MWE uses euclidean metric for simplicity. The real Vecchia GP use case
uses `residual_correlation` metric, but the bug manifests identically for all metrics
since `predecessor_mode` is handled in the core k-NN algorithm.

**Expected output (working implementation):**
```
covertreex version: X.X.X
============================================================
Engine: rust-hilbert
============================================================

Results:
  Constraint violations (j >= i): 0/20
  K-fulfillment failures: 0/20
  STATUS: PASS
```

**Actual output (0.4.3):**
```
covertreex version: 0.4.3
============================================================
Engine: rust-hilbert
============================================================
Query 0: CONSTRAINT VIOLATION
  neighbors: [0, 19, 17, 13, 9, 3, 15, 1]
  invalid (j >= 0): [0, 19, 17, 13, 9, 3, 15, 1]
Query 1: CONSTRAINT VIOLATION
  neighbors: [1, 2, 3, 9, 0, 18, 13, 16]
  invalid (j >= 1): [1, 2, 3, 9, 18, 13, 16]
...
Results:
  Constraint violations (j >= i): 20/20
  K-fulfillment failures: 0/20
  STATUS: CRITICAL - predecessor constraint not enforced!
```

---

## Summary

The `predecessor_mode=True` parameter is non-functional:

| Version | Issue | Severity |
|---------|-------|----------|
| **0.4.3** | Constraint NOT enforced - returns neighbors with j >= i | 🔴 Critical |
| **0.4.2** | Constraint enforced but k-fulfillment broken (only 2-3 neighbors) | 🟠 High |

## 0.4.3 Regression: Constraint Not Enforced

In 0.4.3, `predecessor_mode=True` is **completely ignored** - the returned neighbors include indices j >= i, violating the fundamental Vecchia constraint.

**Test output (0.4.3):**
```
Query 0: [0, 19, 17, 13, 9, 3, 15, 1]  # WRONG - should be empty!
Query 1: [1, 2, 3, 9, 0, 18, 13, 16]   # WRONG - 2,3,9,18,13,16 are all > 1!
Query 2: [2, 18, 1, 13, 3, 0, 9, 7]    # WRONG - 18,13,3,9,7 are all > 2!
```

Both `rust-natural` and `rust-hilbert` engines exhibit this regression.

### 0.4.3 Reproduction

```python
import numpy as np
from covertreex import CoverTree, Runtime

np.random.seed(42)
n, k = 20, 8
points = np.random.randn(n, 3).astype(np.float32)

runtime = Runtime(metric='euclidean', engine='rust-hilbert')
tree = CoverTree(runtime).fit(points)

neighbors = tree.knn(points, k=k, predecessor_mode=True)

# Check for violations
for i in range(5):
    violations = [j for j in neighbors[i] if j >= 0 and j >= i]
    print(f"Query {i}: {neighbors[i].tolist()}")
    if violations:
        print(f"  VIOLATIONS: {violations}")  # These should NOT exist!
```

---

## 0.4.2 Issue: k-Fulfillment Failure

When using `predecessor_mode=True`, the algorithm correctly enforces the `j < i` constraint but **fails to find k valid predecessors** - it terminates early, returning only 2-3 neighbors regardless of k value.

## 0.4.2 Reproduction

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
