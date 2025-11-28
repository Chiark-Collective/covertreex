# Predecessor Mode Status Report

**Date:** 2025-11-28
**covertreex version:** 0.4.2
**Related:** [predecessor-constraint-rfc.md](predecessor-constraint-rfc.md)

## Summary

The `predecessor_mode` parameter is **fully implemented** in covertreex 0.4.2:
- ✅ The j < i constraint IS being enforced (neighbors are valid predecessors)
- ✅ Search continues until k valid predecessors are found
- ✅ Output shape is correct: `(n, k)` for all k values
- ✅ Both `rust-natural` and `rust-hilbert` engines produce correct results

## Current Behavior

### API Signature

```python
tree.knn(k=10, predecessor_mode=True)
```

### What's Working

1. **Constraint IS enforced**: All returned neighbors j satisfy j < i (query index)
2. **Query 0 returns empty**: Correctly returns `[-1, -1, ..., -1]`
3. **Query 1 returns `[0, ...]`**: Correctly returns only index 0
4. **k-fulfillment**: Query i (where i >= k) returns exactly k valid neighbors
5. **Shape is correct**: Returns `(n, k)` as expected
6. **Both engines work**: rust-natural and rust-hilbert produce 0 predecessor violations

Example output with `n=20, k=8`:
```
Shape: (20, 8)  # ✅ Correct

Query 0: [-1, -1, -1, -1, -1, -1, -1, -1]  # ✅ Correct (no predecessors)
Query 1: [0, -1, -1, -1, -1, -1, -1, -1]   # ✅ Correct (only 0 is valid)
Query 5: [4, 3, 2, 1, 0, -1, -1, -1]       # ✅ Correct (5 neighbors, all < 5)
Query 10: [9, 2, 7, 6, 5, 1, 3, 4]         # ✅ Correct (8 neighbors, all < 10)
Query 15: [14, 2, 12, 11, 10, 1, 8, 3]     # ✅ Correct (8 neighbors, all < 15)
```

## Version History

### v0.4.0 (Initial Implementation)
- ❌ Shape was wrong: Returned `(n, 4)` instead of `(n, k)` for k=8
- ❌ Too few neighbors: Query 10 returned only 3-4 valid neighbors instead of 8
- Root cause: Search terminated too early after filtering invalid nodes

### v0.4.1 (k-fulfillment Fix)
- ✅ Search continues until k valid predecessors are found
- ✅ Subtree exploration: When a node fails predecessor constraint but its subtree may contain valid predecessors, search explores the subtree

### v0.4.2 (Engine Consistency Fix)
- ✅ rust-hilbert predecessor_mode correctness via `node_to_dataset` mapping
- ✅ Default `compute_predecessor_bounds=True` for subtree pruning
- ✅ Output shape is now `(n, k)` via `to_py_arrays` fix

## Test Coverage

12 tests in `tests/test_predecessor_mode.py`:

| Test | Description |
|------|-------------|
| `test_predecessor_mode_basic` | Basic constraint enforcement |
| `test_predecessor_mode_early_queries` | Queries 0, 1, 2 return correct counts |
| `test_predecessor_mode_off_by_default` | Default behavior unchanged |
| `test_predecessor_mode_with_distances` | Distance return values |
| `test_predecessor_mode_with_subtree_bounds` | Subtree pruning optimization |
| `test_predecessor_mode_k_fulfillment` | k-fulfillment for i >= k |
| `test_predecessor_correctness_both_engines` | rust-natural and rust-hilbert |
| `test_predecessor_filter_effectiveness` | Control test proving filter works |
| `test_predecessor_mode_gold_standard_scale` | N=32768, D=3, k=50 scale test |
| `test_rust_hilbert_vs_rust_natural_consistency` | Engine consistency |

## Key Implementation Details

### Predecessor Constraint Logic

The predecessor constraint logic in `src/algo.rs`:

- **algo.rs:605-609**: Fast-path for query 0 (returns empty) ✅
- **algo.rs:664-680**: Root constraint check ✅
- **algo.rs:858-868**: Child constraint filtering ✅
- **algo.rs:840-855**: Subtree pruning optimization ✅

### node_to_dataset Mapping

The `node_to_dataset` array maps tree node indices to original dataset indices:

```rust
// In rust-hilbert, nodes are reordered by Hilbert curve
// node_to_dataset[node_idx] gives the original dataset index
let dataset_idx = node_to_dataset[node_idx];

// Predecessor check uses dataset indices, not node indices
if dataset_idx >= query_dataset_idx {
    // Skip this node - not a valid predecessor
}
```

This mapping ensures predecessor correctness regardless of internal tree ordering.

### Why Vecchia Ordering ≠ Hilbert Ordering

The Vecchia constraint (`j < i`) operates on the dataset's natural ordering, not the tree's internal ordering. The tree can use any reordering (Hilbert curve, random, etc.) for efficient construction—the `node_to_dataset` mapping layer ensures the predecessor constraint is evaluated against original indices.

## Notes

- The circular import warning in covertreex plugins doesn't affect functionality
- Telemetry can be enabled with `COVERTREEX_RUST_QUERY_TELEMETRY=1` to debug
- Default engine is `rust-natural` for best predecessor mode support
