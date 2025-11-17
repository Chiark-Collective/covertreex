# Cover Tree Standalone Reference

This document was generated on 2025-11-17 18:05:15 to consolidate every cover tree document and source file into one place.

Each section below mirrors the exact contents of the referenced path in the repository.

## doc/user/core/trees/cover_tree.md

```markdown
# `CoverTree`

The `CoverTree` class implements the cover tree, a hierarchical tree structure
with favorable theoretical properties.  The cover tree is useful for efficient
distance operations (such as [nearest neighbor search](../../methods/knn.md)) in
low to moderate dimensions.

mlpack's `CoverTree` implementation supports three template parameters for
configurable behavior, and implements all the functionality required by the
[TreeType API](../../../developer/trees.md#the-treetype-api), plus some
additional functionality specific to cover trees.

Due to the extra bookkeeping and complexity required to achieve its theoretical
guarantees, the `CoverTree` is often not as fast for nearest neighbor search as
the [`KDTree`](kdtree.md).  However, `CoverTree` is more flexible: it is able to
work with any [distance metric](../distances.md), not just
[`LMetric`](../distances.md#lmetric).

 * [Template parameters](#template-parameters)
 * [Constructors](#constructors)
 * [Basic tree properties](#basic-tree-properties)
 * [Bounding distances with the tree](#bounding-distances-with-the-tree)
 * [Tree traversals](#tree-traversals)
 * [Example usage](#example-usage)

## See also

 * [Cover tree on Wikipedia](https://en.wikipedia.org/wiki/Cover_tree)
 * [`KDTree`](kdtree.md)
 * [mlpack trees](../trees.md)
 * [`KNN`](../../methods/knn.md)
 * [mlpack geometric algorithms](../../modeling.md#geometric-algorithms)
 * [Cover trees for nearest neighbor (pdf)](https://www.hunch.net/~jl/projects/cover_tree/paper/paper.pdf)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

## Template parameters

The `CoverTree` class takes four template parameters, the first three of which
are required by the
[TreeType API](../../../developer/trees.md#template-parameters-required-by-the-treetype-policy)
(see also [this more detailed section](../../../developer/trees.md#template-parameters)).

```
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>
```

 * `DistanceType`: the [distance metric](../distances.md) to use for distance
   computations.  By default, this is
   [`EuclideanDistance`](../distances.md#lmetric).
 * [`StatisticType`](binary_space_tree.md#statistictype): this holds auxiliary
   information in each tree node.  By default,
   [`EmptyStatistic`](binary_space_tree.md#emptystatistic) is used, which holds
   no information.
 * `MatType`: the type of matrix used to represent points.  Must be a type
   matching the [Armadillo API](../../matrices.md).  By default, `arma::mat` is
   used, but other types such as `arma::fmat` or similar will work just fine.
 * `RootPointPolicy`: controls how the root of the tree is selected.  By
   default, `FirstPointIsRoot` is used, which simply uses the first point of the
   dataset as the root of the tree.
   - A custom `RootPointPolicy` must implement the function
     `static size_t ChooseRoot(const MatType& dataset)`, where the `size_t`
     returned indicates the index of the point in `dataset` that should be used
     as the root of the tree.

If no template parameters are explicitly specified, then defaults are used:

```
CoverTree<> = CoverTree<EuclideanDistance, EmptyStatistic, arma::mat,
                        FirstPointIsRoot>
```

## Constructors

`CoverTree`s are constructed level-by-level, without modifying the input
dataset.

---

 * `node = CoverTree(data, base=2.0)`
 * `node = CoverTree(data, distance, base=2.0)`
   - Construct a `CoverTree` on the given `data`, using the given `base` if
     specified.
   - Optionally, specify an instantiated distance metric `distance` to use to
     construct the tree.

---

***Notes:***

 - The name `node` is used here for `CoverTree` objects instead of `tree`,
   because each `CoverTree` object is a single node in the tree.  The
   constructor returns the node that is the root of the tree.

 - In a `CoverTree`, it is not guaranteed that the ball bounds for nodes are
   disjoint; they may be overlapping.  This is because for many datasets, it is
   geometrically impossible to construct disjoint balls that cover the
   entire set of points.

 - Inserting individual points or removing individual points from a `CoverTree`
   is not supported, because this generally results in a cover tree with very
   loose bounding balls.  It is better to simply build a new `CoverTree` on the
   modified dataset.  For trees that support individual insertion and deletions,
   see the [`RectangleTree`](rectangle_tree.md) class and all its variants (e.g.
   [`RTree`](r_tree.md), [`RStarTree`](r_star_tree.md), etc.).

 - See also the
   [developer documentation on tree constructors](../../../developer/trees.md#constructors-and-destructors).

---

### Constructor parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../../matrices.md) | [Column-major](../../matrices.md#representing-data-in-mlpack) matrix to build the tree on.  Optionally pass with `std::move(data)` to transfer ownership to the tree. | _(N/A)_ |
| `distance` | [`DistanceType`](#template-parameters) | Instantiated distance metric (optional). | `EuclideanDistance()` |
| `base` | `double` | Shrinkage factor of each level of the cover tree.  Must be greater than 1. | `2.0` |

***Notes:***

 - According to [the original paper (pdf)](https://www.hunch.net/~jl/projects/cover_tree/paper/paper.pdf),
   sometimes a smaller base (more like 1.3 or 1.5) can provide better empirical
   results in practice.

 - An instantiated `distance` is only necessary when a
   [custom `DistanceType`](#template-parameters) was specified as a template
   parameter, and that distance type that require state.  So, this is not needed
   when using the default `EuclideanDistance`.

## Basic tree properties

Once a `CoverTree` object is constructed, various properties of the tree can be
accessed or inspected.  Many of these functions are required by the [TreeType
API](../../../developer/trees.md#the-treetype-api).

### Navigating the tree

 * `node.NumChildren()` returns the number of children in `node`.  If `0`, then
   `node` is a leaf.

 * `node.IsLeaf()` returns a `bool` indicating whether or not `node` is a leaf.

 * `node.Child(i)` returns a `CoverTree&` that is the `i`th child.
   - This function should only be called if `node.NumChildren()` is not `0`
     (e.g. if `node` is not a leaf).  Note that this returns a valid
     `CoverTree&` that can itself be used just like the root node of the tree!

 * `node.Parent()` will return a `CoverTree*` that points to the parent of
   `node`, or `NULL` if `node` is the root of the `CoverTree`.

 * `node.Base()` will return the value of `base` used to
   [build the tree](#constructors).  This is the same for all nodes in a tree.

 * `node.Scale()` will return an `int` representing the level of the node in the
   cover tree.  Larger values represent higher levels in the tree, and `INT_MIN`
   means that `node` is a leaf.
   - All descendant points are contained within a distance of
     `node.Base()` raised to a power of `node.Scale()`.

---

### Accessing members of a tree

 * `node.Stat()` will return an `EmptyStatistic&` (or a `StatisticType&` if a
   [custom `StatisticType`](#template-parameters) was specified as a template
   parameter) holding the statistics of the node that were computed during tree
   construction.

 * `node.Distance()` will return a
   [`EuclideanDistance&`](../distances.md#lmetric) (or a `DistanceType&` if a
   [custom `DistanceType`](#template-parameters) was specified as a template
   parameter).

See also the
[developer documentation](../../../developer/trees.md#basic-tree-functionality)
for basic tree functionality in mlpack.

---

### Accessing data held in a tree

 * `node.Dataset()` will return a `const arma::mat&` that is the dataset the
   tree was built on.
   - If a [custom `MatType`](#template-parameters) is being used, the return
     type will be `const MatType&` instead of `const arma::mat&`.

 * `node.NumPoints()` returns `1`: all cover tree nodes hold only one point.

 * `node.Point()` returns a `size_t` indicating the index of the point held by
   `node` in `node.Dataset()`.
   - For consistency with other tree types, `node.Point(i)` is also available,
     but `i` must be `0` (because cover tree nodes must hold only one point).
   - The point in `node` can then be accessed as
     `node.Dataset().col(node.Point())`.

 * `node.NumDescendants()` returns a `size_t` indicating the number of points
   held in all descendant leaves of `node`.
   - If `node` is the root of the tree, then `node.NumDescendants()` will be
     equal to `node.Dataset().n_cols`.

 * `node.Descendant(i)` returns a `size_t` indicating the index of the `i`'th
   descendant point in `node.Dataset()`.
   - `i` must be in the range `[0, node.NumDescendants() - 1]` (inclusive).
   - `node` does not need to be a leaf.
   - The `i`'th descendant point in `node` can then be accessed as
     `node.Dataset().col(node.Descendant(i))`.
   - Descendant point indices are not necessarily contiguous for cover trees;
     that is, `node.Descendant(i) + 1` is not necessarily
     `node.Descendant(i + 1)`.

---

### Accessing computed bound quantities of a tree

The following quantities are cached for each node in a `CoverTree`, and so
accessing them does not require any computation.

 * `node.FurthestPointDistance()` returns a `double` representing the distance
   between the center of the bounding ball of `node` and the furthest point held
   by `node`.  This value is always `0` for cover trees, as they only hold one
   point, which is the center of the bounding ball.

 * `node.FurthestDescendantDistance()` returns a `double` representing the
   distance between the center of the bounding ball of `node` and the furthest
   descendant point held by `node`.
   - This will always be less than `node.Base()` raised to the power of
     `node.Scale()`.

 * `node.MinimumBoundDistance()` returns a `double` representing the minimum
   possible distance from the center of the node to any edge of the bounding
   ball of `node`.
   - For cover trees, this quantity is equivalent to
     `node.FurthestDescendantDistance()`.

 * `node.ParentDistance()` returns a `double` representing the distance between
   the center of the bounding ball of `node` and the center of the bounding ball
   of its parent.
   - This is equivalent to the distance between
     `node.Dataset().col(node.Point())` and
     `node.Dataset().col(node.Parent()->Point())`, if `node` is not the root of
     the tree.
   - If `node` is the root of the tree, `0` is returned.

***Notes:***

 - If a [custom `MatType`](#template-parameters) was specified when constructing
   the `CoverTree`, then the return type of each method is the element type of
   the given `MatType` instead of `double`.  (e.g., if `MatType` is
   `arma::fmat`, then the return type is `float`.)

 - For more details on each bound quantity, see the
   [developer documentation](../../../developer/trees.md#complex-tree-functionality-and-bounds)
   on bound quantities for trees.

---

### Other functionality

 * `node.Center(center)` stores the center of the bounding ball of `node` in
   `center`.
   - `center` should be of type `arma::vec&`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec&` when `MatType` is `arma::fmat`.)
   - `center` will be set to have size equivalent to the dimensionality of the
     dataset held by `node`.
   - For cover trees, this sets `center` to have the same values as
     `node.Dataset().col(node.Point())` (e.g. the point held by `node`).

 * A `CoverTree` can be serialized with
   [`data::Save()` and `data::Load()`](../../load_save.md#mlpack-objects).

## Bounding distances with the tree

The primary use of trees in mlpack is bounding distances to points or other tree
nodes.  The following functions can be used for these tasks.

 * `node.GetNearestChild(point)`
 * `node.GetFurthestChild(point)`
   - Return a `size_t` indicating the index of the child that is closest to (or
     furthest from) `point`, with respect to the `MinDistance()` (or
     `MaxDistance()`) function.
   - If there is a tie, the child with the highest index is returned.
   - If `node` is a leaf, `0` is returned.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`; e.g., `arma::fvec` when `MatType` is `arma::fmat`.)

 * `node.GetNearestChild(other)`
 * `node.GetFurthestChild(other)`
   - Return a `size_t` indicating the index of the child that is closest to (or
     furthest from) the `CoverTree` node `other`, with respect to the
     `MinDistance()` (or `MaxDistance()`) function.
   - If there is a tie, the child with the highest index is returned.
   - If `node` is a leaf, `0` is returned.

---

 * `node.MinDistance(point)`
 * `node.MinDistance(other)`
   - Return a `double` indicating the minimum possible distance between `node`
     and `point`, or the `CoverTree` node `other`.
   - This is equivalent to the minimum possible distance between any point
     contained in the bounding ball of `node` and `point`, or between any point
     contained in the bounding ball of `node` and any point contained in the
     bounding ball of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is the element type of `MatType`; e.g.,
     `point` should be `arma::fvec` when `MatType` is `arma::fmat`, and the
     returned distance is `float`).

 * `node.MaxDistance(point)`
 * `node.MaxDistance(other)`
   - Return a `double` indicating the maximum possible distance between `node`
     and `point`, or the `CoverTree` node `other`.
   - This is equivalent to the maximum possible distance between any point
     contained in the bounding ball of `node` and `point`, or between any point
     contained in the bounding ball of `node` and any point contained in the
     bounding ball of `other`.
   - `point` should be of type `arma::vec`.  (If a [custom
     `MatType`](#template-parameters) was specified when constructing the
     `CoverTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is the element type of `MatType`; e.g.,
     `point` should be `arma::fvec` when `MatType` is `arma::fmat`, and the
     returned distance is `float`).

 * `node.RangeDistance(point)`
 * `node.RangeDistance(other)`
   - Return a [`Range`](../math.md#range) whose lower bound is
     `node.MinDistance(point)` or `node.MinDistance(other)`, and whose upper
      bound is `node.MaxDistance(point)` or `node.MaxDistance(other)`.
   - `point` should be of type `arma::vec`.  (If a
     [custom `MatType`](#template-parameters) was specified when constructing
     the `CoverTree`, the type is instead the column vector type for the given
     `MatType`, and the return type is a `RangeType` with element type the same
     as `MatType`; e.g., `point` should be `arma::fvec` when `MatType` is
     `arma::fmat`, and the returned type is
     [`RangeType<float>`](../math.md#range)).

## Tree traversals

Like every mlpack tree, the `CoverTree` class provides a [single-tree and
dual-tree traversal](../../../developer/trees.md#traversals) that can be paired
with a [`RuleType` class](../../../developer/trees.md#rules) to implement a
single-tree or dual-tree algorithm.

 * `CoverTree::SingleTreeTraverser`
   - Implements a breadth-first single-tree traverser: each level (scale) of the
     tree is visited, and base cases are computed and nodes are pruned before
     descending to the next level.

 * `CoverTree::DualTreeTraverser`
   - Implements a joint depth-first and breadth-first traversal as in the
     [original paper (pdf)](https://www.hunch.net/~jl/projects/cover_tree/paper/paper.pdf).
   - The query tree is descended in a depth-first manner; the reference tree is
     descended level-wise in a breadth-first manner, pruning node combinations
     where possible.
   - The level of the query tree and reference tree are held as even as possible
     during the traversal; so, in general, query and reference recursions will
     alternate.

## Example usage

Build a `CoverTree` on the `cloud` dataset and print basic statistics about the
tree.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Build the cover tree with default options.
//
// The std::move() means that `dataset` will be empty after this call, and the
// tree will "own" the dataset.  No data will be copied during tree building,
// regardless of whether we used `std::move()`.
//
// Note that the '<>' isn't necessary if C++20 is being used (e.g.
// `mlpack::CoverTree tree(...)` will work fine in C++20 or newer).
mlpack::CoverTree<> tree(std::move(dataset));

// Print the point held by the root node and the radius of the ball that
// contains all points:
std::cout << "Root node:" << std::endl;
std::cout << " - Base: " << tree.Base() << "." << std::endl;
std::cout << " - Scale: " << tree.Scale() << "." << std::endl;
std::cout << " - Point: " << tree.Dataset().col(tree.Point()).t();
std::cout << std::endl;

// Print the number of descendant points of the root, and of each of its
// children.
std::cout << "Descendant points of root:        "
    << tree.NumDescendants() << "." << std::endl;
std::cout << "Number of children of root: " << tree.NumChildren() << "."
    << std::endl;
for (size_t c = 0; c < tree.NumChildren(); ++c)
{
  std::cout << " - Descendant points of child " << c << ": "
      << tree.Child(c).NumDescendants() << "." << std::endl;
}
```

---

Build two `CoverTree`s on subsets of the corel dataset and compute minimum and
maximum distances between different nodes in the tree.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset, true);

// Build cover trees on the first half and the second half of points.
mlpack::CoverTree<> tree1(dataset.cols(0, dataset.n_cols / 2));
mlpack::CoverTree<> tree2(dataset.cols(dataset.n_cols / 2 + 1,
    dataset.n_cols - 1));

// Compute the maximum distance between the trees.
std::cout << "Maximum distance between tree root nodes: "
    << tree1.MaxDistance(tree2) << "." << std::endl;

// Get a grandchild of the first tree's root---if it exists.
if (!tree1.IsLeaf() && !tree1.Child(0).IsLeaf())
{
  mlpack::CoverTree<>& node1 = tree1.Child(0).Child(0);

  // Get a grandchild of the second tree's root---if it exists.
  if (!tree2.IsLeaf() && !tree2.Child(0).IsLeaf())
  {
    mlpack::CoverTree<>& node2 = tree2.Child(0).Child(0);

    // Print the minimum and maximum distance between the nodes.
    mlpack::Range dists = node1.RangeDistance(node2);
    std::cout << "Possible distances between two grandchild nodes: ["
        << dists.Lo() << ", " << dists.Hi() << "]." << std::endl;

    // Print the minimum distance between the first node and the first
    // descendant point of the second node.
    const size_t descendantIndex = node2.Descendant(0);
    const double descendantMinDist =
        node1.MinDistance(node2.Dataset().col(descendantIndex));
    std::cout << "Minimum distance between grandchild node and descendant "
        << "point: " << descendantMinDist << "." << std::endl;

    // Which child of node2 is closer to node1?
    const size_t closestIndex = node2.GetNearestChild(node1);
    std::cout << "Child " << closestIndex << " of node2 is closest to node1."
        << std::endl;

    // And which child of node1 is further from node2?
    const size_t furthestIndex = node1.GetFurthestChild(node2);
    std::cout << "Child " << furthestIndex << " of node1 is furthest from "
        << "node2." << std::endl;
  }
}
```

---

Build a `CoverTree` on 32-bit floating point data and save it to disk.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::fmat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Build the CoverTree using 32-bit floating point data as the matrix type.
// We will still use the default EmptyStatistic and EuclideanDistance
// parameters.
mlpack::CoverTree<mlpack::EuclideanDistance,
                  mlpack::EmptyStatistic,
                  arma::fmat> tree(dataset);

// Save the CoverTree to disk with the name 'tree'.
mlpack::data::Save("tree.bin", "tree", tree);

std::cout << "Saved tree with " << tree.Dataset().n_cols << " points to "
    << "'tree.bin'." << std::endl;
```

---

Load a 32-bit floating point `CoverTree` from disk, then traverse it manually
and find the number of leaf nodes with fewer than 10 points.

```c++
// This assumes the tree has already been saved to 'tree.bin' (as in the example
// above).

// This convenient typedef saves us a long type name!
using TreeType = mlpack::CoverTree<mlpack::EuclideanDistance,
                                   mlpack::EmptyStatistic,
                                   arma::fmat>;

TreeType tree;
mlpack::data::Load("tree.bin", "tree", tree);
std::cout << "Tree loaded with " << tree.NumDescendants() << " points."
    << std::endl;

// Recurse in a depth-first manner.  Count both the total number of leaves, and
// the number of nodes with more than 100 descendants.
size_t moreThan100Count = 0;
size_t totalLeafCount = 0;
std::stack<TreeType*> stack;
stack.push(&tree);
while (!stack.empty())
{
  TreeType* node = stack.top();
  stack.pop();

  if (node->NumDescendants() > 100)
    ++moreThan100Count;

  if (node->IsLeaf())
    ++totalLeafCount;

  for (size_t c = 0; c < node->NumChildren(); ++c)
    stack.push(&node->Child(c));
}

// Note that it would be possible to use TreeType::SingleTreeTraverser to
// perform the recursion above, but that is more well-suited for more complex
// tasks that require pruning and other non-trivial behavior; so using a simple
// stack is the better option here.

// Print the results.
std::cout << "Tree contains " << totalLeafCount << " leaves." << std::endl;
std::cout << moreThan100Count << " nodes have more than 100 descendants."
    << std::endl;
```
```

## src/mlpack/core/tree/cover_tree.hpp

```cpp
/**
 * @file core/tree/cover_tree.hpp
 * @author Ryan Curtin
 *
 * Includes all the necessary files to use the CoverTree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_HPP
#define MLPACK_CORE_TREE_COVER_TREE_HPP

#include <mlpack/prereqs.hpp>
#include "cover_tree/first_point_is_root.hpp"
#include "cover_tree/cover_tree.hpp"
#include "cover_tree/single_tree_traverser.hpp"
#include "cover_tree/single_tree_traverser_impl.hpp"
#include "cover_tree/dual_tree_traverser.hpp"
#include "cover_tree/dual_tree_traverser_impl.hpp"
#include "cover_tree/traits.hpp"
#include "cover_tree/typedef.hpp"

#endif
```

## src/mlpack/core/tree/cover_tree/cover_tree.hpp

```cpp
/**
 * @file core/tree/cover_tree/cover_tree.hpp
 * @author Ryan Curtin
 *
 * Definition of CoverTree, which can be used in place of the BinarySpaceTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP
#define MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/range.hpp>

#include "../statistic.hpp"
#include "first_point_is_root.hpp"

namespace mlpack {

/**
 * A cover tree is a tree specifically designed to speed up nearest-neighbor
 * computation in high-dimensional spaces.  Each non-leaf node references a
 * point and has a nonzero number of children, including a "self-child" which
 * references the same point.  A leaf node represents only one point.
 *
 * The tree can be thought of as a hierarchy with the root node at the top level
 * and the leaf nodes at the bottom level.  Each level in the tree has an
 * assigned 'scale' i.  The tree follows these two invariants:
 *
 * - nesting: the level C_i is a subset of the level C_{i - 1}.
 * - covering: all node in level C_{i - 1} have at least one node in the
 *     level C_i with distance less than or equal to b^i (exactly one of these
 *     is a parent of the point in level C_{i - 1}.
 *
 * Note that in the cover tree paper, there is a third invariant (the
 * 'separation invariant'), but that does not apply to our implementation,
 * because we have relaxed the invariant.
 *
 * The value 'b' refers to the base, which is a parameter of the tree.  These
 * three properties make the cover tree very good for fast, high-dimensional
 * nearest-neighbor search.
 *
 * The theoretical structure of the tree contains many 'implicit' nodes which
 * only have a "self-child" (a child referencing the same point, but at a lower
 * scale level).  This practical implementation only constructs explicit nodes
 * -- non-leaf nodes with more than one child.  A leaf node has no children, and
 * its scale level is INT_MIN.
 *
 * For more information on cover trees, see
 *
 * @code
 * @inproceedings{
 *   author = {Beygelzimer, Alina and Kakade, Sham and Langford, John},
 *   title = {Cover trees for nearest neighbor},
 *   booktitle = {Proceedings of the 23rd International Conference on Machine
 *     Learning},
 *   series = {ICML '06},
 *   year = {2006},
 *   pages = {97--104]
 * }
 * @endcode
 *
 * For information on runtime bounds of the nearest-neighbor computation using
 * cover trees, see the following paper, presented at NIPS 2009:
 *
 * @code
 * @inproceedings{
 *   author = {Ram, P., and Lee, D., and March, W.B., and Gray, A.G.},
 *   title = {Linear-time Algorithms for Pairwise Statistical Problems},
 *   booktitle = {Advances in Neural Information Processing Systems 22},
 *   editor = {Y. Bengio and D. Schuurmans and J. Lafferty and C.K.I. Williams
 *     and A. Culotta},
 *   pages = {1527--1535},
 *   year = {2009}
 * }
 * @endcode
 *
 * The CoverTree class offers three template parameters; a custom distance
 * metric type can be used with DistanceType (this class defaults to the
 * L2-squared metric).  The root node's point can be chosen with the
 * RootPointPolicy; by default, the FirstPointIsRoot policy is used, meaning the
 * first point in the dataset is used.  The StatisticType policy allows you to
 * define statistics which can be gathered during the creation of the tree.
 *
 * @tparam DistanceType Metric type to use during tree construction.
 * @tparam RootPointPolicy Determines which point to use as the root node.
 * @tparam StatisticType Statistic to be used during tree creation.
 * @tparam MatType Type of matrix to build the tree on (generally mat or
 *      sp_mat).
 */
template<typename DistanceType = LMetric<2, true>,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat,
         typename RootPointPolicy = FirstPointIsRoot>
class CoverTree
{
 public:
  //! So that other classes can access the matrix type.
  using Mat = MatType;
  //! The type held by the matrix type.
  using ElemType = typename MatType::elem_type;

  /**
   * A default constructor.  This returns an empty tree, which is not useful.
   * In general this is only used for serialization or right before copying from
   * a different object.
   */
  CoverTree();

  /**
   * Create the cover tree with the given dataset and given base.
   * The dataset will not be modified during the building procedure (unlike
   * BinarySpaceTree).
   *
   * The last argument will be removed in mlpack 1.1.0 (see #274 and #273).
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building (default 2.0).
   * @param distance Distance metric to use (default NULL).
   */
  CoverTree(const MatType& dataset,
            const ElemType base = 2.0,
            DistanceType* distance = NULL);

  /**
   * Create the cover tree with the given dataset and the given instantiated
   * distance metric.  Optionally, set the base.  The dataset will not be
   * modified during the building procedure (unlike BinarySpaceTree).
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param distance Instantiated distance metric to use during tree building.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(const MatType& dataset,
            DistanceType& distance,
            const ElemType base = 2.0);

  /**
   * Create the cover tree with the given dataset, taking ownership of the
   * dataset.  Optionally, set the base.
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(MatType&& dataset,
            const ElemType base = 2.0);

  /**
   * Create the cover tree with the given dataset and the given instantiated
   * distance metric, taking ownership of the dataset.  Optionally, set the
   * base.
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param distance Instantiated distance metric to use during tree building.
   * @param base Base to use during tree building (default 2.0).
   */
  CoverTree(MatType&& dataset,
            DistanceType& distance,
            const ElemType base = 2.0);

  /**
   * Construct a child cover tree node.  This constructor is not meant to be
   * used externally, but it could be used to insert another node into a tree.
   * This procedure uses only one vector for the near set, the far set, and the
   * used set (this is to prevent unnecessary memory allocation in recursive
   * calls to this constructor).  Therefore, the size of the near set, far set,
   * and used set must be passed in.  The near set will be entirely used up, and
   * some of the far set may be used.  The value of usedSetSize will be set to
   * the number of points used in the construction of this node, and the value
   * of farSetSize will be modified to reflect the number of points in the far
   * set _after_ the construction of this node.
   *
   * If you are calling this manually, be careful that the given scale is
   * as small as possible, or you may be creating an implicit node in your tree.
   *
   * @param dataset Reference to the dataset to build a tree on.
   * @param base Base to use during tree building.
   * @param pointIndex Index of the point this node references.
   * @param scale Scale of this level in the tree.
   * @param parent Parent of this node (NULL indicates no parent).
   * @param parentDistance Distance to the parent node.
   * @param indices Array of indices, ordered [ nearSet | farSet | usedSet ];
   *     will be modified to [ farSet | usedSet ].
   * @param distances Array of distances, ordered the same way as the indices.
   *     These represent the distances between the point specified by pointIndex
   *     and each point in the indices array.
   * @param nearSetSize Size of the near set; if 0, this will be a leaf.
   * @param farSetSize Size of the far set; may be modified (if this node uses
   *     any points in the far set).
   * @param usedSetSize The number of points used will be added to this number.
   * @param distance Distance metric to use (default NULL).
   */
  CoverTree(const MatType& dataset,
            const ElemType base,
            const size_t pointIndex,
            const int scale,
            CoverTree* parent,
            const ElemType parentDistance,
            arma::Col<size_t>& indices,
            arma::vec& distances,
            size_t nearSetSize,
            size_t& farSetSize,
            size_t& usedSetSize,
            DistanceType& distance = NULL);

  /**
   * Manually construct a cover tree node; no tree assembly is done in this
   * constructor, and children must be added manually (use Children()).  This
   * constructor is useful when the tree is being "imported" into the CoverTree
   * class after being created in some other manner.
   *
   * @param dataset Reference to the dataset this node is a part of.
   * @param base Base that was used for tree building.
   * @param pointIndex Index of the point in the dataset which this node refers
   *      to.
   * @param scale Scale of this node's level in the tree.
   * @param parent Parent node (NULL indicates no parent).
   * @param parentDistance Distance to parent node point.
   * @param furthestDescendantDistance Distance to furthest descendant point.
   * @param distance Instantiated distance metric (optional).
   */
  CoverTree(const MatType& dataset,
            const ElemType base,
            const size_t pointIndex,
            const int scale,
            CoverTree* parent,
            const ElemType parentDistance,
            const ElemType furthestDescendantDistance,
            DistanceType* distance = NULL);

  /**
   * Create a cover tree from another tree.  Be careful!  This may use a lot of
   * memory and take a lot of time.  This will also make a copy of the dataset.
   *
   * @param other Cover tree to copy from.
   */
  CoverTree(const CoverTree& other);

  /**
   * Move constructor for a Cover Tree, possess all the members of the given
   * tree.
   *
   * @param other Cover Tree to move.
   */
  CoverTree(CoverTree&& other);

  /**
   * Copy the given Cover Tree.
   *
   * @param other The tree to be copied.
   */
  CoverTree& operator=(const CoverTree& other);

  /**
   * Take ownership of the given Cover Tree.
   *
   * @param other The tree to take ownership of.
   */
  CoverTree& operator=(CoverTree&& other);

  /**
   * Create a cover tree from a cereal archive.
   */
  template<typename Archive>
  CoverTree(
      Archive& ar,
      const typename std::enable_if_t<cereal::is_loading<Archive>()>* = 0);

  /**
   * Delete this cover tree node and its children.
   */
  ~CoverTree();

  //! A single-tree cover tree traverser; see single_tree_traverser.hpp for
  //! implementation.
  template<typename RuleType>
  class SingleTreeTraverser;

  //! A dual-tree cover tree traverser; see dual_tree_traverser.hpp.
  template<typename RuleType>
  class DualTreeTraverser;

  template<typename RuleType>
  using BreadthFirstDualTreeTraverser = DualTreeTraverser<RuleType>;

  //! Get a reference to the dataset.
  const MatType& Dataset() const { return *dataset; }

  //! Get the index of the point which this node represents.
  size_t Point() const { return point; }
  //! For compatibility with other trees; the argument is ignored.
  size_t Point(const size_t) const { return point; }

  bool IsLeaf() const { return (children.size() == 0); }
  size_t NumPoints() const { return 1; }

  //! Get a particular child node.
  const CoverTree& Child(const size_t index) const { return *children[index]; }
  //! Modify a particular child node.
  CoverTree& Child(const size_t index) { return *children[index]; }

  CoverTree*& ChildPtr(const size_t index) { return children[index]; }

  //! Get the number of children.
  size_t NumChildren() const { return children.size(); }

  //! Get the children.
  const std::vector<CoverTree*>& Children() const { return children; }
  //! Modify the children manually (maybe not a great idea).
  std::vector<CoverTree*>& Children() { return children; }

  //! Get the number of descendant points.
  size_t NumDescendants() const;

  //! Get the index of a particular descendant point.
  size_t Descendant(const size_t index) const;

  //! Get the scale of this node.
  int Scale() const { return scale; }
  //! Modify the scale of this node.  Be careful...
  int& Scale() { return scale; }

  //! Get the base.
  ElemType Base() const { return base; }
  //! Modify the base; don't do this, you'll break everything.
  ElemType& Base() { return base; }

  //! Get the statistic for this node.
  const StatisticType& Stat() const { return stat; }
  //! Modify the statistic for this node.
  StatisticType& Stat() { return stat; }

  /**
   * Return the index of the nearest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetNearestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Return the index of the furthest child node to the given query point.  If
   * this is a leaf node, it will return NumChildren() (invalid index).
   */
  template<typename VecType>
  size_t GetFurthestChild(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0);

  /**
   * Return the index of the nearest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetNearestChild(const CoverTree& queryNode);

  /**
   * Return the index of the furthest child node to the given query node.  If it
   * can't decide, it will return NumChildren() (invalid index).
   */
  size_t GetFurthestChild(const CoverTree& queryNode);

  //! Return the minimum distance to another node.
  ElemType MinDistance(const CoverTree& other) const;

  //! Return the minimum distance to another node given that the point-to-point
  //! distance has already been calculated.
  ElemType MinDistance(const CoverTree& other, const ElemType distance) const;

  //! Return the minimum distance to another point.
  ElemType MinDistance(const arma::vec& other) const;

  //! Return the minimum distance to another point given that the distance from
  //! the center to the point has already been calculated.
  ElemType MinDistance(const arma::vec& other, const ElemType distance) const;

  //! Return the maximum distance to another node.
  ElemType MaxDistance(const CoverTree& other) const;

  //! Return the maximum distance to another node given that the point-to-point
  //! distance has already been calculated.
  ElemType MaxDistance(const CoverTree& other, const ElemType distance) const;

  //! Return the maximum distance to another point.
  ElemType MaxDistance(const arma::vec& other) const;

  //! Return the maximum distance to another point given that the distance from
  //! the center to the point has already been calculated.
  ElemType MaxDistance(const arma::vec& other, const ElemType distance) const;

  //! Return the minimum and maximum distance to another node.
  RangeType<ElemType> RangeDistance(const CoverTree& other) const;

  //! Return the minimum and maximum distance to another node given that the
  //! point-to-point distance has already been calculated.
  RangeType<ElemType> RangeDistance(const CoverTree& other,
                                          const ElemType distance) const;

  //! Return the minimum and maximum distance to another point.
  RangeType<ElemType> RangeDistance(const arma::vec& other) const;

  //! Return the minimum and maximum distance to another point given that the
  //! point-to-point distance has already been calculated.
  RangeType<ElemType> RangeDistance(const arma::vec& other,
                                          const ElemType distance) const;

  //! Get the parent node.
  CoverTree* Parent() const { return parent; }
  //! Modify the parent node.
  CoverTree*& Parent() { return parent; }

  //! Get the distance to the parent.
  ElemType ParentDistance() const { return parentDistance; }
  //! Modify the distance to the parent.
  ElemType& ParentDistance() { return parentDistance; }

  //! Get the distance to the furthest point.  This is always 0 for cover trees.
  ElemType FurthestPointDistance() const { return 0.0; }

  //! Get the distance from the center of the node to the furthest descendant.
  ElemType FurthestDescendantDistance() const
  { return furthestDescendantDistance; }
  //! Modify the distance from the center of the node to the furthest
  //! descendant.
  ElemType& FurthestDescendantDistance() { return furthestDescendantDistance; }

  //! Get the minimum distance from the center to any bound edge (this is the
  //! same as furthestDescendantDistance).
  ElemType MinimumBoundDistance() const { return furthestDescendantDistance; }

  //! Get the center of the node and store it in the given vector.
  void Center(arma::vec& center) const
  {
    center = arma::vec(dataset->col(point));
  }

  //! Get the instantiated distance metric.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  DistanceType& Metric() const { return *distance; }

  //! Get the instantiated distance metric.
  DistanceType& Distance() const { return *distance; }

  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  size_t DistanceComps() const { return distanceComps; }
  size_t& DistanceComps() { return distanceComps; }

 private:
  //! Reference to the matrix which this tree is built on.
  const MatType* dataset;
  //! Index of the point in the matrix which this node represents.
  size_t point;
  //! The list of children; the first is the self-child.
  std::vector<CoverTree*> children;
  //! Scale level of the node.
  int scale;
  //! The base used to construct the tree.
  ElemType base;
  //! The instantiated statistic.
  StatisticType stat;
  //! The number of descendant points.
  size_t numDescendants;
  //! The parent node (NULL if this is the root of the tree).
  CoverTree* parent;
  //! Distance to the parent.
  ElemType parentDistance;
  //! Distance to the furthest descendant.
  ElemType furthestDescendantDistance;
  //! Whether or not we need to destroy the distance metric in the destructor.
  bool localDistance;
  //! If true, we own the dataset and need to destroy it in the destructor.
  bool localDataset;
  //! The distance metric used for this tree.
  DistanceType* distance;

  /**
   * Create the children for this node.
   */
  void CreateChildren(arma::Col<size_t>& indices,
                      arma::vec& distances,
                      size_t nearSetSize,
                      size_t& farSetSize,
                      size_t& usedSetSize);

  /**
   * Fill the vector of distances with the distances between the point specified
   * by pointIndex and each point in the indices array.  The distances of the
   * first pointSetSize points in indices are calculated (so, this does not
   * necessarily need to use all of the points in the arrays).
   *
   * @param pointIndex Point to build the distances for.
   * @param indices List of indices to compute distances for.
   * @param distances Vector to store calculated distances in.
   * @param pointSetSize Number of points in arrays to calculate distances for.
   */
  void ComputeDistances(const size_t pointIndex,
                        const arma::Col<size_t>& indices,
                        arma::vec& distances,
                        const size_t pointSetSize);
  /**
   * Split the given indices and distances into a near and a far set, returning
   * the number of points in the near set.  The distances must already be
   * initialized.  This will order the indices and distances such that the
   * points in the near set make up the first part of the array and the far set
   * makes up the rest:  [ nearSet | farSet ].
   *
   * @param indices List of indices; will be reordered.
   * @param distances List of distances; will be reordered.
   * @param bound If the distance is less than or equal to this bound, the point
   *      is placed into the near set.
   * @param pointSetSize Size of point set (because we may be sorting a smaller
   *      list than the indices vector will hold).
   */
  size_t SplitNearFar(arma::Col<size_t>& indices,
                      arma::vec& distances,
                      const ElemType bound,
                      const size_t pointSetSize);

  /**
   * Assuming that the list of indices and distances is sorted as
   * [ childFarSet | childUsedSet | farSet | usedSet ],
   * resort the sets so the organization is
   * [ childFarSet | farSet | childUsedSet | usedSet ].
   *
   * The size_t parameters specify the sizes of each set in the array.  Only the
   * ordering of the indices and distances arrays will be modified (not their
   * actual contents).
   *
   * The size of any of the four sets can be zero and this method will handle
   * that case accordingly.
   *
   * @param indices List of indices to sort.
   * @param distances List of distances to sort.
   * @param childFarSetSize Number of points in child far set (childFarSet).
   * @param childUsedSetSize Number of points in child used set (childUsedSet).
   * @param farSetSize Number of points in far set (farSet).
   */
  size_t SortPointSet(arma::Col<size_t>& indices,
                      arma::vec& distances,
                      const size_t childFarSetSize,
                      const size_t childUsedSetSize,
                      const size_t farSetSize);

  void MoveToUsedSet(arma::Col<size_t>& indices,
                     arma::vec& distances,
                     size_t& nearSetSize,
                     size_t& farSetSize,
                     size_t& usedSetSize,
                     arma::Col<size_t>& childIndices,
                     const size_t childFarSetSize,
                     const size_t childUsedSetSize);
  size_t PruneFarSet(arma::Col<size_t>& indices,
                     arma::vec& distances,
                     const ElemType bound,
                     const size_t nearSetSize,
                     const size_t pointSetSize);

  /**
   * Take a look at the last child (the most recently created one) and remove
   * any implicit nodes that have been created.
   */
  void RemoveNewImplicitNodes();

 private:
  size_t distanceComps;
};

} // namespace mlpack

// Include implementation.
#include "cover_tree_impl.hpp"

// Include the rest of the pieces, if necessary.
#include "../cover_tree.hpp"

#endif
```

## src/mlpack/core/tree/cover_tree/cover_tree_impl.hpp

```cpp
/**
 * @file core/tree/cover_tree/cover_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of CoverTree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_IMPL_HPP

// In case it hasn't already been included.
#include "cover_tree.hpp"

#include <queue>
#include <string>

#include <mlpack/core/util/log.hpp>

namespace mlpack {

// Build the statistics, bottom-up.
template<typename TreeType, typename StatisticType>
void BuildStatistics(TreeType* node)
{
  // Recurse first.
  for (size_t i = 0; i < node->NumChildren(); ++i)
    BuildStatistics<TreeType, StatisticType>(&node->Child(i));

  // Now build the statistic.
  node->Stat() = StatisticType(*node);
}

// Create the cover tree.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    DistanceType* distance) :
    dataset(&dataset),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(distance == NULL),
    localDataset(false),
    distance(distance),
    distanceComps(0)
{
  // If we need to create a distance metric, do that.  We'll just do it on the
  // heap.
  if (localDistance)
    this->distance = new DistanceType();

  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset.n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset.n_cols - 1, dataset.n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset.n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset.n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
  CreateChildren(indices, distances, dataset.n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly.
      old->Child(i).Parent() = this;
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset.n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    DistanceType& distance,
    const ElemType base) :
    dataset(&dataset),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(true),
    localDataset(false),
    distance(new DistanceType(distance)),
    distanceComps(0)
{
  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset.n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset.n_cols - 1, dataset.n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset.n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset.n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
  CreateChildren(indices, distances, dataset.n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly.
      old->Child(i).Parent() = this;
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset.n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    MatType&& data,
    const ElemType base) :
    dataset(new MatType(std::move(data))),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(true),
    localDataset(true),
    distanceComps(0)
{
  // We need to create a distance metric.  We'll just do it on the heap.
  this->distance = new DistanceType();

  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset->n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset->n_cols - 1, dataset->n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset->n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset->n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
  CreateChildren(indices, distances, dataset->n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly.
      old->Child(i).Parent() = this;
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset->n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    MatType&& data,
    DistanceType& distance,
    const ElemType base) :
    dataset(new MatType(std::move(data))),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(true),
    localDataset(true),
    distance(new DistanceType(distance)),
    distanceComps(0)
{
  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset->n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset->n_cols - 1, dataset->n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset->n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset->n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
  CreateChildren(indices, distances, dataset->n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly.
      old->Child(i).Parent() = this;
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset->n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    const size_t pointIndex,
    const int scale,
    CoverTree* parent,
    const ElemType parentDistance,
    arma::Col<size_t>& indices,
    arma::vec& distances,
    size_t nearSetSize,
    size_t& farSetSize,
    size_t& usedSetSize,
    DistanceType& distance) :
    dataset(&dataset),
    point(pointIndex),
    scale(scale),
    base(base),
    numDescendants(0),
    parent(parent),
    parentDistance(parentDistance),
    furthestDescendantDistance(0),
    localDistance(false),
    localDataset(false),
    distance(&distance),
    distanceComps(0)
{
  // If the size of the near set is 0, this is a leaf.
  if (nearSetSize == 0)
  {
    this->scale = INT_MIN;
    numDescendants = 1;
    return;
  }

  // Otherwise, create the children.
  CreateChildren(indices, distances, nearSetSize, farSetSize, usedSetSize);
}

// Manually create a cover tree node.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    const size_t pointIndex,
    const int scale,
    CoverTree* parent,
    const ElemType parentDistance,
    const ElemType furthestDescendantDistance,
    DistanceType* distance) :
    dataset(&dataset),
    point(pointIndex),
    scale(scale),
    base(base),
    numDescendants(0),
    parent(parent),
    parentDistance(parentDistance),
    furthestDescendantDistance(furthestDescendantDistance),
    localDistance(distance == NULL),
    localDataset(false),
    distance(distance),
    distanceComps(0)
{
  // If necessary, create a local distance metric.
  if (localDistance)
    this->distance = new DistanceType();
}

// Copy Constructor.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const CoverTree& other) :
    dataset((other.parent == NULL && other.localDataset) ?
        new MatType(*other.dataset) : other.dataset),
    point(other.point),
    scale(other.scale),
    base(other.base),
    stat(other.stat),
    numDescendants(other.numDescendants),
    parent(other.parent),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    localDistance(other.localDistance),
    localDataset(other.parent == NULL && other.localDataset),
    distance((other.localDistance ? new DistanceType() : other.distance)),
    distanceComps(0)
{
  // Copy each child by hand.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new CoverTree(other.Child(i)));
    children[i]->Parent() = this;
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<CoverTree*> queue;

    for (size_t i = 0; i < NumChildren(); ++i)
      queue.push(children[i]);

    while (!queue.empty())
    {
      CoverTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      for (size_t i = 0; i < node->NumChildren(); ++i)
        queue.push(node->children[i]);
    }
  }
}

// Copy assignment operator: copy the given other tree.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>&
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
operator=(const CoverTree& other)
{
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  if (localDataset)
    delete dataset;

  if (localDistance)
    delete distance;

  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  dataset = ((other.parent == NULL && other.localDataset) ?
      new MatType(*other.dataset) : other.dataset);
  point = other.point;
  scale = other.scale;
  base = other.base;
  stat = other.stat;
  numDescendants = other.numDescendants;
  parent = other.parent;
  parentDistance = other.parentDistance;
  furthestDescendantDistance = other.furthestDescendantDistance;
  localDistance = other.localDistance;
  localDataset = (other.parent == NULL && other.localDataset);
  distance = (other.localDistance ? new DistanceType() : other.distance);
  distanceComps = 0;

  // Copy each child by hand.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new CoverTree(other.Child(i)));
    children[i]->Parent() = this;
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<CoverTree*> queue;

    for (size_t i = 0; i < NumChildren(); ++i)
      queue.push(children[i]);

    while (!queue.empty())
    {
      CoverTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      for (size_t i = 0; i < node->NumChildren(); ++i)
        queue.push(node->children[i]);
    }
  }

  return *this;
}

// Move Constructor.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    CoverTree&& other) :
    dataset(other.dataset),
    point(other.point),
    children(std::move(other.children)),
    scale(other.scale),
    base(other.base),
    stat(std::move(other.stat)),
    numDescendants(other.numDescendants),
    parent(other.parent),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    localDistance(other.localDistance),
    localDataset(other.localDataset),
    distance(other.distance),
    distanceComps(other.distanceComps)
{
  // Set proper parent pointer.
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->Parent() = this;

  other.dataset = NULL;
  other.point = 0;
  other.scale = INT_MIN;
  other.base = 0;
  other.numDescendants = 0;
  other.parent = NULL;
  other.parentDistance = 0;
  other.furthestDescendantDistance = 0;
  other.localDistance = false;
  other.localDataset = false;
  other.distance = NULL;
}

// Move assignment operator: take ownership of the given tree.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>&
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
operator=(CoverTree&& other)
{
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  if (localDataset)
    delete dataset;

  if (localDistance)
    delete distance;

  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];

  dataset = other.dataset;
  point = other.point;
  children = std::move(other.children);
  scale = other.scale;
  base = other.base;
  stat = std::move(other.stat);
  numDescendants = other.numDescendants;
  parent = other.parent;
  parentDistance = other.parentDistance;
  furthestDescendantDistance = other.furthestDescendantDistance;
  localDistance = other.localDistance;
  localDataset = other.localDataset;
  distance = other.distance;
  distanceComps = other.distanceComps;

  // Set proper parent pointer.
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->Parent() = this;

  other.dataset = NULL;
  other.point = 0;
  other.scale = INT_MIN;
  other.base = 0;
  other.numDescendants = 0;
  other.parent = NULL;
  other.parentDistance = 0;
  other.furthestDescendantDistance = 0;
  other.localDistance = false;
  other.localDataset = false;
  other.distance = NULL;

  return *this;
}

// Construct from a cereal archive.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename Archive>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    Archive& ar,
    const typename std::enable_if_t<cereal::is_loading<Archive>()>*) :
    CoverTree() // Create an empty CoverTree.
{
  // Now, serialize to our empty tree.
  ar(cereal::make_nvp("this", *this));
}


template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::~CoverTree()
{
  // Delete each child.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];

  // Delete the local distance metric, if necessary.
  if (localDistance)
    delete distance;

  // Delete the local dataset, if necessary.
  if (localDataset)
    delete dataset;
}

//! Return the number of descendant points.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline size_t
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    NumDescendants() const
{
  return numDescendants;
}

//! Return the index of a particular descendant point.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline size_t
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::Descendant(
    const size_t index) const
{
  // The first descendant is the point contained within this node.
  if (index == 0)
    return point;

  // Is it in the self-child?
  if (index < children[0]->NumDescendants())
    return children[0]->Descendant(index);

  // Now check the other children.
  size_t sum = children[0]->NumDescendants();
  for (size_t i = 1; i < children.size(); ++i)
  {
    if (index - sum < children[i]->NumDescendants())
      return children[i]->Descendant(index - sum);
    sum += children[i]->NumDescendants();
  }

  // This should never happen.
  return (size_t() - 1);
}

/**
 * Return the index of the nearest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
template<typename VecType>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetNearestChild(const VecType& point,
                    typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MinDistance(point);
    if (distance <= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return the index of the furthest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
template<typename VecType>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetFurthestChild(const VecType& point,
                     typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = 0;
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MaxDistance(point);
    if (distance >= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return the index of the nearest child node to the given query node.  If it
 * can't decide, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetNearestChild(const CoverTree& queryNode)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MinDistance(queryNode);
    if (distance <= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return the index of the furthest child node to the given query node.  If it
 * can't decide, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetFurthestChild(const CoverTree& queryNode)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = 0;
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MaxDistance(queryNode);
    if (distance >= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const CoverTree& other) const
{
  // Every cover tree node will contain points up to base^(scale + 1) away.
  return std::max(distance->Evaluate(dataset->col(point),
      other.Dataset().col(other.Point())) -
      furthestDescendantDistance - other.FurthestDescendantDistance(), 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const CoverTree& other, const ElemType distance) const
{
  // We already have the distance as evaluated by the metric.
  return std::max(distance - furthestDescendantDistance -
      other.FurthestDescendantDistance(), 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const arma::vec& other) const
{
  return std::max(distance->Evaluate(dataset->col(point), other) -
      furthestDescendantDistance, 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const arma::vec& /* other */, const ElemType distance) const
{
  return std::max(distance - furthestDescendantDistance, 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const CoverTree& other) const
{
  return distance->Evaluate(dataset->col(point),
      other.Dataset().col(other.Point())) +
      furthestDescendantDistance + other.FurthestDescendantDistance();
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const CoverTree& other, const ElemType distance) const
{
  // We already have the distance as evaluated by the metric.
  return distance + furthestDescendantDistance +
      other.FurthestDescendantDistance();
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const arma::vec& other) const
{
  return distance->Evaluate(dataset->col(point), other) +
      furthestDescendantDistance;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const arma::vec& /* other */, const ElemType distance) const
{
  return distance + furthestDescendantDistance;
}

//! Return the minimum and maximum distance to another node.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const CoverTree& other) const
{
  const ElemType dist = distance->Evaluate(dataset->col(point),
      other.Dataset().col(other.Point()));

  RangeType<ElemType> result;
  result.Lo() = std::max(dist - furthestDescendantDistance -
      other.FurthestDescendantDistance(), 0.0);
  result.Hi() = dist + furthestDescendantDistance +
      other.FurthestDescendantDistance();

  return result;
}

//! Return the minimum and maximum distance to another node given that the
//! point-to-point distance has already been calculated.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const CoverTree& other,
                  const ElemType distance) const
{
  RangeType<ElemType> result;
  result.Lo() = std::max(distance - furthestDescendantDistance -
      other.FurthestDescendantDistance(), 0.0);
  result.Hi() = distance + furthestDescendantDistance +
      other.FurthestDescendantDistance();

  return result;
}

//! Return the minimum and maximum distance to another point.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const arma::vec& other) const
{
  const ElemType dist = distance->Evaluate(dataset->col(point), other);

  return RangeType<ElemType>(
      std::max(dist - furthestDescendantDistance, 0.0),
      dist + furthestDescendantDistance);
}

//! Return the minimum and maximum distance to another point given that the
//! point-to-point distance has already been calculated.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const arma::vec& /* other */,
                  const ElemType distance) const
{
  return RangeType<ElemType>(
      std::max(distance - furthestDescendantDistance, 0.0),
      distance + furthestDescendantDistance);
}

//! For a newly initialized node, create children using the near and far set.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline void
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    CreateChildren(arma::Col<size_t>& indices,
                   arma::vec& distances,
                   size_t nearSetSize,
                   size_t& farSetSize,
                   size_t& usedSetSize)
{
  // Determine the next scale level.  This should be the first level where there
  // are any points in the far set.  So, if we know the maximum distance in the
  // distances array, this will be the largest i such that
  //   maxDistance > pow(base, i)
  // and using this for the scale factor should guarantee we are not creating an
  // implicit node.  If the maximum distance is 0, every point in the near set
  // will be created as a leaf, and a child to this node.  We also do not need
  // to change the furthestChildDistance or furthestDescendantDistance.
  const ElemType maxDistance = max(distances.rows(0,
      nearSetSize + farSetSize - 1));
  if (maxDistance == 0)
  {
    // Make the self child at the lowest possible level.
    // This should not modify farSetSize or usedSetSize.
    size_t tempSize = 0;
    children.push_back(new CoverTree(*dataset, base, point, INT_MIN, this, 0,
        indices, distances, 0, tempSize, usedSetSize, *distance));
    distanceComps += children.back()->DistanceComps();

    // Every point in the near set should be a leaf.
    for (size_t i = 0; i < nearSetSize; ++i)
    {
      // farSetSize and usedSetSize will not be modified.
      children.push_back(new CoverTree(*dataset, base, indices[i],
          INT_MIN, this, distances[i], indices, distances, 0, tempSize,
          usedSetSize, *distance));
      distanceComps += children.back()->DistanceComps();
      usedSetSize++;
    }

    // The number of descendants is just the number of children, because each of
    // them are leaves and contain one point.
    numDescendants = children.size();

    // Re-sort the dataset.  We have
    // [ used | far | other used ]
    // and we want
    // [ far | all used ].
    SortPointSet(indices, distances, 0, usedSetSize, farSetSize);

    return;
  }

  const int nextScale = std::min(scale,
      (int) std::ceil(std::log(maxDistance) / std::log(base))) - 1;
  const ElemType bound = std::pow(base, nextScale);

  // First, make the self child.  We must split the given near set into the near
  // set and far set for the self child.
  size_t childNearSetSize =
      SplitNearFar(indices, distances, bound, nearSetSize);

  // Build the self child (recursively).
  size_t childFarSetSize = nearSetSize - childNearSetSize;
  size_t childUsedSetSize = 0;
  children.push_back(new CoverTree(*dataset, base, point, nextScale, this, 0,
      indices, distances, childNearSetSize, childFarSetSize, childUsedSetSize,
      *distance));
  // Don't double-count the self-child (so, subtract one).
  numDescendants += children[0]->NumDescendants();

  // The self-child can't modify the furthestChildDistance away from 0, but it
  // can modify the furthestDescendantDistance.
  furthestDescendantDistance = children[0]->FurthestDescendantDistance();

  // Remove any implicit nodes we may have created.
  RemoveNewImplicitNodes();

  distanceComps += children[0]->DistanceComps();

  // Now the arrays, in memory, look like this:
  // [ childFar | childUsed | far | used ]
  // but we need to move the used points past our far set:
  // [ childFar | far | childUsed + used ]
  // and keeping in mind that childFar = our near set,
  // [ near | far | childUsed + used ]
  // is what we are trying to make.
  SortPointSet(indices, distances, childFarSetSize, childUsedSetSize,
      farSetSize);

  // Update size of near set and used set.
  nearSetSize -= childUsedSetSize;
  usedSetSize += childUsedSetSize;

  // Now for each point in the near set, we need to make children.  To save
  // computation later, we'll create an array holding the points in the near
  // set, and then after each run we'll check which of those (if any) were used
  // and we will remove them.  ...if that's faster.  I think it is.
  while (nearSetSize > 0)
  {
    size_t newPointIndex = nearSetSize - 1;

    // Swap to front if necessary.
    if (newPointIndex != 0)
    {
      const size_t tempIndex = indices[newPointIndex];
      const ElemType tempDist = distances[newPointIndex];

      indices[newPointIndex] = indices[0];
      distances[newPointIndex] = distances[0];

      indices[0] = tempIndex;
      distances[0] = tempDist;
    }

    // Will this be a new furthest child?
    if (distances[0] > furthestDescendantDistance)
      furthestDescendantDistance = distances[0];

    // If there's only one point left, we don't need this crap.
    if ((nearSetSize == 1) && (farSetSize == 0))
    {
      size_t childNearSetSize = 0;
      children.push_back(new CoverTree(*dataset, base, indices[0], nextScale,
          this, distances[0], indices, distances, childNearSetSize, farSetSize,
          usedSetSize, *distance));
      distanceComps += children.back()->DistanceComps();
      numDescendants += children.back()->NumDescendants();

      // Because the far set size is 0, we don't have to do any swapping to
      // move the point into the used set.
      ++usedSetSize;
      --nearSetSize;

      // And we're done.
      break;
    }

    // Create the near and far set indices and distance vectors.  We don't fill
    // in the self-point, yet.
    arma::Col<size_t> childIndices(nearSetSize + farSetSize);
    childIndices.rows(0, (nearSetSize + farSetSize - 2)) = indices.rows(1,
        nearSetSize + farSetSize - 1);
    arma::vec childDistances(nearSetSize + farSetSize);

    // Build distances for the child.
    ComputeDistances(indices[0], childIndices, childDistances, nearSetSize
        + farSetSize - 1);

    // Split into near and far sets for this point.
    childNearSetSize = SplitNearFar(childIndices, childDistances, bound,
        nearSetSize + farSetSize - 1);
    childFarSetSize = PruneFarSet(childIndices, childDistances,
        base * bound, childNearSetSize,
        (nearSetSize + farSetSize - 1));

    // Now that we know the near and far set sizes, we can put the used point
    // (the self point) in the correct place; now, when we call
    // MoveToUsedSet(), it will move the self-point correctly.  The distance
    // does not matter.
    childIndices(childNearSetSize + childFarSetSize) = indices[0];
    childDistances(childNearSetSize + childFarSetSize) = 0;

    // Build this child (recursively).
    childUsedSetSize = 1; // Mark self point as used.
    children.push_back(new CoverTree(*dataset, base, indices[0], nextScale,
        this, distances[0], childIndices, childDistances, childNearSetSize,
        childFarSetSize, childUsedSetSize, *distance));
    numDescendants += children.back()->NumDescendants();

    // Remove any implicit nodes.
    RemoveNewImplicitNodes();

    distanceComps += children.back()->DistanceComps();

    // Now with the child created, it returns the childIndices and
    // childDistances vectors in this form:
    // [ childFar | childUsed ]
    // For each point in the childUsed set, we must move that point to the used
    // set in our own vector.
    MoveToUsedSet(indices, distances, nearSetSize, farSetSize, usedSetSize,
        childIndices, childFarSetSize, childUsedSetSize);
  }

  // Calculate furthest descendant.
  for (size_t i = (nearSetSize + farSetSize); i < (nearSetSize + farSetSize +
      usedSetSize); ++i)
    if (distances[i] > furthestDescendantDistance)
      furthestDescendantDistance = distances[i];
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SplitNearFar(arma::Col<size_t>& indices,
                 arma::vec& distances,
                 const ElemType bound,
                 const size_t pointSetSize)
{
  // Sanity check; there is no guarantee that this condition will not be true.
  // ...or is there?
  if (pointSetSize <= 1)
    return 0;

  // We'll traverse from both left and right.
  size_t left = 0;
  size_t right = pointSetSize - 1;

  // A modification of quicksort, with the pivot value set to the bound.
  // Everything on the left of the pivot will be less than or equal to the
  // bound; everything on the right will be greater than the bound.
  while ((distances[left] <= bound) && (left != right))
    ++left;
  while ((distances[right] > bound) && (left != right))
    --right;

  while (left != right)
  {
    // Now swap the values and indices.
    const size_t tempPoint = indices[left];
    const ElemType tempDist = distances[left];

    indices[left] = indices[right];
    distances[left] = distances[right];

    indices[right] = tempPoint;
    distances[right] = tempDist;

    // Traverse the left, seeing how many points are correctly on that side.
    // When we encounter an incorrect point, stop.  We will switch it later.
    while ((distances[left] <= bound) && (left != right))
      ++left;

    // Traverse the right, seeing how many points are correctly on that side.
    // When we encounter an incorrect point, stop.  We will switch it with the
    // wrong point from the left side.
    while ((distances[right] > bound) && (left != right))
      --right;
  }

  // The final left value is the index of the first far value.
  return left;
}

// Returns the maximum distance between points.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    ComputeDistances(const size_t pointIndex,
                     const arma::Col<size_t>& indices,
                     arma::vec& distances,
                     const size_t pointSetSize)
{
  // For each point, rebuild the distances.  The indices do not need to be
  // modified.
  distanceComps += pointSetSize;
  for (size_t i = 0; i < pointSetSize; ++i)
  {
    distances[i] = distance->Evaluate(dataset->col(pointIndex),
        dataset->col(indices[i]));
  }
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SortPointSet(arma::Col<size_t>& indices,
                 arma::vec& distances,
                 const size_t childFarSetSize,
                 const size_t childUsedSetSize,
                 const size_t farSetSize)
{
  // We'll use low-level memcpy calls ourselves, just to ensure it's done
  // quickly and the way we want it to be.  Unfortunately this takes up more
  // memory than one-element swaps, but there's not a great way around that.
  const size_t bufferSize = std::min(farSetSize, childUsedSetSize);
  const size_t bigCopySize = std::max(farSetSize, childUsedSetSize);

  // Sanity check: there is no need to sort if the buffer size is going to be
  // zero.
  if (bufferSize == 0)
    return (childFarSetSize + farSetSize);

  size_t* indicesBuffer = new size_t[bufferSize];
  ElemType* distancesBuffer = new ElemType[bufferSize];

  // The start of the memory region to copy to the buffer.
  const size_t bufferFromLocation = ((bufferSize == farSetSize) ?
      (childFarSetSize + childUsedSetSize) : childFarSetSize);
  // The start of the memory region to move directly to the new place.
  const size_t directFromLocation = ((bufferSize == farSetSize) ?
      childFarSetSize : (childFarSetSize + childUsedSetSize));
  // The destination to copy the buffer back to.
  const size_t bufferToLocation = ((bufferSize == farSetSize) ?
      childFarSetSize : (childFarSetSize + farSetSize));
  // The destination of the directly moved memory region.
  const size_t directToLocation = ((bufferSize == farSetSize) ?
      (childFarSetSize + farSetSize) : childFarSetSize);

  // Copy the smaller piece to the buffer.
  memcpy(indicesBuffer, indices.memptr() + bufferFromLocation,
      sizeof(size_t) * bufferSize);
  memcpy(distancesBuffer, distances.memptr() + bufferFromLocation,
      sizeof(ElemType) * bufferSize);

  // Now move the other memory.
  memmove(indices.memptr() + directToLocation,
      indices.memptr() + directFromLocation, sizeof(size_t) * bigCopySize);
  memmove(distances.memptr() + directToLocation,
      distances.memptr() + directFromLocation, sizeof(ElemType) * bigCopySize);

  // Now copy the temporary memory to the right place.
  memcpy(indices.memptr() + bufferToLocation, indicesBuffer,
      sizeof(size_t) * bufferSize);
  memcpy(distances.memptr() + bufferToLocation, distancesBuffer,
      sizeof(ElemType) * bufferSize);

  delete[] indicesBuffer;
  delete[] distancesBuffer;

  // This returns the complete size of the far set.
  return (childFarSetSize + farSetSize);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MoveToUsedSet(arma::Col<size_t>& indices,
                  arma::vec& distances,
                  size_t& nearSetSize,
                  size_t& farSetSize,
                  size_t& usedSetSize,
                  arma::Col<size_t>& childIndices,
                  const size_t childFarSetSize, // childNearSetSize is 0 here.
                  const size_t childUsedSetSize)
{
  const size_t originalSum = nearSetSize + farSetSize + usedSetSize;

  // Loop across the set.  We will swap points as we need.  It should be noted
  // that farSetSize and nearSetSize may change with each iteration of this loop
  // (depending on if we make a swap or not).
  size_t startChildUsedSet = 0; // Where to start in the child set.
  for (size_t i = 0; i < nearSetSize; ++i)
  {
    // Discover if this point was in the child's used set.
    for (size_t j = startChildUsedSet; j < childUsedSetSize; ++j)
    {
      if (childIndices[childFarSetSize + j] == indices[i])
      {
        // We have found a point; a swap is necessary.

        // Since this point is from the near set, to preserve the near set, we
        // must do a swap.
        if (farSetSize > 0)
        {
          if ((nearSetSize - 1) != i)
          {
            // In this case it must be a three-way swap.
            size_t tempIndex = indices[nearSetSize + farSetSize - 1];
            ElemType tempDist = distances[nearSetSize + farSetSize - 1];

            size_t tempNearIndex = indices[nearSetSize - 1];
            ElemType tempNearDist = distances[nearSetSize - 1];

            indices[nearSetSize + farSetSize - 1] = indices[i];
            distances[nearSetSize + farSetSize - 1] = distances[i];

            indices[nearSetSize - 1] = tempIndex;
            distances[nearSetSize - 1] = tempDist;

            indices[i] = tempNearIndex;
            distances[i] = tempNearDist;
          }
          else
          {
            // We can do a two-way swap.
            size_t tempIndex = indices[nearSetSize + farSetSize - 1];
            ElemType tempDist = distances[nearSetSize + farSetSize - 1];

            indices[nearSetSize + farSetSize - 1] = indices[i];
            distances[nearSetSize + farSetSize - 1] = distances[i];

            indices[i] = tempIndex;
            distances[i] = tempDist;
          }
        }
        else if ((nearSetSize - 1) != i)
        {
          // A two-way swap is possible.
          size_t tempIndex = indices[nearSetSize + farSetSize - 1];
          ElemType tempDist = distances[nearSetSize + farSetSize - 1];

          indices[nearSetSize + farSetSize - 1] = indices[i];
          distances[nearSetSize + farSetSize - 1] = distances[i];

          indices[i] = tempIndex;
          distances[i] = tempDist;
        }
        else
        {
          // No swap is necessary.
        }

        // We don't need to do a complete preservation of the child index set,
        // but we want to make sure we only loop over points we haven't seen.
        // So increment the child counter by 1 and move a point if we need.
        if (j != startChildUsedSet)
        {
          childIndices[childFarSetSize + j] = childIndices[childFarSetSize +
              startChildUsedSet];
        }

        // Update all counters from the swaps we have done.
        ++startChildUsedSet;
        --nearSetSize;
        --i; // Since we moved a point out of the near set we must step back.

        break; // Break out of this for loop; back to the first one.
      }
    }
  }

  // Now loop over the far set.  This loop is different because we only require
  // a normal two-way swap instead of the three-way swap to preserve the near
  // set / far set ordering.
  for (size_t i = 0; i < farSetSize; ++i)
  {
    // Discover if this point was in the child's used set.
    for (size_t j = startChildUsedSet; j < childUsedSetSize; ++j)
    {
      if (childIndices[childFarSetSize + j] == indices[i + nearSetSize])
      {
        // We have found a point to swap.

        // Perform the swap.
        size_t tempIndex = indices[nearSetSize + farSetSize - 1];
        ElemType tempDist = distances[nearSetSize + farSetSize - 1];

        indices[nearSetSize + farSetSize - 1] = indices[nearSetSize + i];
        distances[nearSetSize + farSetSize - 1] = distances[nearSetSize + i];

        indices[nearSetSize + i] = tempIndex;
        distances[nearSetSize + i] = tempDist;

        if (j != startChildUsedSet)
        {
          childIndices[childFarSetSize + j] = childIndices[childFarSetSize +
              startChildUsedSet];
        }

        // Update all counters from the swaps we have done.
        ++startChildUsedSet;
        --farSetSize;
        --i;

        break; // Break out of this for loop; back to the first one.
      }
    }
  }

  // Update used set size.
  usedSetSize += childUsedSetSize;

  Log::Assert(originalSum == (nearSetSize + farSetSize + usedSetSize));
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    PruneFarSet(arma::Col<size_t>& indices,
                arma::vec& distances,
                const ElemType bound,
                const size_t nearSetSize,
                const size_t pointSetSize)
{
  // What we are trying to do is remove any points greater than the bound from
  // the far set.  We don't care what happens to those indices and distances...
  // so, we don't need to properly swap points -- just drop new ones in place.
  size_t left = nearSetSize;
  size_t right = pointSetSize - 1;
  while ((distances[left] <= bound) && (left != right))
    ++left;
  while ((distances[right] > bound) && (left != right))
    --right;

  while (left != right)
  {
    // We don't care what happens to the point which should be on the right.
    indices[left] = indices[right];
    distances[left] = distances[right];
    --right; // Since we aren't changing the right.

    // Advance to next location which needs to switch.
    while ((distances[left] <= bound) && (left != right))
      ++left;
    while ((distances[right] > bound) && (left != right))
      --right;
  }

  // The far set size is the left pointer, with the near set size subtracted
  // from it.
  return (left - nearSetSize);
}

/**
 * Take a look at the last child (the most recently created one) and remove any
 * implicit nodes that have been created.
 */
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RemoveNewImplicitNodes()
{
  // If we created an implicit node, take its self-child instead (this could
  // happen multiple times).
  while (children[children.size() - 1]->NumChildren() == 1)
  {
    CoverTree* old = children[children.size() - 1];
    children.erase(children.begin() + children.size() - 1);

    // Now take its child.
    children.push_back(&(old->Child(0)));

    // Set its parent and parameters correctly.
    old->Child(0).Parent() = this;
    old->Child(0).ParentDistance() = old->ParentDistance();
    old->Child(0).DistanceComps() = old->DistanceComps();

    // Remove its child (so it doesn't delete it).
    old->Children().erase(old->Children().begin() + old->Children().size() - 1);

    // Now delete it.
    delete old;
  }
}

/**
 * Default constructor, only for use with cereal.
 */
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree() :
    dataset(NULL),
    point(0),
    scale(INT_MIN),
    base(0.0),
    numDescendants(0),
    parent(NULL),
    parentDistance(0.0),
    furthestDescendantDistance(0.0),
    localDistance(false),
    localDataset(false),
    distance(NULL),
    distanceComps(0)
{
  // Nothing to do.
}

/**
 * Serialize to/from a cereal archive.
 */
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename Archive>
void
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  // If we're loading, and we have children, they need to be deleted.  We may
  // also need to delete the local distance metric and dataset.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];

    if (localDistance && distance)
      delete distance;
    if (localDataset && dataset)
      delete dataset;

    parent = NULL;
  }

  bool hasParent = (parent != NULL);
  ar(CEREAL_NVP(hasParent));
  MatType*& datasetTemp = const_cast<MatType*&>(dataset);
  if (!hasParent)
    ar(CEREAL_POINTER(datasetTemp));

  ar(CEREAL_NVP(point));
  ar(CEREAL_NVP(scale));
  ar(CEREAL_NVP(base));
  ar(CEREAL_NVP(stat));
  ar(CEREAL_NVP(numDescendants));
  ar(CEREAL_NVP(parentDistance));
  ar(CEREAL_NVP(furthestDescendantDistance));
  ar(CEREAL_POINTER(distance));

  if (cereal::is_loading<Archive>() && !hasParent)
  {
    localDistance = true;
    localDataset = true;
  }

  // Lastly, serialize the children.
  ar(CEREAL_VECTOR_POINTER(children));

  if (cereal::is_loading<Archive>())
  {
    // Look through each child individually.
    for (size_t i = 0; i < children.size(); ++i)
    {
      children[i]->localDistance = false;
      children[i]->localDataset = false;
      children[i]->Parent() = this;
    }
  }

  if (!hasParent)
  {
    std::stack<CoverTree*> stack;
    for (size_t i = 0; i < children.size(); ++i)
    {
      stack.push(children[i]);
    }
    while (!stack.empty())
    {
      CoverTree* node = stack.top();
      stack.pop();
      node->dataset = dataset;
      for (size_t i = 0; i < node->children.size(); ++i)
      {
        stack.push(node->children[i]);
      }
    }
  }
}

} // namespace mlpack

#endif
```

## src/mlpack/core/tree/cover_tree/first_point_is_root.hpp

```cpp
/**
 * @file core/tree/cover_tree/first_point_is_root.hpp
 * @author Ryan Curtin
 *
 * A very simple policy for the cover tree; the first point in the dataset is
 * chosen as the root of the cover tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
#define MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class is meant to be used as a choice for the policy class
 * RootPointPolicy of the CoverTree class.  This policy determines which point
 * is used for the root node of the cover tree.  This particular implementation
 * simply chooses the first point in the dataset as the root.  A more complex
 * implementation might choose, for instance, the point with least maximum
 * distance to other points (the closest to the "middle").
 */
class FirstPointIsRoot
{
 public:
  /**
   * Return the point to be used as the root point of the cover tree.  This just
   * returns 0.
   */
  template<typename MatType>
  static size_t ChooseRoot(const MatType& /* dataset */) { return 0; }
};

} // namespace mlpack

#endif // MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
```

## src/mlpack/core/tree/cover_tree/single_tree_traverser.hpp

```cpp
/**
 * @file core/tree/cover_tree/single_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the SingleTreeTraverser for the cover tree.  This implements a
 * single-tree breadth-first recursion with a pruning rule and a base case (two
 * point) rule.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>

#include "cover_tree.hpp"

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
class CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SingleTreeTraverser
{
 public:
  /**
   * Initialize the single tree traverser with the given rule.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is used as
   *      the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, CoverTree& referenceNode);

  //! Get the number of prunes so far.
  size_t NumPrunes() const { return numPrunes; }
  //! Set the number of prunes (good for a reset to 0).
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
```

## src/mlpack/core/tree/cover_tree/single_tree_traverser_impl.hpp

```cpp
/**
 * @file core/tree/cover_tree/single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the single tree traverser for cover trees, which implements
 * a breadth-first traversal.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <queue>

namespace mlpack {

//! This is the structure the cover tree map will use for traversal.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
struct CoverTreeMapEntry
{
  //! The node this entry refers to.
  CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>* node;
  //! The score of the node.
  double score;
  //! The index of the parent node.
  size_t parent;
  //! The base case evaluation.
  double baseCase;

  //! Comparison operator.
  bool operator<(const CoverTreeMapEntry& other) const
  {
    return (score < other.score);
  }
};

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    CoverTree& referenceNode)
{
  // This is a non-recursive implementation (which should be faster than a
  // recursive implementation).
  using MapEntryType = CoverTreeMapEntry<DistanceType, StatisticType, MatType,
      RootPointPolicy>;

  // We will use this map as a priority queue.  Each key represents the scale,
  // and then the vector is all the nodes in that scale which need to be
  // investigated.  Because no point in a scale can add a point in its own
  // scale, we know that the vector for each scale is final when we get to it.
  // In addition, the map is organized in such a way that begin() will return
  // the largest scale.
  std::map<int, std::vector<MapEntryType>, std::greater<int>> mapQueue;

  // Create the score for the children.
  double rootChildScore = rule.Score(queryIndex, referenceNode);

  if (rootChildScore == DBL_MAX)
  {
    numPrunes += referenceNode.NumChildren();
  }
  else
  {
    // Manually add the children of the first node.
    // Often, a ruleset will return without doing any computation on cover trees
    // using TreeTraits::FirstPointIsCentroid; this is an optimization that
    // (theoretically) the compiler should get right.
    double rootBaseCase = rule.BaseCase(queryIndex, referenceNode.Point());

    // Don't add the self-leaf.
    size_t i = 0;
    if (referenceNode.Child(0).NumChildren() == 0)
    {
      ++numPrunes;
      i = 1;
    }

    for (/* i was set above. */; i < referenceNode.NumChildren(); ++i)
    {
      MapEntryType newFrame;
      newFrame.node = &referenceNode.Child(i);
      newFrame.score = rootChildScore;
      newFrame.baseCase = rootBaseCase;
      newFrame.parent = referenceNode.Point();

      // Put it into the map.
      mapQueue[newFrame.node->Scale()].push_back(newFrame);
    }
  }

  // Now begin the iteration through the map, but only if it has anything in it.
  if (mapQueue.empty())
    return;
  int maxScale = mapQueue.cbegin()->first;

  // We will treat the leaves differently (below).
  while (maxScale != INT_MIN)
  {
    // Get a reference to the current scale.
    std::vector<MapEntryType>& scaleVector = mapQueue[maxScale];

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const MapEntryType& frame = scaleVector.at(i);

      CoverTree* node = frame.node;
      const double score = frame.score;
      const size_t parent = frame.parent;
      const size_t point = node->Point();
      double baseCase = frame.baseCase;

      // First we recalculate the score of this node to find if we can prune it.
      if (rule.Rescore(queryIndex, *node, score) == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // Create the score for the children.
      const double childScore = rule.Score(queryIndex, *node);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (childScore == DBL_MAX)
      {
        numPrunes += node->NumChildren();
        continue;
      }

      // If we are a self-child, the base case has already been evaluated.
      // Often, a ruleset will return without doing any computation on cover
      // trees using TreeTraits::FirstPointIsCentroid; this is an optimization
      // that (theoretically) the compiler should get right.
      if (point != parent)
      {
        baseCase = rule.BaseCase(queryIndex, point);
      }

      // Don't add the self-leaf.
      size_t j = 0;
      if (node->Child(0).NumChildren() == 0)
      {
        ++numPrunes;
        j = 1;
      }

      for (/* j is already set. */; j < node->NumChildren(); ++j)
      {
        MapEntryType newFrame;
        newFrame.node = &node->Child(j);
        newFrame.score = childScore;
        newFrame.baseCase = baseCase;
        newFrame.parent = point;

        mapQueue[newFrame.node->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    mapQueue.erase(maxScale);
    maxScale = mapQueue.begin()->first;
  }

  // Now deal with the leaves.
  for (size_t i = 0; i < mapQueue[INT_MIN].size(); ++i)
  {
    const MapEntryType& frame = mapQueue[INT_MIN].at(i);

    CoverTree* node = frame.node;
    const double score = frame.score;
    const size_t point = node->Point();

    // First, recalculate the score of this node to find if we can prune it.
    double rescore = rule.Rescore(queryIndex, *node, score);

    if (rescore == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }

    // For this to be a valid dual-tree algorithm, we *must* evaluate the
    // combination, even if pruning it will make no difference.  It's the
    // definition.
    const double actualScore = rule.Score(queryIndex, *node);

    if (actualScore == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }
    else
    {
      // Evaluate the base case, since the combination was not pruned.
      // Often, a ruleset will return without doing any computation on cover
      // trees using TreeTraits::FirstPointIsCentroid; this is an optimization
      // that (theoretically) the compiler should get right.
      rule.BaseCase(queryIndex, point);
    }
  }
}

} // namespace mlpack

#endif
```

## src/mlpack/core/tree/cover_tree/dual_tree_traverser.hpp

```cpp
/**
 * @file core/tree/cover_tree/dual_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>
#include <queue>

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
class CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    DualTreeTraverser
{
 public:
  /**
   * Initialize the dual tree traverser with the given rule type.
   */
  DualTreeTraverser(RuleType& rule);

  /**
   * Traverse the two specified trees.
   *
   * @param queryNode Root of query tree.
   * @param referenceNode Root of reference tree.
   */
  void Traverse(CoverTree& queryNode, CoverTree& referenceNode);

  //! Get the number of pruned nodes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of pruned nodes.
  size_t& NumPrunes() { return numPrunes; }

  ///// These are all fake because this is a patch for kd-trees only and I still
  ///// want it to compile!
  size_t NumVisited() const { return 0; }
  size_t NumScores() const { return 0; }
  size_t NumBaseCases() const { return 0; }

 private:
  //! The instantiated rule set for pruning branches.
  RuleType& rule;

  //! The number of pruned nodes.
  size_t numPrunes;

  //! Struct used for traversal.
  struct DualCoverTreeMapEntry
  {
    //! The node this entry refers to.
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>*
        referenceNode;
    //! The score of the node.
    double score;
    //! The base case.
    double baseCase;
    //! The traversal info associated with the call to Score() for this entry.
    typename RuleType::TraversalInfoType traversalInfo;

    //! Comparison operator, for sorting within the map.
    bool operator<(const DualCoverTreeMapEntry& other) const
    {
      if (score == other.score)
        return (baseCase < other.baseCase);
      else
        return (score < other.score);
    }
  };

  /**
   * Helper function for traversal of the two trees.
   */
  void Traverse(
      CoverTree& queryNode,
      std::map<int, std::vector<DualCoverTreeMapEntry>,
          std::greater<int>>& referenceMap);

  //! Prepare map for recursion.
  void PruneMap(
      CoverTree& queryNode,
      std::map<int, std::vector<DualCoverTreeMapEntry>,
          std::greater<int>>& referenceMap,
      std::map<int, std::vector<DualCoverTreeMapEntry>,
          std::greater<int>>& childMap);

  void ReferenceRecursion(
      CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry>,
        std::greater<int>>& referenceMap);
};

} // namespace mlpack

// Include implementation.
#include "dual_tree_traverser_impl.hpp"

#endif
```

## src/mlpack/core/tree/cover_tree/dual_tree_traverser_impl.hpp

```cpp
/**
 * @file core/tree/cover_tree/dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>
#include <queue>

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::Traverse(CoverTree& queryNode,
                                      CoverTree& referenceNode)
{
  // Start by creating a map and adding the reference root node to it.
  std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>> refMap;

  DualCoverTreeMapEntry rootRefEntry;

  rootRefEntry.referenceNode = &referenceNode;

  // Perform the evaluation between the roots of either tree.
  rootRefEntry.score = rule.Score(queryNode, referenceNode);
  rootRefEntry.baseCase = rule.BaseCase(queryNode.Point(),
      referenceNode.Point());
  rootRefEntry.traversalInfo = rule.TraversalInfo();

  refMap[referenceNode.Scale()].push_back(rootRefEntry);

  Traverse(queryNode, refMap);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::Traverse(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>>&
        referenceMap)
{
  if (referenceMap.size() == 0)
    return; // Nothing to do!

  // First recurse down the reference nodes as necessary.
  ReferenceRecursion(queryNode, referenceMap);

  // Did the map get emptied?
  if (referenceMap.size() == 0)
    return; // Nothing to do!

  // Now, reduce the scale of the query node by recursing.  But we can't recurse
  // if the query node is a leaf node.
  if ((queryNode.Scale() != INT_MIN) &&
      (queryNode.Scale() >= (*referenceMap.begin()).first))
  {
    // Recurse into the non-self-children first.  The recursion order cannot
    // affect the runtime of the algorithm, because each query child recursion's
    // results are separate and independent.  I don't think this is true in
    // every case, and we may have to modify this section to consider scores in
    // the future.
    for (size_t i = 1; i < queryNode.NumChildren(); ++i)
    {
      // We need a copy of the map for this child.
      std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>>
          childMap;

      PruneMap(queryNode.Child(i), referenceMap, childMap);
      Traverse(queryNode.Child(i), childMap);
    }
    std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>>
        selfChildMap;

    PruneMap(queryNode.Child(0), referenceMap, selfChildMap);
    Traverse(queryNode.Child(0), selfChildMap);
  }

  if (queryNode.Scale() != INT_MIN)
    return; // No need to evaluate base cases at this level.  It's all done.

  // If we have made it this far, all we have is a bunch of base case
  // evaluations to do.
  Log::Assert((*referenceMap.begin()).first == INT_MIN);
  Log::Assert(queryNode.Scale() == INT_MIN);
  std::vector<DualCoverTreeMapEntry>& pointVector = referenceMap[INT_MIN];

  for (size_t i = 0; i < pointVector.size(); ++i)
  {
    // Get a reference to the frame.
    const DualCoverTreeMapEntry& frame = pointVector[i];

    CoverTree* refNode = frame.referenceNode;

    // If the point is the same as both parents, then we have already done this
    // base case.
    if ((refNode->Point() == refNode->Parent()->Point()) &&
        (queryNode.Point() == queryNode.Parent()->Point()))
    {
      ++numPrunes;
      continue;
    }

    // Score the node, to see if we can prune it, after restoring the traversal
    // info.
    rule.TraversalInfo() = frame.traversalInfo;
    double score = rule.Score(queryNode, *refNode);

    if (score == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }

    // If not, compute the base case.
    rule.BaseCase(queryNode.Point(), pointVector[i].referenceNode->Point());
  }
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::PruneMap(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>>&
        referenceMap,
    std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>>&
        childMap)
{
  if (referenceMap.empty())
    return; // Nothing to do.

  // Copy the zero set first.
  if (referenceMap.count(INT_MIN) == 1)
  {
    // Get a reference to the vector representing the entries at this scale.
    std::vector<DualCoverTreeMapEntry>& scaleVector = referenceMap[INT_MIN];

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    childMap[INT_MIN].reserve(scaleVector.size());
    std::vector<DualCoverTreeMapEntry>& newScaleVector = childMap[INT_MIN];

    // Loop over each entry in the vector.
    for (size_t j = 0; j < scaleVector.size(); ++j)
    {
      const DualCoverTreeMapEntry& frame = scaleVector[j];

      // First evaluate if we can prune without performing the base case.
      CoverTree* refNode = frame.referenceNode;

      // Perform the actual scoring, after restoring the traversal info.
      rule.TraversalInfo() = frame.traversalInfo;
      double score = rule.Score(queryNode, *refNode);

      if (score == DBL_MAX)
      {
        // Pruned.  Move on.
        ++numPrunes;
        continue;
      }

      // If it isn't pruned, we must evaluate the base case.
      const double baseCase = rule.BaseCase(queryNode.Point(),
          refNode->Point());

      // Add to child map.
      newScaleVector.push_back(frame);
      newScaleVector.back().score = score;
      newScaleVector.back().baseCase = baseCase;
      newScaleVector.back().traversalInfo = rule.TraversalInfo();
    }

    // If we didn't add anything, then strike this vector from the map.
    if (newScaleVector.size() == 0)
      childMap.erase(INT_MIN);
  }

  typename std::map<int, std::vector<DualCoverTreeMapEntry>,
      std::greater<int>>::iterator it = referenceMap.begin();

  while ((it != referenceMap.end()))
  {
    const int thisScale = (*it).first;
    if (thisScale == INT_MIN) // We already did it.
      break;

    // Get a reference to the vector representing the entries at this scale.
    std::vector<DualCoverTreeMapEntry>& scaleVector = (*it).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    childMap[thisScale].reserve(scaleVector.size());
    std::vector<DualCoverTreeMapEntry>& newScaleVector = childMap[thisScale];

    // Loop over each entry in the vector.
    for (size_t j = 0; j < scaleVector.size(); ++j)
    {
      const DualCoverTreeMapEntry& frame = scaleVector[j];

      // First evaluate if we can prune without performing the base case.
      CoverTree* refNode = frame.referenceNode;

      // Perform the actual scoring, after restoring the traversal info.
      rule.TraversalInfo() = frame.traversalInfo;
      double score = rule.Score(queryNode, *refNode);

      if (score == DBL_MAX)
      {
        // Pruned.  Move on.
        ++numPrunes;
        continue;
      }

      // If it isn't pruned, we must evaluate the base case.
      const double baseCase = rule.BaseCase(queryNode.Point(),
          refNode->Point());

      // Add to child map.
      newScaleVector.push_back(frame);
      newScaleVector.back().score = score;
      newScaleVector.back().baseCase = baseCase;
      newScaleVector.back().traversalInfo = rule.TraversalInfo();
    }

    // If we didn't add anything, then strike this vector from the map.
    if (newScaleVector.size() == 0)
      childMap.erase((*it).first);

    ++it; // Advance to next scale.
  }
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::ReferenceRecursion(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry>, std::greater<int>>&
        referenceMap)
{
  // First, reduce the maximum scale in the reference map down to the scale of
  // the query node.
  while (!referenceMap.empty())
  {
    const int maxScale = ((*referenceMap.begin()).first);
    // Hacky bullshit to imitate jl cover tree.
    if (queryNode.Parent() == NULL && maxScale < queryNode.Scale())
      break;
    if (queryNode.Parent() != NULL && maxScale <= queryNode.Scale())
      break;
    // If the query node's scale is INT_MIN and the reference map's maximum
    // scale is INT_MIN, don't try to recurse...
    if (queryNode.Scale() == INT_MIN && maxScale == INT_MIN)
      break;

    // Get a reference to the current largest scale.
    std::vector<DualCoverTreeMapEntry>& scaleVector = referenceMap[maxScale];

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const DualCoverTreeMapEntry& frame = scaleVector.at(i);
      CoverTree* refNode = frame.referenceNode;

      // Create the score for the children.
      double score = rule.Rescore(queryNode, *refNode, frame.score);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (score == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // If it is not pruned, we must evaluate the base case.

      // Add the children.
      for (size_t j = 0; j < refNode->NumChildren(); ++j)
      {
        rule.TraversalInfo() = frame.traversalInfo;
        double childScore = rule.Score(queryNode, refNode->Child(j));
        if (childScore == DBL_MAX)
        {
          ++numPrunes;
          continue;
        }

        // It wasn't pruned; evaluate the base case.
        const double baseCase = rule.BaseCase(queryNode.Point(),
            refNode->Child(j).Point());

        DualCoverTreeMapEntry newFrame;
        newFrame.referenceNode = &refNode->Child(j);
        newFrame.score = childScore; // Use the score of the parent.
        newFrame.baseCase = baseCase;
        newFrame.traversalInfo = rule.TraversalInfo();
        referenceMap[newFrame.referenceNode->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    referenceMap.erase(maxScale);
  }
}

} // namespace mlpack

#endif
```

## src/mlpack/core/tree/cover_tree/traits.hpp

```cpp
/**
 * @file core/tree/cover_tree/traits.hpp
 * @author Ryan Curtin
 *
 * This file contains the specialization of the TreeTraits class for the
 * CoverTree type of tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_COVER_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {

/**
 * The specialization of the TreeTraits class for the CoverTree tree type.  It
 * defines characteristics of the cover tree, and is used to help write
 * tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
class TreeTraits<CoverTree<
    DistanceType, StatisticType, MatType, RootPointPolicy>>
{
 public:
  /**
   * The cover tree (or, this implementation of it) does not require that
   * children represent non-overlapping subsets of the parent node.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * Cover trees do have self-children, so points can be included in more than
   * one node.
   */
  static const bool HasDuplicatedPoints = true;

  /**
   * Each cover tree node contains only one point, and that point is its
   * centroid.
   */
  static const bool FirstPointIsCentroid = true;

  /**
   * Cover trees do have self-children.
   */
  static const bool HasSelfChildren = true;

  /**
   * Points are not rearranged when the tree is built.
   */
  static const bool RearrangesDataset = false;

  /**
   * The cover tree is not necessarily a binary tree.
   */
  static const bool BinaryTree = false;

  /**
   * NumDescendants() represents the number of unique descendant points.
   */
  static const bool UniqueNumDescendants = true;
};

} // namespace mlpack

#endif
```

## src/mlpack/core/tree/cover_tree/typedef.hpp

```cpp
/**
 * @file core/tree/cover_tree/typedef.hpp
 * @author Ryan Curtin
 *
 * Typedef of cover tree to match TreeType API.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP

#include "cover_tree.hpp"

namespace mlpack {

/**
 * The standard cover tree, as detailed in the original cover tree paper:
 *
 * @code
 * @inproceedings{
 *   author={Beygelzimer, A. and Kakade, S. and Langford, J.},
 *   title={Cover trees for nearest neighbor},
 *   booktitle={Proceedings of the 23rd International Conference on Machine
 *       Learning (ICML 2006)},
 *   pages={97--104},
 *   year={2006}
 * }
 * @endcode
 *
 * This template typedef satisfies the requirements of the TreeType API.
 *
 * @see @ref trees, CoverTree
 */
template<typename DistanceType, typename StatisticType, typename MatType>
using StandardCoverTree = CoverTree<DistanceType,
                                    StatisticType,
                                    MatType,
                                    FirstPointIsRoot>;

} // namespace mlpack

#endif
```

