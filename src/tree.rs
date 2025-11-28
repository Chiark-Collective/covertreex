use ndarray::{Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use std::fmt::Debug;

#[allow(dead_code)]
pub struct CoverTreeData<T> {
    // Flattened points (Row Major): [x0_0, x0_1, ..., x1_0, ...]
    pub points: Vec<T>,
    pub parents: Vec<i64>,
    pub children: Vec<i64>,  // First child
    pub next_node: Vec<i64>, // Next sibling
    pub levels: Vec<i32>,
    pub dimension: usize,
    pub min_level: i32,
    pub max_level: i32,
    pub si_cache: Vec<T>,
}

impl<T> CoverTreeData<T>
where
    T: Float + Debug + Send + Sync,
{
    pub fn new(
        points_array: Array2<T>,
        parents: Vec<i64>,
        children: Vec<i64>,
        next_node: Vec<i64>,
        levels: Vec<i32>,
        min_level: i32,
        max_level: i32,
    ) -> Self {
        let rows = points_array.nrows();
        let dimension = points_array.shape()[1];
        // Convert Array2 to Vec
        let points = points_array
            .as_standard_layout()
            .into_owned()
            .into_raw_vec();

        let si_cache = vec![T::zero(); rows];

        Self {
            points,
            parents,
            children,
            next_node,
            levels,
            dimension,
            min_level,
            max_level,
            si_cache,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len() / self.dimension
    }

    #[allow(dead_code)]
    pub fn get_point(&self, idx: usize) -> ArrayView2<'_, T> {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        ArrayView2::from_shape((1, self.dimension), &self.points[start..end]).unwrap()
    }

    pub fn get_point_row(&self, idx: usize) -> &[T] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.points[start..end]
    }

    pub fn add_point(&mut self, point: ArrayView1<T>, level: i32, parent: i64) -> usize {
        let idx = self.len();
        self.points.extend(point.iter());
        self.parents.push(parent);
        self.children.push(-1);
        self.next_node.push(-1);
        self.levels.push(level);
        self.si_cache.push(T::zero());

        // Link to parent (if valid)
        if parent >= 0 {
            self.link_child(parent as usize, idx);
        }

        if level != i32::MIN {
            if level > self.max_level {
                self.max_level = level;
            }
            if level < self.min_level {
                self.min_level = level;
            }
        }

        idx
    }

    pub fn link_child(&mut self, parent_idx: usize, child_idx: usize) {
        // Prepend to child list: parent.child -> new_node -> old_first_child
        let old_first = self.children[parent_idx];
        self.children[parent_idx] = child_idx as i64;
        self.next_node[child_idx] = old_first;
    }

    pub fn set_level(&mut self, idx: usize, level: i32) {
        self.levels[idx] = level;
        if level > self.max_level {
            self.max_level = level;
        }
        if level < self.min_level {
            self.min_level = level;
        }
    }

    pub fn set_parent(&mut self, idx: usize, parent: i64) {
        self.parents[idx] = parent;
        if parent >= 0 {
            self.link_child(parent as usize, idx);
        }
    }

    pub fn set_si_cache(&mut self, cache: Vec<T>) {
        self.si_cache = cache;
    }
}

/// Compute subtree index bounds for predecessor constraint pruning.
///
/// For each tree node, computes the minimum and maximum dataset indices
/// contained in that node's subtree (including the node itself).
/// This enables aggressive subtree pruning during predecessor-constrained queries:
/// if a subtree's minimum dataset index >= query's max_neighbor_idx, the entire
/// subtree can be skipped.
///
/// Uses iterative propagation from children to parents until convergence.
/// Converges in O(tree_height) iterations, typically O(log n) for cover trees.
///
/// # Arguments
/// * `parents` - Parent index for each node (-1 for root)
/// * `node_to_dataset` - Maps tree node index to dataset index
///
/// # Returns
/// * `(min_bounds, max_bounds)` - Min/max dataset indices in each node's subtree
pub fn compute_subtree_index_bounds(
    parents: &[i64],
    node_to_dataset: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    let n = parents.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Initialize each node with its own dataset index
    let mut min_bounds: Vec<i64> = node_to_dataset.to_vec();
    let mut max_bounds: Vec<i64> = node_to_dataset.to_vec();

    // Propagate from children to parents until convergence
    // Each iteration propagates bounds one level up the tree
    // Converges in O(height) iterations
    let mut changed = true;
    let mut iterations = 0;
    let max_iterations = n; // Safety bound (tree height <= n)

    while changed && iterations < max_iterations {
        changed = false;
        iterations += 1;

        for node in 0..n {
            let parent = parents[node];
            if parent >= 0 {
                let p = parent as usize;
                if min_bounds[node] < min_bounds[p] {
                    min_bounds[p] = min_bounds[node];
                    changed = true;
                }
                if max_bounds[node] > max_bounds[p] {
                    max_bounds[p] = max_bounds[node];
                    changed = true;
                }
            }
        }
    }

    (min_bounds, max_bounds)
}
