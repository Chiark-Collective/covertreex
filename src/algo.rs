use crate::metric::{Euclidean, Metric, ResidualMetric};
use crate::tree::CoverTreeData;
use ndarray::parallel::prelude::*;
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

static DIST_EVALS: AtomicUsize = AtomicUsize::new(0);
static HEAP_PUSHES: AtomicUsize = AtomicUsize::new(0);
static EMIT_STATS: AtomicBool = AtomicBool::new(false);

pub mod batch;

#[derive(Copy, Clone, PartialEq)]
struct OrderedFloat<T>(T);

impl<T: Float> Eq for OrderedFloat<T> {}

impl<T: Float> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: Float> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

struct Candidate<T: Float> {
    dist: OrderedFloat<T>,
    node_idx: i64,
}

impl<T: Float> PartialEq for Candidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node_idx == other.node_idx
    }
}

impl<T: Float> Eq for Candidate<T> {}

impl<T: Float> Ord for Candidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.cmp(&self.dist)
    }
}

impl<T: Float> PartialOrd for Candidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Neighbor<T: Float> {
    dist: OrderedFloat<T>,
    node_idx: i64,
}

impl<T: Float> PartialEq for Neighbor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node_idx == other.node_idx
    }
}

impl<T: Float> Eq for Neighbor<T> {}

impl<T: Float> Ord for Neighbor<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl<T: Float> PartialOrd for Neighbor<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// -----------------------------------------------------------------------------
// Euclidean Query
// -----------------------------------------------------------------------------

pub fn batch_knn_query<T>(
    tree: &CoverTreeData<T>,
    queries: ndarray::ArrayView2<T>,
    k: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    let metric = Euclidean;
    let results: Vec<(Vec<i64>, Vec<T>)> = queries
        .outer_iter()
        .into_par_iter()
        .map(|q_point| single_knn_query(tree, &metric, q_point.as_slice().unwrap(), k))
        .collect();

    let mut indices = Vec::with_capacity(results.len());
    let mut dists = Vec::with_capacity(results.len());
    for (idx, dst) in results {
        indices.push(idx);
        dists.push(dst);
    }
    (indices, dists)
}

fn single_knn_query<T>(
    tree: &CoverTreeData<T>,
    metric: &dyn Metric<T>,
    q_point: &[T],
    k: usize,
) -> (Vec<i64>, Vec<T>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    let mut candidate_heap = BinaryHeap::new();
    let mut result_heap: BinaryHeap<Neighbor<T>> = BinaryHeap::new();

    if tree.len() == 0 {
        return (vec![], vec![]);
    }

    let root_idx = 0;
    let d = metric.distance(q_point, tree.get_point_row(root_idx as usize));
    candidate_heap.push(Candidate {
        dist: OrderedFloat(d),
        node_idx: root_idx,
    });

    let two = T::from(2.0).unwrap();
    let mut kth_dist = T::max_value();

    while let Some(cand) = candidate_heap.pop() {
        let dist = cand.dist.0;
        let node_idx = cand.node_idx;

        // Pruning check: if the lower bound distance to any descendant is greater than
        // the current k-th nearest neighbor distance, we can skip this branch.
        if result_heap.len() == k {
            let level = tree.levels[node_idx as usize];
            let radius = two.powi(level + 1);
            let lower_bound = dist - radius;
            if lower_bound > kth_dist {
                continue;
            }
        }

        result_heap.push(Neighbor {
            dist: OrderedFloat(dist),
            node_idx,
        });
        if result_heap.len() > k {
            result_heap.pop();
        }
        if result_heap.len() == k {
            if let Some(peek) = result_heap.peek() {
                kth_dist = peek.dist.0;
            }
        }

        let mut child = tree.children[node_idx as usize];
        while child != -1 {
            let d_child = metric.distance(q_point, tree.get_point_row(child as usize));
            // Optimization: Early check before pushing to heap?
            // For now, just push. The loop popping will handle the pruning.
            // However, we can prune *insertion* if d_child - child_radius > kth_dist,
            // but child_radius requires looking up child level.
            
            // Tighter check for insertion: if d_child > kth_dist + child_radius?
            // No, simpler to just let the main loop prune.
            
            // Optimization: if d_child > kth_dist + max_possible_radius?
            // Keep it simple for now.
            candidate_heap.push(Candidate {
                dist: OrderedFloat(d_child),
                node_idx: child,
            });

            let next = tree.next_node[child as usize];
            if next == child {
                break;
            }
            child = next;
            if child == tree.children[node_idx as usize] {
                break;
            }
        }
    }

    let sorted_results = result_heap.into_sorted_vec();
    let mut indices = Vec::with_capacity(k);
    let mut dists = Vec::with_capacity(k);
    for n in sorted_results {
        indices.push(n.node_idx);
        dists.push(n.dist.0);
    }
    (indices, dists)
}

// -----------------------------------------------------------------------------
// Residual Query
// -----------------------------------------------------------------------------

pub fn batch_residual_knn_query<'a, T>(
    tree: &'a CoverTreeData<T>,
    query_indices: ndarray::ArrayView1<i64>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
    k: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    let results: Vec<(Vec<i64>, Vec<T>)> = query_indices
        .into_par_iter()
        .map(|&q_idx| single_residual_knn_query(tree, node_to_dataset, metric, q_idx as usize, k))
        .collect();

    let mut indices = Vec::with_capacity(results.len());
    let mut dists = Vec::with_capacity(results.len());
    for (idx, dst) in results {
        indices.push(idx);
        dists.push(dst);
    }
    (indices, dists)
}

fn single_residual_knn_query<'a, T>(
    tree: &'a CoverTreeData<T>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
    q_dataset_idx: usize,
    k: usize,
) -> (Vec<i64>, Vec<T>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    let mut candidate_heap = BinaryHeap::new();
    let mut result_heap: BinaryHeap<Neighbor<T>> = BinaryHeap::new();

    if tree.len() == 0 {
        return (vec![], vec![]);
    }

    // Numba-style Frontier Batching
    // Initial candidates: roots.
    // We push them with priority 0.0 (or max_value? min_value?).
    // Min-heap needed?
    // BinaryHeap is Max-Heap.
    // To get Min-Heap behavior for `dist`, we use `OrderedFloat` which orders by value.
    // We want to pop SMALLEST distance first (Best-First).
    // `Candidate` implements `Ord` as `other.dist.cmp(&self.dist)`.
    // So `pop()` returns SMALLEST distance. Correct.

    // Push root (index 0) with its true distance
    let root_payload = tree.get_point_row(0);
    let root_dataset_idx = root_payload[0].to_usize().unwrap();
    let root_dist = metric.distance_idx(q_dataset_idx, root_dataset_idx);
    candidate_heap.push(Candidate {
        dist: OrderedFloat(root_dist),
        node_idx: 0,
    });
    HEAP_PUSHES.fetch_add(1, AtomicOrdering::Relaxed);

    let mut kth_dist = T::max_value();
    const BATCH_SIZE: usize = 64;
    let mut batch_nodes = Vec::with_capacity(BATCH_SIZE);
    let mut dataset_indices = Vec::with_capacity(BATCH_SIZE);
    let mut distance_buffer: Vec<T> = Vec::with_capacity(BATCH_SIZE);
    let mut child_nodes = Vec::with_capacity(BATCH_SIZE);
    let mut child_dataset_indices = Vec::with_capacity(BATCH_SIZE);
    let mut child_distance_buffer: Vec<T> = Vec::with_capacity(BATCH_SIZE);
    let emit_stats = EMIT_STATS.load(AtomicOrdering::Relaxed);

    while !candidate_heap.is_empty() {
        batch_nodes.clear();
        dataset_indices.clear();

        // 1. Collect Batch
        while batch_nodes.len() < BATCH_SIZE {
            if let Some(cand) = candidate_heap.pop() {
                // Pruning based on parent distance (loose bound)
                // Numba doesn't seem to prune here, but we can?
                // Let's just process.
                batch_nodes.push(cand.node_idx);
                dataset_indices.push(node_to_dataset[cand.node_idx as usize] as usize);
            } else {
                break;
            }
        }

        if batch_nodes.is_empty() {
            break;
        }

        // 2. Batch Compute Distances
        metric.distances_sq_batch_idx_into(q_dataset_idx, &dataset_indices, &mut distance_buffer);
        if emit_stats {
            DIST_EVALS.fetch_add(distance_buffer.len(), AtomicOrdering::Relaxed);
        }

        // 3. Process Results
        for (i, &d_sq) in distance_buffer.iter().enumerate() {
            let dist = d_sq.sqrt();
            let node_idx = batch_nodes[i];

            // Update k-NN
            if result_heap.len() < k || dist < kth_dist {
                result_heap.push(Neighbor {
                    dist: OrderedFloat(dist),
                    node_idx,
                });
                if result_heap.len() > k {
                    result_heap.pop();
                }
                if result_heap.len() == k {
                    if let Some(peek) = result_heap.peek() {
                        kth_dist = peek.dist.0;
                    }
                }
            }

            // Expand Children
            // Pruning: d(q, u) - 2^{level+1} > kth_dist
            let two = T::from(2.0).unwrap();
            let level = tree.levels[node_idx as usize];
            let radius = two.powi(level + 1);
            let lower_bound = dist - radius;

            if result_heap.len() == k && lower_bound > kth_dist {
                continue; 
            }

            // Collect children for batched distance evaluation
            child_nodes.clear();
            child_dataset_indices.clear();
            let mut child = tree.children[node_idx as usize];
            while child != -1 {
                child_nodes.push(child);
                child_dataset_indices.push(node_to_dataset[child as usize] as usize);
                child = tree.next_node[child as usize];
                if child == tree.children[node_idx as usize] {
                    break;
                }
            }

            if !child_nodes.is_empty() {
                metric.distances_sq_batch_idx_into(q_dataset_idx, &child_dataset_indices, &mut child_distance_buffer);
                for (j, &child_node) in child_nodes.iter().enumerate() {
                    let child_dist = child_distance_buffer[j].sqrt();
                    candidate_heap.push(Candidate {
                        dist: OrderedFloat(child_dist),
                        node_idx: child_node,
                    });
                    if emit_stats {
                        HEAP_PUSHES.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
    }

    let sorted_results = result_heap.into_sorted_vec();
    let mut indices = Vec::with_capacity(k);
    let mut dists = Vec::with_capacity(k);
    for n in sorted_results {
        indices.push(n.node_idx);
        dists.push(n.dist.0);
    }

    // Emit simple telemetry for debugging the Rust residual path.
    if emit_stats {
        eprintln!(
            "[rust-hybrid] distances={} heap_pushes={}",
            DIST_EVALS.load(AtomicOrdering::Relaxed),
            HEAP_PUSHES.load(AtomicOrdering::Relaxed)
        );
        DIST_EVALS.store(0, AtomicOrdering::Relaxed);
        HEAP_PUSHES.store(0, AtomicOrdering::Relaxed);
    }
    (indices, dists)
}
