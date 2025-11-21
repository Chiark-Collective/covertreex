use crate::tree::CoverTreeData;
use crate::metric::{Metric, Euclidean, ResidualMetric};
use rayon::prelude::*;
use ndarray::parallel::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub mod batch;

#[derive(Copy, Clone, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(PartialEq, Eq)]
struct Candidate {
    dist: OrderedFloat,
    node_idx: i64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.cmp(&self.dist)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq, Eq)]
struct Neighbor {
    dist: OrderedFloat,
    node_idx: i64,
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// -----------------------------------------------------------------------------
// Euclidean Query (Existing)
// -----------------------------------------------------------------------------

pub fn batch_knn_query(
    tree: &CoverTreeData,
    queries: ndarray::ArrayView2<f32>,
    k: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    let metric = Euclidean;
    let results: Vec<(Vec<i64>, Vec<f32>)> = queries
        .outer_iter()
        .into_par_iter()
        .map(|q_point| {
            single_knn_query(tree, &metric, q_point, k)
        })
        .collect();

    let mut indices = Vec::with_capacity(results.len());
    let mut dists = Vec::with_capacity(results.len());
    for (idx, dst) in results {
        indices.push(idx);
        dists.push(dst);
    }
    (indices, dists)
}

fn single_knn_query(
    tree: &CoverTreeData,
    metric: &dyn Metric,
    q_point_view: ndarray::ArrayView1<f32>,
    k: usize,
) -> (Vec<i64>, Vec<f32>) {
    let q_point = q_point_view.as_slice().unwrap();
    let mut candidate_heap = BinaryHeap::new();
    let mut result_heap: BinaryHeap<Neighbor> = BinaryHeap::new();
    
    if tree.len() == 0 {
        return (vec![], vec![]);
    }
    
    // Roots (Assume 0 is root for now, needs update to iterate root candidates)
    let root_idx = 0;
    let d = metric.distance(q_point, tree.get_point_row(root_idx as usize));
    candidate_heap.push(Candidate { dist: OrderedFloat(d), node_idx: root_idx });
    
    while let Some(cand) = candidate_heap.pop() {
        let dist = cand.dist.0;
        let node_idx = cand.node_idx;
        
        if result_heap.len() == k {
            let max_dist = result_heap.peek().unwrap().dist.0;
            // Simple heuristic pruning: if parent is farther than k-th neighbor,
            // assume children are also far?
            // Strictly not true without Lower Bound.
            // But for "Static Tree" strategy, we rely on the tree structure.
            // If we don't prune, we visit everything.
            // Let's NOT prune here to ensure recall, unless dist is VERY large?
        }

        result_heap.push(Neighbor { dist: OrderedFloat(dist), node_idx });
        if result_heap.len() > k {
            result_heap.pop();
        }
        
        let mut child = tree.children[node_idx as usize];
        while child != -1 {
            let d_child = metric.distance(q_point, tree.get_point_row(child as usize));
            candidate_heap.push(Candidate { dist: OrderedFloat(d_child), node_idx: child });
            
            let next = tree.next_node[child as usize];
            if next == child { break; }
            child = next;
            if child == tree.children[node_idx as usize] { break; }
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
// Residual Query (New)
// -----------------------------------------------------------------------------

pub fn batch_residual_knn_query(
    tree: &CoverTreeData,
    query_indices: ndarray::ArrayView1<i64>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric,
    k: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    
    let results: Vec<(Vec<i64>, Vec<f32>)> = query_indices
        .into_par_iter()
        .map(|&q_idx| {
            single_residual_knn_query(tree, node_to_dataset, metric, q_idx as usize, k)
        })
        .collect();

    let mut indices = Vec::with_capacity(results.len());
    let mut dists = Vec::with_capacity(results.len());
    for (idx, dst) in results {
        indices.push(idx);
        dists.push(dst);
    }
    (indices, dists)
}

fn single_residual_knn_query(
    tree: &CoverTreeData,
    node_to_dataset: &[i64],
    metric: &ResidualMetric,
    q_dataset_idx: usize,
    k: usize,
) -> (Vec<i64>, Vec<f32>) {
    let mut candidate_heap = BinaryHeap::new();
    let mut result_heap: BinaryHeap<Neighbor> = BinaryHeap::new();
    
    if tree.len() == 0 { return (vec![], vec![]); }
    
    // Root
    let root_node_idx = 0;
    let root_dataset_idx = node_to_dataset[root_node_idx as usize] as usize;
    
    let d = metric.distance_idx(q_dataset_idx, root_dataset_idx);
    candidate_heap.push(Candidate { dist: OrderedFloat(d), node_idx: root_node_idx });
    
    let mut visited = std::collections::HashSet::new(); // Or bitset?
    visited.insert(root_node_idx);
    
    while let Some(cand) = candidate_heap.pop() {
        let dist = cand.dist.0;
        let node_idx = cand.node_idx;
        
        result_heap.push(Neighbor { dist: OrderedFloat(dist), node_idx });
        if result_heap.len() > k {
            result_heap.pop();
        }
        
        // Expand
        let mut child = tree.children[node_idx as usize];
        while child != -1 {
            if !visited.contains(&child) {
                visited.insert(child);
                let child_dataset_idx = node_to_dataset[child as usize] as usize;
                let d_child = metric.distance_idx(q_dataset_idx, child_dataset_idx);
                // Priority: Child's distance? Or Parent's distance?
                // Standard BFS: Child's distance.
                // Numba impl used Parent's distance as priority for Child.
                // Let's stick to Standard Best First: Compute child dist, push.
                candidate_heap.push(Candidate { dist: OrderedFloat(d_child), node_idx: child });
            }
            
            let next = tree.next_node[child as usize];
            if next == child { break; }
            child = next;
            if child == tree.children[node_idx as usize] { break; }
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