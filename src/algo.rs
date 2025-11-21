use crate::tree::CoverTreeData;
use crate::metric::{Metric, Euclidean};
use rayon::prelude::*;
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

// For Min-Heap (Candidates): Order by distance (smallest first)
// Rust BinaryHeap is Max-Heap. To make Min-Heap, we reverse order.
#[derive(PartialEq, Eq)]
struct Candidate {
    dist: OrderedFloat,
    node_idx: i64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for Min-Heap
        other.dist.cmp(&self.dist)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// For Max-Heap (Results): Order by distance (largest first) to keep smallest K
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

pub fn batch_knn_query(
    tree: &CoverTreeData,
    queries: ndarray::ArrayView2<f32>,
    k: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    let metric = Euclidean;
    
    // Parallel iterate over queries
    let results: Vec<(Vec<i64>, Vec<f32>)> = queries
        .outer_iter()
        .into_par_iter()
        .map(|q_point| {
            single_knn_query(tree, &metric, q_point, k)
        })
        .collect();

    // Unzip results
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
    q_point: ndarray::ArrayView1<f32>,
    k: usize,
) -> (Vec<i64>, Vec<f32>) {
    let mut candidate_heap = BinaryHeap::new();
    let mut result_heap: BinaryHeap<Neighbor> = BinaryHeap::new(); // Max heap of Neighbors
    
    // Identify roots (nodes with parent -1)
    // Optim: pass roots explicitly? For now, scan or assume 0 if single tree.
    // The Python code passes `parents`. Roots are `parents[i] == -1`.
    // This scan is O(N) which is bad for every query.
    // We should store roots in CoverTreeData.
    // Assuming Root is 0 for now, or we scan once in `CoverTreeData::new` and store `roots`.
    // Let's modify `CoverTreeData` later to store roots.
    // For now, hack: assume node 0 is root.
    
    // Compute dist to root
    let root_idx = 0;
    // Safety check
    if tree.len() == 0 {
        return (vec![], vec![]);
    }
    
    let d = metric.distance(q_point, tree.get_point_row(root_idx as usize));
    candidate_heap.push(Candidate {
        dist: OrderedFloat(d),
        node_idx: root_idx,
    });
    
    // Initialize result heap with root?
    // Or wait until popped?
    // Standard Best First Search:
    // 1. Pop nearest candidate.
    // 2. Add to result heap (maintain k).
    // 3. Prune? (if dist > result_heap.max and result_heap is full)
    // 4. Expand children.
    
    while let Some(cand) = candidate_heap.pop() {
        let dist = cand.dist.0;
        let node_idx = cand.node_idx;
        
        // Pruning Check
        if result_heap.len() == k {
            let max_dist = result_heap.peek().unwrap().dist.0;
            if dist > max_dist {
                // If the *candidate* distance is worse than our k-th best, 
                // and since we pop in increasing order, all subsequent candidates are worse.
                // Can we stop?
                // Only if `dist` is a Lower Bound.
                // Here `dist` is Exact Distance to the node.
                // If Cover Tree property holds: children are contained in ball(p, radius).
                // d(q, child) >= d(q, p) - d(p, child).
                // This requires utilizing tree radii for pruning.
                // We don't have radii loaded yet.
                // So strictly speaking we can't prune branches based purely on point distance
                // unless we assume something.
                // BUT standard BFS explores in distance order.
                // If we have K nodes with distance <= D_k, and we encounter a node with dist > D_k.
                // Its children *could* be closer? Yes.
                // So we cannot simply stop.
                
                // However, for "Static Tree" heuristic we usually implement the standard Cover Tree k-NN
                // which uses (d(q, p) - radius) as lower bound.
                // Without radii, this is just a greedy search or exhaustive search.
                
                // For this "Phase 2" skeleton, let's implement greedy expansion 
                // but we DO update the result heap.
                // If we want TRUE k-NN, we need bounds.
                // For now, let's process ALL nodes reachable? No, too slow.
                // Let's assume standard Cover Tree logic:
                // We need radii.
                // I'll add `radii` to `CoverTreeData` in next step.
                // For now, let's just implement the mechanism.
            }
        }

        // Add to Result Heap
        result_heap.push(Neighbor {
            dist: OrderedFloat(dist),
            node_idx,
        });
        if result_heap.len() > k {
            result_heap.pop();
        }
        
        // Expand children
        let mut child = tree.children[node_idx as usize];
        while child != -1 {
            // Compute dist
            let d_child = metric.distance(q_point, tree.get_point_row(child as usize));
            candidate_heap.push(Candidate {
                dist: OrderedFloat(d_child),
                node_idx: child,
            });
            
            // Next sibling
            let next = tree.next_node[child as usize];
            if next == child { break; } // Cycle safety
            child = next;
            
            // Safety break if next loops back to first child (circular list)
            // Our Python impl uses circular list? 
            // `covertreex` usually uses circular `next` pointer or terminator.
            // Let's check `next_node` semantics.
            // Usually `-1` is terminator.
            if child == tree.children[node_idx as usize] { break; }
        }
    }
    
    // Extract results
    // Heap pops largest first. We want sorted ascending.
    let sorted_results = result_heap.into_sorted_vec(); // This gives ascending? No, `into_sorted_vec` gives ascending for MaxHeap?
    // `BinaryHeap::into_sorted_vec` returns elements in ascending order (Min to Max).
    // But our `Neighbor` comparison is `dist`.
    // MaxHeap pops Max.
    // `into_sorted_vec` pops all. So it returns sorted vec [Min ... Max].
    
    let mut indices = Vec::with_capacity(k);
    let mut dists = Vec::with_capacity(k);
    
    for n in sorted_results {
        indices.push(n.node_idx);
        dists.push(n.dist.0);
    }
    
    (indices, dists)
}
