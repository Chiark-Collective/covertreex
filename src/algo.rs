use crate::metric::{Euclidean, Metric, ResidualMetric};
use crate::tree::CoverTreeData;
use ndarray::parallel::prelude::*;
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

static DIST_EVALS: AtomicUsize = AtomicUsize::new(0);
static HEAP_PUSHES: AtomicUsize = AtomicUsize::new(0);
static EMIT_STATS: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub(crate) fn set_debug_stats_enabled(enabled: bool) {
    EMIT_STATS.store(enabled, AtomicOrdering::Relaxed);
    if enabled {
        DIST_EVALS.store(0, AtomicOrdering::Relaxed);
        HEAP_PUSHES.store(0, AtomicOrdering::Relaxed);
    }
}

#[allow(dead_code)]
#[inline(always)]
fn debug_stats_enabled() -> bool {
    EMIT_STATS.load(AtomicOrdering::Relaxed)
}

#[inline(always)]
pub(crate) fn take_debug_stats() -> (usize, usize) {
    let dist = DIST_EVALS.swap(0, AtomicOrdering::Relaxed);
    let pushes = HEAP_PUSHES.swap(0, AtomicOrdering::Relaxed);
    (dist, pushes)
}

#[inline(always)]
pub(crate) fn debug_stats_snapshot() -> (usize, usize) {
    (
        DIST_EVALS.load(AtomicOrdering::Relaxed),
        HEAP_PUSHES.load(AtomicOrdering::Relaxed),
    )
}

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
    scope_caps: Option<&HashMap<i32, T>>,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    let results: Vec<(Vec<i64>, Vec<T>)> = query_indices
        .into_par_iter()
        .map(|&q_idx| {
            single_residual_knn_query(
                tree,
                node_to_dataset,
                metric,
                q_idx as usize,
                k,
                scope_caps,
            )
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

/// Brute-force block-scanned residual k-NN over all dataset indices.
/// This bypasses the cover tree and is intended as a stopgap PCCT-style query
/// until a full conflict-graph/streamer port lands.
pub fn batch_residual_knn_query_block<'a, T>(
    query_indices: ndarray::ArrayView1<i64>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
    k: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    let n = node_to_dataset.len();
    let mut indices_out: Vec<Vec<i64>> = Vec::with_capacity(query_indices.len());
    let mut dists_out: Vec<Vec<T>> = Vec::with_capacity(query_indices.len());

    const BLOCK: usize = 256;
    let emit_stats = EMIT_STATS.load(AtomicOrdering::Relaxed);
    let prune_sentinel = T::max_value();
    for &q_idx_i64 in query_indices.iter() {
        let q_idx = q_idx_i64 as usize;
        let mut result_heap: BinaryHeap<Neighbor<T>> = BinaryHeap::new();
        let mut kth_dist = T::max_value();
        let mut buffer: Vec<T> = Vec::with_capacity(BLOCK);
        let mut block_indices: Vec<usize> = Vec::with_capacity(BLOCK);

        let mut start = 0;
        while start < n {
            let end = usize::min(start + BLOCK, n);
            block_indices.clear();
            block_indices.extend(start..end);
            metric.distances_sq_batch_idx_into_with_kth(
                q_idx,
                &block_indices,
                Some(kth_dist),
                &mut buffer,
            );
            if emit_stats {
                let evals = buffer.iter().filter(|&&d| d != prune_sentinel).count();
                DIST_EVALS.fetch_add(evals, AtomicOrdering::Relaxed);
            }
            for (j, &dist) in buffer.iter().enumerate() {
                if dist == prune_sentinel {
                    continue;
                }
                let node_idx = block_indices[j] as i64;
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
            }
            start = end;
        }

        let sorted_results = result_heap.into_sorted_vec();
        let mut idxs = Vec::with_capacity(k);
        let mut dsts = Vec::with_capacity(k);
        for nbh in sorted_results {
            idxs.push(node_to_dataset[nbh.node_idx as usize]);
            dsts.push(nbh.dist.0);
        }
        indices_out.push(idxs);
        dists_out.push(dsts);
    }

    (indices_out, dists_out)
}

/// Blocked residual k-NN using per-query heaps and batched distance evaluation.
/// Optimised for float32; falls back to the scalar/block path otherwise.
pub fn batch_residual_knn_query_block_sgemm(
    query_indices: ndarray::ArrayView1<i64>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'_, f32>,
    k: usize,
    survivors: Option<&[usize]>,
) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    let q_count = query_indices.len();
    let targets: Vec<usize> = match survivors {
        Some(s) if !s.is_empty() => s.to_vec(),
        _ => (0..node_to_dataset.len()).collect(),
    };
    if q_count == 0 || targets.is_empty() {
        return (vec![Vec::new(); q_count], vec![Vec::new(); q_count]);
    }

    let coord_dim = metric.scaled_coords.ncols();
    let v_dim = metric.v_matrix.ncols();
    let coords = metric
        .scaled_coords
        .as_slice_memory_order()
        .expect("scaled_coords contiguous");
    let v_mat = metric
        .v_matrix
        .as_slice_memory_order()
        .expect("v_matrix contiguous");

    let mut heaps: Vec<BinaryHeap<Neighbor<f32>>> = Vec::with_capacity(q_count);
    for _ in 0..q_count {
        heaps.push(BinaryHeap::new());
    }
    let mut kth: Vec<f32> = vec![f32::MAX; q_count];

    let t_block = std::env::var("COVERTREEX_RUST_SGEMM_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0 && v <= 2048)
        .unwrap_or(256);
    let mut dist_block: Vec<f32> = vec![0.0; q_count * t_block];
    let prune_sentinel = f32::MAX;

    let mut start = 0;
    while start < targets.len() {
        let end = usize::min(start + t_block, targets.len());
        let t_size = end - start;

        for (qi, &q_idx_i64) in query_indices.iter().enumerate() {
            let q_idx = q_idx_i64 as usize;
            let q_coord = &coords[q_idx * coord_dim..(q_idx + 1) * coord_dim];
            let q_v = &v_mat[q_idx * v_dim..(q_idx + 1) * v_dim];
            let q_norm = metric.scaled_norms[q_idx];
            let q_diag = metric.p_diag[q_idx];
            let row_offset = qi * t_block;

            for (j, tgt_idx) in targets[start..end].iter().enumerate() {
                let tgt = *tgt_idx;
                let t_coord = &coords[tgt * coord_dim..(tgt + 1) * coord_dim];
                let t_v = &v_mat[tgt * v_dim..(tgt + 1) * v_dim];
                let t_norm = metric.scaled_norms[tgt];
                let denom = (q_diag * metric.p_diag[tgt]).sqrt();
                if denom <= 1e-9 {
                    dist_block[row_offset + j] = 1.0;
                    continue;
                }

                // coords dot
                let mut dot_c = 0.0f32;
                for d in 0..coord_dim {
                    dot_c += q_coord[d] * t_coord[d];
                }
                let mut d2 = q_norm + t_norm - 2.0 * dot_c;
                if d2 < 0.0 {
                    d2 = 0.0;
                }
                let k_val = metric.rbf_var * (-0.5 * d2).exp();

                // V dot
                let mut dot_v = 0.0f32;
                for d in 0..v_dim {
                    dot_v += q_v[d] * t_v[d];
                }
                let rho = (k_val - dot_v) / denom;
                let rho_clamped = rho.max(-1.0).min(1.0);
                dist_block[row_offset + j] = 1.0 - rho_clamped.abs();
            }

            for j in t_size..t_block {
                dist_block[row_offset + j] = prune_sentinel;
            }
        }

        for qi in 0..q_count {
            let row_offset = qi * t_block;
            for j in 0..t_size {
                let dist = dist_block[row_offset + j];
                if dist == prune_sentinel {
                    continue;
                }
                let node_idx = targets[start + j] as i64;
                if heaps[qi].len() < k || dist < kth[qi] {
                    heaps[qi].push(Neighbor {
                        dist: OrderedFloat(dist),
                        node_idx,
                    });
                    if heaps[qi].len() > k {
                        heaps[qi].pop();
                    }
                    if let Some(peek) = heaps[qi].peek() {
                        kth[qi] = peek.dist.0;
                    }
                }
            }
        }

        start = end;
    }

    let mut indices_out: Vec<Vec<i64>> = Vec::with_capacity(q_count);
    let mut dists_out: Vec<Vec<f32>> = Vec::with_capacity(q_count);
    for heap in heaps {
        let sorted = heap.into_sorted_vec();
        let mut idxs = Vec::with_capacity(k);
        let mut dsts = Vec::with_capacity(k);
        for nbh in sorted {
            idxs.push(node_to_dataset[nbh.node_idx as usize]);
            dsts.push(nbh.dist.0);
        }
        indices_out.push(idxs);
        dists_out.push(dsts);
    }

    (indices_out, dists_out)
}

fn single_residual_knn_query<'a, T>(
    tree: &'a CoverTreeData<T>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
    q_dataset_idx: usize,
    k: usize,
    scope_caps: Option<&HashMap<i32, T>>,
) -> (Vec<i64>, Vec<T>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    let mut candidate_heap = BinaryHeap::new();
    let mut result_heap: BinaryHeap<Neighbor<T>> = BinaryHeap::new();

    if tree.len() == 0 {
        return (vec![], vec![]);
    }

    // Budget Configuration
    let budget_schedule_str = std::env::var("COVERTREEX_RESIDUAL_BUDGET_SCHEDULE").ok();
    let budget_schedule: Vec<usize> = if let Some(s) = budget_schedule_str {
        s.split(',').filter_map(|v| v.parse().ok()).collect()
    } else {
        vec![32, 64, 96]
    };
    let budget_up: f64 = std::env::var("COVERTREEX_RESIDUAL_BUDGET_UP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6);
    let budget_down: f64 = std::env::var("COVERTREEX_RESIDUAL_BUDGET_DOWN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);

    let mut budget_idx = 0;
    let mut budget_limit = if !budget_schedule.is_empty() {
        budget_schedule[0]
    } else {
        usize::MAX
    };
    let mut survivors_count = 0;
    let mut low_yield_streak = 0;

    // Push root (index 0) with its true distance
    let root_payload = tree.get_point_row(0);
    let root_dataset_idx = root_payload[0].to_usize().unwrap();
    let root_dist = metric.distance_idx(q_dataset_idx, root_dataset_idx);
    candidate_heap.push(Candidate {
        dist: OrderedFloat(root_dist),
        node_idx: 0,
    });
    HEAP_PUSHES.fetch_add(1, AtomicOrdering::Relaxed);
    survivors_count += 1;

    let mut kth_dist = T::max_value();
    const BATCH_SIZE: usize = 64;
    let mut batch_nodes: Vec<(i64, T)> = Vec::with_capacity(BATCH_SIZE);

    // Child buffers
    let stream_tile = std::env::var("COVERTREEX_RESIDUAL_STREAM_TILE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(64);

    let mut child_nodes = Vec::with_capacity(stream_tile);
    let mut child_dataset_indices = Vec::with_capacity(stream_tile);
    let mut child_distance_buffer: Vec<T> = Vec::with_capacity(stream_tile);

    let emit_stats = EMIT_STATS.load(AtomicOrdering::Relaxed);
    let prune_sentinel = T::max_value();
    let use_pruning =
        std::env::var("COVERTREEX_RUST_PRUNE_BOUNDS").map_or(false, |v| v == "1" || v == "true");
    let scope_limit_env = std::env::var("COVERTREEX_SCOPE_CHUNK_TARGET")
        .ok()
        .and_then(|v| v.parse::<isize>().ok())
        .unwrap_or(0);
    let scope_limit: usize = if scope_limit_env > 0 {
        scope_limit_env as usize
    } else {
        16_384
    };
    let radius_floor: T = std::env::var("COVERTREEX_RESIDUAL_RADIUS_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| T::from(v).unwrap_or(T::zero()))
        .unwrap_or_else(T::zero);
    let mut processed_targets: usize = 0;

    while !candidate_heap.is_empty() {
        if survivors_count >= budget_limit && budget_limit != usize::MAX {
            break;
        }

        batch_nodes.clear();

        // 1. Collect Batch
        while batch_nodes.len() < BATCH_SIZE {
            if let Some(cand) = candidate_heap.pop() {
                batch_nodes.push((cand.node_idx, cand.dist.0));
            } else {
                break;
            }
        }

        if batch_nodes.is_empty() {
            break;
        }
        if processed_targets >= scope_limit {
            break;
        }

        // 2. Process Results (using exact distances from heap)
        for &(node_idx, dist) in batch_nodes.iter() {
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
            let mut radius = two.powi(level + 1);

            // Apply Scope Cap
            if let Some(caps) = scope_caps {
                if let Some(&cap) = caps.get(&level) {
                    let cap_t = cap; // Cap is now T
                    if cap_t < radius {
                        radius = cap_t;
                    }
                }
            }

            // Apply Radius Floor
            if radius < radius_floor {
                radius = radius_floor;
            }
            
            let lower_bound = dist - radius;

            if result_heap.len() == k && lower_bound > kth_dist {
                continue;
            }

            // Collect children for batched distance evaluation
            let mut child = tree.children[node_idx as usize];

            // Streamer Loop over Tiles
            while child != -1 {
                child_nodes.clear();
                child_dataset_indices.clear();

                // Collect Tile
                while child != -1 && child_nodes.len() < stream_tile {
                    // Pruning (Insertion Check)
                    let child_level = tree.levels[child as usize];
                    let mut child_radius = two.powi(child_level + 1);

                    // Apply Scope Cap to Child
                    if let Some(caps) = scope_caps {
                        if let Some(&cap) = caps.get(&child_level) {
                            let cap_t = cap; // Cap is now T
                            if cap_t < child_radius {
                                child_radius = cap_t;
                            }
                        }
                    }

                    // Apply Radius Floor
                    if child_radius < radius_floor {
                        child_radius = radius_floor;
                    }

                    let lb_child = dist - child_radius;

                    if result_heap.len() < k || lb_child <= kth_dist {
                        child_nodes.push(child);
                        child_dataset_indices.push(node_to_dataset[child as usize] as usize);
                    }

                    let next = tree.next_node[child as usize];
                    if next == child {
                        child = -1; // Signal end of children
                    } else {
                        child = next;
                        if child == tree.children[node_idx as usize] {
                            child = -1;
                        }
                    }
                }

                if child_nodes.is_empty() {
                    break;
                }

                // Compute Dists
                child_distance_buffer.clear();
                metric.distances_sq_batch_idx_into_with_kth(
                    q_dataset_idx,
                    &child_dataset_indices,
                    if use_pruning { Some(kth_dist) } else { None },
                    &mut child_distance_buffer,
                );

                if emit_stats {
                    let evals = child_distance_buffer
                        .iter()
                        .filter(|&&d| d != prune_sentinel)
                        .count();
                    DIST_EVALS.fetch_add(evals, AtomicOrdering::Relaxed);
                }
                processed_targets = processed_targets.saturating_add(child_dataset_indices.len());

                // Masked Append & Budget Update
                let mut added_in_tile = 0;
                for (j, &child_node) in child_nodes.iter().enumerate() {
                    let child_dist = child_distance_buffer[j];
                    if child_dist == prune_sentinel {
                        continue;
                    }

                    // Post-Distance Pruning
                    if result_heap.len() == k {
                        let child_level = tree.levels[child_node as usize];
                        let mut child_radius = two.powi(child_level + 1);

                        // Apply Scope Cap to Child
                        if let Some(caps) = scope_caps {
                            if let Some(&cap) = caps.get(&child_level) {
                                let cap_t = cap; // Cap is now T
                                if cap_t < child_radius {
                                    child_radius = cap_t;
                                }
                            }
                        }

                        // Apply Radius Floor
                        if child_radius < radius_floor {
                            child_radius = radius_floor;
                        }
                        
                        let lb_child = child_dist - child_radius;
                        if lb_child > kth_dist {
                            continue;
                        }
                    }

                    candidate_heap.push(Candidate {
                        dist: OrderedFloat(child_dist),
                        node_idx: child_node,
                    });
                    if emit_stats {
                        HEAP_PUSHES.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    added_in_tile += 1;
                }

                survivors_count += added_in_tile;

                // Budget Update
                if !budget_schedule.is_empty() {
                    let ratio = if !child_nodes.is_empty() {
                        added_in_tile as f64 / child_nodes.len() as f64
                    } else {
                        0.0
                    };
                    if ratio >= budget_up {
                        if budget_idx + 1 < budget_schedule.len() {
                            budget_idx += 1;
                            budget_limit = budget_schedule[budget_idx];
                        }
                        low_yield_streak = 0;
                    } else if ratio < budget_down {
                        low_yield_streak += 1;
                        if low_yield_streak >= 2 {
                            break;
                        }
                    } else {
                        low_yield_streak = 0;
                    }
                }

                if survivors_count >= budget_limit && budget_limit != usize::MAX {
                    break;
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

    (indices, dists)
}
