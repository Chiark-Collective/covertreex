use crate::metric::{Euclidean, Metric, ResidualMetric};
use crate::telemetry::ResidualQueryTelemetry;
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

// -----------------------------------------------------------------------------
// Separation invariant cache (cover radii)
// -----------------------------------------------------------------------------

pub(crate) fn compute_si_cache_residual<'a, T>(
    tree: &CoverTreeData<T>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
) -> Vec<T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a + 'static,
{
    let mut cache = vec![T::zero(); tree.len()];

    fn dfs<'a, T>(
        node: usize,
        tree: &CoverTreeData<T>,
        node_to_dataset: &[i64],
        metric: &ResidualMetric<'a, T>,
        out: &mut [T],
    ) -> T
    where
        T: Float + Debug + Send + Sync + std::iter::Sum + 'a + 'static,
    {
        let mut max_radius = T::zero();
        let node_ds = node_to_dataset[node] as usize;

        let mut child = tree.children[node];
        if child != -1 {
            loop {
                let c = child as usize;
                let child_radius = dfs(c, tree, node_to_dataset, metric, out);
                let dist = metric.distance_idx(node_ds, node_to_dataset[c] as usize);
                let bound = dist + child_radius;
                if bound > max_radius {
                    max_radius = bound;
                }

                let next = tree.next_node[c];
                if next == -1 || next == child {
                    break;
                }
                child = next;
                if child == tree.children[node] {
                    break;
                }
            }
        }

        out[node] = max_radius;
        max_radius
    }

    if tree.len() > 0 {
        dfs(0, tree, node_to_dataset, metric, &mut cache);
    }

    cache
}

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
    telemetry: Option<&mut ResidualQueryTelemetry>,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a + 'static,
{
    // Opt-in telemetry: disable parallelism to aggregate counts deterministically
    let results: Vec<(Vec<i64>, Vec<T>)> = if let Some(tele) = telemetry {
        let mut out = Vec::with_capacity(query_indices.len());
        for &q_idx in query_indices.iter() {
            let mut local = ResidualQueryTelemetry::default();
            let res = single_residual_knn_query(
                tree,
                node_to_dataset,
                metric,
                q_idx as usize,
                k,
                scope_caps,
                Some(&mut local),
            );
            tele.add_from(&local);
            out.push(res);
        }
        out
    } else {
        query_indices
            .into_par_iter()
            .map(|&q_idx| {
                single_residual_knn_query(
                    tree,
                    node_to_dataset,
                    metric,
                    q_idx as usize,
                    k,
                    scope_caps,
                    None,
                )
            })
            .collect()
    };

    let mut indices = Vec::with_capacity(results.len());
    let mut dists = Vec::with_capacity(results.len());
    for (idx, dst) in results {
        indices.push(idx);
        dists.push(dst);
    }
    (indices, dists)
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
                let dist_sq = 1.0 - rho_clamped.abs();
                dist_block[row_offset + j] = if dist_sq > 0.0 {
                    dist_sq.sqrt()
                } else {
                    0.0
                };
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
    mut telemetry: Option<&mut ResidualQueryTelemetry>,
) -> (Vec<i64>, Vec<T>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a + 'static,
{
    let mut result_heap: BinaryHeap<Neighbor<T>> = BinaryHeap::new();

    if tree.len() == 0 {
        return (vec![], vec![]);
    }

    let parity_mode = std::env::var("COVERTREEX_RESIDUAL_PARITY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Budget Configuration (disable in parity mode to match Python gold behavior)
    let budget_schedule_str = if parity_mode {
        None
    } else {
        std::env::var("COVERTREEX_RESIDUAL_BUDGET_SCHEDULE").ok()
    };
    let budget_schedule: Vec<usize> = if let Some(s) = budget_schedule_str {
        s.split(',').filter_map(|v| v.parse().ok()).collect()
    } else if parity_mode {
        Vec::new()
    } else {
        vec![32, 64, 96]
    };
    // Match Python defaults (scope_budget_up_thresh/down_thresh)
    let budget_up: f64 = if parity_mode {
        1.0
    } else {
        std::env::var("COVERTREEX_RESIDUAL_BUDGET_UP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.015)
    };
    let budget_down: f64 = if parity_mode {
        0.0
    } else {
        std::env::var("COVERTREEX_RESIDUAL_BUDGET_DOWN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.002)
    };

    let mut budget_idx = 0;
    let mut budget_limit = if !budget_schedule.is_empty() {
        budget_schedule[0]
    } else {
        usize::MAX
    };
    let mut survivors_count = 0;
    let mut low_yield_streak = 0;

    // Root distance and seed result heap
    let root_payload = tree.get_point_row(0);
    let root_dataset_idx = root_payload[0].to_usize().unwrap();
    let root_dist = metric.distance_idx(q_dataset_idx, root_dataset_idx);
    result_heap.push(Neighbor {
        dist: OrderedFloat(root_dist),
        node_idx: 0,
    });
    let mut kth_dist = if k > 0 { root_dist } else { T::max_value() };
    survivors_count += 1;

    // Buffers and frontier
    let stream_tile = if parity_mode {
        1
    } else {
        std::env::var("COVERTREEX_RESIDUAL_STREAM_TILE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(64)
    };
    
    // Optimization: Removed per-query O(N) allocations (seen_mask, cached_lb)
    // Assuming tree structure guarantees unique visits per level.
    let _use_masked_append = false;

    let use_visited = parity_mode
        || std::env::var("COVERTREEX_RESIDUAL_VISITED_SET")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    let scope_member_limit_env = std::env::var("COVERTREEX_RESIDUAL_SCOPE_MEMBER_LIMIT")
        .ok()
        .and_then(|v| v.parse::<isize>().ok())
        .unwrap_or(0);
    let scope_limit: usize = if parity_mode {
        usize::MAX
    } else if scope_member_limit_env > 0 {
        scope_member_limit_env as usize
    } else {
        let scope_limit_env = std::env::var("COVERTREEX_SCOPE_CHUNK_TARGET")
            .ok()
            .and_then(|v| v.parse::<isize>().ok())
            .unwrap_or(0);
        if scope_limit_env > 0 {
            scope_limit_env as usize
        } else {
            usize::MAX
        }
    };
    let dynamic_query_block = if parity_mode {
        false
    } else {
        std::env::var("COVERTREEX_RESIDUAL_DYNAMIC_QUERY_BLOCK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    };
    let level_cache_batching = if parity_mode {
        false
    } else {
        std::env::var("COVERTREEX_RESIDUAL_LEVEL_CACHE_BATCHING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    };
    let emit_stats = EMIT_STATS.load(AtomicOrdering::Relaxed);
    let prune_sentinel = T::max_value();
    let use_pruning =
        std::env::var("COVERTREEX_RUST_PRUNE_BOUNDS").map_or(false, |v| v == "1" || v == "true");
    let radius_floor: T = std::env::var("COVERTREEX_RESIDUAL_RADIUS_FLOOR")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| T::from(v).unwrap_or(T::zero()))
        .unwrap_or_else(|| T::from(1e-3).unwrap()); // match gold Numba floor
    let _max_hint = metric.max_distance_hint();
    let two = T::from(2.0).unwrap();

    let mut frontier: Vec<(usize, T)> = vec![(0usize, root_dist)];
    let mut next_frontier: Vec<(usize, T)> = Vec::with_capacity(stream_tile);
    let mut children_nodes: Vec<usize> = Vec::with_capacity(stream_tile);
    let mut children_ds_idx: Vec<usize> = Vec::with_capacity(stream_tile);
    let mut children_parent_lb: Vec<T> = Vec::with_capacity(stream_tile);
    let mut distance_buffer: Vec<T> = Vec::with_capacity(stream_tile);
    
    // Optimization: Removed per-query O(N) allocations (seen_mask, cached_lb)
    // Assuming tree structure guarantees unique visits per level.
    let mut visited_nodes: Vec<bool> = if use_visited {
        vec![false; tree.len()]
    } else {
        Vec::new()
    };
    if use_visited && !visited_nodes.is_empty() {
        visited_nodes[0] = true;
    }

    while !frontier.is_empty() && survivors_count < budget_limit {
        if parity_mode && result_heap.len() == k {
            // Early stop if frontier cannot beat current kth
            let mut min_lb = T::max_value();
            for &(parent, parent_dist) in frontier.iter() {
                let parent_level = tree.levels[parent];
                let mut parent_radius = if !tree.si_cache.is_empty() && parent < tree.si_cache.len()
                {
                    tree.si_cache[parent]
                } else {
                    two.powi(parent_level + 1)
                };
                if parent_radius < radius_floor {
                    parent_radius = radius_floor;
                }
                let lb = parent_dist - parent_radius;
                if lb < min_lb {
                    min_lb = lb;
                }
            }
            if min_lb > kth_dist {
                break;
            }
        }
        if let Some(t) = telemetry.as_mut() {
            t.record_frontier(frontier.len(), children_nodes.len());
        }
        children_nodes.clear();
        children_ds_idx.clear();
        children_parent_lb.clear();

        // Gather children of the current frontier (level cache: process once per level)
        for &(parent, parent_dist) in frontier.iter() {
            let parent_level = tree.levels[parent];
            let mut parent_radius = if !tree.si_cache.is_empty() && parent < tree.si_cache.len() {
                tree.si_cache[parent]
            } else {
                two.powi(parent_level + 1)
            };
            let capped_parent = if parity_mode {
                parent_radius
            } else {
                metric.apply_level_cap(parent_level, scope_caps, parent_radius)
            };
            if let Some(t) = telemetry.as_mut() {
                if capped_parent < parent_radius {
                    t.caps_applied += 1;
                }
            }
            parent_radius = capped_parent;
            if parent_radius < radius_floor {
                parent_radius = radius_floor;
            }
            let parent_lb = parent_dist - parent_radius;
            if result_heap.len() == k && parent_lb > kth_dist {
                if let Some(t) = telemetry.as_mut() {
                    t.prunes_lower_bound += 1;
                }
                continue;
            }

            // Level cache batching: prefetch child distances in blocks to tighten kth early
            let mut child = tree.children[parent];
            let mut parent_children: Vec<usize> = Vec::new();
            while child != -1 {
                parent_children.push(child as usize);
                let next = tree.next_node[child as usize];
                if next == child {
                    break;
                }
                child = next;
                if child == tree.children[parent] {
                    break;
                }
            }
            if level_cache_batching && !parent_children.is_empty() {
                if !parity_mode {
                    // order by dataset distance to parent to prioritize closer children
                    parent_children.sort_by_key(|&c| node_to_dataset[c]);
                }
            }
            for child_idx in parent_children.into_iter() {
                let ds_idx = node_to_dataset[child_idx] as usize;
                if use_visited {
                    if let Some(already) = visited_nodes.get(child_idx).copied() {
                        if already {
                            if let Some(t) = telemetry.as_mut() {
                                t.masked_dedup += 1;
                            }
                            continue;
                        }
                    }
                    if let Some(slot) = visited_nodes.get_mut(child_idx) {
                        *slot = true;
                    }
                }
                
                children_nodes.push(child_idx);
                children_ds_idx.push(ds_idx);
                children_parent_lb.push(parent_lb);
            }
        }

        // Process children in tiles
        let mut start = 0;
        while start < children_nodes.len() && survivors_count < budget_limit {
            let active = children_nodes.len().saturating_sub(start);
            let mut tile = if dynamic_query_block {
                // Adaptive tile: shrink when active set is small, grow when large.
                // Use frontier size + remaining active to scale smoothly.
                let wave = frontier.len().max(1);
                let combined = active + wave;
                let adaptive = if combined <= 32 {
                    16
                } else if combined <= 128 {
                    32
                } else if combined <= 512 {
                    64
                } else {
                    128
                };
                adaptive.min(stream_tile)
            } else {
                stream_tile
            };
            if dynamic_query_block {
                let remaining = scope_limit.saturating_sub(survivors_count);
                tile = tile.min(remaining.max(1));
            }
            if let Some(t) = telemetry.as_mut() {
                t.record_block(tile);
            }
            let end = usize::min(start + tile, children_nodes.len());
            let nodes_chunk = &children_nodes[start..end];
            let ds_chunk = &children_ds_idx[start..end];
            let lb_chunk = &children_parent_lb[start..end];

            // Pre-distance pruning: skip entire chunk if cached parent lower bounds exceed kth.
            if result_heap.len() == k {
                if let Some(&min_lb) = lb_chunk
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
                    if min_lb > kth_dist {
                        if let Some(t) = telemetry.as_mut() {
                            t.prunes_lower_bound_chunks += 1;
                        }
                        start = end;
                        continue;
                    }
                }
            }

            distance_buffer.clear();
            distance_buffer.resize(nodes_chunk.len(), prune_sentinel);
            let mut eval_targets: Vec<usize> = Vec::with_capacity(nodes_chunk.len());
            let mut eval_positions: Vec<usize> = Vec::with_capacity(nodes_chunk.len());
            for (j, &ds_idx) in ds_chunk.iter().enumerate() {
                eval_targets.push(ds_idx);
                eval_positions.push(j);
            }
            let mut tmp: Vec<T> = Vec::with_capacity(eval_targets.len());
            metric.distances_sq_batch_idx_into_with_kth(
                q_dataset_idx,
                &eval_targets,
                if use_pruning { Some(kth_dist) } else { None },
                &mut tmp,
            );
            for (val, pos) in tmp.into_iter().zip(eval_positions.into_iter()) {
                distance_buffer[pos] = val;
            }

            if emit_stats {
                let evals = distance_buffer
                    .iter()
                    .filter(|&&d| d != prune_sentinel)
                    .count();
                DIST_EVALS.fetch_add(evals, AtomicOrdering::Relaxed);
            }
            if let Some(t) = telemetry.as_mut() {
                t.distance_evals += distance_buffer
                    .iter()
                    .filter(|&&d| d != prune_sentinel)
                    .count();
            }

            let mut added_in_chunk = 0;
            // Process closer children first to tighten kth early
            let mut ordered: Vec<(T, usize, T)> = nodes_chunk
                .iter()
                .enumerate()
                .map(|(j, &idx)| (distance_buffer[j], idx, lb_chunk[j]))
                .filter(|(d, _, _)| *d != prune_sentinel)
                .collect();
            ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            for (dist, child_idx, parent_lb_cached) in ordered.into_iter() {
                let child_level = tree.levels[child_idx];
                let mut child_radius =
                    if !tree.si_cache.is_empty() && child_idx < tree.si_cache.len() {
                        tree.si_cache[child_idx]
                    } else {
                        two.powi(child_level + 1)
                    };
                let capped_child = if parity_mode {
                    child_radius
                } else {
                    metric.apply_level_cap(child_level, scope_caps, child_radius)
                };
                if let Some(t) = telemetry.as_mut() {
                    if capped_child < child_radius {
                        t.caps_applied += 1;
                    }
                }
                child_radius = capped_child;
                if child_radius < radius_floor {
                    child_radius = radius_floor;
                }

                if result_heap.len() == k {
                    let lb = parent_lb_cached.min(dist - child_radius);
                    if lb > kth_dist {
                        if let Some(t) = telemetry.as_mut() {
                            t.prunes_cap += 1;
                        }
                        continue;
                    }
                }

                result_heap.push(Neighbor {
                    dist: OrderedFloat(dist),
                    node_idx: child_idx as i64,
                });
                if result_heap.len() > k {
                    result_heap.pop();
                }
                if let Some(peek) = result_heap.peek() {
                    kth_dist = peek.dist.0;
                }

                survivors_count += 1;
                added_in_chunk += 1;

                if survivors_count < budget_limit {
                    next_frontier.push((child_idx, dist));
                } else {
                    break;
                }
            }

            // Budget ladder update (yield-based)
            if !parity_mode && !budget_schedule.is_empty() && !nodes_chunk.is_empty() {
                let ratio = added_in_chunk as f64 / nodes_chunk.len() as f64;
                if let Some(t) = telemetry.as_mut() {
                    t.record_yield(ratio as f32);
                }
                if ratio >= budget_up {
                    if budget_idx + 1 < budget_schedule.len() {
                        budget_idx += 1;
                        budget_limit = budget_schedule[budget_idx];
                        if let Some(t) = telemetry.as_mut() {
                            t.budget_escalations += 1;
                        }
                    }
                    low_yield_streak = 0;
                } else if ratio < budget_down {
                    low_yield_streak += 1;
                    if low_yield_streak >= 2 {
                        if let Some(t) = telemetry.as_mut() {
                            t.budget_early_terminate += 1;
                        }
                        break;
                    }
                } else {
                    low_yield_streak = 0;
                }
            }

            start = end;
        }

        frontier.clear();
        std::mem::swap(&mut frontier, &mut next_frontier);
    }
    if let Some(t) = telemetry {
        // final frontier expansion count (expanded = children processed)
        t.record_frontier(0, children_nodes.len());
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


