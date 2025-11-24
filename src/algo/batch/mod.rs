use crate::algo::batch::mis::compute_mis_greedy;
use crate::metric::Metric;
use crate::tree::CoverTreeData;
use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::time::Instant;

/// Build a CSR scope adjacency from directed conflict edges while enforcing a per-query cap.
///
/// This mirrors the "dense scope streamer" behaviour from the Python path: we dedupe members
/// per owner, apply a hard cap (chunk target), and track chunk-level telemetry used by the
/// telemetry bridge.
fn build_scopes_from_edges(
    batch_len: usize,
    edges: &[(usize, usize)],
    scope_cap: Option<usize>,
    budget_schedule: Option<&[usize]>,
    budget_up: Option<f64>,
    budget_down: Option<f64>,
    masked_scope_append: bool,
    max_segments: Option<usize>,
    pair_merge: bool,
) -> (
    Vec<i64>, // scope_indptr
    Vec<i64>, // scope_indices
    i64,      // scope_chunk_segments
    i64,      // scope_chunk_emitted
    i64,      // scope_chunk_max_members
    i64,      // scope_chunk_points
    i64,      // conflict_scope_chunk_pair_cap
    i64,      // conflict_scope_chunk_pairs_before
    i64,      // conflict_scope_chunk_pairs_after
    i64,      // conflict_scope_chunk_pair_merges
    i64,      // scope_chunk_scans
    i64,      // scope_chunk_dedupe
    i64,      // scope_chunk_saturated
    i64,      // scope_budget_start
    i64,      // scope_budget_final
    i64,      // scope_budget_escalations
    i64,      // scope_budget_early_terminate
) {
    if batch_len == 0 {
        return (
            vec![0],
            Vec::new(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
    }

    let cap = scope_cap.unwrap_or(0);
    let schedule: Vec<usize> = budget_schedule.unwrap_or(&[]).to_vec();
    let base_limit = if !schedule.is_empty() {
        let first = schedule[0];
        if cap > 0 {
            std::cmp::min(first, cap)
        } else {
            first
        }
    } else {
        cap
    };
    let effective_cap = if base_limit == 0 {
        usize::MAX
    } else {
        base_limit
    };
    let budget_up_val = budget_up.unwrap_or(0.0);
    let budget_down_val = budget_down.unwrap_or(0.0);

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); batch_len];
    for &(u, v) in edges.iter() {
        if u < batch_len && v < batch_len && u != v {
            adjacency[u].push(v);
        }
    }

    let pairs_before = (edges.len() / 2) as i64;

    let mut scope_indptr: Vec<i64> = Vec::with_capacity(batch_len + 1);
    let mut scope_indices: Vec<i64> = Vec::new();
    scope_indptr.push(0);

    let mut emitted = 0i64;
    let mut max_members = 0i64;
    let mut total_points = 0i64;
    let mut pairs_after_acc = 0i64;
    let mut scan_count = 0i64;
    let mut dedupe_total = 0i64;
    let mut saturated_count = 0i64;
    let mut budget_escalations_total = 0i64;
    let mut budget_early_total = 0i64;
    let mut budget_final_max = base_limit as i64;
    let max_segments_limit = max_segments.unwrap_or(usize::MAX);

    // Scratch mask reused per owner to avoid allocations when batch sizes are moderate.
    let mut mask = vec![false; batch_len];

    for owners in adjacency.into_iter() {
        if emitted as usize >= max_segments_limit {
            saturated_count += 1;
            break;
        }

        scan_count += owners.len() as i64;
        let mut members: Vec<usize> = if masked_scope_append {
            for &t in owners.iter() {
                mask[t] = true;
            }
            let collected: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(idx, &flag)| if flag { Some(idx) } else { None })
                .collect();
            for &idx in collected.iter() {
                mask[idx] = false;
            }
            collected
        } else {
            owners.clone()
        };

        let scan_len = owners.len() as f64;
        let mut limit = effective_cap;
        let survivors = members.len() as f64;
        if !schedule.is_empty() && scan_len > 0.0 {
            let ratio = survivors / scan_len;
            if budget_up_val > 0.0 && ratio >= budget_up_val && schedule.len() > 1 {
                limit = schedule[1];
                if cap > 0 && limit > 0 {
                    limit = std::cmp::min(limit, cap);
                }
                budget_escalations_total += 1;
            } else if budget_down_val > 0.0 && ratio < budget_down_val {
                budget_early_total += 1;
                members.clear();
            }
        }
        if limit == 0 {
            limit = usize::MAX;
        }
        budget_final_max = budget_final_max.max(limit as i64);

        if members.len() > limit {
            members.truncate(limit);
            saturated_count += 1;
        }

        emitted += 1;
        max_members = max_members.max(members.len() as i64);
        total_points += members.len() as i64;
        let mut pairs_here = ((members.len() as i64) * ((members.len() as i64) - 1)) / 2;
        if pair_merge {
            // clamp to avoid quadratic blow-up when members grow unexpectedly
            let cap_here = (limit as i64).saturating_mul(limit as i64);
            pairs_here = pairs_here.min(cap_here);
        }
        pairs_after_acc += pairs_here;
        dedupe_total += (owners.len().saturating_sub(members.len())) as i64;

        for m in members {
            scope_indices.push(m as i64);
        }
        scope_indptr.push(scope_indices.len() as i64);
    }

    let pair_cap = max_members * max_members;
    let pairs_after = pairs_after_acc.min(pair_cap);
    let pair_merges = (pairs_before - pairs_after).max(0);

    (
        scope_indptr,
        scope_indices,
        emitted, // segments ~ emitted chunks
        emitted,
        max_members,
        total_points,
        pair_cap,
        pairs_before,
        pairs_after,
        pair_merges,
        scan_count,
        dedupe_total,
        saturated_count,
        base_limit as i64,
        budget_final_max,
        budget_escalations_total,
        budget_early_total,
    )
}

/// Build conflict edges using a simple grid hash over candidate coordinates.
///
/// Returns directed edges expressed in the far-candidate index space along with
/// the number of occupied cells and local edges emitted.
fn grid_conflicts_for_candidates<T>(
    coords: ArrayView2<T>,
    candidates: &[usize],
    start_idx: usize,
    scale: T,
) -> (Vec<(usize, usize)>, i64, i64)
where
    T: Float + Debug + Send + Sync,
{
    if candidates.is_empty() {
        return (Vec::new(), 0, 0);
    }

    let dim = coords.ncols();
    let mut cell_map: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();

    for (pos, &global_idx) in candidates.iter().enumerate() {
        let local = global_idx - start_idx;
        let row = coords.row(local);
        let mut key: Vec<i64> = Vec::with_capacity(dim);
        for &v in row.iter() {
            let scaled = v * scale;
            key.push(scaled.floor().to_i64().unwrap_or(0));
        }
        cell_map.entry(key).or_default().push(pos);
    }

    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    let mut local_edges: i64 = 0;

    // Helper to insert directed edges while tracking undirected count once.
    let mut insert_pair = |u: usize, v: usize| {
        if edge_set.insert((u, v)) {
            if u < v {
                local_edges += 1;
            }
        }
        if edge_set.insert((v, u)) {
            if v < u {
                local_edges += 1;
            }
        }
    };

    for (_cell, positions) in cell_map.iter() {
        if positions.len() < 2 {
            continue;
        }
        for a in 0..positions.len() {
            for b in a + 1..positions.len() {
                let u = positions[a];
                let v = positions[b];
                insert_pair(u, v);
            }
        }
    }

    // Neighbor-cell edges (Chebyshev-1) across first 3 dims to mirror Python grid overlap.
    let neighbor_dims = dim.min(3);
    if neighbor_dims > 0 {
        let keys: Vec<Vec<i64>> = cell_map.keys().cloned().collect();
        for key in keys.iter() {
            let positions = match cell_map.get(key) {
                Some(v) => v,
                None => continue,
            };
            let offsets: [i64; 3] = [-1, 0, 1];
            fn build_neighbors(
                idx: usize,
                dims: usize,
                base: &[i64],
                offsets: &[i64; 3],
                cur: &mut Vec<i64>,
                out: &mut Vec<Vec<i64>>,
            ) {
                if idx == dims {
                    out.push(cur.clone());
                    return;
                }
                for off in offsets {
                    cur.push(base[idx] + *off);
                    build_neighbors(idx + 1, dims, base, offsets, cur, out);
                    cur.pop();
                }
            }
            let mut neigh_keys: Vec<Vec<i64>> = Vec::new();
            build_neighbors(
                0,
                neighbor_dims,
                key,
                &offsets,
                &mut Vec::new(),
                &mut neigh_keys,
            );

            for neigh in neigh_keys.into_iter() {
                if neigh == *key {
                    continue;
                }
                if let Some(targets) = cell_map.get(&neigh) {
                    for &u in positions.iter() {
                        for &v in targets.iter() {
                            insert_pair(u, v);
                        }
                    }
                }
            }
        }
    }

    (
        edge_set.into_iter().collect(),
        cell_map.len() as i64,
        local_edges,
    )
}

pub mod mis;

/// Lightweight summary of a single batch insertion, used by the Rust PCCT2 bridge
/// to emit telemetry that mirrors the Python `BatchInsertPlan` schema.
#[derive(Debug, Clone)]
pub struct BatchInsertTelemetry {
    pub parents: Vec<i64>,
    pub levels: Vec<i64>,
    pub selected: Vec<i64>,
    pub dominated: Vec<i64>,
    pub conflict_indptr: Vec<i64>,
    pub conflict_indices: Vec<i64>,
    pub scope_indptr: Vec<i64>,
    pub scope_indices: Vec<i64>,
    pub scope_chunk_segments: i64,
    pub scope_chunk_emitted: i64,
    pub scope_chunk_max_members: i64,
    pub scope_chunk_points: i64,
    pub conflict_scope_chunk_pair_cap: i64,
    pub conflict_scope_chunk_pairs_before: i64,
    pub conflict_scope_chunk_pairs_after: i64,
    pub conflict_scope_chunk_pair_merges: i64,
    pub scope_chunk_scans: i64,
    pub scope_chunk_dedupe: i64,
    pub scope_chunk_saturated: i64,
    pub scope_budget_start: i64,
    pub scope_budget_final: i64,
    pub scope_budget_escalations: i64,
    pub scope_budget_early_terminate: i64,
    pub batch_start_index: i64,
    pub conflict_grid_cells: i64,
    pub conflict_grid_leaders_raw: i64,
    pub conflict_grid_leaders_after: i64,
    pub conflict_grid_local_edges: i64,
    pub degree_cap: i64,
    pub degree_pruned_pairs: i64,
    pub traversal_seconds: f64,
    pub conflict_graph_seconds: f64,
    pub mis_seconds: f64,
}

fn compute_candidate_conflicts<T>(
    points: &Array2<T>,
    radius_sq: T,
    metric: &dyn Metric<T>,
) -> Vec<(usize, usize)>
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    let n = points.nrows();
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            // Don't compute row_i here to avoid lifetime issues with returning iterator
            // Capture points reference instead
            (i + 1..n).filter_map(move |j| {
                let row_i_view = points.row(i);
                let row_i = row_i_view.as_slice().unwrap();

                let row_j_view = points.row(j);
                let row_j = row_j_view.as_slice().unwrap();

                let d_sq = metric.distance_sq(row_i, row_j);
                if d_sq <= radius_sq {
                    Some((i, j))
                } else {
                    None
                }
            })
        })
        .collect()
}

pub fn batch_insert<T>(tree: &mut CoverTreeData<T>, batch: ArrayView2<T>, metric: &dyn Metric<T>)
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'static,
{
    let _ = batch_insert_collect(
        tree, batch, None, metric, None, None, None, None, None, None, None, None, None, false,
    );
}

pub fn batch_insert_with_telemetry<T>(
    tree: &mut CoverTreeData<T>,
    batch: ArrayView2<T>,
    coords: Option<ArrayView2<T>>,
    metric: &dyn Metric<T>,
    grid_whiten_scale: Option<T>,
    scope_chunk_target: Option<usize>,
    conflict_degree_cap: Option<usize>,
    scope_budget_schedule: Option<&[usize]>,
    scope_budget_up: Option<f64>,
    scope_budget_down: Option<f64>,
    masked_scope_append: Option<bool>,
    scope_chunk_max_segments: Option<usize>,
    scope_chunk_pair_merge: Option<bool>,
) -> BatchInsertTelemetry
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'static,
{
    batch_insert_collect(
        tree,
        batch,
        coords,
        metric,
        grid_whiten_scale,
        scope_chunk_target,
        conflict_degree_cap,
        scope_budget_schedule,
        scope_budget_up,
        scope_budget_down,
        masked_scope_append,
        scope_chunk_max_segments,
        scope_chunk_pair_merge,
        true,
    )
}

fn batch_insert_collect<T>(
    tree: &mut CoverTreeData<T>,
    batch: ArrayView2<T>,
    coords: Option<ArrayView2<T>>,
    metric: &dyn Metric<T>,
    grid_whiten_scale: Option<T>,
    scope_chunk_target: Option<usize>,
    conflict_degree_cap: Option<usize>,
    scope_budget_schedule: Option<&[usize]>,
    scope_budget_up: Option<f64>,
    scope_budget_down: Option<f64>,
    masked_scope_append: Option<bool>,
    scope_chunk_max_segments: Option<usize>,
    scope_chunk_pair_merge: Option<bool>,
    collect_stats: bool,
) -> BatchInsertTelemetry
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'static,
{
    let build_start = Instant::now();

    let empty = || BatchInsertTelemetry {
        parents: Vec::new(),
        levels: Vec::new(),
        selected: Vec::new(),
        dominated: Vec::new(),
        conflict_indptr: vec![0],
        conflict_indices: Vec::new(),
        scope_indptr: vec![0],
        scope_indices: Vec::new(),
        scope_chunk_segments: 0,
        scope_chunk_emitted: 0,
        scope_chunk_max_members: 0,
        scope_chunk_points: 0,
        conflict_scope_chunk_pair_cap: 0,
        conflict_scope_chunk_pairs_before: 0,
        conflict_scope_chunk_pairs_after: 0,
        conflict_scope_chunk_pair_merges: 0,
        scope_chunk_scans: 0,
        scope_chunk_dedupe: 0,
        scope_chunk_saturated: 0,
        scope_budget_start: 0,
        scope_budget_final: 0,
        scope_budget_escalations: 0,
        scope_budget_early_terminate: 0,
        batch_start_index: 0,
        conflict_grid_cells: 0,
        conflict_grid_leaders_raw: 0,
        conflict_grid_leaders_after: 0,
        conflict_grid_local_edges: 0,
        degree_cap: 0,
        degree_pruned_pairs: 0,
        traversal_seconds: 0.0,
        conflict_graph_seconds: 0.0,
        mis_seconds: 0.0,
    };

    // 0. Append all points to tree with dummy level
    let start_idx = tree.len();
    let _min_val = T::min_value(); // i32::MIN equivalent not available for Float generally?
                                   // CoverTreeData levels are i32.
                                   // add_point takes level as i32.
                                   // So Float T is for points/metric, but levels are i32.

    for row in batch.outer_iter() {
        tree.add_point(row, i32::MIN, -1);
    }
    let end_idx = tree.len();
    let mut candidates: Vec<usize> = (start_idx..end_idx).collect();

    if candidates.is_empty() {
        return empty();
    }

    // 1. Bootstrap Root if tree was empty
    let mut initial_candidates = candidates.clone();
    if start_idx == 0 {
        let root = initial_candidates[0];
        let max_scale = 10; // TODO: Make configurable
        let min_scale = -10;
        tree.min_level = min_scale;
        tree.max_level = max_scale;
        tree.set_level(root, max_scale);
        initial_candidates.remove(0);
    }

    candidates = initial_candidates;
    if candidates.is_empty() {
        return empty();
    }

    // 2. Initialize Active Sets
    let root_idx = 0;
    let mut active_sets: Vec<Vec<usize>> = vec![vec![root_idx]; candidates.len()];

    let mut current_level: i32 = tree.max_level - 1;
    let min_level: i32 = tree.min_level;

    // Telemetry collectors
    let batch_len = end_idx - start_idx;
    let chunk_target = scope_chunk_target.unwrap_or(batch_len);
    let chunk_target = if chunk_target == 0 {
        batch_len
    } else {
        chunk_target
    };
    let mut conflict_edges: Vec<(usize, usize)> = Vec::new();
    let mut selected_mask = if collect_stats {
        vec![false; batch_len]
    } else {
        Vec::new()
    };
    let mut dominated_mask = if collect_stats {
        vec![false; batch_len]
    } else {
        Vec::new()
    };
    let mut conflict_graph_seconds = 0.0f64;
    let mut mis_seconds = 0.0f64;
    let mut conflict_grid_cells: i64 = 0;
    let mut conflict_grid_leaders_raw: i64 = 0;
    let mut conflict_grid_leaders_after: i64 = 0;
    let mut conflict_grid_local_edges: i64 = 0;

    while current_level >= min_level {
        let radius = T::from(2.0).unwrap().powi(current_level);
        let radius_sq = radius * radius;
        let covers_all = metric
            .max_distance_hint()
            .map(|max_d| radius >= max_d)
            .unwrap_or(false);

        // 3. Filter: Near vs Far
        let filter_results: Vec<(bool, Vec<usize>)> = candidates
            .par_iter()
            .zip(active_sets.par_iter())
            .map(|(&q_idx, covers)| {
                let q_point = tree.get_point_row(q_idx);
                let mut best_dist_sq = T::max_value();
                let mut best_node = usize::MAX;
                let mut is_near = false;

                if covers_all && !covers.is_empty() {
                    // Residual metric distance is bounded; if the current radius already
                    // exceeds that bound, every candidate is automatically "near" its
                    // first cover. This avoids expensive distance evaluations at coarse
                    // levels.
                    return (true, vec![covers[0]]);
                }

                for &p_idx in covers {
                    let p_point = tree.get_point_row(p_idx);
                    let d_p_sq = metric.distance_sq(q_point, p_point);
                    if d_p_sq <= radius_sq {
                        if d_p_sq < best_dist_sq {
                            best_dist_sq = d_p_sq;
                            best_node = p_idx;
                        }
                        is_near = true;
                    }

                    let mut child = tree.children[p_idx];
                    while child != -1 {
                        let c_idx = child as usize;
                        let c_point = tree.get_point_row(c_idx);
                        let d_c_sq = metric.distance_sq(q_point, c_point);
                        if d_c_sq <= radius_sq {
                            if d_c_sq < best_dist_sq {
                                best_dist_sq = d_c_sq;
                                best_node = c_idx;
                            }
                            is_near = true;
                        }

                        let next = tree.next_node[c_idx];
                        if next == child {
                            break;
                        }
                        child = next;
                        if child == tree.children[p_idx] {
                            break;
                        }
                    }
                }

                let next_cover = if is_near { vec![best_node] } else { vec![] };
                (is_near, next_cover)
            })
            .collect();

        let mut far_candidates = Vec::new();
        let mut far_original_indices = Vec::new();

        let mut next_candidates = Vec::new();
        let mut next_active_sets = Vec::new();

        for (i, (is_near, next_cover)) in filter_results.into_iter().enumerate() {
            if is_near {
                next_candidates.push(candidates[i]);
                next_active_sets.push(next_cover);
            } else {
                far_candidates.push(candidates[i]);
                far_original_indices.push(i);
            }
        }

        // 5. Conflict Graph
        if !far_candidates.is_empty() {
            let n_far = far_candidates.len();
            let mut far_points = Array2::<T>::zeros((n_far, tree.dimension));
            for (i, &idx) in far_candidates.iter().enumerate() {
                // Use copy_from_slice for fast copy
                far_points
                    .row_mut(i)
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(tree.get_point_row(idx));
            }

            // Lightweight grid-based adjacency to short-circuit obvious conflicts
            // before the full pairwise pass. This mirrors the Python grid builder used
            // to seed MIS with dense-local edges.
            let grid_scale = grid_whiten_scale.unwrap_or_else(T::one);
            let (grid_edges, grid_cells_level, grid_local_edges_level) = if let Some(coord_batch) =
                coords
            {
                grid_conflicts_for_candidates(coord_batch, &far_candidates, start_idx, grid_scale)
            } else {
                (Vec::new(), 0, 0)
            };

            let conflict_start = Instant::now();
            let mut conflicts = compute_candidate_conflicts(&far_points, radius_sq, metric);
            if !grid_edges.is_empty() {
                conflicts.extend_from_slice(&grid_edges);
            }
            // MIS uses f32 for priorities? Or T?
            // compute_mis_greedy takes &[f32].
            // Let's use f32 for priorities regardless of T.
            let priorities = vec![1.0f32; n_far];
            let conflict_elapsed = conflict_start.elapsed();
            conflict_graph_seconds += conflict_elapsed.as_secs_f64();

            if collect_stats {
                conflict_grid_cells = conflict_grid_cells.max(grid_cells_level);
                conflict_grid_local_edges += grid_local_edges_level;
                conflict_grid_leaders_raw = conflict_grid_leaders_raw.max(n_far as i64);
                conflict_grid_leaders_after = conflict_grid_leaders_after.max(grid_cells_level);
            }

            let mis_start = Instant::now();
            let mis_mask = compute_mis_greedy(n_far, &conflicts, &priorities);
            mis_seconds += mis_start.elapsed().as_secs_f64();

            // Identify Leaders
            let mut leaders = Vec::new();

            for (i, &is_leader) in mis_mask.iter().enumerate() {
                if is_leader {
                    leaders.push(far_candidates[i]);

                    let q_idx = far_candidates[i];
                    tree.set_level(q_idx, current_level);

                    // Mark selected in local batch space
                    if collect_stats && q_idx >= start_idx && q_idx < end_idx {
                        selected_mask[q_idx - start_idx] = true;
                    }

                    let original_idx = far_original_indices[i];
                    let parents = &active_sets[original_idx];
                    if !parents.is_empty() {
                        tree.set_parent(q_idx, parents[0] as i64);
                    }
                }
            }

            // Handle Non-Leaders
            let leader_points_arr = Array2::from_shape_vec(
                (leaders.len(), tree.dimension),
                leaders
                    .iter()
                    .flat_map(|&idx| tree.get_point_row(idx).to_vec())
                    .collect(),
            )
            .unwrap();

            let non_leader_indices: Vec<usize> = mis_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_l)| !is_l)
                .map(|(i, _)| i)
                .collect();

            let non_leader_results: Vec<Vec<usize>> = non_leader_indices
                .par_iter()
                .map(|&i| {
                    let q_idx = far_candidates[i];
                    let q_point = tree.get_point_row(q_idx);
                    let mut best_dist = T::max_value();
                    let mut best_leader = usize::MAX;

                    for l_i in 0..leaders.len() {
                        let l_point_view = leader_points_arr.row(l_i);
                        let l_point = l_point_view.as_slice().unwrap();
                        let d_sq = metric.distance_sq(q_point, l_point);
                        if d_sq <= radius_sq && d_sq < best_dist {
                            best_dist = d_sq;
                            best_leader = leaders[l_i];
                        }
                    }

                    if best_leader != usize::MAX {
                        vec![best_leader]
                    } else {
                        Vec::<usize>::new()
                    }
                })
                .collect();

            for (k, &i) in non_leader_indices.iter().enumerate() {
                let q_idx = far_candidates[i];
                let covered_by: &Vec<usize> = &non_leader_results[k];
                if !covered_by.is_empty() {
                    next_candidates.push(q_idx);
                    next_active_sets.push(covered_by.clone());
                    if collect_stats && q_idx >= start_idx && q_idx < end_idx {
                        dominated_mask[q_idx - start_idx] = true;
                    }
                }
            }

            // Capture edges for telemetry using batch-local indices
            if collect_stats {
                for (u, v) in conflicts {
                    let u_global = far_candidates[u];
                    let v_global = far_candidates[v];
                    if (start_idx..end_idx).contains(&u_global)
                        && (start_idx..end_idx).contains(&v_global)
                    {
                        conflict_edges.push((u_global - start_idx, v_global - start_idx));
                        conflict_edges.push((v_global - start_idx, u_global - start_idx));
                    }
                }
            }
        }

        candidates = next_candidates;
        active_sets = next_active_sets;
        current_level -= 1;
    }

    for (i, &q_idx) in candidates.iter().enumerate() {
        tree.set_level(q_idx, min_level);
        if !active_sets[i].is_empty() {
            tree.set_parent(q_idx, active_sets[i][0] as i64);
        }
        if collect_stats && q_idx >= start_idx && q_idx < end_idx {
            // Candidates that survive to the bottom are effectively selected.
            selected_mask[q_idx - start_idx] = true;
        }
    }

    // Build CSR adjacency for conflicts within this batch
    let (
        conflict_indptr,
        conflict_indices,
        parents,
        levels,
        selected,
        dominated,
        scope_indptr,
        scope_indices,
        scope_chunk_segments,
        scope_chunk_emitted,
        scope_chunk_max_members,
        scope_chunk_points,
        conflict_scope_chunk_pair_cap,
        conflict_scope_chunk_pairs_before,
        conflict_scope_chunk_pairs_after,
        conflict_scope_chunk_pair_merges,
        scope_chunk_scans,
        scope_chunk_dedupe,
        scope_chunk_saturated,
        scope_budget_start,
        scope_budget_final,
        scope_budget_escalations,
        scope_budget_early,
        degree_pruned_pairs,
        batch_start_index,
    ) = if collect_stats {
        let mut conflict_indptr = Vec::with_capacity(end_idx - start_idx + 1);
        conflict_indptr.push(0);
        let mut conflict_indices: Vec<i64> = Vec::new();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); end_idx - start_idx];
        for (u, v) in conflict_edges.iter() {
            if *u < adjacency.len() && *v < adjacency.len() {
                adjacency[*u].push(*v);
            }
        }
        let mut degree_pruned_pairs = 0i64;
        if let Some(cap) = conflict_degree_cap {
            for neigh in adjacency.iter_mut() {
                if neigh.len() > cap {
                    let removed = neigh.len() - cap;
                    degree_pruned_pairs += removed as i64;
                    neigh.truncate(cap);
                }
            }
        }
        for neigh in adjacency.iter() {
            conflict_indices.extend(neigh.iter().map(|&x| x as i64));
            conflict_indptr.push(conflict_indices.len() as i64);
        }

        let parents: Vec<i64> = tree.parents[start_idx..end_idx].iter().copied().collect();
        let levels: Vec<i64> = tree.levels[start_idx..end_idx]
            .iter()
            .map(|&l| l as i64)
            .collect();

        let selected: Vec<i64> = selected_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &is_sel)| if is_sel { Some(i as i64) } else { None })
            .collect();
        let dominated: Vec<i64> = dominated_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &is_dom)| if is_dom { Some(i as i64) } else { None })
            .collect();

        let (
            scope_indptr,
            scope_indices,
            scope_chunk_segments,
            scope_chunk_emitted,
            scope_chunk_max_members,
            scope_chunk_points,
            conflict_scope_chunk_pair_cap,
            conflict_scope_chunk_pairs_before,
            conflict_scope_chunk_pairs_after,
            conflict_scope_chunk_pair_merges,
            scope_chunk_scans,
            scope_chunk_dedupe,
            scope_chunk_saturated,
            scope_budget_start,
            scope_budget_final,
            scope_budget_escalations,
            scope_budget_early,
        ) = build_scopes_from_edges(
            batch_len,
            &conflict_edges,
            Some(chunk_target),
            scope_budget_schedule,
            scope_budget_up,
            scope_budget_down,
            masked_scope_append.unwrap_or(true),
            scope_chunk_max_segments,
            scope_chunk_pair_merge.unwrap_or(true),
        );

        (
            conflict_indptr,
            conflict_indices,
            parents,
            levels,
            selected,
            dominated,
            scope_indptr,
            scope_indices,
            scope_chunk_segments,
            scope_chunk_emitted,
            scope_chunk_max_members,
            scope_chunk_points,
            conflict_scope_chunk_pair_cap,
            conflict_scope_chunk_pairs_before,
            conflict_scope_chunk_pairs_after,
            conflict_scope_chunk_pair_merges,
            scope_chunk_scans,
            scope_chunk_dedupe,
            scope_chunk_saturated,
            scope_budget_start,
            scope_budget_final,
            scope_budget_escalations,
            scope_budget_early,
            degree_pruned_pairs,
            start_idx as i64,
        )
    } else {
        (
            vec![0i64; 1],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![0i64; 1],
            Vec::new(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0, // conflict_scope_chunk_pair_merges
            0, // scope_chunk_scans
            0, // scope_chunk_dedupe
            0, // scope_chunk_saturated
            0, // scope_budget_start
            0, // scope_budget_final
            0, // scope_budget_escalations
            0, // scope_budget_early
            0, // degree_pruned_pairs
            0, // batch_start_index
        )
    };

    BatchInsertTelemetry {
        parents,
        levels,
        selected,
        dominated,
        conflict_indptr,
        conflict_indices,
        scope_indptr,
        scope_indices: scope_indices.clone(),
        scope_chunk_segments,
        scope_chunk_emitted,
        scope_chunk_max_members,
        scope_chunk_points,
        conflict_scope_chunk_pair_cap,
        conflict_scope_chunk_pairs_before,
        conflict_scope_chunk_pairs_after,
        conflict_scope_chunk_pair_merges,
        scope_chunk_scans,
        scope_chunk_dedupe,
        scope_chunk_saturated,
        scope_budget_start,
        scope_budget_final,
        scope_budget_escalations,
        scope_budget_early_terminate: scope_budget_early,
        batch_start_index,
        conflict_grid_cells,
        conflict_grid_leaders_raw,
        conflict_grid_leaders_after: if conflict_grid_leaders_after > 0 {
            conflict_grid_leaders_after
        } else if conflict_grid_cells > 0 {
            conflict_grid_cells
        } else {
            batch_len as i64
        },
        conflict_grid_local_edges,
        degree_cap: conflict_degree_cap.map(|v| v as i64).unwrap_or(0),
        degree_pruned_pairs,
        traversal_seconds: build_start.elapsed().as_secs_f64(),
        conflict_graph_seconds,
        mis_seconds,
    }
}
