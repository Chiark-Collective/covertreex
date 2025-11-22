use crate::algo::batch::mis::compute_mis_greedy;
use crate::metric::Metric;
use crate::tree::CoverTreeData;
use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use rayon::prelude::*;
use std::fmt::Debug;

pub mod mis;

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
        return;
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
        return;
    }

    // 2. Initialize Active Sets
    let root_idx = 0;
    let mut active_sets: Vec<Vec<usize>> = vec![vec![root_idx]; candidates.len()];

    let mut current_level: i32 = tree.max_level - 1;
    let min_level: i32 = tree.min_level;

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

            let conflicts = compute_candidate_conflicts(&far_points, radius_sq, metric);
            // MIS uses f32 for priorities? Or T?
            // compute_mis_greedy takes &[f32].
            // Let's use f32 for priorities regardless of T.
            let priorities = vec![1.0f32; n_far];
            let mis_mask = compute_mis_greedy(n_far, &conflicts, &priorities);

            // Identify Leaders
            let mut leaders = Vec::new();

            for (i, &is_leader) in mis_mask.iter().enumerate() {
                if is_leader {
                    leaders.push(far_candidates[i]);

                    let q_idx = far_candidates[i];
                    tree.set_level(q_idx, current_level);

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
    }
}
