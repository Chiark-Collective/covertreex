use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use crate::tree::CoverTreeData;
use crate::metric::Metric;
use crate::algo::batch::mis::compute_mis_greedy;

pub mod mis;

fn compute_candidate_conflicts(
    points: &Array2<f32>, 
    radius: f32, 
    metric: &dyn Metric
) -> Vec<(usize, usize)> {
    let n = points.nrows();
    (0..n).into_par_iter()
        .flat_map_iter(|i| {
            let row_i = points.row(i);
            (i + 1..n).filter_map(move |j| {
                let d = metric.distance(row_i, points.row(j));
                if d <= radius {
                    Some((i, j))
                }
                else {
                    None
                }
            })
        })
        .collect()
}

pub fn batch_insert(
    tree: &mut CoverTreeData,
    batch: ArrayView2<f32>,
    metric: &dyn Metric,
) {
    // 0. Append all points to tree with dummy level

    let start_idx = tree.len();
    for row in batch.outer_iter() {
        tree.add_point(row, i32::MIN, -1);
    }
    let end_idx = tree.len();
    let mut candidates: Vec<usize> = (start_idx..end_idx).collect();
    
    if candidates.is_empty() { return; }

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
    if candidates.is_empty() { return; }

    // 2. Initialize Active Sets
    // For Phase 3, we assume single root 0 is valid cover.
    let root_idx = 0; 
    let mut active_sets: Vec<Vec<usize>> = vec![vec![root_idx]; candidates.len()];
    
    let mut current_level: i32 = tree.max_level - 1;
    let min_level: i32 = tree.min_level;
    
    while current_level >= min_level {
        let radius = 2.0_f32.powi(current_level);
        
        // 3. Filter: Near vs Far
        let filter_results: Vec<(bool, Vec<usize>)> = candidates.par_iter().zip(active_sets.par_iter())
            .map(|(&q_idx, covers)| {
                let q_point = tree.get_point_row(q_idx);
                let mut best_dist = f32::MAX;
                let mut best_node = usize::MAX;
                let mut is_near = false;
                
                for &p_idx in covers {
                    // Check p itself
                    let p_point = tree.get_point_row(p_idx);
                    let d_p = metric.distance(q_point, p_point);
                    if d_p <= radius {
                        if d_p < best_dist {
                            best_dist = d_p;
                            best_node = p_idx;
                        }
                        is_near = true;
                    }
                    
                    // Check children
                    let mut child = tree.children[p_idx];
                    while child != -1 {
                        let c_idx = child as usize;
                        let c_point = tree.get_point_row(c_idx);
                        let d_c = metric.distance(q_point, c_point);
                        if d_c <= radius {
                            if d_c < best_dist {
                                best_dist = d_c;
                                best_node = c_idx;
                            }
                            is_near = true;
                        }
                        
                        let next = tree.next_node[c_idx];
                        if next == child { break; }
                        child = next;
                        if child == tree.children[p_idx] { break; }
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
            let mut far_points = Array2::<f32>::zeros((n_far, tree.dimension));
            for (i, &idx) in far_candidates.iter().enumerate() {
                far_points.row_mut(i).assign(&tree.get_point_row(idx));
            }
            
            let conflicts = compute_candidate_conflicts(&far_points, radius, metric);
            let priorities = vec![1.0; n_far];
            let mis_mask = compute_mis_greedy(n_far, &conflicts, &priorities);
            
            // Identify Leaders
            let mut leaders = Vec::new();
            let mut leader_indices = Vec::new(); // Index in far_candidates
            
            for (i, &is_leader) in mis_mask.iter().enumerate() {
                if is_leader {
                    leaders.push(far_candidates[i]);
                    leader_indices.push(i);
                    
                    let q_idx = far_candidates[i];
                    tree.set_level(q_idx, current_level);
                    
                    let original_idx = far_original_indices[i];
                    let parents = &active_sets[original_idx];
                    if !parents.is_empty() {
                        tree.set_parent(q_idx, parents[0] as i64);
                    }
                }
            }
            
            // Handle Non-Leaders (must be covered by a Leader)
            let leader_points_arr = Array2::from_shape_vec(
                (leaders.len(), tree.dimension),
                leaders.iter().flat_map(|&idx| tree.get_point_row(idx).to_vec()).collect()
            ).unwrap();
            
            // For each non-leader, find covering leader
            // This can be parallelized
            // We need to map back to next_candidates
            
            // Collect non-leaders to process
            let non_leader_indices: Vec<usize> = mis_mask.iter().enumerate()
                .filter(|(_, &is_l)| !is_l)
                .map(|(i, _)| i)
                .collect();
                
            let non_leader_results: Vec<Vec<usize>> = non_leader_indices.par_iter()
                .map(|&i| {
                    let q_idx = far_candidates[i];
                    let q_point = tree.get_point_row(q_idx);
                    let mut covered_by = Vec::new();
                    // Check all leaders
                    for (l_i, &l_idx) in leaders.iter().enumerate() {
                        if metric.distance(q_point, leader_points_arr.row(l_i)) <= radius {
                            covered_by.push(l_idx);
                        }
                    }
                    covered_by
                })
                .collect();
                
            // Push non-leaders to next
            for (k, &i) in non_leader_indices.iter().enumerate() {
                let q_idx = far_candidates[i];
                let covered_by = &non_leader_results[k];
                if !covered_by.is_empty() {
                    next_candidates.push(q_idx);
                    next_active_sets.push(covered_by.clone());
                } else {
                    // Should not happen if MIS/Conflict is correct?
                    // Unless rounding errors?
                    // Force insert or drop? 
                    // For Phase 3, force insert as leaf later?
                    // Or force link to parent?
                    // Let's add to next but keep active set from before?
                    // Re-use parent cover? No, "Far" means not covered by parents.
                    // So it must be covered by new nodes.
                    // If not covered, it's an Orphan.
                    // Panic? Or log.
                    // eprintln!("Orphan node {}", q_idx);
                }
            }
        }
        
        candidates = next_candidates;
        active_sets = next_active_sets;
        current_level -= 1;
    }
    
    // Finalize remaining
    for (i, &q_idx) in candidates.iter().enumerate() {
        tree.set_level(q_idx, min_level);
        if !active_sets[i].is_empty() {
            tree.set_parent(q_idx, active_sets[i][0] as i64);
        }
    }
}
