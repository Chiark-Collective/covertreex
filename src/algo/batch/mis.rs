use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Simple greedy MIS on the conflict graph.
/// 
/// Inputs:
/// - adjacency: Vec of (u, v) edges representing conflicts between queries.
/// - priorities: Score for each query (higher = better leader).
/// 
/// Output:
/// - Boolean mask (true = selected as center).
pub fn compute_mis_greedy(
    num_queries: usize,
    adjacency: &[(usize, usize)], // Edges between queries
    priorities: &[f32],
) -> Vec<bool> {
    // Python impl uses Luby's algorithm or Greedy.
    // Greedy (Serial) is O(E).
    // Parallel Luby is O(log N).
    // For CPU, simple parallel greedy with "peaks" is often good enough.
    
    // Let's implement a "peaks" approach:
    // 1. A node is a local peak if its priority > neighbors.
    // 2. All peaks join MIS.
    // 3. Remove neighbors.
    // 4. Repeat.
    
    // Or simple serial greedy for Phase 3 MVP to ensure correctness.
    // "Rust Serial" is still faster than "Python Serial".
    
    // Let's do Serial Greedy first.
    let mut state = vec![0i8; num_queries]; // 0=unknown, 1=in, -1=out
    
    // Build adjacency list
    let mut adj_list = vec![Vec::new(); num_queries];
    for &(u, v) in adjacency {
        adj_list[u].push(v);
        adj_list[v].push(u);
    }
    
    // Sort queries by priority (descending)
    let mut indices: Vec<usize> = (0..num_queries).collect();
    indices.sort_by(|&a, &b| priorities[b].partial_cmp(&priorities[a]).unwrap());
    
    for &idx in &indices {
        if state[idx] != 0 {
            continue;
        }
        
        // Select idx
        state[idx] = 1;
        
        // Mark neighbors as out
        for &nbr in &adj_list[idx] {
            if state[nbr] == 0 {
                state[nbr] = -1;
            }
        }
    }
    
    state.into_iter().map(|s| s == 1).collect()
}
