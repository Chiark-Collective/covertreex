#[derive(Debug, Default, Clone)]
pub struct ResidualQueryTelemetry {
    pub frontier_levels: Vec<usize>,
    pub frontier_expanded: Vec<usize>,
    pub yields: Vec<f32>,
    pub caps_applied: usize,
    pub prunes_lower_bound: usize,
    pub prunes_lower_bound_chunks: usize,
    pub prunes_cap: usize,
    pub masked_dedup: usize,
    pub distance_evals: usize,
    pub budget_escalations: usize,
    pub budget_early_terminate: usize,
    pub level_cache_hits: usize,
    pub level_cache_misses: usize,
    pub block_sizes: Vec<usize>,
    pub predecessor_filtered: usize,
    pub subtrees_pruned: usize,
}

impl ResidualQueryTelemetry {
    pub fn record_frontier(&mut self, level_size: usize, expanded: usize) {
        self.frontier_levels.push(level_size);
        self.frontier_expanded.push(expanded);
    }

    pub fn record_yield(&mut self, yield_ratio: f32) {
        self.yields.push(yield_ratio);
    }

    pub fn record_block(&mut self, block: usize) {
        self.block_sizes.push(block);
    }

    pub fn clone_empty(&self) -> ResidualQueryTelemetry {
        ResidualQueryTelemetry::default()
    }

    pub fn add_from(&mut self, other: &ResidualQueryTelemetry) {
        self.frontier_levels
            .extend_from_slice(other.frontier_levels.as_slice());
        self.frontier_expanded
            .extend_from_slice(other.frontier_expanded.as_slice());
        self.yields.extend_from_slice(other.yields.as_slice());
        self.caps_applied += other.caps_applied;
        self.prunes_lower_bound += other.prunes_lower_bound;
        self.prunes_lower_bound_chunks += other.prunes_lower_bound_chunks;
        self.prunes_cap += other.prunes_cap;
        self.masked_dedup += other.masked_dedup;
        self.distance_evals += other.distance_evals;
        self.budget_escalations += other.budget_escalations;
        self.budget_early_terminate += other.budget_early_terminate;
        self.level_cache_hits += other.level_cache_hits;
        self.level_cache_misses += other.level_cache_misses;
        self.block_sizes
            .extend_from_slice(other.block_sizes.as_slice());
        self.predecessor_filtered += other.predecessor_filtered;
        self.subtrees_pruned += other.subtrees_pruned;
    }
}
