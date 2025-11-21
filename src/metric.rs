use ndarray::ArrayView2;

pub trait Metric: Sync + Send {
    // Standard coord-based metrics
    fn distance(&self, p1: &[f32], p2: &[f32]) -> f32;
    fn distance_sq(&self, p1: &[f32], p2: &[f32]) -> f32;
}

#[derive(Copy, Clone)]
pub struct Euclidean;

impl Metric for Euclidean {
    fn distance(&self, p1: &[f32], p2: &[f32]) -> f32 {
        self.distance_sq(p1, p2).sqrt()
    }

    fn distance_sq(&self, p1: &[f32], p2: &[f32]) -> f32 {
        p1.iter()
            .zip(p2.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }
}

// Residual Metric operates on INDICES into the V-Matrix and Coords
#[derive(Copy, Clone)]
pub struct ResidualMetric<'a> {
    pub v_matrix: ArrayView2<'a, f32>,
    pub p_diag: &'a [f32], // Changed from ArrayView1 for direct access
    pub coords: ArrayView2<'a, f32>,
    pub rbf_var: f32,
    pub rbf_ls_sq: &'a [f32], // Changed from ArrayView1
}

impl<'a> ResidualMetric<'a> {
    pub fn distance_idx(&self, idx_1: usize, idx_2: usize) -> f32 {
        // 1. RBF Kernel K(x, y)
        let x_view = self.coords.row(idx_1);
        let x = x_view.as_slice().unwrap();
        
        let y_view = self.coords.row(idx_2);
        let y = y_view.as_slice().unwrap();
        
        let d2: f32 = x.iter()
            .zip(y.iter())
            .zip(self.rbf_ls_sq.iter())
            .map(|((&xi, &yi), &ls_sq)| {
                let diff = xi - yi;
                (diff * diff) / ls_sq
            })
            .sum();
        
        let k_val = self.rbf_var * (-0.5 * d2).exp();
        
        // 2. Dot Product V_i . V_j
        let v1_view = self.v_matrix.row(idx_1);
        let v1 = v1_view.as_slice().unwrap();
        
        let v2_view = self.v_matrix.row(idx_2);
        let v2 = v2_view.as_slice().unwrap();
        
        let dot: f32 = v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        
        // 3. Residual Distance
        // p_diag is slice
        let denom = (self.p_diag[idx_1] * self.p_diag[idx_2]).sqrt();
        
        if denom < 1e-9 {
            return 1.0; 
        }
        
        let rho = (k_val - dot) / denom;
        let rho_clamped = rho.max(-1.0).min(1.0);
        (1.0 - rho_clamped.abs()).sqrt()
    }
}

impl<'a> Metric for ResidualMetric<'a> {
    fn distance(&self, p1: &[f32], p2: &[f32]) -> f32 {
        // Assume points are 1D arrays containing a single float which is the index
        let idx1 = p1[0] as usize;
        let idx2 = p2[0] as usize;
        self.distance_idx(idx1, idx2)
    }
    
    fn distance_sq(&self, p1: &[f32], p2: &[f32]) -> f32 {
        let d = self.distance(p1, p2);
        d * d
    }
}
    