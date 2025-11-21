use ndarray::{ArrayView1, ArrayView2};

pub trait Metric: Sync + Send {
    // For standard coord-based metrics
    fn distance(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32;
    fn distance_sq(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32;
}

#[derive(Copy, Clone)]
pub struct Euclidean;

impl Metric for Euclidean {
    fn distance(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32 {
        let s1 = p1.as_slice().unwrap();
        let s2 = p2.as_slice().unwrap();
        let mut sum = 0.0;
        for i in 0..s1.len() {
            let diff = s1[i] - s2[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn distance_sq(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32 {
        let s1 = p1.as_slice().unwrap();
        let s2 = p2.as_slice().unwrap();
        let mut sum = 0.0;
        for i in 0..s1.len() {
            let diff = s1[i] - s2[i];
            sum += diff * diff;
        }
        sum
    }
}

// Residual Metric operates on INDICES into the V-Matrix and Coords
#[derive(Copy, Clone)]
pub struct ResidualMetric<'a> {
    pub v_matrix: ArrayView2<'a, f32>,
    pub p_diag: ArrayView1<'a, f32>,
    pub coords: ArrayView2<'a, f32>,
    pub rbf_var: f32,
    pub rbf_ls_sq: ArrayView1<'a, f32>,
}

impl<'a> ResidualMetric<'a> {
        pub fn distance_idx(&self, idx_1: usize, idx_2: usize) -> f32 {
            // 1. RBF Kernel K(x, y)
            let x = self.coords.row(idx_1);
            let y = self.coords.row(idx_2);
            
            // Weighted Euclidean Sq
            let sx = x.as_slice().unwrap();
            let sy = y.as_slice().unwrap();
            let sl = self.rbf_ls_sq.as_slice().unwrap();
            
            let mut d2 = 0.0;
            for i in 0..sx.len() {
                let diff = sx[i] - sy[i];
                d2 += (diff * diff) / sl[i];
            }
            
            let k_val = self.rbf_var * (-0.5 * d2).exp();
            
            // 2. Dot Product V_i . V_j
            let v1 = self.v_matrix.row(idx_1);
            let v2 = self.v_matrix.row(idx_2);
            
            // Manual SIMD-friendly loop
            let sv1 = v1.as_slice().unwrap();
            let sv2 = v2.as_slice().unwrap();
            let mut dot = 0.0;
            for i in 0..sv1.len() {
                dot += sv1[i] * sv2[i];
            }
            
            // 3. Residual Distance
            let denom = (self.p_diag[idx_1] * self.p_diag[idx_2]).sqrt();
            
            if denom < 1e-9 {
                return 1.0; // Safety
            }
            
            let rho = (k_val - dot) / denom;
            
            // Clamp
            let rho_clamped = rho.max(-1.0).min(1.0);
            (1.0 - rho_clamped.abs()).sqrt()
        }
    }
    
    impl<'a> Metric for ResidualMetric<'a> {
        fn distance(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32 {
            // Assume points are 1D arrays containing a single float which is the index
            let idx1 = p1[0] as usize;
            let idx2 = p2[0] as usize;
            self.distance_idx(idx1, idx2)
        }
        
        fn distance_sq(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32 {
            let d = self.distance(p1, p2);
            d * d
        }
    }
    