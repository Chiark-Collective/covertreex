use ndarray::ArrayView2;
use num_traits::Float;
use std::fmt::Debug;

pub trait Metric<T>: Sync + Send {
    fn distance(&self, p1: &[T], p2: &[T]) -> T;
    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T;
}

#[derive(Copy, Clone)]
pub struct Euclidean;

impl<T> Metric<T> for Euclidean 
where T: Float + Debug + Send + Sync + std::iter::Sum
{
    fn distance(&self, p1: &[T], p2: &[T]) -> T {
        self.distance_sq(p1, p2).sqrt()
    }

    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T {
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
pub struct ResidualMetric<'a, T> {
    pub v_matrix: ArrayView2<'a, T>,
    pub p_diag: &'a [T],
    pub coords: ArrayView2<'a, T>,
    pub rbf_var: T,
    pub rbf_ls_sq: &'a [T],
}

impl<'a, T> ResidualMetric<'a, T> 
where T: Float + Debug + Send + Sync + std::iter::Sum + 'a
{
    pub fn distance_idx(&self, idx_1: usize, idx_2: usize) -> T {
        let x_view = self.coords.row(idx_1);
        let x = x_view.as_slice().unwrap();
        
        let y_view = self.coords.row(idx_2);
        let y = y_view.as_slice().unwrap();
        
        let d2: T = x.iter()
            .zip(y.iter())
            .zip(self.rbf_ls_sq.iter())
            .map(|((&xi, &yi), &ls_sq)| {
                let diff = xi - yi;
                (diff * diff) / ls_sq
            })
            .sum();
        
        let _two = T::from(2.0).unwrap();
        let neg_half = T::from(-0.5).unwrap();
        let k_val = self.rbf_var * (neg_half * d2).exp();
        
        let v1_view = self.v_matrix.row(idx_1);
        let v1 = v1_view.as_slice().unwrap();
        
        let v2_view = self.v_matrix.row(idx_2);
        let v2 = v2_view.as_slice().unwrap();
        
        let dot: T = v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        
        let denom = (self.p_diag[idx_1] * self.p_diag[idx_2]).sqrt();
        let eps = T::from(1e-9).unwrap();
        
        if denom < eps {
            return T::one(); 
        }
        
        let rho = (k_val - dot) / denom;
        let one = T::one();
        let neg_one = -one;
        let rho_clamped = rho.max(neg_one).min(one);
        (one - rho_clamped.abs()).sqrt()
    }
}

impl<'a, T> Metric<T> for ResidualMetric<'a, T> 
where T: Float + Debug + Send + Sync + std::iter::Sum + 'a
{
    fn distance(&self, p1: &[T], p2: &[T]) -> T {
        // Assume points are 1D arrays containing a single value which is the index
        let idx1 = p1[0].to_usize().unwrap();
        let idx2 = p2[0].to_usize().unwrap();
        self.distance_idx(idx1, idx2)
    }
    
    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T {
        let d = self.distance(p1, p2);
        d * d
    }
}