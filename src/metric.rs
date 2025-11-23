use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumCast};
use std::fmt::Debug;
use wide::{f32x8, f64x4};

pub trait Metric<T>: Sync + Send {
    fn distance(&self, p1: &[T], p2: &[T]) -> T;
    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T;

    /// Optional upper bound on the metric distance.
    ///
    /// If provided, callers can bypass expensive distance calculations when
    /// the current radius is larger than this bound (e.g., residual
    /// correlation is capped by √2). Default is `None`.
    fn max_distance_hint(&self) -> Option<T> {
        None
    }
}

#[derive(Copy, Clone)]
pub struct Euclidean;

impl<T> Metric<T> for Euclidean
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
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
pub struct ResidualMetric<'a, T> {
    pub v_matrix: ArrayView2<'a, T>,
    pub p_diag: &'a [T],
    pub rbf_var: T,
    pub scaled_coords: Array2<T>,
    pub scaled_norms: Vec<T>,
    pub neg_half: T,
}

impl<'a, T> ResidualMetric<'a, T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    pub fn new(
        v_matrix: ArrayView2<'a, T>,
        p_diag: &'a [T],
        coords: ArrayView2<'a, T>,
        rbf_var: T,
        rbf_ls: &'a [T],
    ) -> Self {
        let dim = coords.ncols();
        let eps = T::from(1e-6).unwrap();
        let fallback_ls = *rbf_ls.get(0).unwrap_or(&T::one());
        let mut inv_ls: Vec<T> = Vec::with_capacity(dim);
        for i in 0..dim {
            let raw = *rbf_ls.get(i).unwrap_or(&fallback_ls);
            let safe = if raw.abs() < eps { eps } else { raw };
            inv_ls.push(T::one() / safe);
        }

        let mut scaled_coords = coords.as_standard_layout().to_owned();
        for mut row in scaled_coords.outer_iter_mut() {
            for (val, inv) in row.iter_mut().zip(inv_ls.iter()) {
                *val = *val * *inv;
            }
        }

        let mut scaled_norms: Vec<T> = Vec::with_capacity(scaled_coords.nrows());
        for row in scaled_coords.outer_iter() {
            let norm: T = row.iter().map(|v| *v * *v).sum();
            scaled_norms.push(norm);
        }

        let neg_half = T::from(-0.5).unwrap();

        ResidualMetric {
            v_matrix,
            p_diag,
            rbf_var,
            scaled_coords,
            scaled_norms,
            neg_half,
        }
    }

    #[inline(always)]
    pub fn distance_sq_idx(&self, idx_1: usize, idx_2: usize) -> T {
        // Coords dot product (small dimension, typically 2-10)
        let x_view = self.scaled_coords.row(idx_1);
        let y_view = self.scaled_coords.row(idx_2);
        let dot_scaled = dot_product_simd(x_view, y_view);

        let two = T::from(2.0).unwrap();
        let mut d2 = self.scaled_norms[idx_1] + self.scaled_norms[idx_2] - two * dot_scaled;
        if d2 < T::zero() {
            d2 = T::zero();
        }

        let k_val = self.rbf_var * (self.neg_half * d2).exp();

        // V-Matrix dot product (hot loop)
        let v1_view = self.v_matrix.row(idx_1);
        let v2_view = self.v_matrix.row(idx_2);
        let dot = dot_product_simd(v1_view, v2_view);

        let denom = (self.p_diag[idx_1] * self.p_diag[idx_2]).sqrt();
        let eps = T::from(1e-9).unwrap();

        if denom < eps {
            return T::one();
        }

        let rho = (k_val - dot) / denom;
        let one = T::one();
        let neg_one = -one;
        let rho_clamped = rho.max(neg_one).min(one);
        one - rho_clamped.abs()
    }

    #[allow(dead_code)]
    pub fn distances_sq_batch_idx(&self, q_idx: usize, p_indices: &[usize]) -> Vec<T> {
        let mut results = Vec::with_capacity(p_indices.len());
        self.distances_sq_batch_idx_into(q_idx, p_indices, &mut results);
        results
    }

    pub fn distances_sq_batch_idx_into(&self, q_idx: usize, p_indices: &[usize], out: &mut Vec<T>) {
        out.clear();
        out.reserve(p_indices.len());
        // Pre-fetch query data
        let q_norm = self.scaled_norms[q_idx];
        let q_diag = self.p_diag[q_idx];
        let two = T::from(2.0).unwrap();
        let one = T::one();
        let neg_one = -one;
        let eps = T::from(1e-9).unwrap();

        let coords_stride = self.scaled_coords.ncols();
        let v_stride = self.v_matrix.ncols();

        // Fast path when both arrays are standard layout: process targets in tiles to amortize overhead.
        if let (Some(coords_flat), Some(v_flat)) = (
            self.scaled_coords.as_slice_memory_order(),
            self.v_matrix.as_slice_memory_order(),
        ) {
            let q_coords = &coords_flat[q_idx * coords_stride..(q_idx + 1) * coords_stride];
            let q_v = &v_flat[q_idx * v_stride..(q_idx + 1) * v_stride];
            const TILE: usize = 64;
            let mut idx = 0;
            while idx < p_indices.len() {
                let end = usize::min(idx + TILE, p_indices.len());
                for &idx_2 in &p_indices[idx..end] {
                    // Coords dot
                    let y = &coords_flat[idx_2 * coords_stride..(idx_2 + 1) * coords_stride];
                    let dot_scaled = dot_product_simd_slice(q_coords, y);

                    let mut d2 = q_norm + self.scaled_norms[idx_2] - two * dot_scaled;
                    if d2 < T::zero() {
                        d2 = T::zero();
                    }

                    let k_val = self.rbf_var * (self.neg_half * d2).exp();

                    // V-matrix dot
                    let v2 = &v_flat[idx_2 * v_stride..(idx_2 + 1) * v_stride];
                    let dot = dot_product_simd_slice(q_v, v2);

                    let denom = (q_diag * self.p_diag[idx_2]).sqrt();

                    if denom < eps {
                        out.push(T::one());
                    } else {
                        let rho = (k_val - dot) / denom;
                        let rho_clamped = rho.max(neg_one).min(one);
                        out.push(one - rho_clamped.abs());
                    }
                }
                idx = end;
            }
            return;
        }

        // Fallback path using ndarray row views
        let x_view = self.scaled_coords.row(q_idx);
        let v1_view = self.v_matrix.row(q_idx);
        for &idx_2 in p_indices {
            // 1. Coords Part (SIMD)
            let y_view = self.scaled_coords.row(idx_2);
            let dot_scaled = dot_product_simd(x_view, y_view);

            let mut d2 = q_norm + self.scaled_norms[idx_2] - two * dot_scaled;
            if d2 < T::zero() {
                d2 = T::zero();
            }

            let k_val = self.rbf_var * (self.neg_half * d2).exp();

            // 2. V-Matrix Part (SIMD)
            let v2_view = self.v_matrix.row(idx_2);
            let dot = dot_product_simd(v1_view, v2_view);

            let denom = (q_diag * self.p_diag[idx_2]).sqrt();

            if denom < eps {
                out.push(T::one());
            } else {
                let rho = (k_val - dot) / denom;
                let rho_clamped = rho.max(neg_one).min(one);
                out.push(one - rho_clamped.abs());
            }
        }
    }

    pub fn distance_idx(&self, idx_1: usize, idx_2: usize) -> T {
        self.distance_sq_idx(idx_1, idx_2).sqrt()
    }
}

impl<'a, T> Metric<T> for ResidualMetric<'a, T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a,
{
    fn distance(&self, p1: &[T], p2: &[T]) -> T {
        // Assume points are 1D arrays containing a single value which is the index
        let idx1 = p1[0].to_usize().unwrap();
        let idx2 = p2[0].to_usize().unwrap();
        self.distance_idx(idx1, idx2)
    }

    fn distance_sq(&self, p1: &[T], p2: &[T]) -> T {
        let idx1 = p1[0].to_usize().unwrap();
        let idx2 = p2[0].to_usize().unwrap();
        self.distance_sq_idx(idx1, idx2)
    }

    fn max_distance_hint(&self) -> Option<T> {
        // Residual correlation distance is bounded by sqrt(2).
        Some(T::from(2.0).unwrap().sqrt())
    }
}

#[inline(always)]
fn dot_product_simd<T>(a: ndarray::ArrayView1<T>, b: ndarray::ArrayView1<T>) -> T
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    if let (Some(av), Some(bv)) = (a.as_slice(), b.as_slice()) {
        return dot_product_simd_slice(av, bv);
    }
    // Fallback scalar (non-contiguous)
    let mut dot = T::zero();
    let len = a.len();
    for i in 0..len {
        unsafe {
            dot = dot + *a.uget(i) * *b.uget(i);
        }
    }
    dot
}

#[inline(always)]
fn dot_product_simd_slice<T>(a: &[T], b: &[T]) -> T
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    debug_assert_eq!(a.len(), b.len());
    if std::mem::size_of::<T>() == 4 {
        let avf: &[f32] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let bvf: &[f32] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
        let mut acc = 0.0f32;
        let chunks = avf.len() / 8;
        let tail_start = chunks * 8;
        let mut i = 0;
        while i < tail_start {
            let va = f32x8::from([
                avf[i],
                avf[i + 1],
                avf[i + 2],
                avf[i + 3],
                avf[i + 4],
                avf[i + 5],
                avf[i + 6],
                avf[i + 7],
            ]);
            let vb = f32x8::from([
                bvf[i],
                bvf[i + 1],
                bvf[i + 2],
                bvf[i + 3],
                bvf[i + 4],
                bvf[i + 5],
                bvf[i + 6],
                bvf[i + 7],
            ]);
            acc += (va * vb).reduce_add();
            i += 8;
        }
        for j in tail_start..avf.len() {
            acc += avf[j] * bvf[j];
        }
        return NumCast::from(acc).unwrap();
    }
    if std::mem::size_of::<T>() == 8 {
        let avf: &[f64] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let bvf: &[f64] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, b.len()) };
        let mut acc = 0.0f64;
        let chunks = avf.len() / 4;
        let tail_start = chunks * 4;
        let mut i = 0;
        while i < tail_start {
            let va = f64x4::from([avf[i], avf[i + 1], avf[i + 2], avf[i + 3]]);
            let vb = f64x4::from([bvf[i], bvf[i + 1], bvf[i + 2], bvf[i + 3]]);
            acc += (va * vb).reduce_add();
            i += 4;
        }
        for j in tail_start..avf.len() {
            acc += avf[j] * bvf[j];
        }
        return NumCast::from(acc).unwrap();
    }

    // Fallback scalar
    let mut dot = T::zero();
    for i in 0..a.len() {
        dot = dot + a[i] * b[i];
    }
    dot
}
