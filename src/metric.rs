use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ResidualKernelType {
    RBF = 0,
    Matern52 = 1,
}

impl From<i32> for ResidualKernelType {
    fn from(val: i32) -> Self {
        match val {
            1 => ResidualKernelType::Matern52,
            _ => ResidualKernelType::RBF,
        }
    }
}

// Residual Metric operates on INDICES into the V-Matrix and Coords
pub struct ResidualMetric<'a, T> {
    pub(crate) v_matrix: ArrayView2<'a, T>,
    pub(crate) p_diag: &'a [T],
    pub(crate) rbf_var: T,
    pub(crate) kernel_type: ResidualKernelType,
    pub(crate) scaled_coords: Array2<T>,
    pub(crate) scaled_norms: Vec<T>,
    pub(crate) v_norms: Vec<T>,
    pub(crate) neg_half: T,
    pub(crate) cap_default: T,
    pub(crate) disable_fast_paths: bool,
    pub(crate) parity_mode: bool,
    pub(crate) use_f32_math: bool,
    pub(crate) scaled_coords_f32: Option<Array2<f32>>,
    pub(crate) scaled_norms_f32: Option<Vec<f32>>,
    pub(crate) v_matrix_f32: Option<Array2<f32>>,
    pub(crate) p_diag_f32: Option<Vec<f32>>,
    pub(crate) v_norms_f32: Option<Vec<f32>>,
}

impl<'a, T> ResidualMetric<'a, T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a + 'static,
{
    pub fn new(
        v_matrix: ArrayView2<'a, T>,
        p_diag: &'a [T],
        coords: ArrayView2<'a, T>,
        rbf_var: T,
        rbf_ls: &'a [T],
        kernel_type_int: i32,
        cap_default: Option<T>,
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

        let mut v_norms: Vec<T> = Vec::with_capacity(v_matrix.nrows());
        for row in v_matrix.outer_iter() {
            let norm_sq: T = row.iter().map(|v| *v * *v).sum();
            v_norms.push(norm_sq.sqrt());
        }

        let parity_mode = std::env::var("COVERTREEX_RESIDUAL_PARITY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let disable_fast_paths = parity_mode
            || std::env::var("COVERTREEX_RESIDUAL_DISABLE_FAST_PATHS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        let force_f32_math = std::env::var("COVERTREEX_RESIDUAL_FORCE_F32_MATH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        
        let use_f32_math =
            std::mem::size_of::<T>() == 8 && (parity_mode || force_f32_math) || std::mem::size_of::<T>() == 4;

        let (scaled_coords_f32, scaled_norms_f32, v_matrix_f32, p_diag_f32, v_norms_f32) =
            if use_f32_math {
                let coords_f32 = scaled_coords.mapv(|v| v.to_f32().unwrap());
                let norms_f32 = scaled_norms.iter().map(|v| v.to_f32().unwrap()).collect();
                let v_mat_f32 = v_matrix.mapv(|v| v.to_f32().unwrap());
                let p_diag_f32: Vec<f32> = p_diag.iter().map(|v| v.to_f32().unwrap()).collect();
                let v_norms_f32: Vec<f32> = v_norms.iter().map(|v| v.to_f32().unwrap()).collect();
                (
                    Some(coords_f32),
                    Some(norms_f32),
                    Some(v_mat_f32),
                    Some(p_diag_f32),
                    Some(v_norms_f32),
                )
            } else {
                (None, None, None, None, None)
            };

        let neg_half = T::from(-0.5).unwrap();
        let cap_default = cap_default.unwrap_or_else(|| T::from(2.0).unwrap());
        let kernel_type = ResidualKernelType::from(kernel_type_int);

        ResidualMetric {
            v_matrix,
            p_diag,
            rbf_var,
            kernel_type,
            scaled_coords,
            scaled_norms,
            v_norms,
            neg_half,
            cap_default,
            disable_fast_paths,
            parity_mode,
            use_f32_math,
            scaled_coords_f32,
            scaled_norms_f32,
            v_matrix_f32,
            p_diag_f32,
            v_norms_f32,
        }
    }

    #[inline(always)]
    fn compute_kernel_value(&self, d2: T) -> T {
        match self.kernel_type {
            ResidualKernelType::RBF => {
                self.rbf_var * (self.neg_half * d2).exp()
            }
            ResidualKernelType::Matern52 => {
                 // d2 is squared distance in scaled space.
                 // r = sqrt(d2)
                 // a = sqrt(5) * r
                 // K = var * (1 + a + a^2/3) * exp(-a)
                 let r = if d2 > T::zero() { d2.sqrt() } else { T::zero() };
                 let sqrt5 = T::from(2.2360679775).unwrap();
                 let a = sqrt5 * r;
                 let one = T::one();
                 let three = T::from(3.0).unwrap();
                 let term_poly = one + a + (a * a) / three;
                 self.rbf_var * term_poly * (-a).exp()
            }
        }
    }

    #[inline(always)]
    pub fn apply_level_cap(&self, level: i32, caps: Option<&HashMap<i32, T>>, radius: T) -> T {
        if let Some(map) = caps {
            if let Some(&cap) = map.get(&level) {
                return if cap < radius { cap } else { radius };
            }
        }
        let default_cap = self.cap_default;
        if default_cap > T::zero() && default_cap < radius {
            return default_cap;
        }
        radius
    }

    #[inline(always)]
    pub fn distance_sq_idx(&self, idx_1: usize, idx_2: usize) -> T {
        if self.use_f32_math {
            return self.distance_sq_idx_f32(idx_1, idx_2);
        }
        // Coords dot product (small dimension, typically 2-10)
        let x_view = self.scaled_coords.row(idx_1);
        let y_view = self.scaled_coords.row(idx_2);
        let dot_scaled = if self.parity_mode {
            parity_dot_f32(x_view, y_view).unwrap_or_else(|| x_view.dot(&y_view))
        } else {
            dot_product_simd(x_view, y_view)
        };

        let two = T::from(2.0).unwrap();
        let mut d2 = self.scaled_norms[idx_1] + self.scaled_norms[idx_2] - two * dot_scaled;
        if d2 < T::zero() {
            d2 = T::zero();
        }

        let k_val = self.compute_kernel_value(d2);

        // V-Matrix dot product (hot loop)
        let v1_view = self.v_matrix.row(idx_1);
        let v2_view = self.v_matrix.row(idx_2);
        let dot = if self.parity_mode {
            parity_dot_f32(v1_view, v2_view).unwrap_or_else(|| v1_view.dot(&v2_view))
        } else {
            dot_product_simd(v1_view, v2_view)
        };

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
        self.distances_sq_batch_idx_into_with_kth(q_idx, p_indices, None, &mut results);
        results
    }

    #[allow(dead_code)]
    pub fn distances_sq_batch_idx_into(&self, q_idx: usize, p_indices: &[usize], out: &mut Vec<T>) {
        self.distances_sq_batch_idx_into_with_kth(q_idx, p_indices, None, out);
    }

    // NOTE: Despite the name, this returns the residual **distance** (sqrt of
    // 1 - |rho|) to align with the Python/Numba implementation. The interface
    // is kept to avoid broader refactors while preserving parity.
    pub fn distances_sq_batch_idx_into_with_kth(
        &self,
        q_idx: usize,
        p_indices: &[usize],
        kth: Option<T>,
        out: &mut Vec<T>,
    ) {
        out.clear();
        out.reserve(p_indices.len());
        if self.use_f32_math {
            self.distances_f32_batch_idx_into_with_kth(q_idx, p_indices, kth, out);
            return;
        }
        if self.disable_fast_paths {
            self.distances_sq_batch_idx_into_with_kth_fallback(q_idx, p_indices, kth, out);
            return;
        }
        // Fallback to default implementation for f64 or other types
        self.distances_sq_batch_idx_into_with_kth_fallback(q_idx, p_indices, kth, out);
    }

    fn distances_sq_batch_idx_into_with_kth_fallback(
        &self,
        q_idx: usize,
        p_indices: &[usize],
        kth: Option<T>,
        out: &mut Vec<T>,
    ) {
        let q_norm = self.scaled_norms[q_idx];
        let two = T::from(2.0).unwrap();
        let one = T::one();
        let neg_one = -one;
        let eps = T::from(1e-9).unwrap();
        let prune_sentinel = T::max_value();
        let kth_cutoff = kth.unwrap_or(prune_sentinel);

        let x_view = self.scaled_coords.row(q_idx);
        let v1_view = self.v_matrix.row(q_idx);
        for &idx_2 in p_indices {
            if q_idx == idx_2 {
                out.push(T::zero());
                continue;
            }

            // 1. Coords Part (SIMD)
            let y_view = self.scaled_coords.row(idx_2);
            let dot_scaled = if self.parity_mode {
                parity_dot_f32(x_view, y_view).unwrap_or_else(|| x_view.dot(&y_view))
            } else {
                dot_product_simd(x_view, y_view)
            };

            let mut d2 = q_norm + self.scaled_norms[idx_2] - two * dot_scaled;
            if d2 < T::zero() {
                d2 = T::zero();
            }

            let k_val = self.compute_kernel_value(d2);

            let denom = (self.p_diag[q_idx] * self.p_diag[idx_2]).sqrt();

            if denom < eps {
                out.push(one);
                continue;
            }

            if kth.is_some() {
                let cap = self.v_norms[q_idx] * self.v_norms[idx_2];
                let max_abs_rho = ((k_val - cap).abs()).max((k_val + cap).abs()) / denom;
                let min_dist_sq = one - max_abs_rho.min(one);
                let min_dist = if min_dist_sq > T::zero() {
                    min_dist_sq.sqrt()
                } else {
                    T::zero()
                };
                if min_dist > kth_cutoff {
                    out.push(prune_sentinel);
                    continue;
                }
            }

            // 2. V-Matrix Part (SIMD)
            let v2_view = self.v_matrix.row(idx_2);
            let dot = if self.parity_mode {
                parity_dot_f32(v1_view, v2_view).unwrap_or_else(|| v1_view.dot(&v2_view))
            } else {
                dot_product_simd(v1_view, v2_view)
            };

            let rho = (k_val - dot) / denom;
            let rho_clamped = rho.max(neg_one).min(one);
            let dist_sq = one - rho_clamped.abs();
            let dist = if dist_sq > T::zero() {
                dist_sq.sqrt()
            } else {
                T::zero()
            };
            out.push(dist);
        }
    }

    fn distances_f32_batch_idx_into_with_kth(
        &self,
        q_idx: usize,
        p_indices: &[usize],
        kth: Option<T>,
        out: &mut Vec<T>,
    ) {
        let coords_arr = self
            .scaled_coords_f32
            .as_ref()
            .expect("f32 coords requested but not initialised");
        let v_mat_arr = self
            .v_matrix_f32
            .as_ref()
            .expect("f32 v_matrix requested but not initialised");
        let p_diag = self
            .p_diag_f32
            .as_ref()
            .expect("f32 p_diag requested but not initialised");
        let scaled_norms = self
            .scaled_norms_f32
            .as_ref()
            .expect("f32 norms requested but not initialised");
        // v_norms available but V-norm pruning found ineffective (see note below)
        let _v_norms = &self.v_norms_f32;

        // Use flat slice for faster unchecked access
        let coords = coords_arr.as_slice_memory_order().unwrap();
        let v_mat = v_mat_arr.as_slice_memory_order().unwrap();

        let dim_c = coords_arr.ncols();
        let dim_v = v_mat_arr.ncols();

        let q_norm = scaled_norms[q_idx];
        let q_diag = p_diag[q_idx];
        let two = 2.0f32;
        let one = 1.0f32;
        let neg_one = -one;
        let eps = 1e-9f32;

        let rbf_var = self.rbf_var.to_f32().unwrap();
        let neg_half = self.neg_half.to_f32().unwrap();
        let sqrt5 = 2.2360679775f32;
        let inv_three = 0.3333333333f32;

        // Pruning threshold (kept for potential future use)
        let kth_f32 = kth.map(|k| k.to_f32().unwrap());

        let kernel_id = match self.kernel_type {
             ResidualKernelType::RBF => 0,
             ResidualKernelType::Matern52 => 1,
        };

        // Pre-fetch query vectors
        let q_coords = &coords[q_idx * dim_c..(q_idx + 1) * dim_c];
        let q_v = &v_mat[q_idx * dim_v..(q_idx + 1) * dim_v];

        for &idx_2 in p_indices {
            if q_idx == idx_2 {
                out.push(NumCast::from(0.0).unwrap());
                continue;
            }

            // Coords Dot (Optimized for small D)
            let mut dot_c = 0.0f32;
            let p_coords = &coords[idx_2 * dim_c..(idx_2 + 1) * dim_c];

            if dim_c == 3 {
                // Unroll for D=3
                dot_c = q_coords[0] * p_coords[0] + q_coords[1] * p_coords[1] + q_coords[2] * p_coords[2];
            } else {
                // Fallback loop
                for d in 0..dim_c {
                    dot_c += q_coords[d] * p_coords[d];
                }
            }

            let mut d2 = q_norm + scaled_norms[idx_2] - two * dot_c;
            if d2 < 0.0 { d2 = 0.0; }

            let k_val = if kernel_id == 0 {
                // RBF
                rbf_var * (neg_half * d2).exp()
            } else {
                // Matern 5/2
                let r = d2.sqrt();
                let a = sqrt5 * r;
                let term_poly = one + a + (a * a) * inv_three;
                rbf_var * term_poly * (-a).exp()
            };

            let p_d = p_diag[idx_2];
            let denom = (q_diag * p_d).max(eps * eps).sqrt();

            // NOTE: V-norm pruning was evaluated but found ineffective for this metric.
            // The denominator sqrt(P[q]·P[p]) is typically ~10^-6, causing rho bounds
            // to explode and always clamp to ±1, giving min_dist=0 (0% prune rate).
            // See opt/v-norm-pruning branch for analysis. The kth parameter is kept
            // for potential future use with different metric configurations.
            let _ = kth_f32; // Suppress unused warning

            // V Dot (SIMD f32x8) - only computed if not pruned
            let p_v = &v_mat[idx_2 * dim_v..(idx_2 + 1) * dim_v];
            let dot_v = dot_f32_simd_inline(q_v, p_v);

            let rho = (k_val - dot_v) / denom;
            let rho_clamped = rho.max(neg_one).min(one);
            let dist_sq = one - rho_clamped.abs();
            let dist = if dist_sq > 0.0 { dist_sq.sqrt() } else { 0.0 };
            out.push(NumCast::from(dist).unwrap());
        }
    }

    fn distance_sq_idx_f32(&self, idx_1: usize, idx_2: usize) -> T {
        if idx_1 == idx_2 {
            return NumCast::from(0.0).unwrap();
        }

        let coords_arr = self
            .scaled_coords_f32
            .as_ref()
            .expect("f32 coords requested but not initialised");
        let v_mat_arr = self
            .v_matrix_f32
            .as_ref()
            .expect("f32 v_matrix requested but not initialised");
        let p_diag = self
            .p_diag_f32
            .as_ref()
            .expect("f32 p_diag requested but not initialised");
        let scaled_norms = self
            .scaled_norms_f32
            .as_ref()
            .expect("f32 norms requested but not initialised");

        let two = 2.0f32;
        let one = 1.0f32;
        let neg_one = -one;
        let eps = 1e-9f32;

        let x_view = coords_arr.row(idx_1);
        let y_view = coords_arr.row(idx_2);
        
        let dot_scaled = if self.parity_mode {
            parity_dot_f32(x_view, y_view)
                .unwrap_or_else(|| x_view.dot(&y_view))
        } else {
            // Use slice dot
            dot_product_simd(x_view, y_view)
        };
        
        let mut d2 = scaled_norms[idx_1] + scaled_norms[idx_2] - two * dot_scaled;
        if d2 < 0.0 {
            d2 = 0.0;
        }
        
        let k_val = match self.kernel_type {
            ResidualKernelType::RBF => {
                self.rbf_var.to_f32().unwrap() * (self.neg_half.to_f32().unwrap() * d2).exp()
            }
            ResidualKernelType::Matern52 => {
                let r = d2.sqrt();
                let sqrt5 = 2.2360679775f32;
                let a = sqrt5 * r;
                let term_poly = 1.0 + a + (a * a) / 3.0;
                self.rbf_var.to_f32().unwrap() * term_poly * (-a).exp()
            }
        };

        let denom_sq = (p_diag[idx_1] * p_diag[idx_2]).max(eps * eps);
        let denom = denom_sq.sqrt();

        let v1_view = v_mat_arr.row(idx_1);
        let v2_view = v_mat_arr.row(idx_2);
        
        let dot = if self.parity_mode {
            parity_dot_f32(v1_view, v2_view)
                .unwrap_or_else(|| v1_view.dot(&v2_view))
        } else {
            dot_product_simd(v1_view, v2_view)
        };
        
        let rho = if denom > 0.0 { (k_val - dot) / denom } else { 0.0 };
        let rho_clamped = rho.max(neg_one).min(one);
        let dist_sq = one - rho_clamped.abs();
        let dist_sq = if dist_sq > 0.0 { dist_sq } else { 0.0 };
        NumCast::from(dist_sq).unwrap()
    }

    pub fn distance_idx(&self, idx_1: usize, idx_2: usize) -> T {
        self.distance_sq_idx(idx_1, idx_2).sqrt()
    }
}

impl<'a, T> Metric<T> for ResidualMetric<'a, T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + 'a + 'static,
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
        // Residual correlation distance is bounded by 1 (sqrt(1 - |rho|), |rho|<=1).
        Some(T::one())
    }
}

/// Optimized f32 dot product using SIMD (AVX2 f32x8).
/// Uses 2x loop unrolling for better instruction-level parallelism.
#[inline(always)]
pub(crate) fn dot_f32_simd_inline(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let tail_start = chunks * 8;

    let mut acc0 = f32x8::ZERO;
    let mut acc1 = f32x8::ZERO;

    // Process 16 elements per iteration (2x unroll for better ILP)
    let double_chunks = chunks / 2;
    let mut i = 0;
    for _ in 0..double_chunks {
        // First 8 elements
        let va0 = f32x8::from(unsafe { *(a.as_ptr().add(i) as *const [f32; 8]) });
        let vb0 = f32x8::from(unsafe { *(b.as_ptr().add(i) as *const [f32; 8]) });
        acc0 = va0.mul_add(vb0, acc0);

        // Second 8 elements
        let va1 = f32x8::from(unsafe { *(a.as_ptr().add(i + 8) as *const [f32; 8]) });
        let vb1 = f32x8::from(unsafe { *(b.as_ptr().add(i + 8) as *const [f32; 8]) });
        acc1 = va1.mul_add(vb1, acc1);

        i += 16;
    }

    // Handle remaining complete 8-element chunk
    if chunks % 2 == 1 {
        let va = f32x8::from(unsafe { *(a.as_ptr().add(i) as *const [f32; 8]) });
        let vb = f32x8::from(unsafe { *(b.as_ptr().add(i) as *const [f32; 8]) });
        acc0 = va.mul_add(vb, acc0);
    }

    // Combine accumulators and reduce
    let combined = acc0 + acc1;
    let mut result = combined.reduce_add();

    // Handle tail elements
    for j in tail_start..len {
        result += a[j] * b[j];
    }

    result
}

#[inline(always)]
pub(crate) fn dot_product_simd<T>(a: ndarray::ArrayView1<T>, b: ndarray::ArrayView1<T>) -> T
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
pub(crate) fn dot_parity_f32(a: ndarray::ArrayView1<f32>, b: ndarray::ArrayView1<f32>) -> f32 {
    // Use f64 accumulation to minimize rounding errors, hoping to align better with BLAS
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum::<f64>() as f32
}

#[inline(always)]
pub(crate) fn parity_dot_f32<T>(a: ndarray::ArrayView1<T>, b: ndarray::ArrayView1<T>) -> Option<T>
where
    T: Float + Debug + Send + Sync + std::iter::Sum + NumCast,
{
    if std::mem::size_of::<T>() != 4 {
        return None;
    }
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        // Safe because we only reinterpret f32 data.
        let a_f = unsafe {
            std::slice::from_raw_parts(a_slice.as_ptr() as *const f32, a_slice.len())
        };
        let b_f = unsafe {
            std::slice::from_raw_parts(b_slice.as_ptr() as *const f32, b_slice.len())
        };
        let dot = dot_parity_f32(ndarray::ArrayView1::from(a_f), ndarray::ArrayView1::from(b_f));
        return NumCast::from(dot);
    }
    None
}

#[inline(always)]
pub(crate) fn dot_product_simd_slice<T>(a: &[T], b: &[T]) -> T
where
    T: Float + Debug + Send + Sync + std::iter::Sum,
{
    debug_assert_eq!(a.len(), b.len());
    if std::mem::size_of::<T>() == 4 {
        let avf: &[f32] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let bvf: &[f32] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
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
        let avf: &[f64] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let bvf: &[f64] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, b.len()) };
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

#[inline(always)]
pub(crate) fn dot_tile_f32(
    q: &[f32],
    matrix_flat: &[f32],
    stride: usize,
    indices: &[usize],
    out: &mut [f32; 64],
) {
    let tile_len = indices.len();
    for k in 0..tile_len {
        out[k] = 0.0;
    }
    let chunks = stride / 8;
    let tail_start = chunks * 8;
    for c in 0..chunks {
        let base = c * 8;
        let qv = f32x8::from([
            q[base],
            q[base + 1],
            q[base + 2],
            q[base + 3],
            q[base + 4],
            q[base + 5],
            q[base + 6],
            q[base + 7],
        ]);
        for (pos, &idx) in indices.iter().enumerate() {
            let row = &matrix_flat[idx * stride + base..idx * stride + base + 8];
            let rv = f32x8::from([
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
            ]);
            out[pos] += (qv * rv).reduce_add();
        }
    }
    for d in tail_start..stride {
        let qd = q[d];
        for (pos, &idx) in indices.iter().enumerate() {
            out[pos] += qd * matrix_flat[idx * stride + d];
        }
    }
}

#[inline(always)]
pub(crate) fn dot_tile_f64(
    q: &[f64],
    matrix_flat: &[f64],
    stride: usize,
    indices: &[usize],
    out: &mut [f64; 64],
) {
    let tile_len = indices.len();
    for k in 0..tile_len {
        out[k] = 0.0;
    }
    let chunks = stride / 4;
    let tail_start = chunks * 4;
    for c in 0..chunks {
        let base = c * 4;
        let qv = f64x4::from([q[base], q[base + 1], q[base + 2], q[base + 3]]);
        for (pos, &idx) in indices.iter().enumerate() {
            let row = &matrix_flat[idx * stride + base..idx * stride + base + 4];
            let rv = f64x4::from([row[0], row[1], row[2], row[3]]);
            out[pos] += (qv * rv).reduce_add();
        }
    }
    for d in tail_start..stride {
        let qd = q[d];
        for (pos, &idx) in indices.iter().enumerate() {
            out[pos] += qd * matrix_flat[idx * stride + d];
        }
    }
}
