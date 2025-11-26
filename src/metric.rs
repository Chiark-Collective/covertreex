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

// Residual Metric operates on INDICES into the V-Matrix and Coords
pub struct ResidualMetric<'a, T> {
    pub(crate) v_matrix: ArrayView2<'a, T>,
    pub(crate) p_diag: &'a [T],
    pub(crate) rbf_var: T,
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
        // Rust f64 accumulation deviates from the float32 baseline used by the Python
        // residual backend. In parity/forced modes, run the metric arithmetic in f32
        // and cast results back to T to match the gold distances.
        let use_f32_math =
            std::mem::size_of::<T>() == 8 && (parity_mode || force_f32_math);

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

        ResidualMetric {
            v_matrix,
            p_diag,
            rbf_var,
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

        let k_val = self.rbf_var * (self.neg_half * d2).exp();

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
        // Pre-fetch query data
        let q_norm = self.scaled_norms[q_idx];
        let q_diag = self.p_diag[q_idx];
        let two = T::from(2.0).unwrap();
        let one = T::one();
        let neg_one = -one;
        let eps = T::from(1e-9).unwrap();
        let prune_sentinel = T::max_value();
        let kth_cutoff = kth.unwrap_or(prune_sentinel);
        let pruning_enabled = kth.is_some();

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
                let size = std::mem::size_of::<T>();
                if !pruning_enabled {
                    let mut dot_coords_f32 = [0f32; TILE];
                    let mut dot_v_f32 = [0f32; TILE];
                    let mut dot_coords_f64 = [0f64; TILE];
                    let mut dot_v_f64 = [0f64; TILE];
                    if size == 4 {
                        let q_coords_f = unsafe {
                            std::slice::from_raw_parts(
                                q_coords.as_ptr() as *const f32,
                                q_coords.len(),
                            )
                        };
                        let coords_f = unsafe {
                            std::slice::from_raw_parts(
                                coords_flat.as_ptr() as *const f32,
                                coords_flat.len(),
                            )
                        };
                        let q_v_f = unsafe {
                            std::slice::from_raw_parts(q_v.as_ptr() as *const f32, q_v.len())
                        };
                        let v_flat_f = unsafe {
                            std::slice::from_raw_parts(v_flat.as_ptr() as *const f32, v_flat.len())
                        };
                        dot_tile_f32(
                            q_coords_f,
                            coords_f,
                            coords_stride,
                            &p_indices[idx..end],
                            &mut dot_coords_f32,
                        );
                        dot_tile_f32(
                            q_v_f,
                            v_flat_f,
                            v_stride,
                            &p_indices[idx..end],
                            &mut dot_v_f32,
                        );
                    } else if size == 8 {
                        let q_coords_f = unsafe {
                            std::slice::from_raw_parts(
                                q_coords.as_ptr() as *const f64,
                                q_coords.len(),
                            )
                        };
                        let coords_f = unsafe {
                            std::slice::from_raw_parts(
                                coords_flat.as_ptr() as *const f64,
                                coords_flat.len(),
                            )
                        };
                        let q_v_f = unsafe {
                            std::slice::from_raw_parts(q_v.as_ptr() as *const f64, q_v.len())
                        };
                        let v_flat_f = unsafe {
                            std::slice::from_raw_parts(v_flat.as_ptr() as *const f64, v_flat.len())
                        };
                        dot_tile_f64(
                            q_coords_f,
                            coords_f,
                            coords_stride,
                            &p_indices[idx..end],
                            &mut dot_coords_f64,
                        );
                        dot_tile_f64(
                            q_v_f,
                            v_flat_f,
                            v_stride,
                            &p_indices[idx..end],
                            &mut dot_v_f64,
                        );
                    }

                    for (local_offset, &idx_2) in p_indices[idx..end].iter().enumerate() {
                        if q_idx == idx_2 {
                            out.push(T::zero());
                            continue;
                        }

                        let dot_scaled = if size == 4 {
                            NumCast::from(dot_coords_f32[local_offset]).unwrap()
                        } else if size == 8 {
                            NumCast::from(dot_coords_f64[local_offset]).unwrap()
                        } else {
                            let y =
                                &coords_flat[idx_2 * coords_stride..(idx_2 + 1) * coords_stride];
                            dot_product_simd_slice(q_coords, y)
                        };

                        let mut d2 = q_norm + self.scaled_norms[idx_2] - two * dot_scaled;
                        if d2 < T::zero() {
                            d2 = T::zero();
                        }

                        let k_val = self.rbf_var * (self.neg_half * d2).exp();

                        let dot = if size == 4 {
                            NumCast::from(dot_v_f32[local_offset]).unwrap()
                        } else if size == 8 {
                            NumCast::from(dot_v_f64[local_offset]).unwrap()
                        } else {
                            let v2 = &v_flat[idx_2 * v_stride..(idx_2 + 1) * v_stride];
                            dot_product_simd_slice(q_v, v2)
                        };

                        let denom = (q_diag * self.p_diag[idx_2]).sqrt();

                        if denom < eps {
                            out.push(one);
                        } else {
                            let rho = (k_val - dot) / denom;
                            let rho_clamped = rho.max(neg_one).min(one);
                            out.push(one - rho_clamped.abs());
                        }
                    }
                    idx = end;
                    continue;
                }

                let mut dot_coords_f32 = [0f32; TILE];
                let mut dot_coords_f64 = [0f64; TILE];
                if size == 4 {
                    let q_coords_f = unsafe {
                        std::slice::from_raw_parts(q_coords.as_ptr() as *const f32, q_coords.len())
                    };
                    let coords_f = unsafe {
                        std::slice::from_raw_parts(
                            coords_flat.as_ptr() as *const f32,
                            coords_flat.len(),
                        )
                    };
                    dot_tile_f32(
                        q_coords_f,
                        coords_f,
                        coords_stride,
                        &p_indices[idx..end],
                        &mut dot_coords_f32,
                    );
                } else if size == 8 {
                    let q_coords_f = unsafe {
                        std::slice::from_raw_parts(q_coords.as_ptr() as *const f64, q_coords.len())
                    };
                    let coords_f = unsafe {
                        std::slice::from_raw_parts(
                            coords_flat.as_ptr() as *const f64,
                            coords_flat.len(),
                        )
                    };
                    dot_tile_f64(
                        q_coords_f,
                        coords_f,
                        coords_stride,
                        &p_indices[idx..end],
                        &mut dot_coords_f64,
                    );
                }

                for (local_offset, &idx_2) in p_indices[idx..end].iter().enumerate() {
                    let dot_scaled = if size == 4 {
                        NumCast::from(dot_coords_f32[local_offset]).unwrap()
                    } else if size == 8 {
                        NumCast::from(dot_coords_f64[local_offset]).unwrap()
                    } else {
                        let y = &coords_flat[idx_2 * coords_stride..(idx_2 + 1) * coords_stride];
                        dot_product_simd_slice(q_coords, y)
                    };

                    let mut d2 = q_norm + self.scaled_norms[idx_2] - two * dot_scaled;
                    if d2 < T::zero() {
                        d2 = T::zero();
                    }

                    let k_val = self.rbf_var * (self.neg_half * d2).exp();

                    let denom = (q_diag * self.p_diag[idx_2]).sqrt();

                    if denom < eps {
                        out.push(one);
                        continue;
                    }

                    if kth_cutoff < prune_sentinel {
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

                    let v2 = &v_flat[idx_2 * v_stride..(idx_2 + 1) * v_stride];
                    let dot = dot_product_simd_slice(q_v, v2);

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
                idx = end;
            }
            return;
        }

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

            let k_val = self.rbf_var * (self.neg_half * d2).exp();

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
        let coords = self
            .scaled_coords_f32
            .as_ref()
            .expect("f32 coords requested but not initialised");
        let v_mat = self
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
        let v_norms = self
            .v_norms_f32
            .as_ref()
            .expect("f32 v_norms requested but not initialised");

        let q_norm = scaled_norms[q_idx];
        let q_diag = p_diag[q_idx];
        let two = 2.0f32;
        let one = 1.0f32;
        let neg_one = -one;
        let eps = 1e-9f32;
        let prune_sentinel = f32::MAX;
        let kth_cutoff: f32 = kth
            .and_then(|v| v.to_f32())
            .unwrap_or(prune_sentinel);

        // SIMD constants
        let v_two = f32x8::splat(2.0);
        let v_one = f32x8::splat(1.0);
        let v_neg_one = f32x8::splat(-1.0);
        let v_zero = f32x8::splat(0.0);
        let v_eps = f32x8::splat(eps);
        let v_q_norm = f32x8::splat(q_norm);
        let v_q_diag = f32x8::splat(q_diag);
        let v_rbf_var = f32x8::splat(self.rbf_var.to_f32().unwrap());
        let v_neg_half = f32x8::splat(self.neg_half.to_f32().unwrap());
        let v_v_norm_q = f32x8::splat(v_norms[q_idx]);

        let coords_stride = coords.ncols();
        let v_stride = v_mat.ncols();
        
        let (coords_flat, v_flat) = match (coords.as_slice_memory_order(), v_mat.as_slice_memory_order()) {
            (Some(c), Some(v)) => (c, v),
            _ => {
                (coords.as_slice().unwrap(), v_mat.as_slice().unwrap())
            }
        };

        out.clear();
        out.reserve(p_indices.len());

        let mut dot_coords_f32 = [0f32; 64];
        let mut dot_v_f32 = [0f32; 64];
        
        let mut idx = 0;
        while idx < p_indices.len() {
            let end = usize::min(idx + 64, p_indices.len());
            let chunk_len = end - idx;
            
            // 1. Compute Dot Products (Batch 64)
            dot_tile_f32(
                &coords_flat[q_idx * coords_stride..(q_idx + 1) * coords_stride],
                coords_flat,
                coords_stride,
                &p_indices[idx..end],
                &mut dot_coords_f32,
            );
            dot_tile_f32(
                &v_flat[q_idx * v_stride..(q_idx + 1) * v_stride],
                v_flat,
                v_stride,
                &p_indices[idx..end],
                &mut dot_v_f32,
            );

            // 2. Compute Distances (SIMD x 8)
            let mut i = 0;
            while i < chunk_len {
                // Check if we have a full vector of 8
                if i + 8 <= chunk_len {
                    // Gather
                    let p_idxs = &p_indices[idx + i..idx + i + 8];
                    let mut norms = [0.0f32; 8];
                    let mut diags = [0.0f32; 8];
                    let mut v_norms_p = [0.0f32; 8];
                    
                    for k in 0..8 {
                        norms[k] = scaled_norms[p_idxs[k]];
                        diags[k] = p_diag[p_idxs[k]];
                        v_norms_p[k] = v_norms[p_idxs[k]];
                    }
                    
                    let v_norms_p_vec = f32x8::from(norms);
                    let v_p_diag = f32x8::from(diags);
                    
                    let v_dot_c = f32x8::from(&dot_coords_f32[i..i+8]);
                    let v_dot_v = f32x8::from(&dot_v_f32[i..i+8]);

                    // RBF Kernel
                    // d2 = q_norm + p_norm - 2 * dot
                    let mut v_d2 = v_q_norm + v_norms_p_vec - v_two * v_dot_c;
                    v_d2 = v_d2.max(v_zero);
                    
                    let v_k_val = v_rbf_var * (v_neg_half * v_d2).exp();
                    
                    // Denom
                    let v_denom_sq = (v_q_diag * v_p_diag).max(v_eps * v_eps);
                    let v_denom = v_denom_sq.sqrt();
                    
                    // Result
                    // rho = (k_val - dot_v) / denom
                    let v_rho = (v_k_val - v_dot_v) / v_denom;
                    let v_rho_clamped = v_rho.max(v_neg_one).min(v_one);
                    let v_dist_sq = v_one - v_rho_clamped.abs();
                    // mask out small neg
                    let v_dist = v_dist_sq.max(v_zero).sqrt();
                    
                    let res_arr = v_dist.to_array();
                    
                    // Write back
                    for k in 0..8 {
                        if p_idxs[k] == q_idx {
                            out.push(NumCast::from(0.0).unwrap());
                        } else {
                            out.push(NumCast::from(res_arr[k]).unwrap());
                        }
                    }
                    i += 8;
                } else {
                    // Scalar tail
                    let p_idx = p_indices[idx + i];
                    if p_idx == q_idx {
                        out.push(NumCast::from(0.0).unwrap());
                        i += 1;
                        continue;
                    }
                    
                    let dot_c = dot_coords_f32[i];
                    let dot_v = dot_v_f32[i];
                    let p_norm = scaled_norms[p_idx];
                    let p_d = p_diag[p_idx];
                    
                    let mut d2 = q_norm + p_norm - two * dot_c;
                    if d2 < 0.0 { d2 = 0.0; }
                    let k_val = self.rbf_var.to_f32().unwrap() * (self.neg_half.to_f32().unwrap() * d2).exp();
                    let denom = (q_diag * p_d).max(eps * eps).sqrt();
                    let rho = (k_val - dot_v) / denom;
                    let rho_clamped = rho.max(neg_one).min(one);
                    let dist_sq = one - rho_clamped.abs();
                    let dist = if dist_sq > 0.0 { dist_sq.sqrt() } else { 0.0 };
                    out.push(NumCast::from(dist).unwrap());
                    i += 1;
                }
            }
            
            idx = end;
        }
    }

    fn distance_sq_idx_f32(&self, idx_1: usize, idx_2: usize) -> T {
        if idx_1 == idx_2 {
            return NumCast::from(0.0).unwrap();
        }

        let coords = self
            .scaled_coords_f32
            .as_ref()
            .expect("f32 coords requested but not initialised");
        let v_mat = self
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

        let dot_scaled = if self.parity_mode {
            parity_dot_f32(coords.row(idx_1), coords.row(idx_2))
                .unwrap_or_else(|| coords.row(idx_1).dot(&coords.row(idx_2)))
        } else {
            dot_product_simd(coords.row(idx_1), coords.row(idx_2))
        };
        let mut d2 = scaled_norms[idx_1] + scaled_norms[idx_2] - two * dot_scaled;
        if d2 < 0.0 {
            d2 = 0.0;
        }
        let k_val =
            self.rbf_var.to_f32().unwrap() * (self.neg_half.to_f32().unwrap() * d2).exp();

        let denom_sq = (p_diag[idx_1] * p_diag[idx_2]).max(eps * eps);
        let denom = denom_sq.sqrt();

        let dot = if self.parity_mode {
            parity_dot_f32(v_mat.row(idx_1), v_mat.row(idx_2))
                .unwrap_or_else(|| v_mat.row(idx_1).dot(&v_mat.row(idx_2)))
        } else {
            dot_product_simd(v_mat.row(idx_1), v_mat.row(idx_2))
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
