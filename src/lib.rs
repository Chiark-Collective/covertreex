use crate::algo::batch::{batch_insert, batch_insert_with_telemetry, BatchInsertTelemetry};
use crate::algo::{
    batch_knn_query, batch_residual_knn_query, batch_residual_knn_query_block_sgemm,
    compute_si_cache_residual, debug_stats_snapshot, set_debug_stats_enabled, take_debug_stats,
};
use crate::metric::{Euclidean, ResidualMetric};
use crate::pcct::hilbert_like_order;
use crate::telemetry::ResidualQueryTelemetry;
use crate::tree::{compute_subtree_index_bounds, CoverTreeData};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use std::collections::HashMap;

mod algo;
mod metric;
mod pcct;
mod telemetry;
mod tree;

fn load_scope_caps(py: Python<'_>) -> Option<HashMap<i32, f32>> {
    let path = std::env::var("COVERTREEX_RESIDUAL_SCOPE_CAP_PATH").ok()?;
    if path.is_empty() {
        return None;
    }

    let module = PyModule::import(py, "covertreex.metrics.residual.scope_caps").ok()?;
    let caps_obj = module.call_method1("get_scope_cap_table", (path,)).ok()?;

    if caps_obj.is_none() {
        return None;
    }

    let level_caps_attr = caps_obj.getattr("level_caps").ok()?;
    let dict: Bound<'_, PyDict> = level_caps_attr.downcast().ok()?.clone();

    let mut caps = HashMap::new();
    for (k, v) in dict.iter() {
        if let (Ok(level), Ok(cap)) = (k.extract::<i32>(), v.extract::<f32>()) {
            caps.insert(level, cap);
        }
    }

    if !caps.is_empty() {
        Some(caps)
    } else {
        None
    }
}

struct CachedResidualData {
    v_matrix: Array2<f32>,
    p_diag: Vec<f32>,
    coords: Array2<f32>,
    rbf_ls: Vec<f32>,
    rbf_var: f32,
    kernel_type: i32,
}

/// A simple wrapper for the Cover Tree core logic
#[pyclass]
struct CoverTreeWrapper {
    inner: CoverTreeInner,
    survivors: Vec<i64>,
    last_query_telemetry: Option<ResidualQueryTelemetry>,
    inv_order: Option<Vec<i64>>,
    cached_data: Option<CachedResidualData>,
    order: Option<Vec<i64>>,
}

enum CoverTreeInner {
    F32(CoverTreeData<f32>),
    F64(CoverTreeData<f64>),
}

fn to_array2_f32(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f32>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f32>>() {
        return Ok(arr.as_array().to_owned());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr.as_array().mapv(|v| v as f32));
    }
    Err(PyTypeError::new_err(
        "expected a float32 or float64 2D array",
    ))
}

fn to_array1_f32(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array1<f32>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(arr.as_array().to_owned());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr.as_array().mapv(|v| v as f32));
    }
    Err(PyTypeError::new_err(
        "expected a float32 or float64 1D array",
    ))
}

fn to_array2_f64(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr.as_array().to_owned());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f32>>() {
        return Ok(arr.as_array().mapv(|v| v as f64));
    }
    Err(PyTypeError::new_err(
        "expected a float32 or float64 2D array",
    ))
}

fn to_array1_f64(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array1<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr.as_array().to_owned());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f32>>() {
        return Ok(arr.as_array().mapv(|v| v as f64));
    }
    Err(PyTypeError::new_err(
        "expected a float32 or float64 1D array",
    ))
}

fn telemetry_to_pydict(py: Python<'_>, telem: &ResidualQueryTelemetry) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("frontier_levels", &telem.frontier_levels)?;
    dict.set_item("frontier_expanded", &telem.frontier_expanded)?;
    dict.set_item("yields", &telem.yields)?;
    dict.set_item("caps_applied", telem.caps_applied)?;
    dict.set_item("prunes_lower_bound", telem.prunes_lower_bound)?;
    dict.set_item("prunes_lower_bound_chunks", telem.prunes_lower_bound_chunks)?;
    dict.set_item("prunes_cap", telem.prunes_cap)?;
    dict.set_item("masked_dedup", telem.masked_dedup)?;
    dict.set_item("distance_evals", telem.distance_evals)?;
    dict.set_item("budget_escalations", telem.budget_escalations)?;
    dict.set_item("budget_early_terminate", telem.budget_early_terminate)?;
    dict.set_item("level_cache_hits", telem.level_cache_hits)?;
    dict.set_item("level_cache_misses", telem.level_cache_misses)?;
    dict.set_item("block_sizes", &telem.block_sizes)?;
    dict.set_item("predecessor_filtered", telem.predecessor_filtered)?;
    dict.set_item("subtrees_pruned", telem.subtrees_pruned)?;
    Ok(dict.into_any().into())
}

#[pymethods]
impl CoverTreeWrapper {
    #[new]
    fn new(
        py: Python<'_>,
        points: PyObject,
        parents: Vec<i64>,
        children: Vec<i64>,
        next_node: Vec<i64>,
        levels: Vec<i32>,
        min_level: i32,
        max_level: i32,
    ) -> PyResult<Self> {
        if let Ok(points_f32) = points.extract::<PyReadonlyArray2<f32>>(py) {
            let points_owned = points_f32.as_array().to_owned();
            let data = CoverTreeData::new(
                points_owned,
                parents,
                children,
                next_node,
                levels,
                min_level,
                max_level,
            );
            return Ok(CoverTreeWrapper {
                inner: CoverTreeInner::F32(data),
                survivors: Vec::new(),
                last_query_telemetry: None,
                inv_order: None,
                cached_data: None,
                order: None,
            });
        }

        if let Ok(points_f64) = points.extract::<PyReadonlyArray2<f64>>(py) {
            let points_owned = points_f64.as_array().to_owned();
            let data = CoverTreeData::new(
                points_owned,
                parents,
                children,
                next_node,
                levels,
                min_level,
                max_level,
            );
            return Ok(CoverTreeWrapper {
                inner: CoverTreeInner::F64(data),
                survivors: Vec::new(),
                last_query_telemetry: None,
                inv_order: None,
                cached_data: None,
                order: None,
            });
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "points must be float32 or float64 array",
        ))
    }

    fn insert(&mut self, py: Python<'_>, batch: PyObject) -> PyResult<()> {
        match &mut self.inner {
            CoverTreeInner::F32(data) => {
                let batch_obj = batch.extract::<PyReadonlyArray2<f32>>(py)?;
                let batch_view = batch_obj.as_array();
                let metric = Euclidean;
                batch_insert(data, batch_view, &metric);
            }
            CoverTreeInner::F64(data) => {
                let batch_obj = batch.extract::<PyReadonlyArray2<f64>>(py)?;
                let batch_view = batch_obj.as_array();
                let metric = Euclidean;
                batch_insert(data, batch_view, &metric);
            }
        }
        Ok(())
    }

    fn set_si_cache(&mut self, py: Python<'_>, cache: PyObject) -> PyResult<()> {
        if let Ok(arr) = cache.extract::<PyReadonlyArray1<f32>>(py) {
            match &mut self.inner {
                CoverTreeInner::F32(data) => {
                    data.set_si_cache(arr.as_slice()?.to_vec());
                    return Ok(());
                }
                CoverTreeInner::F64(_) => { /* fall through */ }
            }
        }
        if let Ok(arr) = cache.extract::<PyReadonlyArray1<f64>>(py) {
            match &mut self.inner {
                CoverTreeInner::F64(data) => {
                    data.set_si_cache(arr.as_slice()?.to_vec());
                    return Ok(());
                }
                CoverTreeInner::F32(data) => {
                    let cache_f32: Vec<f32> = arr.as_slice()?.iter().map(|v| *v as f32).collect();
                    data.set_si_cache(cache_f32);
                    return Ok(());
                }
            }
        }
        Err(PyTypeError::new_err(
            "si_cache must be a float32 or float64 1D array",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (batch_indices, v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size=None, kernel_type=None))]
    fn insert_residual(
        &mut self,
        py: Python<'_>,
        batch_indices: PyObject,
        v_matrix: PyObject,
        p_diag: PyObject,
        coords: PyObject,
        rbf_var: f64,
        rbf_ls: PyObject,
        chunk_size: Option<usize>,
        kernel_type: Option<i32>,
    ) -> PyResult<()> {
        let k_type = kernel_type.unwrap_or(0);
        match &mut self.inner {
            CoverTreeInner::F32(data) => {
                let batch_arr = to_array2_f32(&batch_indices.bind(py))?;
                let v_matrix_arr = to_array2_f32(&v_matrix.bind(py))?;
                let p_diag_arr = to_array1_f32(&p_diag.bind(py))?;
                let coords_arr = to_array2_f32(&coords.bind(py))?;
                let rbf_ls_arr = to_array1_f32(&rbf_ls.bind(py))?;

                let metric = ResidualMetric::new(
                    v_matrix_arr.view(),
                    p_diag_arr.as_slice().unwrap(),
                    coords_arr.view(),
                    rbf_var as f32,
                    rbf_ls_arr.as_slice().unwrap(),
                    k_type,
                    None,
                );
                let chunk = chunk_size.unwrap_or_else(|| batch_arr.nrows());
                let mut start = 0;
                while start < batch_arr.nrows() {
                    let end = std::cmp::min(start + chunk, batch_arr.nrows());
                    let view = batch_arr.slice(ndarray::s![start..end, ..]);
                    batch_insert(data, view, &metric);
                    start = end;
                }
            }
            CoverTreeInner::F64(data) => {
                let batch_arr = to_array2_f64(&batch_indices.bind(py))?;
                let v_matrix_arr = to_array2_f64(&v_matrix.bind(py))?;
                let p_diag_arr = to_array1_f64(&p_diag.bind(py))?;
                let coords_arr = to_array2_f64(&coords.bind(py))?;
                let rbf_ls_arr = to_array1_f64(&rbf_ls.bind(py))?;

                let metric = ResidualMetric::new(
                    v_matrix_arr.view(),
                    p_diag_arr.as_slice().unwrap(),
                    coords_arr.view(),
                    rbf_var,
                    rbf_ls_arr.as_slice().unwrap(),
                    k_type,
                    None,
                );
                let chunk = chunk_size.unwrap_or_else(|| batch_arr.nrows());
                let mut start = 0;
                while start < batch_arr.nrows() {
                    let end = std::cmp::min(start + chunk, batch_arr.nrows());
                    let view = batch_arr.slice(ndarray::s![start..end, ..]);
                    batch_insert(data, view, &metric);
                    start = end;
                }
            }
        }
        Ok(())
    }

    fn point_count(&self) -> usize {
        match &self.inner {
            CoverTreeInner::F32(data) => data.len(),
            CoverTreeInner::F64(data) => data.len(),
        }
    }

    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        queries: PyObject,
        k: usize,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
        match &self.inner {
            CoverTreeInner::F32(data) => {
                let q_obj = queries.extract::<PyReadonlyArray2<f32>>(py)?;
                let q_view = q_obj.as_array();
                let (indices, dists) = batch_knn_query(data, q_view, k);
                let (idx, dst) = to_py_arrays(py, indices, dists, k);
                Ok((idx, dst.into_any().into()))
            }
            CoverTreeInner::F64(data) => {
                let q_obj = queries.extract::<PyReadonlyArray2<f64>>(py)?;
                let q_view = q_obj.as_array();
                let (indices, dists) = batch_knn_query(data, q_view, k);
                let (idx, dst) = to_py_arrays(py, indices, dists, k);
                Ok((idx, dst.into_any().into()))
            }
        }
    }

    fn last_query_telemetry<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if let Some(t) = &self.last_query_telemetry {
            Ok(Some(telemetry_to_pydict(py, t)?))
        } else {
            Ok(None)
        }
    }

    fn clear_last_query_telemetry(&mut self) {
        self.last_query_telemetry = None;
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query_indices, node_to_dataset, v_matrix, p_diag, coords, rbf_var, rbf_ls, k, kernel_type=None, predecessor_mode=None, subtree_min_bounds=None))]
    fn knn_query_residual<'py>(
        &mut self,
        py: Python<'py>,
        query_indices: numpy::PyReadonlyArray1<i64>,
        node_to_dataset: Vec<i64>,
        v_matrix: PyObject,
        p_diag: PyObject,
        coords: PyObject,
        rbf_var: f64,
        rbf_ls: PyObject,
        k: usize,
        kernel_type: Option<i32>,
        predecessor_mode: Option<bool>,
        subtree_min_bounds: Option<numpy::PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
        let pred_mode = predecessor_mode.unwrap_or(false);
        let k_type = kernel_type.unwrap_or(0);
        let bounds_vec: Option<Vec<i64>> = subtree_min_bounds.map(|arr| arr.as_slice().unwrap().to_vec());
        // Reuse cached data if available (fast path)
        if let Some(cached) = &self.cached_data {
            match &self.inner {
                CoverTreeInner::F32(data) => {
                    let metric = ResidualMetric::new(
                        cached.v_matrix.view(),
                        &cached.p_diag,
                        cached.coords.view(),
                        cached.rbf_var,
                        &cached.rbf_ls,
                        cached.kernel_type,
                        None,
                    );

                    let scope_caps = load_scope_caps(py);
                    let telemetry_enabled = std::env::var("COVERTREEX_RUST_QUERY_TELEMETRY")
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                    let mut telemetry_rec: Option<ResidualQueryTelemetry> = None;

                    let queries = query_indices.as_array();
                    let q_mapped_owned = if let Some(inv) = &self.inv_order {
                        // Map queries: q_old -> q_new
                        let mut mapped = Vec::with_capacity(queries.len());
                        for &q in queries {
                            if q >= 0 && (q as usize) < inv.len() {
                                mapped.push(inv[q as usize]);
                            } else {
                                mapped.push(0); // Fallback
                            }
                        }
                        Some(ndarray::Array1::from_vec(mapped))
                    } else {
                        None
                    };

                    let q_input = if let Some(qm) = &q_mapped_owned {
                        qm.view()
                    } else {
                        queries
                    };

                    // For predecessor mode with Hilbert ordering:
                    // - Use self.order as node_to_dataset so constraint checks use dataset indices
                    // - Results will already be dataset indices (no remapping needed)
                    // Without predecessor mode: use identity_map, remap results at end
                    let identity_map: Vec<i64> = (0..data.len() as i64).collect();
                    let use_order_for_constraint = pred_mode && self.order.is_some();
                    let node_to_dataset_ref: &[i64] = if use_order_for_constraint {
                        self.order.as_ref().unwrap()
                    } else {
                        &identity_map
                    };

                    // For predecessor mode, use original dataset indices as constraint bounds
                    let predecessor_indices: Option<Vec<i64>> = if pred_mode {
                        Some(queries.as_slice().expect("contiguous queries").to_vec())
                    } else {
                        None
                    };

                    let (mut indices, dists) = if telemetry_enabled {
                        let mut telem = ResidualQueryTelemetry::default();
                        let res = batch_residual_knn_query(
                            data,
                            q_input,
                            node_to_dataset_ref,
                            &metric,
                            k,
                            scope_caps.as_ref(),
                            predecessor_indices.as_deref(),
                            bounds_vec.as_deref(),
                            Some(&mut telem),
                        );
                        telemetry_rec = Some(telem);
                        res
                    } else {
                        batch_residual_knn_query(
                            data,
                            q_input,
                            node_to_dataset_ref,
                            &metric,
                            k,
                            scope_caps.as_ref(),
                            predecessor_indices.as_deref(),
                            bounds_vec.as_deref(),
                            None,
                        )
                    };

                    // Map results back: Hilbert_idx -> dataset_idx
                    // Skip if we used order for constraint (results are already dataset indices)
                    if !use_order_for_constraint {
                        if let Some(order) = &self.order {
                            for row in indices.iter_mut() {
                                for idx in row.iter_mut() {
                                    if *idx >= 0 && (*idx as usize) < order.len() {
                                        *idx = order[*idx as usize];
                                    }
                                }
                            }
                        } else {
                            for row in indices.iter_mut() {
                                for idx in row.iter_mut() {
                                    if *idx >= 0 && (*idx as usize) < node_to_dataset.len() {
                                        *idx = node_to_dataset[*idx as usize];
                                    }
                                }
                            }
                        }
                    }

                    self.last_query_telemetry = telemetry_rec;
                    let (idx, dst) = to_py_arrays(py, indices, dists, k);
                    return Ok((idx, dst.into_any().into()));
                }
                _ => return Err(PyTypeError::new_err("Cached data only supported for F32 tree")),
            }
        }

        match &self.inner {
            CoverTreeInner::F32(data) => {
                let v_matrix_arr = to_array2_f32(&v_matrix.bind(py))?;
                let p_diag_arr = to_array1_f32(&p_diag.bind(py))?;
                let coords_arr = to_array2_f32(&coords.bind(py))?;
                let rbf_ls_arr = to_array1_f32(&rbf_ls.bind(py))?;

                let metric = ResidualMetric::new(
                    v_matrix_arr.view(),
                    p_diag_arr.as_slice().unwrap(),
                    coords_arr.view(),
                    rbf_var as f32,
                    rbf_ls_arr.as_slice().unwrap(),
                    k_type,
                    None,
                );

                let scope_caps = load_scope_caps(py);
                let telemetry_enabled = std::env::var("COVERTREEX_RUST_QUERY_TELEMETRY")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let mut telemetry_rec: Option<ResidualQueryTelemetry> = None;

                // Query indices from Python are dataset indices.
                // If tree has Hilbert ordering (inv_order available), map to tree node indices.
                let query_arr = query_indices.as_array();
                let query_mapped_owned: Option<ndarray::Array1<i64>> = if let Some(inv) = &self.inv_order {
                    let mut mapped = Vec::with_capacity(query_arr.len());
                    for &q in query_arr.iter() {
                        if q >= 0 && (q as usize) < inv.len() {
                            mapped.push(inv[q as usize]);
                        } else {
                            mapped.push(0);
                        }
                    }
                    Some(ndarray::Array1::from_vec(mapped))
                } else {
                    None
                };
                let query_input = if let Some(qm) = &query_mapped_owned {
                    qm.view()
                } else {
                    query_arr
                };

                // For predecessor mode, use ORIGINAL dataset indices (not mapped)
                let predecessor_indices: Option<Vec<i64>> = if pred_mode {
                    Some(query_arr.as_slice().expect("contiguous queries").to_vec())
                } else {
                    None
                };

                let (mut indices, dists) = if telemetry_enabled {
                    let mut telem = ResidualQueryTelemetry::default();
                    let res = batch_residual_knn_query(
                        data,
                        query_input,
                        &node_to_dataset,
                        &metric,
                        k,
                        scope_caps.as_ref(),
                        predecessor_indices.as_deref(),
                        bounds_vec.as_deref(),
                        Some(&mut telem),
                    );
                    telemetry_rec = Some(telem);
                    res
                } else {
                    batch_residual_knn_query(
                        data,
                        query_input,
                        &node_to_dataset,
                        &metric,
                        k,
                        scope_caps.as_ref(),
                        predecessor_indices.as_deref(),
                        bounds_vec.as_deref(),
                        None,
                    )
                };

                // Map internal node indices back to original dataset indices
                for row in indices.iter_mut() {
                    for idx in row.iter_mut() {
                        *idx = node_to_dataset[*idx as usize];
                    }
                }

                self.last_query_telemetry = telemetry_rec;
                let (idx, dst) = to_py_arrays(py, indices, dists, k);
                Ok((idx, dst.into_any().into()))
            }
            CoverTreeInner::F64(data) => {
                let v_matrix_arr = to_array2_f64(&v_matrix.bind(py))?;
                let p_diag_arr = to_array1_f64(&p_diag.bind(py))?;
                let coords_arr = to_array2_f64(&coords.bind(py))?;
                let rbf_ls_arr = to_array1_f64(&rbf_ls.bind(py))?;

                let metric = ResidualMetric::new(
                    v_matrix_arr.view(),
                    p_diag_arr.as_slice().unwrap(),
                    coords_arr.view(),
                    rbf_var,
                    rbf_ls_arr.as_slice().unwrap(),
                    k_type,
                    None,
                );

                let scope_caps = load_scope_caps(py);
                let caps_f64: Option<HashMap<i32, f64>> =
                    scope_caps.map(|m| m.into_iter().map(|(k, v)| (k, v as f64)).collect());

                let telemetry_enabled = std::env::var("COVERTREEX_RUST_QUERY_TELEMETRY")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let mut telemetry_rec: Option<ResidualQueryTelemetry> = None;

                // Query indices from Python are dataset indices.
                // If tree has Hilbert ordering (inv_order available), map to tree node indices.
                let query_arr = query_indices.as_array();
                let query_mapped_owned: Option<ndarray::Array1<i64>> = if let Some(inv) = &self.inv_order {
                    let mut mapped = Vec::with_capacity(query_arr.len());
                    for &q in query_arr.iter() {
                        if q >= 0 && (q as usize) < inv.len() {
                            mapped.push(inv[q as usize]);
                        } else {
                            mapped.push(0);
                        }
                    }
                    Some(ndarray::Array1::from_vec(mapped))
                } else {
                    None
                };
                let query_input = if let Some(qm) = &query_mapped_owned {
                    qm.view()
                } else {
                    query_arr
                };

                // For predecessor mode, use ORIGINAL dataset indices (not mapped)
                let predecessor_indices: Option<Vec<i64>> = if pred_mode {
                    Some(query_arr.as_slice().expect("contiguous queries").to_vec())
                } else {
                    None
                };

                let (mut indices, dists) = if telemetry_enabled {
                    let mut telem = ResidualQueryTelemetry::default();
                    let res = batch_residual_knn_query(
                        data,
                        query_input,
                        &node_to_dataset,
                        &metric,
                        k,
                        caps_f64.as_ref(),
                        predecessor_indices.as_deref(),
                        bounds_vec.as_deref(),
                        Some(&mut telem),
                    );
                    telemetry_rec = Some(telem);
                    res
                } else {
                    batch_residual_knn_query(
                        data,
                        query_input,
                        &node_to_dataset,
                        &metric,
                        k,
                        caps_f64.as_ref(),
                        predecessor_indices.as_deref(),
                        bounds_vec.as_deref(),
                        None,
                    )
                };

                // Map internal node indices back to original dataset indices
                for row in indices.iter_mut() {
                    for idx in row.iter_mut() {
                        *idx = node_to_dataset[*idx as usize];
                    }
                }

                self.last_query_telemetry = telemetry_rec;
                let (idx, dst) = to_py_arrays(py, indices, dists, k);
                Ok((idx, dst.into_any().into()))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query_indices, node_to_dataset, v_matrix, p_diag, coords, rbf_var, rbf_ls, k, kernel_type=None))]
    fn knn_query_residual_block<'py>(
        &self,
        py: Python<'py>,
        query_indices: numpy::PyReadonlyArray1<i64>,
        node_to_dataset: Vec<i64>,
        v_matrix: PyObject,
        p_diag: PyObject,
        coords: PyObject,
        rbf_var: f64,
        rbf_ls: PyObject,
        k: usize,
        kernel_type: Option<i32>,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
        let k_type = kernel_type.unwrap_or(0);
        match &self.inner {
            CoverTreeInner::F32(_) => {
                let v_matrix_arr = to_array2_f32(&v_matrix.bind(py))?;
                let p_diag_arr = to_array1_f32(&p_diag.bind(py))?;
                let coords_arr = to_array2_f32(&coords.bind(py))?;
                let rbf_ls_arr = to_array1_f32(&rbf_ls.bind(py))?;

                let metric = ResidualMetric::new(
                    v_matrix_arr.view(),
                    p_diag_arr.as_slice().unwrap(),
                    coords_arr.view(),
                    rbf_var as f32,
                    rbf_ls_arr.as_slice().unwrap(),
                    k_type,
                    None,
                );

                let scope_caps = load_scope_caps(py);

                let mode = std::env::var("COVERTREEX_RUST_PCCT2_SGEMM")
                    .unwrap_or_else(|_| "tree".to_string());
                let (indices, dists) = match mode.as_str() {
                    "1" | "block" | "sgemm" => batch_residual_knn_query_block_sgemm(
                        query_indices.as_array(),
                        &node_to_dataset,
                        &metric,
                        k,
                        None,
                    ),
                    "survivor" | "survivors" => {
                        let mut survivors: Vec<usize> =
                            self.survivors.iter().map(|v| *v as usize).collect();
                        if let Ok(limit_str) = std::env::var("COVERTREEX_RUST_SURVIVOR_LIMIT") {
                            if let Ok(limit) = limit_str.parse::<usize>() {
                                if limit > 0 && survivors.len() > limit {
                                    survivors.truncate(limit);
                                }
                            }
                        }
                        batch_residual_knn_query_block_sgemm(
                            query_indices.as_array(),
                            &node_to_dataset,
                            &metric,
                            k,
                            Some(&survivors),
                        )
                    }
                    "auto" => {
                        let mut survivors: Vec<usize> =
                            self.survivors.iter().map(|v| *v as usize).collect();
                        if let Ok(limit_str) = std::env::var("COVERTREEX_RUST_SURVIVOR_LIMIT") {
                            if let Ok(limit) = limit_str.parse::<usize>() {
                                if limit > 0 && survivors.len() > limit {
                                    survivors.truncate(limit);
                                }
                            }
                        }
                        // Prefer tree traversal; fall back to survivor SGEMM if available.
                        let tree_result = batch_residual_knn_query(
                            match &self.inner {
                                CoverTreeInner::F32(data) => data,
                                _ => unreachable!(),
                            },
                            query_indices.as_array(),
                            &node_to_dataset,
                            &metric,
                            k,
                            scope_caps.as_ref(),
                            None,  // predecessor_mode not supported in block path
                            None,  // subtree_min_bounds
                            None,
                        );
                        if !survivors.is_empty() {
                            batch_residual_knn_query_block_sgemm(
                                query_indices.as_array(),
                                &node_to_dataset,
                                &metric,
                                k,
                                Some(&survivors),
                            )
                        } else {
                            tree_result
                        }
                    }
                    _ => batch_residual_knn_query(
                        match &self.inner {
                            CoverTreeInner::F32(data) => data,
                            _ => unreachable!(),
                        },
                        query_indices.as_array(),
                        &node_to_dataset,
                        &metric,
                        k,
                        scope_caps.as_ref(),
                        None,  // predecessor_mode not supported in block path
                        None,  // subtree_min_bounds
                        None,
                    ),
                };
                let (idx, dst) = to_py_arrays(py, indices, dists, k);
                Ok((idx, dst.into_any().into()))
            }
            CoverTreeInner::F64(_) => {
                // For now only float32 is supported for the blocked path; fall back to error.
                Err(PyTypeError::new_err(
                    "knn_query_residual_block supports only float32 payloads",
                ))
            }
        }
    }

    fn get_points<'py>(&self, py: Python<'py>) -> PyObject {
        match &self.inner {
            CoverTreeInner::F32(data) => {
                let dims = (data.len(), data.dimension);
                let array = ndarray::Array2::from_shape_vec(dims, data.points.clone()).unwrap();
                array.into_pyarray(py).into_any().into()
            }
            CoverTreeInner::F64(data) => {
                let dims = (data.len(), data.dimension);
                let array = ndarray::Array2::from_shape_vec(dims, data.points.clone()).unwrap();
                array.into_pyarray(py).into_any().into()
            }
        }
    }

    fn get_parents<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        match &self.inner {
            CoverTreeInner::F32(d) => {
                let array = ndarray::Array1::from_vec(d.parents.clone());
                array.into_pyarray(py)
            }
            CoverTreeInner::F64(d) => {
                let array = ndarray::Array1::from_vec(d.parents.clone());
                array.into_pyarray(py)
            }
        }
    }

    fn get_children<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        match &self.inner {
            CoverTreeInner::F32(d) => {
                let array = ndarray::Array1::from_vec(d.children.clone());
                array.into_pyarray(py)
            }
            CoverTreeInner::F64(d) => {
                let array = ndarray::Array1::from_vec(d.children.clone());
                array.into_pyarray(py)
            }
        }
    }

    fn get_next_node<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i64>> {
        match &self.inner {
            CoverTreeInner::F32(d) => {
                let array = ndarray::Array1::from_vec(d.next_node.clone());
                array.into_pyarray(py)
            }
            CoverTreeInner::F64(d) => {
                let array = ndarray::Array1::from_vec(d.next_node.clone());
                array.into_pyarray(py)
            }
        }
    }

    fn get_levels<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<i32>> {
        match &self.inner {
            CoverTreeInner::F32(d) => {
                let array = ndarray::Array1::from_vec(d.levels.clone());
                array.into_pyarray(py)
            }
            CoverTreeInner::F64(d) => {
                let array = ndarray::Array1::from_vec(d.levels.clone());
                array.into_pyarray(py)
            }
        }
    }

    fn get_min_level(&self) -> i32 {
        match &self.inner {
            CoverTreeInner::F32(d) => d.min_level,
            CoverTreeInner::F64(d) => d.min_level,
        }
    }

    fn get_max_level(&self) -> i32 {
        match &self.inner {
            CoverTreeInner::F32(d) => d.max_level,
            CoverTreeInner::F64(d) => d.max_level,
        }
    }
}

fn to_py_arrays<'py, T: numpy::Element + Copy + num_traits::Zero + num_traits::Float>(
    py: Python<'py>,
    indices: Vec<Vec<i64>>,
    dists: Vec<Vec<T>>,
    requested_k: usize,
) -> (
    Bound<'py, numpy::PyArray2<i64>>,
    Bound<'py, numpy::PyArray2<T>>,
) {
    let n_queries = indices.len();
    // Use requested_k for output dimension, ensuring consistent shape (n, k)
    // even when some queries return fewer results (e.g., predecessor_mode)
    let dim_k = if requested_k > 0 {
        requested_k
    } else {
        // Fallback: find maximum row length
        indices.iter().map(|v| v.len()).max().unwrap_or(0)
    };

    // Initialize with padding values: -1 for indices, max value for distances
    let mut idx_array = ndarray::Array2::<i64>::from_elem((n_queries, dim_k), -1);
    let mut dst_array = ndarray::Array2::<T>::from_elem((n_queries, dim_k), T::max_value());

    for i in 0..n_queries {
        for j in 0..indices[i].len().min(dim_k) {
            idx_array[[i, j]] = indices[i][j];
            dst_array[[i, j]] = dists[i][j];
        }
    }

    (idx_array.into_pyarray(py), dst_array.into_pyarray(py))
}

/// The Rust backend module for CoverTreeX
#[pymodule]
fn covertreex_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoverTreeWrapper>()?;
    m.add_function(wrap_pyfunction!(set_rust_debug_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get_rust_debug_stats, m)?)?;
    m.add_function(wrap_pyfunction!(build_pcct_residual_tree, m)?)?;
    m.add_function(wrap_pyfunction!(build_pcct2_residual_tree, m)?)?;
    m.add_function(wrap_pyfunction!(knn_query_residual_block, m)?)?;
    m.add_function(wrap_pyfunction!(compute_subtree_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert_order, m)?)?;
    Ok(())
}

/// Compute Hilbert-curve-like ordering for spatial locality.
/// Returns indices that sort points by their Hilbert curve position.
#[pyfunction]
fn hilbert_order(py: Python<'_>, coords: PyObject) -> PyResult<Py<PyArray1<i64>>> {
    let coords_arr = to_array2_f32(&coords.bind(py))?;
    let order = hilbert_like_order(coords_arr.view());
    let order_i64: Vec<i64> = order.into_iter().map(|i| i as i64).collect();
    Ok(PyArray1::from_vec(py, order_i64).into())
}

#[pyfunction]
fn set_rust_debug_stats(enable: bool) {
    set_debug_stats_enabled(enable);
}

#[pyfunction(signature = (reset=None))]
fn get_rust_debug_stats(reset: Option<bool>) -> (usize, usize) {
    let should_reset = reset.unwrap_or(true);
    if should_reset {
        take_debug_stats()
    } else {
        debug_stats_snapshot()
    }
}

/// Compute subtree index bounds for predecessor constraint pruning.
///
/// For each tree node, computes the minimum and maximum dataset indices
/// contained in that node's subtree. This enables aggressive subtree pruning
/// during predecessor-constrained queries.
///
/// # Arguments
/// * `parents` - Parent index for each node (-1 for root)
/// * `node_to_dataset` - Maps tree node index to dataset index
///
/// # Returns
/// * `(min_bounds, max_bounds)` - numpy arrays of min/max dataset indices per subtree
#[pyfunction]
fn compute_subtree_bounds_py<'py>(
    py: Python<'py>,
    parents: numpy::PyReadonlyArray1<'py, i64>,
    node_to_dataset: numpy::PyReadonlyArray1<'py, i64>,
) -> (Bound<'py, numpy::PyArray1<i64>>, Bound<'py, numpy::PyArray1<i64>>) {
    let parents_slice = parents.as_slice().expect("contiguous parents array");
    let n2d_slice = node_to_dataset.as_slice().expect("contiguous node_to_dataset array");

    let (min_bounds, max_bounds) = compute_subtree_index_bounds(parents_slice, n2d_slice);

    let min_arr = ndarray::Array1::from_vec(min_bounds);
    let max_arr = ndarray::Array1::from_vec(max_bounds);

    (min_arr.into_pyarray(py), max_arr.into_pyarray(py))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size=None, batch_order=None, kernel_type=None))]
fn build_pcct_residual_tree(
    py: Python<'_>,
    v_matrix: PyObject,
    p_diag: PyObject,
    coords: PyObject,
    rbf_var: f64,
    rbf_ls: PyObject,
    chunk_size: Option<usize>,
    batch_order: Option<String>,
    kernel_type: Option<i32>,
) -> PyResult<(CoverTreeWrapper, Vec<i64>)> {
    let k_type = kernel_type.unwrap_or(0);
    let parity_mode = std::env::var("COVERTREEX_RESIDUAL_PARITY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let (
        v_matrix_f32,
        v_matrix_f64,
        p_diag_f32,
        p_diag_f64,
        coords_f32,
        coords_f64,
        rbf_ls_f32,
        rbf_ls_f64,
    ) = if parity_mode {
        (
            None,
            Some(to_array2_f64(&v_matrix.bind(py))?),
            None,
            Some(to_array1_f64(&p_diag.bind(py))?),
            None,
            Some(to_array2_f64(&coords.bind(py))?),
            None,
            Some(to_array1_f64(&rbf_ls.bind(py))?),
        )
    } else {
        (
            Some(to_array2_f32(&v_matrix.bind(py))?),
            None,
            Some(to_array1_f32(&p_diag.bind(py))?),
            None,
            Some(to_array2_f32(&coords.bind(py))?),
            None,
            Some(to_array1_f32(&rbf_ls.bind(py))?),
            None,
        )
    };

    let n_rows = coords_f32
        .as_ref()
        .map(|c| c.nrows())
        .unwrap_or_else(|| coords_f64.as_ref().unwrap().nrows());
    let mut _coords_for_order_owned: Option<ndarray::Array2<f32>> = None;
    let coords_for_order = if let Some(c) = coords_f32.as_ref() {
        c.view()
    } else {
        let tmp: ndarray::Array2<f32> = coords_f64.as_ref().unwrap().mapv(|v| v as f32);
        _coords_for_order_owned = Some(tmp);
        _coords_for_order_owned.as_ref().unwrap().view()
    };
    let order = match batch_order.as_deref() {
        Some(s) if s.eq_ignore_ascii_case("natural") => (0..n_rows).collect(),
        Some(s) if s.eq_ignore_ascii_case("hilbert") => hilbert_like_order(coords_for_order),
        Some(s) if s.eq_ignore_ascii_case("hilbert-morton") => hilbert_like_order(coords_for_order),
        _ => hilbert_like_order(coords_for_order),
    };

    // Index payloads (consistent with existing Rust path)
    let indices_arr_f32 = {
        let mut v = Vec::with_capacity(order.len());
        for &idx in &order {
            v.push(idx as f32);
        }
        ndarray::Array2::from_shape_vec((order.len(), 1), v).ok()
    };
    let indices_arr_f64 = {
        let mut v = Vec::with_capacity(order.len());
        for &idx in &order {
            v.push(idx as f64);
        }
        ndarray::Array2::from_shape_vec((order.len(), 1), v).ok()
    };

    // Empty cover tree wrapper
    let empty_i64 = Array1::<i64>::zeros(0);
    let empty_i32 = Array1::<i32>::zeros(0);
    let mut tree = if parity_mode {
        let dummy = Array2::<f64>::zeros((0, 1));
        CoverTreeWrapper {
            inner: CoverTreeInner::F64(CoverTreeData::new(
                dummy,
                empty_i64.to_vec(),
                empty_i64.to_vec(),
                empty_i64.to_vec(),
                empty_i32.to_vec(),
                -20,
                20,
            )),
            survivors: Vec::new(),
            last_query_telemetry: None,
            inv_order: None,
            cached_data: None,
            order: None,
        }
    } else {
        let dummy = Array2::<f32>::zeros((0, 1));
        CoverTreeWrapper {
            inner: CoverTreeInner::F32(CoverTreeData::new(
                dummy,
                empty_i64.to_vec(),
                empty_i64.to_vec(),
                empty_i64.to_vec(),
                empty_i32.to_vec(),
                -20,
                20,
            )),
            survivors: Vec::new(),
            last_query_telemetry: None,
            inv_order: None,
            cached_data: None,
            order: None,
        }
    };

    let metric_f32 = v_matrix_f32.as_ref().map(|v| {
        ResidualMetric::new(
            v.view(),
            p_diag_f32.as_ref().unwrap().as_slice().unwrap(),
            coords_f32.as_ref().unwrap().view(),
            rbf_var as f32,
            rbf_ls_f32.as_ref().unwrap().as_slice().unwrap(),
            k_type,
            None,
        )
    });
    let metric_f64 = v_matrix_f64.as_ref().map(|v| {
        ResidualMetric::new(
            v.view(),
            p_diag_f64.as_ref().unwrap().as_slice().unwrap(),
            coords_f64.as_ref().unwrap().view(),
            rbf_var,
            rbf_ls_f64.as_ref().unwrap().as_slice().unwrap(),
            k_type,
            None,
        )
    });
    let chunk = chunk_size.unwrap_or_else(|| {
        if parity_mode {
            indices_arr_f64.as_ref().unwrap().nrows()
        } else {
            indices_arr_f32.as_ref().unwrap().nrows()
        }
    });
    let mut start = 0;
    let mut survivors: Vec<i64> = Vec::new();
    while start
        < if parity_mode {
            indices_arr_f64.as_ref().unwrap().nrows()
        } else {
            indices_arr_f32.as_ref().unwrap().nrows()
        }
    {
        let end = std::cmp::min(
            start + chunk,
            if parity_mode {
                indices_arr_f64.as_ref().unwrap().nrows()
            } else {
                indices_arr_f32.as_ref().unwrap().nrows()
            },
        );

        if parity_mode {
            let view = indices_arr_f64
                .as_ref()
                .unwrap()
                .slice(ndarray::s![start..end, ..]);
            let telemetry = batch_insert_with_telemetry(
                match &mut tree.inner {
                    CoverTreeInner::F64(data) => data,
                    _ => unreachable!(),
                },
                view,
                None,
                metric_f64.as_ref().unwrap(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            );
            for sel in telemetry.selected.iter() {
                survivors.push((telemetry.batch_start_index + *sel) as i64);
            }
        } else {
            let view = indices_arr_f32
                .as_ref()
                .unwrap()
                .slice(ndarray::s![start..end, ..]);
            let telemetry = batch_insert_with_telemetry(
                match &mut tree.inner {
                    CoverTreeInner::F32(data) => data,
                    _ => unreachable!(),
                },
                view,
                None,
                metric_f32.as_ref().unwrap(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            );
            for sel in telemetry.selected.iter() {
                survivors.push((telemetry.batch_start_index + *sel) as i64);
            }
        }

        start = end;
    }

    let node_to_dataset: Vec<i64> = order.iter().map(|&i| i as i64).collect();
    tree.survivors = survivors;

    // Compute and persist separation-invariant cache (cover radii) for residual traversal.
    if parity_mode {
        let si_cache = compute_si_cache_residual(
            match &tree.inner {
                CoverTreeInner::F64(data) => data,
                _ => unreachable!(),
            },
            node_to_dataset.as_slice(),
            metric_f64.as_ref().unwrap(),
        );
        match &mut tree.inner {
            CoverTreeInner::F64(data) => data.set_si_cache(si_cache),
            _ => unreachable!(),
        }
    } else {
        let si_cache = compute_si_cache_residual(
            match &tree.inner {
                CoverTreeInner::F32(data) => data,
                _ => unreachable!(),
            },
            node_to_dataset.as_slice(),
            metric_f32.as_ref().unwrap(),
        );
        match &mut tree.inner {
            CoverTreeInner::F32(data) => data.set_si_cache(si_cache),
            _ => unreachable!(),
        }
    }

    Ok((tree, node_to_dataset))
}

fn emit_rust_batch(
    py: Python<'_>,
    log_writer: &PyObject,
    batch_index: usize,
    telemetry: BatchInsertTelemetry,
) -> PyResult<()> {
    let batch_size = telemetry.parents.len();
    let BatchInsertTelemetry {
        parents,
        levels,
        selected,
        dominated,
        conflict_indptr,
        conflict_indices,
        scope_indptr,
        scope_indices,
        traversal_seconds,
        conflict_graph_seconds,
        mis_seconds,
        scope_chunk_segments,
        scope_chunk_emitted,
        scope_chunk_max_members,
        scope_chunk_points,
        conflict_scope_chunk_pair_cap,
        conflict_scope_chunk_pairs_before,
        conflict_scope_chunk_pairs_after,
        conflict_scope_chunk_pair_merges,
        scope_chunk_scans,
        scope_chunk_dedupe,
        scope_chunk_saturated,
        scope_budget_start,
        scope_budget_final,
        scope_budget_escalations,
        scope_budget_early_terminate,
        conflict_grid_cells,
        conflict_grid_leaders_raw,
        conflict_grid_leaders_after,
        conflict_grid_local_edges,
        degree_cap,
        degree_pruned_pairs,
        batch_start_index: _,
    } = telemetry;

    let payload = PyDict::new(py);
    payload.set_item("batch_index", batch_index)?;
    payload.set_item("batch_size", batch_size)?;
    payload.set_item("parents", parents.clone())?;
    payload.set_item("levels", levels)?;
    payload.set_item("selected", selected)?;
    payload.set_item("dominated", dominated)?;
    payload.set_item("conflict_indptr", conflict_indptr.clone())?;
    payload.set_item("conflict_indices", conflict_indices)?;
    payload.set_item("scope_indptr", scope_indptr.clone())?;
    payload.set_item("scope_indices", scope_indices)?;

    let timings = PyDict::new(py);
    timings.set_item("traversal_seconds", traversal_seconds)?;
    timings.set_item("conflict_graph_seconds", conflict_graph_seconds)?;
    timings.set_item("mis_seconds", mis_seconds)?;
    timings.set_item("pairwise_seconds", conflict_graph_seconds)?;
    timings.set_item("mask_seconds", 0.0f64)?;
    timings.set_item("semisort_seconds", 0.0f64)?;
    timings.set_item("scope_chunk_segments", scope_chunk_segments)?;
    timings.set_item("scope_chunk_emitted", scope_chunk_emitted)?;
    timings.set_item("scope_chunk_max_members", scope_chunk_max_members)?;
    timings.set_item("scope_chunk_scans", scope_chunk_scans)?;
    timings.set_item("scope_chunk_points", scope_chunk_points)?;
    timings.set_item("scope_chunk_dedupe", scope_chunk_dedupe)?;
    timings.set_item("scope_chunk_saturated", scope_chunk_saturated)?;
    timings.set_item("scope_budget_start", scope_budget_start)?;
    timings.set_item("scope_budget_final", scope_budget_final)?;
    timings.set_item("scope_budget_escalations", scope_budget_escalations)?;
    timings.set_item("scope_budget_early_terminate", scope_budget_early_terminate)?;
    timings.set_item("scope_chunk_pair_cap", conflict_scope_chunk_pair_cap)?;
    timings.set_item(
        "scope_chunk_pairs_before",
        conflict_scope_chunk_pairs_before,
    )?;
    timings.set_item("scope_chunk_pairs_after", conflict_scope_chunk_pairs_after)?;
    timings.set_item("scope_chunk_pair_merges", conflict_scope_chunk_pair_merges)?;
    timings.set_item("degree_cap", degree_cap)?;
    timings.set_item("degree_pruned_pairs", degree_pruned_pairs)?;
    timings.set_item("mis_iterations", 1)?;
    payload.set_item("timings", timings)?;

    payload.set_item("batch_order_strategy", "hilbert-morton")?;
    payload.set_item("batch_order_metrics", PyDict::new(py))?;
    payload.set_item("engine", "rust-pcct2")?;
    payload.set_item("conflict_grid_cells", conflict_grid_cells)?;
    payload.set_item("conflict_grid_leaders_raw", conflict_grid_leaders_raw)?;
    payload.set_item("conflict_grid_leaders_after", conflict_grid_leaders_after)?;
    payload.set_item("conflict_grid_local_edges", conflict_grid_local_edges)?;

    let bridge = PyModule::import(py, "covertreex.telemetry.rust_bridge")?;
    let recorder = bridge.getattr("record_rust_batch")?;
    recorder.call1((log_writer, payload))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size=None, batch_order=None, log_writer=None, grid_whiten_scale=None, scope_chunk_target=None, conflict_degree_cap=None, scope_budget_schedule=None, scope_budget_up=None, scope_budget_down=None, masked_scope_append=None, scope_chunk_max_segments=None, scope_chunk_pair_merge=None, kernel_type=None))]
fn build_pcct2_residual_tree(
    py: Python<'_>,
    v_matrix: PyObject,
    p_diag: PyObject,
    coords: PyObject,
    rbf_var: f64,
    rbf_ls: PyObject,
    chunk_size: Option<usize>,
    batch_order: Option<String>,
    log_writer: Option<PyObject>,
    grid_whiten_scale: Option<f64>,
    scope_chunk_target: Option<usize>,
    conflict_degree_cap: Option<usize>,
    scope_budget_schedule: Option<Vec<usize>>,
    scope_budget_up: Option<f64>,
    scope_budget_down: Option<f64>,
    masked_scope_append: Option<bool>,
    scope_chunk_max_segments: Option<usize>,
    scope_chunk_pair_merge: Option<bool>,
    kernel_type: Option<i32>,
) -> PyResult<(CoverTreeWrapper, Vec<i64>)> {
    // Force float32 payloads to mirror python-numba fast path.
    let v_matrix_arr = to_array2_f32(&v_matrix.bind(py))?;
    let p_diag_arr = to_array1_f32(&p_diag.bind(py))?;
    let coords_arr = to_array2_f32(&coords.bind(py))?;
    let rbf_ls_arr = to_array1_f32(&rbf_ls.bind(py))?;

    let k_type = kernel_type.unwrap_or(0);

    let order = match batch_order.as_deref() {
        Some(s) if s.eq_ignore_ascii_case("natural") => (0..coords_arr.nrows()).collect(),
        Some(s) if s.eq_ignore_ascii_case("hilbert") => hilbert_like_order(coords_arr.view()),
        Some(s) if s.eq_ignore_ascii_case("hilbert-morton") => {
            hilbert_like_order(coords_arr.view())
        }
        _ => hilbert_like_order(coords_arr.view()),
    };

    // Compute inverse order for query mapping
    let mut inv_order = vec![0i64; order.len()];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        if old_idx < inv_order.len() {
            inv_order[old_idx] = new_idx as i64;
        }
    }

    // Prepare reordered data for caching and building
    let mut v_ordered = Array2::<f32>::zeros((order.len(), v_matrix_arr.ncols()));
    let mut c_ordered = Array2::<f32>::zeros((order.len(), coords_arr.ncols()));
    let mut p_ordered = Vec::with_capacity(order.len());
    
    for (dst, &src_idx) in order.iter().enumerate() {
        v_ordered.row_mut(dst).assign(&v_matrix_arr.row(src_idx));
        c_ordered.row_mut(dst).assign(&coords_arr.row(src_idx));
        p_ordered.push(p_diag_arr[src_idx]);
    }

    let cached_data = CachedResidualData {
        v_matrix: v_ordered.clone(),
        p_diag: p_ordered.clone(),
        coords: c_ordered.clone(),
        rbf_ls: rbf_ls_arr.as_slice().unwrap().to_vec(),
        rbf_var: rbf_var as f32,
        kernel_type: k_type,
    };

    // Indices array now contains 0..N (new indices), matching the reordered data
    let indices_f32: Vec<f32> = (0..order.len()).map(|i| i as f32).collect();
    let indices_arr = Array2::from_shape_vec((order.len(), 1), indices_f32).expect("shape for indices");

    let parse_bool_env = |key: &str, default: Option<bool>| -> Option<bool> {
        std::env::var(key)
            .ok()
            .and_then(|v| {
                let norm = v.to_lowercase();
                if norm == "1" || norm == "true" || norm == "yes" || norm == "on" {
                    Some(true)
                } else if norm == "0" || norm == "false" || norm == "no" || norm == "off" {
                    Some(false)
                } else {
                    default
                }
            })
            .or(default)
    };

    let scope_chunk_pair_merge = scope_chunk_pair_merge
        .or_else(|| parse_bool_env("COVERTREEX_SCOPE_CHUNK_PAIR_MERGE", None))
        .unwrap_or(true);
    let masked_scope_append = masked_scope_append
        .or_else(|| parse_bool_env("COVERTREEX_RESIDUAL_MASKED_SCOPE_APPEND", None));
    let scope_chunk_max_segments = scope_chunk_max_segments.or_else(|| {
        std::env::var("COVERTREEX_SCOPE_CHUNK_MAX_SEGMENTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    });

    let dummy = Array2::<f32>::zeros((0, 1));
    let empty_i64 = Array1::<i64>::zeros(0);
    let empty_i32 = Array1::<i32>::zeros(0);
    let mut tree = CoverTreeWrapper {
        inner: CoverTreeInner::F32(CoverTreeData::new(
            dummy,
            empty_i64.to_vec(),
            empty_i64.to_vec(),
            empty_i64.to_vec(),
            empty_i32.to_vec(),
            -20,
            20,
        )),
        survivors: Vec::new(),
        last_query_telemetry: None,
        inv_order: Some(inv_order),
        cached_data: Some(cached_data),
        order: Some(order.iter().map(|&i| i as i64).collect()),
    };

    let metric = ResidualMetric::new(
        v_ordered.view(),
        &p_ordered,
        c_ordered.view(),
        rbf_var as f32,
        rbf_ls_arr.as_slice().unwrap(),
        k_type,
        None,
    );

    let chunk = chunk_size.unwrap_or_else(|| indices_arr.nrows());
    let mut start = 0usize;
    let mut batch_idx = 0usize;
    let mut survivors: Vec<i64> = Vec::new();
    while start < indices_arr.nrows() {
        let end = std::cmp::min(start + chunk, indices_arr.nrows());
        let view = indices_arr.slice(ndarray::s![start..end, ..]);
        let coords_view = c_ordered.slice(ndarray::s![start..end, ..]);
        let telemetry = batch_insert_with_telemetry(
            match &mut tree.inner {
                CoverTreeInner::F32(data) => data,
                _ => unreachable!(),
            },
            view,
            Some(coords_view),
            &metric,
            grid_whiten_scale.map(|v| v as f32),
            scope_chunk_target,
            conflict_degree_cap,
            scope_budget_schedule.as_deref(),
            scope_budget_up,
            scope_budget_down,
            masked_scope_append,
            scope_chunk_max_segments,
            Some(scope_chunk_pair_merge),
        );

        for sel in telemetry.selected.iter() {
            survivors.push((telemetry.batch_start_index + *sel) as i64);
        }

        if let Some(ref writer) = log_writer {
            emit_rust_batch(py, writer, batch_idx, telemetry)?;
        }

        batch_idx += 1;
        start = end;
    }

    // Correct node_to_dataset mapping: node_i corresponds to batch row i (new_idx).
    // Map new_idx -> old_idx via order.
    let node_to_dataset: Vec<i64> = order.iter().map(|&i| i as i64).collect();
    tree.survivors = survivors;

    // Compute si_cache using 0..N indices (new_idx), since tree nodes contain 0..N and metric uses 0..N data.
    // We pass IDENTITY mapping for si_cache calculation because the metric expects new_idx.
    let identity_map: Vec<i64> = (0..order.len() as i64).collect();
    
    let si_cache = compute_si_cache_residual(
        match &tree.inner {
            CoverTreeInner::F32(data) => data,
            _ => unreachable!(),
        },
        identity_map.as_slice(),
        &metric,
    );
    match &mut tree.inner {
        CoverTreeInner::F32(data) => data.set_si_cache(si_cache),
        _ => unreachable!(),
    }

    Ok((tree, node_to_dataset))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn knn_query_residual_block<'py>(
    py: Python<'py>,
    tree: &CoverTreeWrapper,
    query_indices: numpy::PyReadonlyArray1<'py, i64>,
    node_to_dataset: Vec<i64>,
    v_matrix: PyObject,
    p_diag: PyObject,
    coords: PyObject,
    rbf_var: f64,
    rbf_ls: PyObject,
    k: usize,
    kernel_type: Option<i32>,
) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
    tree.knn_query_residual_block(
        py,
        query_indices,
        node_to_dataset,
        v_matrix,
        p_diag,
        coords,
        rbf_var,
        rbf_ls,
        k,
        kernel_type,
    )
}
