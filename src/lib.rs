use crate::algo::batch::{batch_insert, batch_insert_with_telemetry, BatchInsertTelemetry};
use crate::algo::{
    batch_knn_query, batch_residual_knn_query, batch_residual_knn_query_block_sgemm,
    compute_si_cache_residual, debug_stats_snapshot, set_debug_stats_enabled, take_debug_stats,
};
use crate::metric::{Euclidean, ResidualMetric};
use crate::pcct::hilbert_like_order;
use crate::telemetry::ResidualQueryTelemetry;
use crate::tree::CoverTreeData;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
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

/// A simple wrapper for the Cover Tree core logic
#[pyclass]
struct CoverTreeWrapper {
    inner: CoverTreeInner,
    survivors: Vec<i64>,
    last_query_telemetry: Option<ResidualQueryTelemetry>,
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

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (batch_indices, v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size=None))]
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
    ) -> PyResult<()> {
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
                let (idx, dst) = to_py_arrays(py, indices, dists);
                Ok((idx, dst.into_any().into()))
            }
            CoverTreeInner::F64(data) => {
                let q_obj = queries.extract::<PyReadonlyArray2<f64>>(py)?;
                let q_view = q_obj.as_array();
                let (indices, dists) = batch_knn_query(data, q_view, k);
                let (idx, dst) = to_py_arrays(py, indices, dists);
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
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
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
                    None,
                );

                let scope_caps = load_scope_caps(py);
                let telemetry_enabled = std::env::var("COVERTREEX_RUST_QUERY_TELEMETRY")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let mut telemetry_rec: Option<ResidualQueryTelemetry> = None;

                let (indices, dists) = if telemetry_enabled {
                    let mut telem = ResidualQueryTelemetry::default();
                    let res = batch_residual_knn_query(
                        data,
                        query_indices.as_array(),
                        &node_to_dataset,
                        &metric,
                        k,
                        scope_caps.as_ref(),
                        Some(&mut telem),
                    );
                    telemetry_rec = Some(telem);
                    res
                } else {
                    batch_residual_knn_query(
                        data,
                        query_indices.as_array(),
                        &node_to_dataset,
                        &metric,
                        k,
                        scope_caps.as_ref(),
                        None,
                    )
                };
                self.last_query_telemetry = telemetry_rec;
                let (idx, dst) = to_py_arrays(py, indices, dists);
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
                    None,
                );

                let scope_caps = load_scope_caps(py);
                // Convert f32 caps to f64 if needed, or just implement similar logic for F64
                // Since scope_caps is HashMap<i32, f32>, we might need to map it or change signature.
                // batch_residual_knn_query is generic on T.
                // But scope_caps map is f32.
                // We need to convert it or make it generic?
                // Simpler: just map it for f64 case.
                let caps_f64: Option<HashMap<i32, f64>> =
                    scope_caps.map(|m| m.into_iter().map(|(k, v)| (k, v as f64)).collect());

                let telemetry_enabled = std::env::var("COVERTREEX_RUST_QUERY_TELEMETRY")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let mut telemetry_rec: Option<ResidualQueryTelemetry> = None;

                let (indices, dists) = if telemetry_enabled {
                    let mut telem = ResidualQueryTelemetry::default();
                    let res = batch_residual_knn_query(
                        data,
                        query_indices.as_array(),
                        &node_to_dataset,
                        &metric,
                        k,
                        caps_f64.as_ref(),
                        Some(&mut telem),
                    );
                    telemetry_rec = Some(telem);
                    res
                } else {
                    batch_residual_knn_query(
                        data,
                        query_indices.as_array(),
                        &node_to_dataset,
                        &metric,
                        k,
                        caps_f64.as_ref(),
                        None,
                    )
                };
                self.last_query_telemetry = telemetry_rec;
                let (idx, dst) = to_py_arrays(py, indices, dists);
                Ok((idx, dst.into_any().into()))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
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
                        None,
                    ),
                };
                let (idx, dst) = to_py_arrays(py, indices, dists);
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

fn to_py_arrays<'py, T: numpy::Element + Copy + num_traits::Zero>(
    py: Python<'py>,
    indices: Vec<Vec<i64>>,
    dists: Vec<Vec<T>>,
) -> (
    Bound<'py, numpy::PyArray2<i64>>,
    Bound<'py, numpy::PyArray2<T>>,
) {
    let n_queries = indices.len();
    let dim_k = if n_queries > 0 { indices[0].len() } else { 0 };

    let mut idx_array = ndarray::Array2::<i64>::zeros((n_queries, dim_k));
    let mut dst_array = ndarray::Array2::<T>::zeros((n_queries, dim_k));

    for i in 0..n_queries {
        for j in 0..indices[i].len() {
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
    Ok(())
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

#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size=None, batch_order=None))]
fn build_pcct_residual_tree(
    py: Python<'_>,
    v_matrix: PyObject,
    p_diag: PyObject,
    coords: PyObject,
    rbf_var: f64,
    rbf_ls: PyObject,
    chunk_size: Option<usize>,
    batch_order: Option<String>,
) -> PyResult<(CoverTreeWrapper, Vec<i64>)> {
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
    let mut coords_for_order_owned: Option<ndarray::Array2<f32>> = None;
    let coords_for_order = if let Some(c) = coords_f32.as_ref() {
        c.view()
    } else {
        let tmp: ndarray::Array2<f32> = coords_f64.as_ref().unwrap().mapv(|v| v as f32);
        coords_for_order_owned = Some(tmp);
        coords_for_order_owned.as_ref().unwrap().view()
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
        }
    };

    let metric_f32 = v_matrix_f32.as_ref().map(|v| {
        ResidualMetric::new(
            v.view(),
            p_diag_f32.as_ref().unwrap().as_slice().unwrap(),
            coords_f32.as_ref().unwrap().view(),
            rbf_var as f32,
            rbf_ls_f32.as_ref().unwrap().as_slice().unwrap(),
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
#[pyfunction(signature = (v_matrix, p_diag, coords, rbf_var, rbf_ls, chunk_size=None, batch_order=None, log_writer=None, grid_whiten_scale=None, scope_chunk_target=None, conflict_degree_cap=None, scope_budget_schedule=None, scope_budget_up=None, scope_budget_down=None, masked_scope_append=None, scope_chunk_max_segments=None, scope_chunk_pair_merge=None))]
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
) -> PyResult<(CoverTreeWrapper, Vec<i64>)> {
    // Force float32 payloads to mirror python-numba fast path.
    let v_matrix_arr = to_array2_f32(&v_matrix.bind(py))?;
    let p_diag_arr = to_array1_f32(&p_diag.bind(py))?;
    let coords_arr = to_array2_f32(&coords.bind(py))?;
    let rbf_ls_arr = to_array1_f32(&rbf_ls.bind(py))?;

    let order = match batch_order.as_deref() {
        Some(s) if s.eq_ignore_ascii_case("natural") => (0..coords_arr.nrows()).collect(),
        Some(s) if s.eq_ignore_ascii_case("hilbert") => hilbert_like_order(coords_arr.view()),
        Some(s) if s.eq_ignore_ascii_case("hilbert-morton") => {
            hilbert_like_order(coords_arr.view())
        }
        _ => hilbert_like_order(coords_arr.view()),
    };

    let mut indices_f32 = Vec::with_capacity(order.len());
    for &idx in &order {
        indices_f32.push(idx as f32);
    }
    let indices_arr =
        Array2::from_shape_vec((order.len(), 1), indices_f32).expect("shape for indices");

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
    };

    let metric = ResidualMetric::new(
        v_matrix_arr.view(),
        p_diag_arr.as_slice().unwrap(),
        coords_arr.view(),
        rbf_var as f32,
        rbf_ls_arr.as_slice().unwrap(),
        None,
    );

    // Reorder coords to match insertion permutation
    let mut coords_ordered = Array2::<f32>::zeros((order.len(), coords_arr.ncols()));
    for (dst, &src_idx) in order.iter().enumerate() {
        coords_ordered.row_mut(dst).assign(&coords_arr.row(src_idx));
    }

    let chunk = chunk_size.unwrap_or_else(|| indices_arr.nrows());
    let mut start = 0usize;
    let mut batch_idx = 0usize;
    let mut survivors: Vec<i64> = Vec::new();
    while start < indices_arr.nrows() {
        let end = std::cmp::min(start + chunk, indices_arr.nrows());
        let view = indices_arr.slice(ndarray::s![start..end, ..]);
        let coords_view = coords_ordered.slice(ndarray::s![start..end, ..]);
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

    let node_to_dataset: Vec<i64> = order.iter().map(|&i| i as i64).collect();
    tree.survivors = survivors;

    let si_cache = compute_si_cache_residual(
        match &tree.inner {
            CoverTreeInner::F32(data) => data,
            _ => unreachable!(),
        },
        node_to_dataset.as_slice(),
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
    )
}
