use crate::algo::batch::batch_insert;
use crate::algo::{
    batch_knn_query, batch_residual_knn_query, debug_stats_snapshot, set_debug_stats_enabled,
    take_debug_stats,
};
use crate::metric::{Euclidean, ResidualMetric};
use crate::tree::CoverTreeData;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

mod algo;
mod metric;
mod tree;

/// A simple wrapper for the Cover Tree core logic
#[pyclass]
struct CoverTreeWrapper {
    inner: CoverTreeInner,
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

    #[allow(clippy::too_many_arguments)]
    fn knn_query_residual<'py>(
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
                );

                let (indices, dists) = batch_residual_knn_query(
                    data,
                    query_indices.as_array(),
                    &node_to_dataset,
                    &metric,
                    k,
                );
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
                );

                let (indices, dists) = batch_residual_knn_query(
                    data,
                    query_indices.as_array(),
                    &node_to_dataset,
                    &metric,
                    k,
                );
                let (idx, dst) = to_py_arrays(py, indices, dists);
                Ok((idx, dst.into_any().into()))
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
