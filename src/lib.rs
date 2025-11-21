use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, IntoPyArray};
use crate::tree::CoverTreeData;
use crate::algo::batch::batch_insert;
use crate::algo::batch_knn_query;
use crate::algo::batch_residual_knn_query;
use crate::metric::ResidualMetric;

mod tree;
mod metric;
mod algo;

/// A simple wrapper for the Cover Tree core logic
#[pyclass]
struct CoverTreeWrapper {
    data: CoverTreeData,
}

#[pymethods]
impl CoverTreeWrapper {
    // ... existing new/insert ...
    #[new]
    fn new(
        points: PyReadonlyArray2<f32>,
        parents: Vec<i64>,
        children: Vec<i64>,
        next_node: Vec<i64>,
        levels: Vec<i32>,
        min_level: i32,
        max_level: i32,
    ) -> PyResult<Self> {
        let points_owned = points.as_array().to_owned();
        let data = CoverTreeData::new(
            points_owned, parents, children, next_node, levels, min_level, max_level
        );
        Ok(CoverTreeWrapper { data })
    }
    
    fn insert(&mut self, batch: PyReadonlyArray2<f32>) -> PyResult<()> {
        let batch_view = batch.as_array();
        batch_insert(&mut self.data, batch_view);
        Ok(())
    }

    fn point_count(&self) -> usize {
        self.data.len()
    }
    
    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, Bound<'py, numpy::PyArray2<f32>>)> {
        let q_view = queries.as_array();
        let (indices, dists) = batch_knn_query(&self.data, q_view, k);
        
        // Convert Vec<Vec<T>> to 2D Array
        let n_queries = indices.len();
        let dim_k = if n_queries > 0 { indices[0].len() } else { 0 };
        
        let mut idx_array = ndarray::Array2::<i64>::zeros((n_queries, dim_k));
        let mut dst_array = ndarray::Array2::<f32>::zeros((n_queries, dim_k));
        
        for i in 0..n_queries {
            for j in 0..indices[i].len() {
                idx_array[[i, j]] = indices[i][j];
                dst_array[[i, j]] = dists[i][j];
            }
        }
        
        Ok((
            idx_array.into_pyarray(py),
            dst_array.into_pyarray(py),
        ))
    }
    
    #[allow(clippy::too_many_arguments)]
    fn knn_query_residual<'py>(
        &self,
        py: Python<'py>,
        query_indices: numpy::PyReadonlyArray1<i64>,
        node_to_dataset: Vec<i64>,
        v_matrix: PyReadonlyArray2<f32>,
        p_diag: numpy::PyReadonlyArray1<f32>,
        coords: PyReadonlyArray2<f32>,
        rbf_var: f32,
        rbf_ls: numpy::PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, Bound<'py, numpy::PyArray2<f32>>)> {
        
        let metric = ResidualMetric {
            v_matrix: v_matrix.as_array(),
            p_diag: p_diag.as_array(),
            coords: coords.as_array(),
            rbf_var,
            rbf_ls_sq: rbf_ls.as_array(), // Assumes squared passed from Python
        };
        
        let (indices, dists) = batch_residual_knn_query(
            &self.data,
            query_indices.as_array(),
            &node_to_dataset,
            &metric,
            k
        );
        
        let n_queries = indices.len();
        let dim_k = if n_queries > 0 { indices[0].len() } else { 0 };
        
        let mut idx_array = ndarray::Array2::<i64>::zeros((n_queries, dim_k));
        let mut dst_array = ndarray::Array2::<f32>::zeros((n_queries, dim_k));
        
        for i in 0..n_queries {
            for j in 0..indices[i].len() {
                idx_array[[i, j]] = indices[i][j];
                dst_array[[i, j]] = dists[i][j];
            }
        }
        
        Ok((
            idx_array.into_pyarray(py),
            dst_array.into_pyarray(py),
        ))
    }
}

/// The Rust backend module for CoverTreeX
#[pymodule]
fn covertreex_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoverTreeWrapper>()?;
    Ok(())
}