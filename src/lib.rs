use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray1};

/// A simple wrapper for the Cover Tree core logic
#[pyclass]
struct CoverTreeCore {
    // In Phase 1, we just store points to verify ownership
    points: Vec<f32>,
    dimension: usize,
}

#[pymethods]
impl CoverTreeCore {
    #[new]
    fn new(points: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let points_array = points.as_array();
        let shape = points_array.shape();
        let n = shape[0];
        let d = shape[1];
        
        // Copy data into Rust Vector (Row Major)
        // This confirms we can read NumPy data
        let mut vec = Vec::with_capacity(n * d);
        if let Some(slice) = points_array.as_slice() {
            vec.extend_from_slice(slice);
        } else {
            // Fallback for non-contiguous arrays
             for row in points_array.outer_iter() {
                 for &val in row {
                     vec.push(val);
                 }
             }
        }

        Ok(CoverTreeCore {
            points: vec,
            dimension: d,
        })
    }

    fn point_count(&self) -> usize {
        self.points.len() / self.dimension
    }
    
    fn get_point(&self, idx: usize) -> PyResult<Vec<f32>> {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        if end > self.points.len() {
             return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }
        Ok(self.points[start..end].to_vec())
    }
}

/// The Rust backend module for CoverTreeX
#[pymodule]
fn covertreex_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoverTreeCore>()?;
    Ok(())
}
