use ndarray::ArrayView1;

pub trait Metric: Sync + Send {
    fn distance(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32;
    fn distance_sq(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32;
}

#[derive(Copy, Clone)]
pub struct Euclidean;

impl Metric for Euclidean {
    fn distance(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32 {
        self.distance_sq(p1, p2).sqrt()
    }

    fn distance_sq(&self, p1: ArrayView1<f32>, p2: ArrayView1<f32>) -> f32 {
        // Manual loop often vectorizes better than iterators for simple reductions
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
