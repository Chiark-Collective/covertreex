use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};

#[allow(dead_code)]
pub struct CoverTreeData {
    // Flattened points (Row Major): [x0_0, x0_1, ..., x1_0, ...]
    pub points: Vec<f32>, 
    pub parents: Vec<i64>,
    pub children: Vec<i64>, // First child
    pub next_node: Vec<i64>, // Next sibling
    pub levels: Vec<i32>,
    pub dimension: usize,
    pub min_level: i32,
    pub max_level: i32,
}

impl CoverTreeData {
    pub fn new(
        points_array: Array2<f32>,
        parents: Vec<i64>,
        children: Vec<i64>,
        next_node: Vec<i64>,
        levels: Vec<i32>,
        min_level: i32,
        max_level: i32,
    ) -> Self {
        let dimension = points_array.shape()[1];
        // Convert Array2 to Vec
        let points = points_array.as_standard_layout().into_owned().into_raw_vec();
        
        Self {
            points,
            parents,
            children,
            next_node,
            levels,
            dimension,
            min_level,
            max_level,
        }
    }
    
    // ... existing len/get_point ...

    pub fn len(&self) -> usize {
        self.points.len() / self.dimension
    }

    #[allow(dead_code)]
    pub fn get_point(&self, idx: usize) -> ArrayView2<'_, f32> {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        ArrayView2::from_shape((1, self.dimension), &self.points[start..end]).unwrap()
    }
    
    pub fn get_point_row(&self, idx: usize) -> ndarray::ArrayView1<'_, f32> {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        ArrayView1::from_shape((self.dimension,), &self.points[start..end]).unwrap()
    }
    
    pub fn add_point(&mut self, point: ArrayView1<f32>, level: i32, parent: i64) -> usize {
        let idx = self.len();
        self.points.extend(point.iter());
        self.parents.push(parent);
        self.children.push(-1);
        self.next_node.push(-1);
        self.levels.push(level);
        
        // Link to parent (if valid)
        if parent >= 0 {
            self.link_child(parent as usize, idx);
        }
        
        if level != i32::MIN {
            if level > self.max_level { self.max_level = level; }
            if level < self.min_level { self.min_level = level; }
        }
        
        idx
    }
    
    pub fn link_child(&mut self, parent_idx: usize, child_idx: usize) {
        // Prepend to child list: parent.child -> new_node -> old_first_child
        let old_first = self.children[parent_idx];
        self.children[parent_idx] = child_idx as i64;
        self.next_node[child_idx] = old_first;
    }
    
    pub fn set_level(&mut self, idx: usize, level: i32) {
        self.levels[idx] = level;
        if level > self.max_level { self.max_level = level; }
        if level < self.min_level { self.min_level = level; }
    }
    
    pub fn set_parent(&mut self, idx: usize, parent: i64) {
        self.parents[idx] = parent;
        if parent >= 0 {
            self.link_child(parent as usize, idx);
        }
    }
}
