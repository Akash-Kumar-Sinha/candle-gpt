use crate::{Result, TensorError, TensorOps};

#[derive(Debug)]
pub struct Tensor<const R: usize> {
    pub data: Vec<f32>,
    pub shape: [usize; R],
}

impl<const R: usize> Tensor<R> {
    pub fn new(data: Vec<f32>, shape: [usize; R]) -> Result<Self> {
        let expected_size = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::LengthMismatch {
                actual: data.len(),
                expected: expected_size,
                shape: shape.to_vec(),
                ndim: R,
            });
        }
        Ok(Tensor { data, shape })
    }

    fn raw_new_val(data: f32, shape: [usize; R]) -> Result<Self> {
        let length = shape.iter().product();
        let data = vec![data; length];

        Self::new(data, shape)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                ops: String::from("add"),
            });
        }

        let shape = self.shape;

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<f32>>();

        Self::new(data, shape)
    }
}

impl<const R: usize> TensorOps<R> for Tensor<R> {
    fn ones(shape: [usize; R]) -> Result<Self> {
        Self::raw_new_val(1.0, shape)
    }

    fn zeros(shape: [usize; R]) -> Result<Self> {
        Self::raw_new_val(0.0, shape)
    }
    fn add(&self, other: &Self) -> Result<Self> {
        Self::add(&self, other)
    }
}
