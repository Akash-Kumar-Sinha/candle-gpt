use crate::{Result, Tensor};

pub trait TensorOps<const R: usize>: Sized {
    fn ones(shape: [usize; R]) -> Result<Self>;
    fn zeros(shape: [usize; R]) -> Result<Self>;
    fn add(&self, other: &Self) -> Result<Self>;
    fn matmul(a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
    fn transpose(&self) -> Result<Self>;
}
