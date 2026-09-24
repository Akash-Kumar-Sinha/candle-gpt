use crate::Result;

pub trait TensorOps<const R: usize>: Sized {
    fn ones(shape: [usize; R]) -> Result<Self>;
    fn zeros(shape: [usize; R]) -> Result<Self>;
    fn add(&self, other: &Self) -> Result<Self>;
}
