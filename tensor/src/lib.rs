mod error;
mod tensor;
mod traits;

pub use error::{Result, TensorError};
pub use tensor::{Tensor, matmul, transpose};
pub use traits::TensorOps;
