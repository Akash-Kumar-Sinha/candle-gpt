mod error;
mod tensor;
mod traits;

pub use error::{Result, TensorError};
pub use tensor::Tensor;
pub use traits::TensorOps;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_tensor() {
        let tensor1 = Tensor::ones([2, 3]).expect("Failed to create ones tensor");
        let tensor2 = Tensor::ones([2, 3]).expect("Failed to create ones tensor");

        let result = tensor1.add(&tensor2).expect("Failed to add two tensor");

        assert!(result.data.iter().all(|a| *a == 2.0));
        println!("[tensor]: {:?}", result);
    }
}
