use std::fs;

use tensor::{Result, Tensor, TensorError};

pub fn load_1d_tensor_from_txt(path: &str, size: usize) -> Result<Tensor<1>> {
    let content = fs::read_to_string(path).expect("Failed to read the content");
    let mut data = Vec::with_capacity(size);

    for line in content.lines() {
        for val in line.split_whitespace() {
            let num = val.parse::<f32>().expect("Failed to parse float");
            data.push(num);
        }
    }

    if data.len() != size {
        return Err(TensorError::LengthMismatch {
            actual: data.len(),
            expected: size,
            shape: vec![size],
            ndim: 1,
        });
    }
    Tensor::new(data, [size])
}

pub fn load_2d_tensor_from_txt(path: &str, rows: usize, cols: usize) -> Result<Tensor<2>> {
    let content = fs::read_to_string(path).expect("Failed to read the content");
    let mut data = Vec::with_capacity(rows * cols);

    for line in content.lines() {
        for val in line.split_whitespace() {
            let num = val.parse::<f32>().expect("Failed to parse float");
            data.push(num);
        }
    }

    if data.len() != rows * cols {
        return Err(TensorError::LengthMismatch {
            actual: data.len(),
            expected: rows * cols,
            shape: vec![rows, cols],
            ndim: 2,
        });
    }
    Tensor::new(data, [rows, cols])
}

pub use load_1d_tensor_from_txt as load_1d_tensor_from_text;
pub use load_2d_tensor_from_txt as load_2d_tensor_from_text;
