use rayon::prelude::*;

use crate::{Result, TensorError, TensorOps};

#[derive(Debug, Clone, PartialEq)]
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

    // C_i = A_i + B_i
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                ops: String::from("add"),
            });
        }

        let mut data = vec![0.0f32; self.data.len()];
        if self.data.len() >= 4096 {
            data.par_iter_mut()
                .zip(self.data.par_iter())
                .zip(other.data.par_iter())
                .for_each(|((out, &a), &b)| *out = a + b);
        } else {
            for i in 0..self.data.len() {
                data[i] = self.data[i] + other.data[i];
            }
        }

        Ok(Tensor {
            data,
            shape: self.shape,
        })
    }

    pub fn matmul(a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        matmul(a, b)
    }

    pub fn transpose(&self) -> Result<Self> {
        transpose(self)
    }
}

impl Tensor<2> {
    pub fn row_slice(&self, row: usize) -> &[f32] {
        let cols = self.shape[1];
        let start = row * cols;
        &self.data[start..start + cols]
    }
}

pub fn transpose<const R: usize>(tensor: &Tensor<R>) -> Result<Tensor<R>> {
    if R < 2 {
        return Err(TensorError::InvalidRank { ndim: R });
    }

    let rows = tensor.shape[R - 2];
    let cols = tensor.shape[R - 1];

    let expected_len: usize = tensor.shape.iter().product();
    if tensor.data.len() != expected_len {
        return Err(TensorError::LengthMismatch {
            actual: tensor.data.len(),
            expected: expected_len,
            shape: tensor.shape.to_vec(),
            ndim: R,
        });
    }

    let mut out_shape = tensor.shape;
    out_shape[R - 2] = cols;
    out_shape[R - 1] = rows;

    if rows == 0 || cols == 0 || tensor.data.is_empty() {
        return Ok(Tensor {
            data: Vec::new(),
            shape: out_shape,
        });
    }

    let batch_count: usize = tensor.shape[..R - 2].iter().product();
    let matrix_size = rows * cols;
    let mut output_data = vec![0.0f32; tensor.data.len()];

    const PARALLEL_THRESHOLD: usize = 4096;
    if output_data.len() < PARALLEL_THRESHOLD {
        for b in 0..batch_count {
            let in_offset = b * matrix_size;
            let out_offset = b * matrix_size;
            for r in 0..rows {
                let r_in = in_offset + r * cols;
                for c in 0..cols {
                    output_data[out_offset + c * rows + r] = tensor.data[r_in + c];
                }
            }
        }
    } else {
        output_data
            .par_chunks_mut(rows)
            .enumerate()
            .for_each(|(global_col_idx, out_row)| {
                let b = global_col_idx / cols;
                let c = global_col_idx % cols;
                let in_batch_offset = b * matrix_size;
                for (r, val) in out_row.iter_mut().enumerate() {
                    *val = tensor.data[in_batch_offset + r * cols + c];
                }
            });
    }

    Ok(Tensor::new(output_data, out_shape).unwrap())
}

pub fn matmul<const R: usize>(a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
    if R < 2 {
        return Err(TensorError::InvalidRank { ndim: R });
    }

    let m = a.shape[R - 2];
    let k_a = a.shape[R - 1];
    let k_b = b.shape[R - 2];
    let n = b.shape[R - 1];

    if k_a != k_b {
        return Err(TensorError::MatmulShapeMismatch {
            a_cols: k_a,
            b_rows: k_b,
            lhs: a.shape.to_vec(),
            rhs: b.shape.to_vec(),
        });
    }

    for i in 0..R - 2 {
        if a.shape[i] != b.shape[i] {
            return Err(TensorError::BatchDimMismatch {
                dim_idx: i,
                lhs_dim: a.shape[i],
                rhs_dim: b.shape[i],
                lhs: a.shape.to_vec(),
                rhs: b.shape.to_vec(),
            });
        }
    }

    let k = k_a;
    let expected_a_len: usize = a.shape.iter().product();
    if a.data.len() != expected_a_len {
        return Err(TensorError::LengthMismatch {
            actual: a.data.len(),
            expected: expected_a_len,
            shape: a.shape.to_vec(),
            ndim: R,
        });
    }

    let expected_b_len: usize = b.shape.iter().product();
    if b.data.len() != expected_b_len {
        return Err(TensorError::LengthMismatch {
            actual: b.data.len(),
            expected: expected_b_len,
            shape: b.shape.to_vec(),
            ndim: R,
        });
    }

    let mut out_shape = a.shape;
    out_shape[R - 2] = m;
    out_shape[R - 1] = n;
    let out_len: usize = out_shape.iter().product();

    if out_len == 0 {
        return Ok(Tensor {
            data: Vec::new(),
            shape: out_shape,
        });
    }

    let mut output_data = vec![0.0f32; out_len];
    if k == 0 {
        return Ok(Tensor {
            data: output_data,
            shape: out_shape,
        });
    }

    let batch_count: usize = a.shape[..R - 2].iter().product();
    let total_rows = batch_count * m;
    let total_ops = total_rows * n * k;
    const PARALLEL_THRESHOLD: usize = 4096;

    if total_ops < PARALLEL_THRESHOLD {
        for batch_idx in 0..batch_count {
            let a_batch = &a.data[batch_idx * (m * k)..(batch_idx + 1) * (m * k)];
            let b_batch = &b.data[batch_idx * (k * n)..(batch_idx + 1) * (k * n)];
            let c_batch = &mut output_data[batch_idx * (m * n)..(batch_idx + 1) * (m * n)];

            for i in 0..m {
                let a_row = &a_batch[i * k..(i + 1) * k];
                let c_row = &mut c_batch[i * n..(i + 1) * n];

                for (p, &a_val) in a_row.iter().enumerate() {
                    let b_row = &b_batch[p * n..(p + 1) * n];
                    for (c_elem, &b_elem) in c_row.iter_mut().zip(b_row.iter()) {
                        *c_elem += a_val * b_elem;
                    }
                }
            }
        }
    } else if m == 1 && n >= 512 {
        const CHUNK_SIZE: usize = 256;
        for batch_idx in 0..batch_count {
            let a_slice = &a.data[batch_idx * k..(batch_idx + 1) * k];
            let b_slice = &b.data[batch_idx * (k * n)..(batch_idx + 1) * (k * n)];
            let out_slice = &mut output_data[batch_idx * n..(batch_idx + 1) * n];

            out_slice
                .par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(c_idx, chunk)| {
                    let j_start = c_idx * CHUNK_SIZE;
                    let chunk_len = chunk.len();
                    for p in 0..k {
                        let a_val = a_slice[p];
                        if a_val != 0.0 {
                            let b_row = &b_slice[p * n + j_start..p * n + j_start + chunk_len];
                            for j in 0..chunk_len {
                                chunk[j] += a_val * b_row[j];
                            }
                        }
                    }
                });
        }
    } else {
        output_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(global_row, c_row)| {
                let batch_idx = global_row / m;
                let i = global_row % m;

                let a_offset = batch_idx * (m * k) + i * k;
                let a_row = &a.data[a_offset..a_offset + k];

                let b_offset = batch_idx * (k * n);
                let b_matrix = &b.data[b_offset..b_offset + (k * n)];

                for (p, &a_val) in a_row.iter().enumerate() {
                    if a_val != 0.0 {
                        let b_slice = &b_matrix[p * n..(p + 1) * n];
                        for (c_elem, &b_elem) in c_row.iter_mut().zip(b_slice.iter()) {
                            *c_elem += a_val * b_elem;
                        }
                    }
                }
            });
    }

    Ok(Tensor::new(output_data, out_shape).unwrap())
}

impl<const R: usize> TensorOps<R> for Tensor<R> {
    fn ones(shape: [usize; R]) -> Result<Self> {
        Self::raw_new_val(1.0, shape)
    }

    fn zeros(shape: [usize; R]) -> Result<Self> {
        Self::raw_new_val(0.0, shape)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        Self::add(self, other)
    }

    fn matmul(a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        Self::matmul(a, b)
    }

    fn transpose(&self) -> Result<Self> {
        Self::transpose(self)
    }
}
