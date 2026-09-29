use rayon::prelude::*;
use tensor::{Tensor, TensorOps};

const PI: f32 = std::f32::consts::PI;

pub fn softmax(x: &Tensor<2>) -> Tensor<2> {
    let rows = x.shape[0];
    let cols = x.shape[1];

    let mut output = Tensor::zeros([rows, cols]).expect("Failed to create softmax output");

    output
        .data
        .par_chunks_mut(cols)
        .zip(x.data.par_chunks(cols))
        .for_each(|(out_row, in_row)| {
            let max = in_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let mut sum = 0.0;
            for (out_val, &in_val) in out_row.iter_mut().zip(in_row.iter()) {
                let value = (in_val - max).exp();
                *out_val = value;
                sum += value;
            }

            let inv_sum = 1.0 / sum;
            for out_val in out_row.iter_mut() {
                *out_val *= inv_sum;
            }
        });

    output
}

pub fn gelu(x: &Tensor<2>) -> Tensor<2> {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    let coeff = 0.044715f32;

    let data = x
        .data
        .par_iter()
        .map(|&v| {
            let cube = v * v * v;
            let inner = sqrt_2_over_pi * (v + coeff * cube);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();

    Tensor {
        data,
        shape: x.shape,
    }
}
