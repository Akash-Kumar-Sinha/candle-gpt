use rayon::prelude::*;
use tensor::Tensor;

use crate::gelu;

pub fn mlp(
    x: &Tensor<2>,
    c_fc_weight: &Tensor<2>,
    c_fc_bias: &Tensor<1>,
    c_proj_weight: &Tensor<2>,
    c_proj_bias: &Tensor<1>,
) -> Tensor<2> {
    let h = linear(x, c_fc_weight, c_fc_bias);

    let a = gelu(&h);

    linear(&a, c_proj_weight, c_proj_bias)
}

pub fn linear(x: &Tensor<2>, weight: &Tensor<2>, bias: &Tensor<1>) -> Tensor<2> {
    let output_dim = weight.shape[1];
    let mut output = Tensor::matmul(x, weight).expect("linear matmul failed");

    output.data.par_chunks_mut(output_dim).for_each(|row| {
        for (out, &b) in row.iter_mut().zip(bias.data.iter()) {
            *out += b;
        }
    });

    output
}
