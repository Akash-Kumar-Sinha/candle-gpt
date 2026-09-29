use rayon::prelude::*;
use tensor::Tensor;

pub fn layer_norm(x: &Tensor<2>, weight: &Tensor<1>, bias: &Tensor<1>) -> Tensor<2> {
    let embedding_dim = x.shape[1];
    let eps = 1e-5;

    let mut output = Tensor {
        data: vec![0.0; x.data.len()],
        shape: x.shape,
    };

    output
        .data
        .par_chunks_mut(embedding_dim)
        .zip(x.data.par_chunks(embedding_dim))
        .for_each(|(out_row, in_row)| {
            let mean = in_row.iter().sum::<f32>() / embedding_dim as f32;

            let variance = in_row
                .iter()
                .map(|value| {
                    let diff = value - mean;
                    diff * diff
                })
                .sum::<f32>()
                / embedding_dim as f32;

            let inv_std = 1.0 / (variance + eps).sqrt();

            for i in 0..embedding_dim {
                let normalized = (in_row[i] - mean) * inv_std;
                out_row[i] = normalized * weight.data[i] + bias.data[i];
            }
        });

    output
}
