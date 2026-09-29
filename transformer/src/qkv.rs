use tensor::Tensor;

use crate::linear;

pub fn qkv(x: &Tensor<2>, weight: &Tensor<2>, bias: &Tensor<1>) -> Tensor<2> {
    linear(x, weight, bias)
}

pub fn split_qkv(qkv: &Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    let seq_len = qkv.shape[0];
    let qkv_dim = qkv.shape[1];

    let embedding_dim = qkv_dim / 3;

    let mut query = Vec::with_capacity(seq_len * embedding_dim);
    let mut key = Vec::with_capacity(seq_len * embedding_dim);
    let mut value = Vec::with_capacity(seq_len * embedding_dim);

    for row in 0..seq_len {
        let start = row * qkv_dim;

        query.extend_from_slice(&qkv.data[start..start + embedding_dim]);

        key.extend_from_slice(&qkv.data[start + embedding_dim..start + 2 * embedding_dim]);

        value.extend_from_slice(&qkv.data[start + 2 * embedding_dim..start + 3 * embedding_dim]);
    }

    (
        Tensor::new(query, [seq_len, embedding_dim]).unwrap(),
        Tensor::new(key, [seq_len, embedding_dim]).unwrap(),
        Tensor::new(value, [seq_len, embedding_dim]).unwrap(),
    )
}
