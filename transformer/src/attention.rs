use rayon::prelude::*;
use tensor::{Tensor, TensorOps};

use crate::activation::softmax;

#[derive(Debug, Clone, Default)]
pub struct LayerKVCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub seq_len: usize,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            k: Vec::with_capacity(1024 * 768),
            v: Vec::with_capacity(1024 * 768),
            seq_len: 0,
        }
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.seq_len = 0;
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKVCache::new()).collect(),
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn seq_len(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers[0].seq_len
        }
    }
}

pub fn first_attention_head(
    query: &Tensor<2>,
    key: &Tensor<2>,
    value: &Tensor<2>,
) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    let seq_len = query.shape[0];
    let head_dim = 64;

    let mut query_head = Tensor::zeros([seq_len, head_dim]).expect("Failed to create query head");

    let mut key_head = Tensor::zeros([seq_len, head_dim]).expect("Failed to create key head");

    let mut value_head = Tensor::zeros([seq_len, head_dim]).expect("Failed to create value head");

    for row in 0..seq_len {
        let src_start = row * 768;
        let dst_start = row * head_dim;

        query_head.data[dst_start..dst_start + head_dim]
            .copy_from_slice(&query.data[src_start..src_start + head_dim]);

        key_head.data[dst_start..dst_start + head_dim]
            .copy_from_slice(&key.data[src_start..src_start + head_dim]);

        value_head.data[dst_start..dst_start + head_dim]
            .copy_from_slice(&value.data[src_start..src_start + head_dim]);
    }

    (query_head, key_head, value_head)
}

pub fn scale_attention_scores(scores: &Tensor<2>) -> Tensor<2> {
    let head_dim = 64;
    let scale = (head_dim as f32).sqrt();

    let mut scaled = scores.clone();
    scaled.data.par_iter_mut().for_each(|v| *v /= scale);
    scaled
}

pub fn causal_mask(scores: &mut Tensor<2>) {
    let seq_len = scores.shape[0];

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                scores.data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
}

pub fn attention_head(
    query_head: &Tensor<2>,
    key_head: &Tensor<2>,
    value_head: &Tensor<2>,
) -> Tensor<2> {
    let key_head_t = key_head.transpose().expect("Failed to transpose key head");

    let scores =
        Tensor::matmul(query_head, &key_head_t).expect("Failed to calculate attention scores");

    let mut scaled_scores = scale_attention_scores(&scores);

    causal_mask(&mut scaled_scores);

    let attention_weights = softmax(&scaled_scores);

    Tensor::matmul(&attention_weights, value_head).expect("Failed to calculate attention output")
}

pub fn get_attention_head(
    q: &Tensor<2>,
    k: &Tensor<2>,
    v: &Tensor<2>,
    head_index: usize,
) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    let seq_len = q.shape[0];
    let head_dim = 64;

    let start = head_index * head_dim;
    let end = start + head_dim;

    let mut query = Tensor::zeros([seq_len, head_dim]).unwrap();
    let mut key = Tensor::zeros([seq_len, head_dim]).unwrap();
    let mut value = Tensor::zeros([seq_len, head_dim]).unwrap();

    for row in 0..seq_len {
        let src_start = row * q.shape[1];
        let dst_start = row * head_dim;

        query.data[dst_start..dst_start + head_dim]
            .copy_from_slice(&q.data[src_start + start..src_start + end]);

        key.data[dst_start..dst_start + head_dim]
            .copy_from_slice(&k.data[src_start + start..src_start + end]);

        value.data[dst_start..dst_start + head_dim]
            .copy_from_slice(&v.data[src_start + start..src_start + end]);
    }

    (query, key, value)
}
