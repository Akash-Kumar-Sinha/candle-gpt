use candle_weights::{LayerWeights, TransformerWeights};
use tensor::{Result, Tensor};

use rayon::prelude::*;

use crate::{
    attention::{KVCache, LayerKVCache, attention_head, get_attention_head},
    layer_norm::layer_norm,
    linear,
    mlp::mlp,
    qkv::{qkv, split_qkv},
};

pub struct TransformerBlock {
    pub weights: LayerWeights,
}

impl TransformerBlock {
    pub fn new(weights: LayerWeights) -> Self {
        Self { weights }
    }

    pub fn forward(&self, x: &Tensor<2>) -> Tensor<2> {
        let x_norm1 = layer_norm(x, &self.weights.ln_1_weight, &self.weights.ln_1_bias);

        let qkv_output = qkv(
            &x_norm1,
            &self.weights.attn_c_attn_weight,
            &self.weights.attn_c_attn_bias,
        );
        let (q, k, v) = split_qkv(&qkv_output);

        let seq_len = q.shape[0];
        let head_dim = 64;
        let num_heads = 12;

        let head_outputs: Vec<Tensor<2>> = (0..num_heads)
            .into_par_iter()
            .map(|head_index| {
                let (query_head, key_head, value_head) = get_attention_head(&q, &k, &v, head_index);
                attention_head(&query_head, &key_head, &value_head)
            })
            .collect();

        let mut multi_head_output = vec![0.0f32; seq_len * head_dim * num_heads];

        for (head_index, head_output) in head_outputs.into_iter().enumerate() {
            for row in 0..seq_len {
                let dst_start = row * (head_dim * num_heads) + head_index * head_dim;
                let src_start = row * head_dim;
                multi_head_output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&head_output.data[src_start..src_start + head_dim]);
            }
        }

        let multi_head = Tensor::new(multi_head_output, [seq_len, head_dim * num_heads]).unwrap();

        let attn_out = linear(
            &multi_head,
            &self.weights.attn_c_proj_weight,
            &self.weights.attn_c_proj_bias,
        );

        let x_res1 = Tensor::add(x, &attn_out).expect("residual add failed");

        let x_norm2 = layer_norm(&x_res1, &self.weights.ln_2_weight, &self.weights.ln_2_bias);

        let mlp_out = mlp(
            &x_norm2,
            &self.weights.mlp_c_fc_weight,
            &self.weights.mlp_c_fc_bias,
            &self.weights.mlp_c_proj_weight,
            &self.weights.mlp_c_proj_bias,
        );

        Tensor::add(&x_res1, &mlp_out).expect("residual add failed")
    }

    pub fn forward_with_cache(&self, x: &Tensor<2>, cache: &mut LayerKVCache) -> Tensor<2> {
        let x_norm1 = layer_norm(x, &self.weights.ln_1_weight, &self.weights.ln_1_bias);

        let qkv_output = qkv(
            &x_norm1,
            &self.weights.attn_c_attn_weight,
            &self.weights.attn_c_attn_bias,
        );

        let head_dim = 64;
        let num_heads = 12;
        let n_embd = head_dim * num_heads;
        let qkv_stride = 3 * n_embd;

        let seq_len = x.shape[0];
        let mut multi_head_output = vec![0.0f32; seq_len * n_embd];

        for row in 0..seq_len {
            let row_offset = row * qkv_stride;
            let q_token = &qkv_output.data[row_offset..row_offset + n_embd];
            let k_token = &qkv_output.data[row_offset + n_embd..row_offset + 2 * n_embd];
            let v_token = &qkv_output.data[row_offset + 2 * n_embd..row_offset + 3 * n_embd];

            cache.k.extend_from_slice(k_token);
            cache.v.extend_from_slice(v_token);
            cache.seq_len += 1;

            let total_seq_len = cache.seq_len;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let head_results: Vec<(usize, Vec<f32>)> = (0..num_heads)
                .into_par_iter()
                .map(|head_index| {
                    let head_offset = head_index * head_dim;
                    let q_head = &q_token[head_offset..head_offset + head_dim];

                    let mut scores = vec![0.0f32; total_seq_len];
                    for j in 0..total_seq_len {
                        let k_head_offset = j * n_embd + head_offset;
                        let k_head = &cache.k[k_head_offset..k_head_offset + head_dim];

                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_head[d] * k_head[d];
                        }
                        scores[j] = dot * scale;
                    }

                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        sum_exp += *s;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for s in scores.iter_mut() {
                        *s *= inv_sum;
                    }

                    let mut head_out = vec![0.0f32; head_dim];
                    for d in 0..head_dim {
                        let mut sum_val = 0.0f32;
                        for j in 0..total_seq_len {
                            let v_head_offset = j * n_embd + head_offset;
                            sum_val += scores[j] * cache.v[v_head_offset + d];
                        }
                        head_out[d] = sum_val;
                    }

                    (head_index, head_out)
                })
                .collect();

            let dst_row_start = row * n_embd;
            for (head_index, head_out) in head_results {
                let dst_head_start = dst_row_start + head_index * head_dim;
                multi_head_output[dst_head_start..dst_head_start + head_dim]
                    .copy_from_slice(&head_out);
            }
        }

        let multi_head = Tensor {
            data: multi_head_output,
            shape: [seq_len, n_embd],
        };

        let attn_out = linear(
            &multi_head,
            &self.weights.attn_c_proj_weight,
            &self.weights.attn_c_proj_bias,
        );

        let x_res1 = Tensor::add(x, &attn_out).expect("residual add failed");

        let x_norm2 = layer_norm(&x_res1, &self.weights.ln_2_weight, &self.weights.ln_2_bias);

        let mlp_out = mlp(
            &x_norm2,
            &self.weights.mlp_c_fc_weight,
            &self.weights.mlp_c_fc_bias,
            &self.weights.mlp_c_proj_weight,
            &self.weights.mlp_c_proj_bias,
        );

        Tensor::add(&x_res1, &mlp_out).expect("residual add failed")
    }
}

pub struct Transformer {
    pub blocks: Vec<TransformerBlock>,
    pub ln_f_weight: Tensor<1>,
    pub ln_f_bias: Tensor<1>,
}

impl Transformer {
    pub fn new(weights: TransformerWeights) -> Self {
        let blocks = weights
            .layers
            .into_iter()
            .map(TransformerBlock::new)
            .collect();
        Self {
            blocks,
            ln_f_weight: weights.ln_f_weight,
            ln_f_bias: weights.ln_f_bias,
        }
    }

    pub fn load_from_dir(weights_dir: &str, num_layers: usize) -> Result<Self> {
        let weights = TransformerWeights::load_from_dir(weights_dir, num_layers)?;
        Ok(Self::new(weights))
    }

    pub fn forward(&self, x: &Tensor<2>) -> Tensor<2> {
        let mut hidden = x.clone();

        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        layer_norm(&hidden, &self.ln_f_weight, &self.ln_f_bias)
    }

    pub fn forward_with_cache(&self, x: &Tensor<2>, kv_cache: &mut KVCache) -> Tensor<2> {
        let mut hidden = x.clone();

        for (idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_with_cache(&hidden, &mut kv_cache.layers[idx]);
        }

        layer_norm(&hidden, &self.ln_f_weight, &self.ln_f_bias)
    }

    pub fn compute_logits(&self, x_final: &Tensor<2>, wte: &Tensor<2>) -> Tensor<2> {
        let wte_t = wte.transpose().expect("Failed to transpose wte");
        Tensor::matmul(x_final, &wte_t).expect("Failed to compute logits")
    }

    pub fn compute_last_logits(&self, x_final: &Tensor<2>, wte: &Tensor<2>) -> Vec<f32> {
        let seq_len = x_final.shape[0];
        let embedding_dim = x_final.shape[1];
        let vocab_size = wte.shape[0];

        let last_row_start = (seq_len - 1) * embedding_dim;
        let last_hidden = &x_final.data[last_row_start..last_row_start + embedding_dim];

        let mut logits = vec![0.0f32; vocab_size];
        const CHUNK_SIZE: usize = 512;
        logits
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(c_idx, chunk)| {
                let start_tok = c_idx * CHUNK_SIZE;
                for (offset, logit) in chunk.iter_mut().enumerate() {
                    let tok = start_tok + offset;
                    let tok_start = tok * embedding_dim;
                    let tok_embedding = &wte.data[tok_start..tok_start + embedding_dim];

                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let chunks = embedding_dim / 4;
                    for i in 0..chunks {
                        let idx = i * 4;
                        s0 += last_hidden[idx] * tok_embedding[idx];
                        s1 += last_hidden[idx + 1] * tok_embedding[idx + 1];
                        s2 += last_hidden[idx + 2] * tok_embedding[idx + 2];
                        s3 += last_hidden[idx + 3] * tok_embedding[idx + 3];
                    }
                    let mut score = s0 + s1 + s2 + s3;
                    for i in (chunks * 4)..embedding_dim {
                        score += last_hidden[i] * tok_embedding[i];
                    }
                    *logit = score;
                }
            });

        logits
    }

    pub fn generate_next_token(&self, x_final: &Tensor<2>, wte: &Tensor<2>) -> u32 {
        let logits = self.compute_last_logits(x_final, wte);

        let mut best_token = 0u32;
        let mut max_score = f32::NEG_INFINITY;
        for (tok, &score) in logits.iter().enumerate() {
            if score > max_score {
                max_score = score;
                best_token = tok as u32;
            }
        }
        best_token
    }
}
