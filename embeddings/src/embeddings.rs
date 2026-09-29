use rayon::prelude::*;

use crate::{EmbeddingError, Result};
use tensor::Tensor;

pub struct Embeddings {
    pub wte: Tensor<2>,
    pub wpe: Tensor<2>,
}

impl Embeddings {
    pub fn new() -> Result<Self> {
        Self::load_embeddings_from_txt()
    }

    pub fn token_embeddings(&self, token_ids: &[u32]) -> Result<Tensor<2>> {
        let embedding_dim = self.wte.shape[1];
        let max_positions = self.wpe.shape[0];

        if token_ids.len() > max_positions {
            return Err(EmbeddingError::SequenceTooLong {
                sequence_length: token_ids.len(),
                max_positions,
            });
        }

        let data: Vec<f32> = token_ids
            .par_iter()
            .enumerate()
            .flat_map_iter(|(position, &token_id)| {
                let token_start = (token_id as usize) * embedding_dim;
                let position_start = position * embedding_dim;

                (0..embedding_dim).map(move |i| {
                    self.wte.data[token_start + i] + self.wpe.data[position_start + i]
                })
            })
            .collect();

        Ok(Tensor {
            data,
            shape: [token_ids.len(), embedding_dim],
        })
    }

    // h_0 = wte[token] + wpe[pos]
    pub fn forward_single_token(&self, token_id: u32, pos: usize) -> Tensor<2> {
        let embedding_dim = self.wte.shape[1];
        let tok_start = (token_id as usize) * embedding_dim;
        let pos_start = pos * embedding_dim;
        let mut data = vec![0.0f32; embedding_dim];
        for i in 0..embedding_dim {
            data[i] = self.wte.data[tok_start + i] + self.wpe.data[pos_start + i];
        }
        Tensor::new(data, [1, embedding_dim]).unwrap()
    }

    pub fn embeddings_to_token(&self, embeddings: &Tensor<2>) -> Result<Vec<u32>> {
        let vocab_size = self.wte.shape[0];
        let embedding_dim = self.wte.shape[1];
        let max_positions = self.wpe.shape[0];

        if embeddings.shape[1] != embedding_dim {
            return Err(EmbeddingError::DimensionMismatch {
                wte_dim: embedding_dim,
                wpe_dim: embeddings.shape[1],
            });
        }

        let sequence_length = embeddings.shape[0];

        if sequence_length > max_positions {
            return Err(EmbeddingError::SequenceTooLong {
                sequence_length,
                max_positions,
            });
        }

        let token_ids: Vec<u32> = (0..sequence_length)
            .into_par_iter()
            .map(|position| {
                let start = position * embedding_dim;
                let end = start + embedding_dim;
                let embedding_slice = &embeddings.data[start..end];

                let position_start = position * embedding_dim;
                let position_end = position_start + embedding_dim;
                let position_embedding = &self.wpe.data[position_start..position_end];

                let mut diff = vec![0.0f32; embedding_dim];
                for i in 0..embedding_dim {
                    diff[i] = embedding_slice[i] - position_embedding[i];
                }

                let mut closest_token_id = 0;
                let mut closest_distance = f32::MAX;

                for token in 0..vocab_size {
                    let token_start = token * embedding_dim;
                    let token_end = token_start + embedding_dim;
                    let token_embedding = &self.wte.data[token_start..token_end];

                    let distance: f32 = diff
                        .iter()
                        .zip(token_embedding.iter())
                        .map(|(&d, &tok)| {
                            let diff_val = d - tok;
                            diff_val * diff_val
                        })
                        .sum();

                    if distance < closest_distance {
                        closest_distance = distance;
                        closest_token_id = token as u32;
                    }
                }

                closest_token_id
            })
            .collect();

        Ok(token_ids)
    }

    pub fn load_embeddings_from_txt() -> Result<Self> {
        let (wte, wpe) = rayon::join(
            || {
                candle_weights::load_2d_tensor_from_txt(
                    "weights/transformer.wte.weight.txt",
                    50257,
                    768,
                )
            },
            || {
                candle_weights::load_2d_tensor_from_txt(
                    "weights/transformer.wpe.weight.txt",
                    1024,
                    768,
                )
            },
        );

        Ok(Embeddings {
            wte: wte?,
            wpe: wpe?,
        })
    }
}
