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
        let vocab_size = self.wte.shape[0];
        let embedding_dim = self.wte.shape[1];
        let max_positions = self.wpe.shape[0];

        if token_ids.len() > max_positions {
            return Err(EmbeddingError::SequenceTooLong {
                sequence_length: token_ids.len(),
                max_positions,
            });
        }

        let mut data = Vec::with_capacity(token_ids.len() * embedding_dim);

        for (position, &token_id) in token_ids.iter().enumerate() {
            let token_id = token_id as usize;

            if token_id >= vocab_size {
                return Err(EmbeddingError::TokenIdOutOfBounds {
                    token_id: token_id as u32,
                    vocab_size,
                });
            }

            let token_start = token_id * embedding_dim;
            let position_start = position * embedding_dim;

            for i in 0..embedding_dim {
                data.push(self.wte.data[token_start + i] + self.wpe.data[position_start + i]);
            }
        }

        Ok(Tensor {
            data,
            shape: [token_ids.len(), embedding_dim],
        })
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

        let mut token_ids = Vec::with_capacity(sequence_length);

        for position in 0..sequence_length {
            let start = position * embedding_dim;
            let end = start + embedding_dim;

            let embedding_slice = &embeddings.data[start..end];

            // Positional embedding for this position
            let position_start = position * embedding_dim;
            let position_end = position_start + embedding_dim;

            let position_embedding = &self.wpe.data[position_start..position_end];

            let mut closest_token_id = 0;
            let mut closest_distance = f32::MAX;

            for token in 0..vocab_size {
                let token_start = token * embedding_dim;
                let token_end = token_start + embedding_dim;

                let token_embedding = &self.wte.data[token_start..token_end];

                let distance: f32 = embedding_slice
                    .iter()
                    .zip(position_embedding.iter())
                    .zip(token_embedding.iter())
                    .map(|((input, position), token)| {
                        let d = (input - position) - token;
                        d * d
                    })
                    .sum();

                if distance < closest_distance {
                    closest_distance = distance;
                    closest_token_id = token as u32;
                }
            }

            token_ids.push(closest_token_id);
        }

        Ok(token_ids)
    }

    pub fn load_embeddings_from_txt() -> Result<Self> {
        let wte = candle_weights::load_2d_tensor_from_txt(
            "weights/transformer.wte.weight.txt",
            50257,
            768,
        )?;
        let wpe = candle_weights::load_2d_tensor_from_txt(
            "weights/transformer.wpe.weight.txt",
            1024,
            768,
        )?;

        Ok(Embeddings { wte, wpe })
    }
}
