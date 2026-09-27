use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] tensor::TensorError),

    #[error("Token ID {token_id} exceeds vocabulary size {vocab_size}")]
    TokenIdOutOfBounds { token_id: u32, vocab_size: usize },

    #[error("Position {pos} exceeds maximum sequence length {max_seq_len}")]
    PositionOutOfBounds { pos: usize, max_seq_len: usize },

    #[error(
        "Embedding dimension mismatch: token embeddings have dim {wte_dim}, but position embeddings have dim {wpe_dim}"
    )]
    DimensionMismatch { wte_dim: usize, wpe_dim: usize },

    #[error("Sequence is too long: length {sequence_length}, maximum {max_positions}")]
    SequenceTooLong {
        sequence_length: usize,
        max_positions: usize,
    },
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;
