use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer from '{path}': {source}")]
    LoadFile {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Failed to load tokenizer: {0}")]
    Load(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Failed to encode text: {0}")]
    Encode(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Failed to decode token IDs: {0}")]
    Decode(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("Token not found in vocabulary: '{0}'")]
    TokenNotFound(String),

    #[error("Token ID not found in vocabulary: {0}")]
    IdNotFound(u32),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
