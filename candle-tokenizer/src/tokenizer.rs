use std::path::Path;
use tokenizers::Tokenizer as GP2Tokenizer;

use crate::error::{Result, TokenizerError};

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}

#[derive(Debug, Clone)]
pub struct CandleTokenizer {
    inner: GP2Tokenizer,
}

const DEFAULT_TOKENIZER_PATH: &str = "weights/tokenizer/tokenizer.json";

impl CandleTokenizer {
    pub fn new() -> Result<Self> {
        Self::from_file(DEFAULT_TOKENIZER_PATH)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let inner = GP2Tokenizer::from_file(path_ref).map_err(|e| TokenizerError::LoadFile {
            path: path_ref.to_path_buf(),
            source: e,
        })?;
        Ok(Self { inner })
    }
}

impl Tokenizer for CandleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(TokenizerError::Encode)?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(TokenizerError::Decode)
    }
}
