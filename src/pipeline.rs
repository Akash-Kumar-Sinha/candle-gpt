use candle_tokenizer::{CandleTokenizer, Tokenizer, TokenizerError};
use embeddings::{EmbeddingError, Embeddings};
use tensor::Tensor;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}

#[derive(Debug, Clone)]
pub struct PipelineOutput {
    #[allow(dead_code)]
    pub prompt: String,
    pub tokens: Vec<u32>,
    pub embeddings_len: usize,
    pub reconstructed_tokens: Vec<u32>,
    pub decoded_text: String,
}

pub struct CandlePipeline {
    tokenizer: CandleTokenizer,
    embeddings: Embeddings,
}

impl CandlePipeline {
    pub fn new() -> Result<Self, PipelineError> {
        let tokenizer = CandleTokenizer::new()?;
        let embeddings = Embeddings::new()?;
        Ok(Self {
            tokenizer,
            embeddings,
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        Ok(self.tokenizer.encode(prompt)?)
    }

    pub fn get_embeddings(&self, tokens: &[u32]) -> Result<Tensor<2>, PipelineError> {
        Ok(self.embeddings.token_embeddings(tokens)?)
    }

    pub fn reconstruct_tokens(&self, embeddings: &Tensor<2>) -> Result<Vec<u32>, PipelineError> {
        Ok(self.embeddings.embeddings_to_token(embeddings)?)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, PipelineError> {
        Ok(self.tokenizer.decode(tokens)?)
    }

    pub fn process(&self, prompt: &str) -> Result<PipelineOutput, PipelineError> {
        let tokens = self.encode(prompt)?;
        let token_embeddings = self.get_embeddings(&tokens)?;
        let embeddings_len = token_embeddings.data.len();
        let reconstructed_tokens = self.reconstruct_tokens(&token_embeddings)?;
        let decoded_text = self.decode(&reconstructed_tokens)?;

        Ok(PipelineOutput {
            prompt: prompt.to_string(),
            tokens,
            embeddings_len,
            reconstructed_tokens,
            decoded_text,
        })
    }
}
