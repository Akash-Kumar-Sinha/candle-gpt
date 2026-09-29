use candle_tokenizer::{CandleTokenizer, Tokenizer, TokenizerError};
use embeddings::{EmbeddingError, Embeddings};
use tensor::TensorError;
use thiserror::Error;
use transformer::{KVCache, Transformer};

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub prompt: String,
    pub input_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub full_text: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub min_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub eos_bias: f32,
    pub eos_punct_bias: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            min_tokens: 20,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.15,
            eos_bias: 2.5,
            eos_punct_bias: 10.0,
        }
    }
}

pub struct SimpleRng(u64);

impl SimpleRng {
    pub fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(123456789);
        Self(nanos ^ 0x517cc1b727220a95)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / 16777216.0
    }
}

pub struct CandlePipeline {
    tokenizer: CandleTokenizer,
    embeddings: Embeddings,
    transformer: Transformer,
}

impl CandlePipeline {
    pub fn new() -> Result<Self, PipelineError> {
        let tokenizer = CandleTokenizer::new()?;
        let embeddings = Embeddings::new()?;
        let transformer = Transformer::load_from_dir("weights", 12)?;
        Ok(Self {
            tokenizer,
            embeddings,
            transformer,
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        Ok(self.tokenizer.encode(prompt)?)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, PipelineError> {
        Ok(self.tokenizer.decode(tokens)?)
    }

    pub fn compute_next_logits(&self, tokens: &[u32]) -> Result<Vec<f32>, PipelineError> {
        let h_0 = self.embeddings.token_embeddings(tokens)?;

        let h_final = self.transformer.forward(&h_0);

        Ok(self
            .transformer
            .compute_last_logits(&h_final, &self.embeddings.wte))
    }

    #[allow(dead_code)]
    pub fn predict_next_token(&self, tokens: &[u32]) -> Result<u32, PipelineError> {
        let logits = self.compute_next_logits(tokens)?;
        let (best_token, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        Ok(best_token as u32)
    }

    pub fn sample_next_token_from_hidden(
        &self,
        x_final: &tensor::Tensor<2>,
        tokens: &[u32],
        generated_count: usize,
        config: &GenerationConfig,
        rng: &mut SimpleRng,
    ) -> Result<u32, PipelineError> {
        let mut logits = self
            .transformer
            .compute_last_logits(x_final, &self.embeddings.wte);

        let eos_token_id = 50256usize;

        let effective_min = config.min_tokens.min(config.max_new_tokens / 2);

        if generated_count < effective_min {
            if eos_token_id < logits.len() {
                logits[eos_token_id] = f32::NEG_INFINITY;
            }
        } else if eos_token_id < logits.len() {
            let rho = (generated_count as f32) / (config.max_new_tokens as f32).max(1.0);
            let rho2 = rho * rho;

            let is_punct = tokens
                .last()
                .and_then(|&tok| {
                    self.decode(&[tok]).ok().map(|s| {
                        let trimmed = s.trim_end();
                        trimmed.ends_with('.')
                            || trimmed.ends_with('?')
                            || trimmed.ends_with('!')
                            || trimmed.ends_with('\n')
                            || trimmed.ends_with('"')
                            || trimmed.ends_with('\'')
                    })
                })
                .unwrap_or(false);

            let eos_boost = if is_punct {
                config.eos_bias * rho + config.eos_punct_bias * rho2
            } else {
                config.eos_bias * rho
            };

            logits[eos_token_id] += eos_boost;
        }

        if config.repetition_penalty != 1.0 {
            for &token in tokens {
                let idx = token as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= config.repetition_penalty;
                    } else {
                        logits[idx] *= config.repetition_penalty;
                    }
                }
            }
        }

        if config.temperature <= 0.0 {
            let (best_token, _) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            return Ok(best_token as u32);
        }

        for val in logits.iter_mut() {
            *val /= config.temperature;
        }

        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();

        if config.top_k > 0 && config.top_k < indexed.len() {
            indexed.select_nth_unstable_by(config.top_k - 1, |a, b| b.1.total_cmp(&a.1));
            indexed.truncate(config.top_k);
        }
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let max_logit = indexed
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs: Vec<(u32, f32)> = indexed
            .into_iter()
            .map(|(tok, logit)| {
                let p = (logit - max_logit).exp();
                sum += p;
                (tok, p)
            })
            .collect();

        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }

        if config.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (idx, (_, p)) in probs.iter().enumerate() {
                cumsum += *p;
                if cumsum >= config.top_p {
                    cutoff = idx + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            let new_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
            for (_, p) in probs.iter_mut() {
                *p /= new_sum;
            }
        }

        let u = rng.next_f32();
        let mut cum = 0.0f32;
        let mut selected = probs[0].0;
        for (tok, p) in probs {
            cum += p;
            if u < cum {
                selected = tok;
                break;
            }
        }

        Ok(selected)
    }

    #[allow(dead_code)]
    pub fn sample_next_token(
        &self,
        tokens: &[u32],
        generated_count: usize,
        config: &GenerationConfig,
        rng: &mut SimpleRng,
    ) -> Result<u32, PipelineError> {
        let h_0 = self.embeddings.token_embeddings(tokens)?;
        let h_final = self.transformer.forward(&h_0);
        self.sample_next_token_from_hidden(&h_final, tokens, generated_count, config, rng)
    }

    pub fn generate_stream_with_config<F>(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        mut on_token: F,
    ) -> Result<PipelineOutput, PipelineError>
    where
        F: FnMut(&str, u32),
    {
        let mut rng = SimpleRng::new();
        let input_tokens = self.encode(prompt)?;
        if input_tokens.is_empty() {
            return Ok(PipelineOutput {
                prompt: prompt.to_string(),
                input_tokens: Vec::new(),
                generated_tokens: Vec::new(),
                generated_text: String::new(),
                full_text: String::new(),
            });
        }

        let mut current_tokens = input_tokens.clone();
        let mut generated_tokens = Vec::new();
        let mut prev_text = String::new();

        let mut kv_cache = KVCache::new(self.transformer.blocks.len());
        let prompt_len = input_tokens.len();
        let mut next_token = 50256u32;

        for (pos, &tok_id) in input_tokens.iter().enumerate() {
            let x = self.embeddings.forward_single_token(tok_id, pos);
            let x_out = self.transformer.forward_with_cache(&x, &mut kv_cache);
            if pos == prompt_len - 1 {
                next_token = self.sample_next_token_from_hidden(
                    &x_out,
                    &current_tokens,
                    0,
                    config,
                    &mut rng,
                )?;
            }
        }

        while generated_tokens.len() < config.max_new_tokens && kv_cache.seq_len() < 1024 {
            if next_token == 50256 {
                break;
            }

            current_tokens.push(next_token);
            generated_tokens.push(next_token);

            let curr_text = self.decode(&generated_tokens)?;
            if curr_text.len() > prev_text.len() {
                let chunk = &curr_text[prev_text.len()..];
                on_token(chunk, next_token);
                prev_text = curr_text;
            }

            let curr_pos = kv_cache.seq_len();
            let x = self.embeddings.forward_single_token(next_token, curr_pos);
            let x_out = self.transformer.forward_with_cache(&x, &mut kv_cache);
            next_token = self.sample_next_token_from_hidden(
                &x_out,
                &current_tokens,
                generated_tokens.len(),
                config,
                &mut rng,
            )?;
        }

        let generated_text = self.decode(&generated_tokens)?;
        if generated_text.len() > prev_text.len() {
            let chunk = &generated_text[prev_text.len()..];
            on_token(chunk, *generated_tokens.last().unwrap_or(&0));
        }

        let full_text = self.decode(&current_tokens)?;

        Ok(PipelineOutput {
            prompt: prompt.to_string(),
            input_tokens,
            generated_tokens,
            generated_text,
            full_text,
        })
    }

    #[allow(dead_code)]
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        on_token: F,
    ) -> Result<PipelineOutput, PipelineError>
    where
        F: FnMut(&str, u32),
    {
        let config = GenerationConfig {
            max_new_tokens,
            ..GenerationConfig::default()
        };
        self.generate_stream_with_config(prompt, &config, on_token)
    }

    #[allow(dead_code)]
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<PipelineOutput, PipelineError> {
        self.generate_stream(prompt, max_new_tokens, |_, _| {})
    }

    #[allow(dead_code)]
    pub fn process_stream<F>(
        &self,
        prompt: &str,
        on_token: F,
    ) -> Result<PipelineOutput, PipelineError>
    where
        F: FnMut(&str, u32),
    {
        let config = GenerationConfig::default();
        self.generate_stream_with_config(prompt, &config, on_token)
    }

    #[allow(dead_code)]
    pub fn process(&self, prompt: &str) -> Result<PipelineOutput, PipelineError> {
        self.generate(prompt, 30)
    }
}
