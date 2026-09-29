use thiserror::Error;

#[allow(dead_code)]
#[derive(Error, Debug)]
pub enum TransformerError {}

#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, TransformerError>;
