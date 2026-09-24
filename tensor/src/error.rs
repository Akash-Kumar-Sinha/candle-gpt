use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    #[error(
        "Data length [{actual}] does not match shape {shape:?} (expected {expected} elements across {ndim} dimensions)"
    )]
    LengthMismatch {
        actual: usize,
        expected: usize,
        shape: Vec<usize>,
        ndim: usize,
    },

    #[error("To perform [{ops}] operations, the shapes of both matrices must be the same")]
    ShapeMismatch { ops: String },
}

pub type Result<T> = std::result::Result<T, TensorError>;
