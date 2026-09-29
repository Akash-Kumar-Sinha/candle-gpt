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

    #[error(
        "Matrix multiplication shape mismatch: lhs columns ({a_cols}) must match rhs rows ({b_rows}) (lhs: {lhs:?}, rhs: {rhs:?})"
    )]
    MatmulShapeMismatch {
        a_cols: usize,
        b_rows: usize,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    #[error(
        "Matrix multiplication requires at least 2 dimensions, found tensor with {ndim} dimensions"
    )]
    InvalidRank { ndim: usize },

    #[error(
        "Batch dimension mismatch for matmul: batch dim {dim_idx} differs (lhs: {lhs_dim}, rhs: {rhs_dim})"
    )]
    BatchDimMismatch {
        dim_idx: usize,
        lhs_dim: usize,
        rhs_dim: usize,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
}

pub type Result<T> = std::result::Result<T, TensorError>;
