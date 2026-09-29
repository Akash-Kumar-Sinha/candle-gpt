mod error;
mod tensor;
mod traits;

pub use error::{Result, TensorError};
pub use tensor::{matmul, transpose, Tensor};
pub use traits::TensorOps;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_tensor() {
        let tensor1 = Tensor::ones([2, 3]).expect("Failed to create ones tensor");
        let tensor2 = Tensor::ones([2, 3]).expect("Failed to create ones tensor");

        let result = tensor1.add(&tensor2).expect("Failed to add two tensor");

        assert!(result.data.iter().all(|a| *a == 2.0));
        println!("[tensor]: {:?}", result);
    }

    #[test]
    fn test_matmul_basic() {
        // [2, 2] x [2, 2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();

        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, [2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);

        // Associated function syntax
        let c_assoc = Tensor::matmul(&a, &b).unwrap();
        assert_eq!(c_assoc.data, c.data);
    }

    #[test]
    fn test_matmul_rectangular() {
        // [2, 3] x [3, 2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0], [3, 2]).unwrap();

        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, [2, 2]);
        // [1*7 + 2*9 + 3*2, 1*8 + 2*1 + 3*3] = [7 + 18 + 6, 8 + 2 + 9] = [31, 19]
        // [4*7 + 5*9 + 6*2, 4*8 + 5*1 + 6*3] = [28 + 45 + 12, 32 + 5 + 18] = [85, 55]
        assert_eq!(c.data, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_matmul_batched_3d() {
        // [2, 2, 2] x [2, 2, 2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 2.0], [2, 2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], [2, 2, 2]).unwrap();

        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, [2, 2, 2]);

        // Batch 0: [[1, 2], [3, 4]] x [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        assert_eq!(&c.data[0..4], &[19.0, 22.0, 43.0, 50.0]);
        // Batch 1: [[2, 0], [1, 2]] x [[1, 2], [3, 4]] = [[2, 4], [7, 10]]
        assert_eq!(&c.data[4..8], &[2.0, 4.0, 7.0, 10.0]);
    }

    #[test]
    fn test_matmul_parallel_and_single_row() {
        // Test parallel row execution and single-row column-chunking
        let m = 32;
        let k = 64;
        let n = 32;

        let a_data: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|x| (x % 11) as f32).collect();

        let a = Tensor::new(a_data, [m, k]).unwrap();
        let b = Tensor::new(b_data, [k, n]).unwrap();

        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape, [m, n]);

        // Verify with ground truth calculation
        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for p in 0..k {
                    expected += a.data[i * k + p] * b.data[p * n + j];
                }
                assert!((c.data[i * n + j] - expected).abs() < 1e-4);
            }
        }

        // Test M=1 (vector x matrix) parallel column chunking
        let a_single = Tensor::new((0..k).map(|x| (x % 5) as f32).collect(), [1, k]).unwrap();
        let c_single = matmul(&a_single, &b).unwrap();
        assert_eq!(c_single.shape, [1, n]);
        for j in 0..n {
            let mut expected = 0.0f32;
            for p in 0..k {
                expected += a_single.data[p] * b.data[p * n + j];
            }
            assert!((c_single.data[j] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_matmul_shape_mismatch_error() {
        let a = Tensor::ones([2, 3]).unwrap();
        let b = Tensor::ones([4, 2]).unwrap();

        let err = matmul(&a, &b).unwrap_err();
        match err {
            TensorError::MatmulShapeMismatch { a_cols, b_rows, lhs, rhs } => {
                assert_eq!(a_cols, 3);
                assert_eq!(b_rows, 4);
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![4, 2]);
            }
            _ => panic!("Expected MatmulShapeMismatch, got {:?}", err),
        }
    }

    #[test]
    fn test_transpose_basic() {
        // [2, 3] -> [3, 2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let a_t = a.transpose().unwrap();
        assert_eq!(a_t.shape, [3, 2]);
        assert_eq!(a_t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Double transpose is identity
        let a_tt = a_t.transpose().unwrap();
        assert_eq!(a_tt.shape, [2, 3]);
        assert_eq!(a_tt.data, a.data);
    }

    #[test]
    fn test_transpose_batched_3d() {
        // [2, 2, 3] -> [2, 3, 2]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let a = Tensor::new(data, [2, 2, 3]).unwrap();
        let a_t = a.transpose().unwrap();
        assert_eq!(a_t.shape, [2, 3, 2]);
        // Batch 0: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(&a_t.data[0..6], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Batch 1: [[7, 10], [8, 11], [9, 12]]
        assert_eq!(&a_t.data[6..12], &[7.0, 10.0, 8.0, 11.0, 9.0, 12.0]);
    }

    #[test]
    fn test_transpose_parallel() {
        let rows = 64;
        let cols = 128;
        let data: Vec<f32> = (0..rows * cols).map(|x| x as f32).collect();
        let a = Tensor::new(data, [rows, cols]).unwrap();

        let a_t = a.transpose().unwrap();
        assert_eq!(a_t.shape, [cols, rows]);

        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(a_t.data[c * rows + r], a.data[r * cols + c]);
            }
        }
    }

    #[test]
    fn test_transpose_rank_error() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let err = a.transpose().unwrap_err();
        match err {
            TensorError::InvalidRank { ndim } => assert_eq!(ndim, 1),
            _ => panic!("Expected InvalidRank error, got {:?}", err),
        }
    }
}

