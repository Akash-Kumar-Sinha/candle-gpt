use std::fs;

use rayon::prelude::*;
use tensor::{Result, Tensor, TensorError};

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub ln_1_weight: Tensor<1>,
    pub ln_1_bias: Tensor<1>,
    pub attn_c_attn_weight: Tensor<2>,
    pub attn_c_attn_bias: Tensor<1>,
    pub attn_c_proj_weight: Tensor<2>,
    pub attn_c_proj_bias: Tensor<1>,
    pub ln_2_weight: Tensor<1>,
    pub ln_2_bias: Tensor<1>,
    pub mlp_c_fc_weight: Tensor<2>,
    pub mlp_c_fc_bias: Tensor<1>,
    pub mlp_c_proj_weight: Tensor<2>,
    pub mlp_c_proj_bias: Tensor<1>,
}

impl LayerWeights {
    pub fn load_from_dir(weights_dir: &str, layer_idx: usize) -> Result<Self> {
        let prefix = format!("{weights_dir}/transformer.h.{layer_idx}");
        let ln_1_weight = load_1d_tensor_from_txt(&format!("{prefix}.ln_1.weight.txt"), 768)?;
        let ln_1_bias = load_1d_tensor_from_txt(&format!("{prefix}.ln_1.bias.txt"), 768)?;
        let attn_c_attn_weight =
            load_2d_tensor_from_txt(&format!("{prefix}.attn.c_attn.weight.txt"), 768, 2304)?;
        let attn_c_attn_bias =
            load_1d_tensor_from_txt(&format!("{prefix}.attn.c_attn.bias.txt"), 2304)?;
        let attn_c_proj_weight =
            load_2d_tensor_from_txt(&format!("{prefix}.attn.c_proj.weight.txt"), 768, 768)?;
        let attn_c_proj_bias =
            load_1d_tensor_from_txt(&format!("{prefix}.attn.c_proj.bias.txt"), 768)?;
        let ln_2_weight = load_1d_tensor_from_txt(&format!("{prefix}.ln_2.weight.txt"), 768)?;
        let ln_2_bias = load_1d_tensor_from_txt(&format!("{prefix}.ln_2.bias.txt"), 768)?;
        let mlp_c_fc_weight =
            load_2d_tensor_from_txt(&format!("{prefix}.mlp.c_fc.weight.txt"), 768, 3072)?;
        let mlp_c_fc_bias = load_1d_tensor_from_txt(&format!("{prefix}.mlp.c_fc.bias.txt"), 3072)?;
        let mlp_c_proj_weight =
            load_2d_tensor_from_txt(&format!("{prefix}.mlp.c_proj.weight.txt"), 3072, 768)?;
        let mlp_c_proj_bias =
            load_1d_tensor_from_txt(&format!("{prefix}.mlp.c_proj.bias.txt"), 768)?;

        Ok(Self {
            ln_1_weight,
            ln_1_bias,
            attn_c_attn_weight,
            attn_c_attn_bias,
            attn_c_proj_weight,
            attn_c_proj_bias,
            ln_2_weight,
            ln_2_bias,
            mlp_c_fc_weight,
            mlp_c_fc_bias,
            mlp_c_proj_weight,
            mlp_c_proj_bias,
        })
    }

    pub fn load_for_layer(layer_idx: usize) -> Result<Self> {
        Self::load_from_dir("weights", layer_idx)
    }
}

pub type Weights = LayerWeights;

#[derive(Debug, Clone)]
pub struct TransformerWeights {
    pub layers: Vec<LayerWeights>,
    pub ln_f_weight: Tensor<1>,
    pub ln_f_bias: Tensor<1>,
}

impl TransformerWeights {
    pub fn load_from_dir(weights_dir: &str, num_layers: usize) -> Result<Self> {
        let (layers_res, ln_f_res) = rayon::join(
            || {
                (0..num_layers)
                    .into_par_iter()
                    .map(|i| LayerWeights::load_from_dir(weights_dir, i))
                    .collect::<Result<Vec<_>>>()
            },
            || {
                let ln_f_weight = load_1d_tensor_from_txt(
                    &format!("{weights_dir}/transformer.ln_f.weight.txt"),
                    768,
                )?;
                let ln_f_bias = load_1d_tensor_from_txt(
                    &format!("{weights_dir}/transformer.ln_f.bias.txt"),
                    768,
                )?;
                Ok((ln_f_weight, ln_f_bias))
            },
        );

        let layers = layers_res?;
        let (ln_f_weight, ln_f_bias) = ln_f_res?;

        Ok(Self {
            layers,
            ln_f_weight,
            ln_f_bias,
        })
    }
}

pub fn load_1d_tensor_from_txt(path: &str, size: usize) -> Result<Tensor<1>> {
    let content = fs::read_to_string(path).expect("Failed to read the content");
    let mut data = Vec::with_capacity(size);

    for line in content.lines() {
        for val in line.split_whitespace() {
            let num = val.parse::<f32>().expect("Failed to parse float");
            data.push(num);
        }
    }

    if data.len() != size {
        return Err(TensorError::LengthMismatch {
            actual: data.len(),
            expected: size,
            shape: vec![size],
            ndim: 1,
        });
    }
    Tensor::new(data, [size])
}

pub fn load_2d_tensor_from_txt(path: &str, rows: usize, cols: usize) -> Result<Tensor<2>> {
    let content = fs::read_to_string(path).expect("Failed to read the content");
    let mut data = Vec::with_capacity(rows * cols);

    for line in content.lines() {
        for val in line.split_whitespace() {
            let num = val.parse::<f32>().expect("Failed to parse float");
            data.push(num);
        }
    }

    if data.len() != rows * cols {
        return Err(TensorError::LengthMismatch {
            actual: data.len(),
            expected: rows * cols,
            shape: vec![rows, cols],
            ndim: 2,
        });
    }
    Tensor::new(data, [rows, cols])
}
