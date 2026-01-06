//! Unified normalization layers
//!
//! Provides common normalization components adapted from z_image:
//! - QkNorm: Query-Key normalization with RMSNorm
//! - RmsNormNoWeight: RMSNorm without learnable weights
//! - LayerNormNoParams: LayerNorm without learnable parameters
//!
//! Adapted from z_image (candle-transformers)

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// RMSNorm without learnable weights (elementwise_affine=False)
///
/// Used when no learnable scale is needed, matching PyTorch RMSNorm(elementwise_affine=False)
#[derive(Debug, Clone)]
pub struct RmsNormNoWeight {
    eps: f64,
}

impl RmsNormNoWeight {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for RmsNormNoWeight {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();

        let internal_dtype = match input_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let x_f32 = x.to_dtype(internal_dtype)?;
        let hidden_size = x.dim(D::Minus1)?;
        let variance = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        x_normed.to_dtype(input_dtype)
    }
}

/// LayerNorm without learnable parameters (elementwise_affine=False)
///
/// Normalizes input to zero mean and unit variance without learnable scale/bias.
/// Matches PyTorch LayerNorm(elementwise_affine=False)
#[derive(Debug, Clone)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoParams {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;

        // Subtract mean
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;

        // Divide by std
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;

        x_normed.to_dtype(x_dtype)
    }
}

/// RMSNorm wrapper for candle-transformers compatibility
///
/// Wraps the standard RmsNorm from candle-transformers with a simpler interface
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::init::ONE)?;
        Ok(Self { weight, eps })
    }

    /// Create RmsNorm from an existing weight tensor
    /// Useful for quantized models where weights are already dequantized
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();

        let internal_dtype = match input_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let x_f32 = x.to_dtype(internal_dtype)?;
        let hidden_size = x.dim(D::Minus1)?;
        let variance = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        let weight = self.weight.to_dtype(internal_dtype)?;
        let result = x_normed.broadcast_mul(&weight)?;

        result.to_dtype(input_dtype)
    }
}

/// Query-Key Normalization using RMSNorm
///
/// Normalizes Q and K tensors separately before attention computation.
/// Improves training stability, especially for large models.
#[derive(Debug, Clone)]
pub struct QkNorm {
    norm_q: RmsNorm,
    norm_k: RmsNorm,
}

impl QkNorm {
    pub fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm_q = RmsNorm::new(head_dim, eps, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(head_dim, eps, vb.pp("norm_k"))?;
        Ok(Self { norm_q, norm_k })
    }

    /// Apply normalization to Q and K tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor (B, seq_len, n_heads, head_dim)
    /// * `k` - Key tensor (B, seq_len, n_heads, head_dim)
    ///
    /// # Returns
    /// Tuple of normalized (Q, K) tensors
    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q = self.norm_q.forward(q)?;
        let k = self.norm_k.forward(k)?;
        Ok((q, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rms_norm_no_weight() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 64), &device)?;

        let norm = RmsNormNoWeight::new(1e-5);
        let output = norm.forward(&x)?;

        assert_eq!(output.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_layer_norm_no_params() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 64), &device)?;

        let norm = LayerNormNoParams::new(1e-6);
        let output = norm.forward(&x)?;

        assert_eq!(output.dims(), x.dims());

        // Check that output is approximately normalized (mean ~0)
        let mean = output.mean_all()?.to_scalar::<f32>()?;
        assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
        Ok(())
    }

    #[test]
    fn test_dtype_preservation() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 64), &device)?.to_dtype(DType::F32)?;

        let norm = RmsNormNoWeight::new(1e-5);
        let output = norm.forward(&x)?;

        assert_eq!(output.dtype(), x.dtype());
        Ok(())
    }
}
