//! Cross-platform attention dispatch
//!
//! Provides unified attention API with optimal implementation per platform:
//! - CUDA: Flash Attention (when compiled with flash-attn feature)
//! - Metal: SDPA kernel via candle_nn::ops::sdpa
//! - CPU: Basic scaled dot-product attention fallback
//!
//! Adapted from z_image (candle-transformers)

use candle_core::{Device, Result, Tensor};

// Flash Attention wrapper for CUDA
#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    candle_core::bail!("flash-attn feature not enabled, compile with '--features flash-attn'")
}

/// Configuration for cross-platform attention
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to use accelerated attention (Flash-Attn/SDPA)
    pub use_accelerated: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            n_heads: 8,
            head_dim: 64,
            use_accelerated: true,
        }
    }
}

/// Cross-platform attention with automatic dispatch
#[derive(Debug, Clone)]
pub struct CrossPlatformAttention {
    config: AttentionConfig,
}

impl CrossPlatformAttention {
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Dispatch attention to optimal implementation based on device
    ///
    /// # Arguments
    /// * `q` - Query tensor (B, n_heads, seq_len, head_dim)
    /// * `k` - Key tensor (B, n_heads, seq_len, head_dim)
    /// * `v` - Value tensor (B, n_heads, seq_len, head_dim)
    /// * `mask` - Optional attention mask (B, seq_len)
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let device = q.device();

        if !self.config.use_accelerated {
            return self.attention_basic(q, k, v, mask, scale);
        }

        match device {
            Device::Cuda(_) => self.attention_cuda(q, k, v, mask, scale),
            Device::Metal(_) => self.attention_metal(q, k, v, mask, scale),
            Device::Cpu => self.attention_basic(q, k, v, mask, scale),
        }
    }

    /// CUDA: Use Flash Attention
    #[allow(unused_variables)]
    fn attention_cuda(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            // Flash-attn doesn't support custom masks, fallback when mask present
            if mask.is_some() {
                return self.attention_basic(q, k, v, mask, scale);
            }

            // Flash-attn expects (batch, seq_len, num_heads, head_size)
            // Current format: (batch, num_heads, seq_len, head_size)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            let result = flash_attn(&q, &k, &v, scale as f32, false)?;
            result.transpose(1, 2)
        }

        #[cfg(not(feature = "flash-attn"))]
        {
            self.attention_basic(q, k, v, mask, scale)
        }
    }

    /// Metal: Use fused SDPA kernel
    fn attention_metal(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let sdpa_mask = self.prepare_sdpa_mask(mask, q)?;

        // candle_nn::ops::sdpa expects (bs, qhead, seq, hidden)
        candle_nn::ops::sdpa(q, k, v, sdpa_mask.as_ref(), false, scale as f32, 1.0)
    }

    /// CPU fallback: Basic scaled dot-product attention
    fn attention_basic(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            // mask: (B, seq_len) -> (B, 1, 1, seq_len)
            let m = m.unsqueeze(1)?.unsqueeze(2)?;
            let m = m.to_dtype(attn_weights.dtype())?;
            // 1=valid, 0=padding -> 0=valid, -inf=padding
            let m = ((m - 1.0)? * 1e9)?;
            attn_weights = attn_weights.broadcast_add(&m)?;
        }

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_probs.matmul(v)
    }

    /// Prepare mask for SDPA format
    fn prepare_sdpa_mask(&self, mask: Option<&Tensor>, q: &Tensor) -> Result<Option<Tensor>> {
        match mask {
            Some(m) => {
                let (b, _, seq_len, _) = q.dims4()?;
                let m = m.unsqueeze(1)?.unsqueeze(2)?;
                let m = m.to_dtype(q.dtype())?;
                // SDPA uses additive mask: 0=valid, -inf=masked
                let m = ((m - 1.0)? * 1e9)?;
                let m = m.broadcast_as((b, self.config.n_heads, seq_len, seq_len))?;
                Ok(Some(m))
            }
            None => Ok(None),
        }
    }
}

/// Standalone function for quick attention dispatch
pub fn attention_dispatch(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    use_accelerated: bool,
) -> Result<Tensor> {
    let (_, n_heads, _, head_dim) = q.dims4()?;
    let config = AttentionConfig {
        n_heads,
        head_dim,
        use_accelerated,
    };
    let attn = CrossPlatformAttention::new(config);
    attn.forward(q, k, v, mask, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_attention_basic() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let seq_len = 8;
        let head_dim = 32;

        let q = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;

        let config = AttentionConfig {
            n_heads: heads,
            head_dim,
            use_accelerated: false,
        };
        let attn = CrossPlatformAttention::new(config);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let output = attn.forward(&q, &k, &v, None, scale)?;

        assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
        Ok(())
    }

    #[test]
    fn test_attention_with_mask() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let seq_len = 8;
        let head_dim = 32;

        let q = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let v = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let mask = Tensor::ones((batch, seq_len), DType::F32, &device)?;

        let config = AttentionConfig {
            n_heads: heads,
            head_dim,
            use_accelerated: false,
        };
        let attn = CrossPlatformAttention::new(config);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let output = attn.forward(&q, &k, &v, Some(&mask), scale)?;

        assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
        Ok(())
    }
}
