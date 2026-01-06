//! RoPE with pre-computed caching and fused kernel support
//!
//! Provides 3D Rotary Position Embedding with:
//! - Pre-computed cos/sin cache per axis (from Z-Image)
//! - Fused CUDA/Metal/CPU kernels for apply operation (from candle-nn)
//! - Support for 3D coordinates (frame, height, width)
//!
//! This hybrid approach gives us:
//! - 3D spatiotemporal structure (critical for video)
//! - Pre-computed cache (faster inference)  
//! - Optimized kernels (reduced GPU overhead)

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};

// Re-export candle-nn's optimized rope functions for 4D tensors
pub use candle_nn::rotary_emb::{rope, rope_i, rope_thd};

/// 3D Rotary Position Embedding with pre-computed cache
///
/// Pre-computes cos/sin values for each axis dimension to avoid
/// repeated computation during inference.
///
/// This struct handles:
/// - 3D position IDs (frame, height, width)
/// - Axis-specific frequency computation
/// - Cached cos/sin lookup
#[derive(Debug, Clone)]
pub struct RopeEmbedder {
    theta: f64,
    axes_dims: Vec<usize>,
    axes_lens: Vec<usize>,
    /// Pre-computed cos cache per axis: [axis_idx] -> (max_len, axis_dim/2)
    cos_cached: Vec<Tensor>,
    /// Pre-computed sin cache per axis: [axis_idx] -> (max_len, axis_dim/2)
    sin_cached: Vec<Tensor>,
}

impl RopeEmbedder {
    /// Create a new RopeEmbedder with pre-computed cache
    ///
    /// # Arguments
    /// * `theta` - Base frequency for rotary embeddings (e.g., 10000.0)
    /// * `axes_dims` - Dimension allocated to each axis (e.g., [32, 48, 48] for 3D)
    /// * `axes_lens` - Maximum sequence length per axis (e.g., [128, 64, 64])
    /// * `device` - Device to create tensors on
    /// * `dtype` - Data type for the cached tensors
    pub fn new(
        theta: f64,
        axes_dims: Vec<usize>,
        axes_lens: Vec<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        assert_eq!(axes_dims.len(), axes_lens.len());
        let mut cos_cached = Vec::with_capacity(axes_dims.len());
        let mut sin_cached = Vec::with_capacity(axes_dims.len());

        for (d, e) in axes_dims.iter().zip(axes_lens.iter()) {
            let half_d = d / 2;
            let inv_freq: Vec<f32> = (0..half_d)
                .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / *d as f32))
                .collect();
            let inv_freq = Tensor::from_vec(inv_freq, half_d, device)?;

            let positions = Tensor::arange(0u32, *e as u32, device)?.to_dtype(DType::F32)?;
            let freqs = positions
                .unsqueeze(1)?
                .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

            cos_cached.push(freqs.cos()?.to_dtype(dtype)?);
            sin_cached.push(freqs.sin()?.to_dtype(dtype)?);
        }

        Ok(Self {
            theta,
            axes_dims,
            axes_lens,
            cos_cached,
            sin_cached,
        })
    }

    /// Get RoPE cos/sin from 3D position IDs
    ///
    /// # Arguments
    /// * `ids` - Position IDs tensor (seq_len, num_axes) where num_axes typically = 3
    ///
    /// # Returns
    /// Tuple of (cos, sin) each of shape (seq_len, head_dim/2)
    pub fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut cos_parts = Vec::with_capacity(self.axes_dims.len());
        let mut sin_parts = Vec::with_capacity(self.axes_dims.len());

        for (i, _) in self.axes_dims.iter().enumerate() {
            // Must be contiguous for Metal
            let axis_ids = ids.i((.., i))?.contiguous()?;
            let cos_i = self.cos_cached[i].index_select(&axis_ids, 0)?;
            let sin_i = self.sin_cached[i].index_select(&axis_ids, 0)?;
            cos_parts.push(cos_i);
            sin_parts.push(sin_i);
        }

        let cos = Tensor::cat(&cos_parts, D::Minus1)?;
        let sin = Tensor::cat(&sin_parts, D::Minus1)?;
        Ok((cos, sin))
    }

    /// Get the total head dimension (sum of axes_dims)
    pub fn head_dim(&self) -> usize {
        self.axes_dims.iter().sum()
    }

    /// Get theta value
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Get axes dimensions
    pub fn axes_dims(&self) -> &[usize] {
        &self.axes_dims
    }

    /// Get axes lengths
    pub fn axes_lens(&self) -> &[usize] {
        &self.axes_lens
    }
}

/// Apply RoPE using fused kernel when available, tensor ops fallback otherwise
///
/// Automatically selects between:
/// - candle-nn fused kernel (CUDA/Metal) - fastest
/// - Tensor ops fallback - portable
///
/// # Arguments
/// * `x` - Input tensor (B, seq_len, n_heads, head_dim)  
/// * `cos` - Cosine values (seq_len, head_dim/2)
/// * `sin` - Sine values (seq_len, head_dim/2)
///
/// # Returns
/// Rotated tensor with same shape as input
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, n_heads, head_dim) = x.dims4()?;

    // Try to use fused kernel if input format is compatible
    // candle-nn expects (B, heads, seq, dim) but we have (B, seq, heads, dim)
    // Transpose, apply fused kernel, transpose back

    let use_fused = x.is_contiguous() && cos.is_contiguous() && sin.is_contiguous();

    if use_fused {
        // Transpose to (B, heads, seq, dim) for candle-nn kernel
        let x_t = x.transpose(1, 2)?.contiguous()?;

        // The fused kernel expects cos/sin in (seq_len, dim/2) format
        // which matches what we have
        match rope_fused(&x_t, cos, sin) {
            Ok(result) => {
                // Transpose back to (B, seq, heads, dim)
                return result.transpose(1, 2);
            }
            Err(_) => {
                // Fall through to tensor ops
            }
        }
    }

    // Fallback: Pure tensor ops (always works)
    apply_rotary_emb_tensor_ops(x, cos, sin)
}

/// Apply RoPE using fused candle-nn kernel
/// Input: (B, heads, seq, dim)
fn rope_fused(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // candle-nn's rope expects cos/sin shape (seq_len, dim/2)
    // and doubles them internally to match head_dim
    candle_nn::rotary_emb::rope(x, cos, sin)
}

/// Apply RoPE using pure tensor ops (fallback)
///
/// Uses real-number form equivalent to complex multiplication
fn apply_rotary_emb_tensor_ops(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, seq_len, n_heads, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    // Reshape x to interleaved real/imag form: (B, seq_len, n_heads, half_dim, 2)
    let x = x.reshape((b, seq_len, n_heads, half_dim, 2))?;

    // Extract real and imag parts
    let x_real = x.i((.., .., .., .., 0))?;
    let x_imag = x.i((.., .., .., .., 1))?;

    // Expand cos/sin for broadcasting: (seq_len, half_dim) -> (1, seq_len, 1, half_dim)
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let y_real = (x_real.broadcast_mul(&cos)? - x_imag.broadcast_mul(&sin)?)?;
    let y_imag = (x_real.broadcast_mul(&sin)? + x_imag.broadcast_mul(&cos)?)?;

    // Interleave back
    Tensor::stack(&[y_real, y_imag], D::Minus1)?.reshape((b, seq_len, n_heads, head_dim))
}

/// Apply RoPE to tensor in contiguous half format
///
/// This variant handles the case where x has shape (B, heads, seq, dim)
/// and first half and second half of dim are rotated together (not interleaved).
///
/// # Arguments
/// * `x` - Input tensor (B, heads, seq_len, dim)
/// * `cos` - Cosine values (seq_len, dim/2)
/// * `sin` - Sine values (seq_len, dim/2)
pub fn apply_rotary_emb_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Try fused kernel first
    if x.is_contiguous()
        && cos.is_contiguous()
        && sin.is_contiguous()
        && let Ok(result) = candle_nn::rotary_emb::rope(x, cos, sin)
    {
        return Ok(result);
    }

    // Fallback to slow version
    candle_nn::rotary_emb::rope_slow(x, cos, sin)
}

/// Apply RoPE to tensor in interleaved format
///
/// Uses the interleaved variant where x0, x1 pairs are adjacent in memory.
///
/// # Arguments
/// * `x` - Input tensor (B, heads, seq_len, dim)
/// * `cos` - Cosine values (seq_len, dim/2)  
/// * `sin` - Sine values (seq_len, dim/2)
pub fn apply_rotary_emb_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Try fused kernel first
    if x.is_contiguous()
        && cos.is_contiguous()
        && sin.is_contiguous()
        && let Ok(result) = candle_nn::rotary_emb::rope_i(x, cos, sin)
    {
        return Ok(result);
    }

    // Fallback to slow version
    candle_nn::rotary_emb::rope_i_slow(x, cos, sin)
}

/// Apply RoPE to tensor in THD format (seq, heads, dim)
///
/// Used when input has shape (B, seq_len, heads, dim) instead of (B, heads, seq_len, dim)
///
/// # Arguments
/// * `x` - Input tensor (B, seq_len, heads, dim)
/// * `cos` - Cosine values (seq_len, dim/2)
/// * `sin` - Sine values (seq_len, dim/2)
pub fn apply_rotary_emb_thd(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Try fused kernel first
    if x.is_contiguous()
        && cos.is_contiguous()
        && sin.is_contiguous()
        && let Ok(result) = candle_nn::rotary_emb::rope_thd(x, cos, sin)
    {
        return Ok(result);
    }

    // Fallback: transpose, apply, transpose back
    let x_transposed = x.transpose(1, 2)?.contiguous()?;
    let result = apply_rotary_emb_half(&x_transposed, cos, sin)?;
    result.transpose(1, 2)
}

/// Apply RoPE to linear tensor format (B, seq, inner_dim)
///
/// This variant operates on Q/K before reshaping to multi-head format,
/// matching diffusers apply_rotary_emb behavior.
///
/// # Arguments
/// * `x` - Input tensor (B, seq_len, inner_dim) where inner_dim = heads * head_dim
/// * `cos` - Cosine values (B, seq_len, inner_dim) or broadcastable
/// * `sin` - Sine values (B, seq_len, inner_dim) or broadcastable
pub fn apply_rotary_emb_linear(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x_shape = x.dims();
    let batch = x_shape[0];
    let seq_len = x_shape[1];
    let inner_dim = x_shape[2];

    // Pair up dimensions: (batch, seq_len, inner_dim/2, 2)
    let x_reshaped = x.reshape((batch, seq_len, inner_dim / 2, 2))?;

    // Get real and imag parts
    let x_real = x_reshaped.i((.., .., .., 0))?;
    let x_imag = x_reshaped.i((.., .., .., 1))?;

    // Create rotated version: stack(-x_imag, x_real) then flatten
    let x_rot = Tensor::stack(&[x_imag.neg()?, x_real], D::Minus1)?;
    let x_rot = x_rot.flatten_from(D::Minus2)?;

    // Apply rotation: x * cos + x_rot * sin
    x.broadcast_mul(cos)?.add(&x_rot.broadcast_mul(sin)?)
}

/// Generate position IDs grid for 3D spatiotemporal data
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `t` - Number of temporal positions (frames)
/// * `h` - Number of height positions  
/// * `w` - Number of width positions
/// * `device` - Device to create tensor on
///
/// # Returns
/// Position IDs tensor of shape (t*h*w, 3) for use with RopeEmbedder
pub fn generate_position_ids(t: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let seq_len = t * h * w;
    let mut t_ids = Vec::with_capacity(seq_len);
    let mut h_ids = Vec::with_capacity(seq_len);
    let mut w_ids = Vec::with_capacity(seq_len);

    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                t_ids.push(ti as u32);
                h_ids.push(hi as u32);
                w_ids.push(wi as u32);
            }
        }
    }

    let t_tensor = Tensor::from_vec(t_ids, seq_len, device)?;
    let h_tensor = Tensor::from_vec(h_ids, seq_len, device)?;
    let w_tensor = Tensor::from_vec(w_ids, seq_len, device)?;

    // Stack to (seq_len, 3)
    Tensor::stack(&[t_tensor, h_tensor, w_tensor], 1)
}

/// Generate position IDs grid with batch dimension
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `t` - Number of temporal positions
/// * `h` - Number of height positions  
/// * `w` - Number of width positions
/// * `device` - Device to create tensor on
///
/// # Returns
/// Position IDs tensor of shape (batch, 3, t*h*w)
pub fn generate_position_ids_batched(
    batch_size: usize,
    t: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> Result<Tensor> {
    let ids = generate_position_ids(t, h, w, device)?;
    // (seq_len, 3) -> (1, seq_len, 3) -> (batch, seq_len, 3) -> (batch, 3, seq_len)
    ids.unsqueeze(0)?
        .expand((batch_size, t * h * w, 3))?
        .transpose(1, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_embedder_creation() -> Result<()> {
        let device = Device::Cpu;
        let rope = RopeEmbedder::new(
            256.0,
            vec![32, 48, 48],
            vec![64, 64, 64],
            &device,
            DType::F32,
        )?;

        assert_eq!(rope.head_dim(), 128);
        assert_eq!(rope.theta(), 256.0);
        assert_eq!(rope.cos_cached.len(), 3);
        Ok(())
    }

    #[test]
    fn test_rope_forward() -> Result<()> {
        let device = Device::Cpu;
        let rope = RopeEmbedder::new(
            256.0,
            vec![32, 32, 32],
            vec![64, 64, 64],
            &device,
            DType::F32,
        )?;

        let ids = generate_position_ids(2, 4, 4, &device)?;
        let (cos, sin) = rope.forward(&ids)?;

        // seq_len = 2*4*4 = 32
        // half_dim = 16 + 16 + 16 = 48
        assert_eq!(cos.dim(0)?, 32);
        assert_eq!(cos.dim(1)?, 48);
        assert_eq!(sin.dims(), cos.dims());
        Ok(())
    }

    #[test]
    fn test_apply_rotary_emb() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 8;
        let n_heads = 4;
        let head_dim = 64;
        let half_dim = head_dim / 2;

        let x = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
        let cos = Tensor::randn(0f32, 1.0, (seq_len, half_dim), &device)?;
        let sin = Tensor::randn(0f32, 1.0, (seq_len, half_dim), &device)?;

        let output = apply_rotary_emb(&x, &cos, &sin)?;

        assert_eq!(output.dims(), &[batch, seq_len, n_heads, head_dim]);
        Ok(())
    }

    #[test]
    fn test_generate_position_ids() -> Result<()> {
        let device = Device::Cpu;
        let ids = generate_position_ids(2, 4, 4, &device)?;

        assert_eq!(ids.dim(0)?, 2 * 4 * 4);
        assert_eq!(ids.dim(1)?, 3);
        Ok(())
    }

    #[test]
    fn test_generate_position_ids_batched() -> Result<()> {
        let device = Device::Cpu;
        let ids = generate_position_ids_batched(2, 4, 8, 8, &device)?;

        assert_eq!(ids.dim(0)?, 2);
        assert_eq!(ids.dim(1)?, 3);
        assert_eq!(ids.dim(2)?, 4 * 8 * 8);
        Ok(())
    }

    #[test]
    fn test_apply_rotary_emb_half() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let seq_len = 8;
        let head_dim = 64;
        let half_dim = head_dim / 2;

        // Shape for candle-nn: (B, heads, seq, dim)
        let x = Tensor::randn(0f32, 1.0, (batch, heads, seq_len, head_dim), &device)?;
        let cos = Tensor::randn(0f32, 1.0, (seq_len, half_dim), &device)?;
        let sin = Tensor::randn(0f32, 1.0, (seq_len, half_dim), &device)?;

        let output = apply_rotary_emb_half(&x, &cos, &sin)?;

        assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
        Ok(())
    }
}
