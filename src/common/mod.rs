//! Common utilities for video generation models
//!
//! This module provides shared components adapted from z_image (candle-transformers)
//! and candle-nn for optimal performance:
//!
//! - **attention**: Cross-platform attention dispatch (CUDA Flash-Attn / Metal SDPA / CPU)
//! - **rope**: 3D RoPE with pre-computed caching + fused CUDA/Metal kernels
//! - **norms**: Unified normalizations (QkNorm, RmsNorm, LayerNormNoParams)

pub mod attention;
pub mod norms;
pub mod rope;

// Attention exports
pub use attention::{AttentionConfig, CrossPlatformAttention, attention_dispatch};

// Normalization exports
pub use norms::{LayerNormNoParams, QkNorm, RmsNorm, RmsNormNoWeight};

// RoPE exports
pub use rope::{
    RopeEmbedder,
    apply_rotary_emb,
    apply_rotary_emb_half,
    apply_rotary_emb_interleaved,
    apply_rotary_emb_linear,
    apply_rotary_emb_thd,
    generate_position_ids,
    generate_position_ids_batched,
    // Re-exports from candle-nn for direct fused kernel access
    rope,
    rope_i,
    rope_thd,
};
