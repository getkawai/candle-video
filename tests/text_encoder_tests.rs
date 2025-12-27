//! Comprehensive tests for T5-XXL Text Encoder Wrapper
//!
//! Tests cover:
//! - T5 encoder configuration (T5-XXL specific)
//! - Prompt encoding with tokenization
//! - Embedding shape validation
//! - Embedding caching
//! - CPU offloading
//! - Batch processing

use candle_core::{DType, Device, Result};
use candle_video::text_encoder::*;
use std::fs;
use tempfile::tempdir;

// =============================================================================
// Configuration Tests
// =============================================================================

#[test]
fn test_t5_config_creation() {
    let config = T5EncoderConfig::default();

    // T5-XXL has specific dimensions
    assert_eq!(config.d_model, 4096);
    assert_eq!(config.num_heads, 64);
    assert_eq!(config.num_layers, 24);
}

#[test]
fn test_t5_config_t5_xxl_preset() {
    let config = T5EncoderConfig::t5_xxl();

    assert_eq!(config.d_model, 4096);
    assert_eq!(config.d_ff, 10240);
    assert_eq!(config.num_heads, 64);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.d_kv, 64);
}

#[test]
fn test_t5_config_custom() {
    let config = T5EncoderConfig::new(2048, 16, 12);

    assert_eq!(config.d_model, 2048);
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.num_layers, 12);
}

#[test]
fn test_t5_config_from_json() {
    let dir = tempdir().unwrap();

    let json_config = serde_json::json!({
        "d_model": 4096,
        "d_ff": 10240,
        "d_kv": 64,
        "num_heads": 64,
        "num_layers": 24,
        "vocab_size": 32128,
        "layer_norm_epsilon": 1e-6
    });

    fs::write(
        dir.path().join("config.json"),
        serde_json::to_string_pretty(&json_config).unwrap(),
    )
    .unwrap();

    let config = T5EncoderConfig::from_json(dir.path().join("config.json")).unwrap();

    assert_eq!(config.d_model, 4096);
    assert_eq!(config.num_heads, 64);
}

// =============================================================================
// Text Encoder Wrapper Tests
// =============================================================================

#[test]
fn test_text_encoder_wrapper_creation() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32);

    assert!(wrapper.is_ok());
}

#[test]
fn test_text_encoder_device() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

    assert!(matches!(wrapper.device(), Device::Cpu));
}

#[test]
fn test_text_encoder_dtype() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::BF16).unwrap();

    assert_eq!(wrapper.dtype(), DType::BF16);
}

#[test]
fn test_text_encoder_config() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config.clone(), Device::Cpu, DType::F32).unwrap();

    assert_eq!(wrapper.config().d_model, config.d_model);
}

// =============================================================================
// Tokenization Tests (without model weights)
// =============================================================================

#[test]
fn test_tokenize_simple_prompt() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

    // Without actual tokenizer, use mock tokens
    let prompt = "A beautiful sunset over the ocean";
    let token_ids = wrapper.mock_tokenize(prompt);

    // Should return non-empty tokens
    assert!(!token_ids.is_empty());
    // Token count should be reasonable for this prompt
    assert!(token_ids.len() >= 5 && token_ids.len() <= 20);
}

#[test]
fn test_tokenize_empty_prompt() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

    let prompt = "";
    let token_ids = wrapper.mock_tokenize(prompt);

    // Empty prompt should produce minimal tokens (maybe just EOS)
    assert!(token_ids.len() <= 2);
}

#[test]
fn test_token_ids_to_tensor() {
    let config = T5EncoderConfig::t5_xxl();
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

    let token_ids = vec![1u32, 2, 3, 4, 5];
    let tensor = wrapper.token_ids_to_tensor(&token_ids).unwrap();

    assert_eq!(tensor.dims(), &[1, 5]); // [batch, seq_len]
}

// =============================================================================
// Embedding Shape Tests (using mock embeddings)
// =============================================================================

#[test]
fn test_mock_encode_output_shape() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config.clone(), Device::Cpu, DType::F32)?;

    let prompt = "A cat sitting on a mat";
    let embeddings = wrapper.mock_encode(prompt)?;

    // Check output shape: [batch=1, seq_len, d_model=4096]
    let dims = embeddings.dims();
    assert_eq!(dims.len(), 3);
    assert_eq!(dims[0], 1); // batch size
    assert_eq!(dims[2], config.d_model); // embedding dimension

    Ok(())
}

#[test]
fn test_mock_encode_batch() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config.clone(), Device::Cpu, DType::F32)?;

    let prompts = vec!["A beautiful landscape", "A cat playing with a ball"];
    let embeddings = wrapper.mock_encode_batch(&prompts)?;

    // Check output shape: [batch=2, seq_len, d_model=4096]
    let dims = embeddings.dims();
    assert_eq!(dims.len(), 3);
    assert_eq!(dims[0], 2); // batch size
    assert_eq!(dims[2], config.d_model); // embedding dimension

    Ok(())
}

// =============================================================================
// Embedding Caching Tests
// =============================================================================

#[test]
fn test_embedding_cache_hit() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32)?;
    wrapper.enable_cache(true);

    let prompt = "A repeating prompt for caching";

    // First encode - should be cached
    let _embeddings1 = wrapper.mock_encode(prompt)?;
    assert!(wrapper.cache_contains(prompt));

    // Second encode - should use cache
    let _embeddings2 = wrapper.mock_encode(prompt)?;
    assert_eq!(wrapper.cache_hits(), 1);

    Ok(())
}

#[test]
fn test_embedding_cache_miss() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32)?;
    wrapper.enable_cache(true);

    let prompt1 = "First prompt";
    let prompt2 = "Different prompt";

    let _embeddings1 = wrapper.mock_encode(prompt1)?;
    let _embeddings2 = wrapper.mock_encode(prompt2)?;

    assert_eq!(wrapper.cache_hits(), 0);
    assert_eq!(wrapper.cache_size(), 2);

    Ok(())
}

#[test]
fn test_embedding_cache_clear() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32)?;
    wrapper.enable_cache(true);

    let _embeddings = wrapper.mock_encode("Some prompt")?;
    assert_eq!(wrapper.cache_size(), 1);

    wrapper.clear_cache();
    assert_eq!(wrapper.cache_size(), 0);

    Ok(())
}

#[test]
fn test_embedding_cache_disabled() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32)?;
    wrapper.enable_cache(false);

    let _embeddings = wrapper.mock_encode("Some prompt")?;
    assert_eq!(wrapper.cache_size(), 0);

    Ok(())
}

// =============================================================================
// CPU Offloading Tests
// =============================================================================

#[test]
fn test_offload_config() {
    let config = T5EncoderConfig::t5_xxl().with_cpu_offload(true);

    assert!(config.cpu_offload);
}

#[test]
fn test_encoder_with_cpu_offload() {
    let config = T5EncoderConfig::t5_xxl().with_cpu_offload(true);
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

    assert!(wrapper.is_cpu_offload_enabled());
}

// =============================================================================
// Max Sequence Length Tests
// =============================================================================

#[test]
fn test_max_sequence_length_default() {
    let config = T5EncoderConfig::t5_xxl();

    assert_eq!(config.max_seq_len, 256);
}

#[test]
fn test_max_sequence_length_custom() {
    let config = T5EncoderConfig::t5_xxl().with_max_seq_len(512);

    assert_eq!(config.max_seq_len, 512);
}

#[test]
fn test_truncate_to_max_length() {
    let config = T5EncoderConfig::t5_xxl().with_max_seq_len(10);
    let wrapper = T5TextEncoderWrapper::new(config, Device::Cpu, DType::F32).unwrap();

    // Create a very long "prompt" worth many tokens
    let long_prompt = "word ".repeat(100);
    let token_ids = wrapper.mock_tokenize(&long_prompt);

    // Should be truncated to max_seq_len
    assert!(token_ids.len() <= 10);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_config_file_not_found() {
    let result = T5EncoderConfig::from_json("/nonexistent/config.json");
    assert!(result.is_err());
}

// =============================================================================
// T5 Candle Integration Tests (configuration compatibility)
// =============================================================================

#[test]
fn test_to_candle_config() {
    let config = T5EncoderConfig::t5_xxl();
    let candle_config = config.to_candle_t5_config();

    assert_eq!(candle_config.d_model, 4096);
    assert_eq!(candle_config.num_heads, 64);
    assert_eq!(candle_config.num_layers, 24);
    assert_eq!(candle_config.d_kv, 64);
}

// =============================================================================
// Negative Prompt Tests
// =============================================================================

#[test]
fn test_encode_for_cfg() -> Result<()> {
    let config = T5EncoderConfig::t5_xxl();
    let mut wrapper = T5TextEncoderWrapper::new(config.clone(), Device::Cpu, DType::F32)?;

    let prompt = "A beautiful sunset";
    let negative_prompt = "";

    let (positive_emb, negative_emb) = wrapper.mock_encode_for_cfg(prompt, negative_prompt)?;

    // Both should have same shape
    assert_eq!(positive_emb.dims(), negative_emb.dims());
    assert_eq!(positive_emb.dims()[2], config.d_model);

    Ok(())
}
