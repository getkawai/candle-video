//! Comprehensive tests for safetensors loader improvements
//!
//! Tests cover:
//! - Single file loading
//! - Sharded file loading with auto-detection
//! - JSON config parsing
//! - Name mapping (Python → Rust)
//! - Tensor name validation

use candle_core::{DType, Device};
use candle_video::loader::*;
use std::collections::HashMap;
use std::fs;
use tempfile::tempdir;

// =============================================================================
// Test Data Setup Helpers
// =============================================================================

/// Creates a minimal valid safetensors file for testing
fn create_test_safetensors(path: &std::path::Path, tensors: &[(&str, Vec<f32>, Vec<usize>)]) {
    use safetensors::serialize;
    use safetensors::tensor::TensorView;

    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(name, data, shape)| {
            let bytes = bytemuck::cast_slice(data);
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).unwrap();
            (name.to_string(), view)
        })
        .collect();

    let data = serialize(views, None).unwrap();
    fs::write(path, data).unwrap();
}

/// Creates a test model.safetensors.index.json file
fn create_test_index_json(
    path: &std::path::Path,
    weight_map: &HashMap<String, String>,
    metadata: Option<serde_json::Value>,
) {
    let mut json = serde_json::json!({
        "weight_map": weight_map
    });

    if let Some(meta) = metadata {
        json["metadata"] = meta;
    }

    let content = serde_json::to_string_pretty(&json).unwrap();
    fs::write(path, content).unwrap();
}

/// Creates a test config.json file
fn create_test_config_json(path: &std::path::Path, config: &serde_json::Value) {
    let content = serde_json::to_string_pretty(config).unwrap();
    fs::write(path, content).unwrap();
}

// =============================================================================
// WeightLoader Creation Tests
// =============================================================================

#[test]
fn test_weight_loader_creation() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    assert_eq!(loader.dtype(), DType::F32);
}

#[test]
fn test_weight_loader_with_bf16() {
    let loader = WeightLoader::new(Device::Cpu, DType::BF16);
    assert_eq!(loader.dtype(), DType::BF16);
}

// =============================================================================
// Name Mapping Tests
// =============================================================================

#[test]
fn test_single_name_mapping() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32)
        .add_mapping("model.diffusion_model", "diffusion_model");

    assert!(loader.has_mapping("model.diffusion_model"));
    assert_eq!(loader.map_name("model.diffusion_model"), "diffusion_model");
}

#[test]
fn test_multiple_name_mappings() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32)
        .add_mapping("model.transformer.blocks", "blocks")
        .add_mapping("model.vae.encoder", "vae_encoder")
        .add_mapping("model.text_enc", "t5_encoder");

    assert_eq!(loader.map_name("model.transformer.blocks"), "blocks");
    assert_eq!(loader.map_name("model.vae.encoder"), "vae_encoder");
    assert_eq!(loader.map_name("model.text_enc"), "t5_encoder");
}

#[test]
fn test_unmapped_name_returns_original() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32).add_mapping("model.layer1", "layer1");

    // Unmapped name should return itself
    assert_eq!(loader.map_name("model.layer2"), "model.layer2");
}

#[test]
fn test_prefix_mapping() {
    let loader =
        WeightLoader::new(Device::Cpu, DType::F32).add_prefix_mapping("model.diffusion_model.", "");

    assert_eq!(
        loader.map_name("model.diffusion_model.transformer.layer.0.weight"),
        "transformer.layer.0.weight"
    );
}

#[test]
fn test_suffix_mapping() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32)
        .add_suffix_mapping(".gamma", ".weight")
        .add_suffix_mapping(".beta", ".bias");

    assert_eq!(loader.map_name("layer_norm.gamma"), "layer_norm.weight");
    assert_eq!(loader.map_name("layer_norm.beta"), "layer_norm.bias");
}

#[test]
fn test_chain_mapping() {
    // Test that mappings can be chained together
    let loader = WeightLoader::new(Device::Cpu, DType::F32)
        .add_prefix_mapping("model.", "")
        .add_mapping("encoder.weight", "enc_w");

    // First apply prefix, then exact match
    // "model.encoder.weight" → "encoder.weight" → "enc_w"
    assert_eq!(loader.map_name("model.encoder.weight"), "enc_w");
}

// =============================================================================
// Single File Loading Tests
// =============================================================================

#[test]
fn test_load_single_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");

    // Create a test safetensors file
    create_test_safetensors(
        &file_path,
        &[
            ("weight1", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            ("weight2", vec![0.5, 0.5], vec![2]),
        ],
    );

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let result = loader.load_single(&file_path);

    assert!(result.is_ok());
}

#[test]
fn test_load_single_file_not_found() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let result = loader.load_single("/nonexistent/path/model.safetensors");

    assert!(result.is_err());
}

// =============================================================================
// Sharded File Loading Tests
// =============================================================================

#[test]
fn test_load_sharded_files() {
    let dir = tempdir().unwrap();

    // Create sharded safetensors files
    create_test_safetensors(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("layer1.weight", vec![1.0, 2.0], vec![2])],
    );
    create_test_safetensors(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("layer2.weight", vec![3.0, 4.0], vec![2])],
    );

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let paths = vec![
        dir.path().join("model-00001-of-00002.safetensors"),
        dir.path().join("model-00002-of-00002.safetensors"),
    ];
    let result = loader.load_sharded(&paths);

    assert!(result.is_ok());
}

#[test]
fn test_find_sharded_files_by_prefix() {
    let dir = tempdir().unwrap();

    // Create sharded files with pattern
    fs::write(dir.path().join("model-00001-of-00003.safetensors"), "dummy").unwrap();
    fs::write(dir.path().join("model-00002-of-00003.safetensors"), "dummy").unwrap();
    fs::write(dir.path().join("model-00003-of-00003.safetensors"), "dummy").unwrap();
    fs::write(dir.path().join("other_model.safetensors"), "dummy").unwrap();

    let files = find_sharded_files(dir.path(), "model-").unwrap();

    assert_eq!(files.len(), 3);
    assert!(files[0].to_str().unwrap().contains("model-00001-of-00003"));
    assert!(files[2].to_str().unwrap().contains("model-00003-of-00003"));
}

// =============================================================================
// Auto Shard Detection Tests (model.safetensors.index.json)
// =============================================================================

#[test]
fn test_auto_detect_shards_from_index() {
    let dir = tempdir().unwrap();

    // Create shard files
    create_test_safetensors(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("layer1.weight", vec![1.0, 2.0], vec![2])],
    );
    create_test_safetensors(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("layer2.weight", vec![3.0, 4.0], vec![2])],
    );

    // Create index.json
    let mut weight_map = HashMap::new();
    weight_map.insert(
        "layer1.weight".to_string(),
        "model-00001-of-00002.safetensors".to_string(),
    );
    weight_map.insert(
        "layer2.weight".to_string(),
        "model-00002-of-00002.safetensors".to_string(),
    );

    create_test_index_json(
        &dir.path().join("model.safetensors.index.json"),
        &weight_map,
        Some(serde_json::json!({
            "format": "safetensors",
            "total_size": 16
        })),
    );

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let result = loader.load_from_directory(dir.path());

    assert!(result.is_ok());
}

#[test]
fn test_auto_detect_single_file() {
    let dir = tempdir().unwrap();

    // Create single model file (no sharding)
    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[("weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
    );

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let result = loader.load_from_directory(dir.path());

    assert!(result.is_ok());
}

#[test]
fn test_parse_index_json() {
    let dir = tempdir().unwrap();

    let mut weight_map = HashMap::new();
    weight_map.insert("a.weight".to_string(), "shard1.safetensors".to_string());
    weight_map.insert("b.weight".to_string(), "shard2.safetensors".to_string());

    create_test_index_json(
        &dir.path().join("model.safetensors.index.json"),
        &weight_map,
        Some(serde_json::json!({
            "format": "safetensors",
            "total_size": 1024
        })),
    );

    let index = SafetensorsIndex::load(dir.path().join("model.safetensors.index.json")).unwrap();

    assert_eq!(index.weight_map.len(), 2);
    assert_eq!(
        index.weight_map.get("a.weight"),
        Some(&"shard1.safetensors".to_string())
    );
    assert_eq!(index.shard_files().len(), 2);
}

#[test]
fn test_get_tensor_location() {
    let dir = tempdir().unwrap();

    let mut weight_map = HashMap::new();
    weight_map.insert(
        "encoder.layer.0.weight".to_string(),
        "model-00001-of-00002.safetensors".to_string(),
    );
    weight_map.insert(
        "encoder.layer.1.weight".to_string(),
        "model-00002-of-00002.safetensors".to_string(),
    );

    create_test_index_json(
        &dir.path().join("model.safetensors.index.json"),
        &weight_map,
        None,
    );

    let index = SafetensorsIndex::load(dir.path().join("model.safetensors.index.json")).unwrap();

    assert_eq!(
        index.get_file_for_tensor("encoder.layer.0.weight"),
        Some("model-00001-of-00002.safetensors")
    );
    assert_eq!(index.get_file_for_tensor("nonexistent"), None);
}

// =============================================================================
// Config Loading Tests
// =============================================================================

#[test]
fn test_load_vae_config() {
    let dir = tempdir().unwrap();

    let config = serde_json::json!({
        "_class_name": "CausalVideoAutoencoder",
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "temporal_downsample": 8,
        "spatial_downsample": 32,
        "causal": true
    });

    create_test_config_json(&dir.path().join("config.json"), &config);

    let loaded = load_model_config::<serde_json::Value>(dir.path().join("config.json")).unwrap();

    assert_eq!(loaded["in_channels"], 3);
    assert_eq!(loaded["latent_channels"], 128);
}

#[test]
fn test_load_dit_config() {
    let dir = tempdir().unwrap();

    let config = serde_json::json!({
        "_class_name": "LTXVideoTransformer3DModel",
        "patch_size": 2,
        "in_channels": 128,
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "caption_channels": 4096,
        "mlp_ratio": 4.0
    });

    create_test_config_json(&dir.path().join("config.json"), &config);

    let loaded = load_model_config::<serde_json::Value>(dir.path().join("config.json")).unwrap();

    assert_eq!(loaded["patch_size"], 2);
    assert_eq!(loaded["hidden_size"], 1152);
    assert_eq!(loaded["depth"], 28);
}

#[test]
fn test_config_not_found() {
    let result = load_model_config::<serde_json::Value>("/nonexistent/config.json");
    assert!(result.is_err());
}

// =============================================================================
// Tensor Name Validation Tests
// =============================================================================

#[test]
fn test_validate_tensor_names() {
    let expected = vec![
        "encoder.layer.0.weight".to_string(),
        "encoder.layer.0.bias".to_string(),
        "decoder.layer.0.weight".to_string(),
    ];

    let actual = vec!["encoder.layer.0.weight", "encoder.layer.0.bias"];

    let missing = validate_tensor_names(&expected, &actual);

    assert_eq!(missing.len(), 1);
    assert!(missing.contains(&"decoder.layer.0.weight".to_string()));
}

#[test]
fn test_validate_all_tensors_present() {
    let expected = vec!["weight1".to_string(), "weight2".to_string()];
    let actual = vec!["weight1", "weight2", "weight3"]; // Extra is OK

    let missing = validate_tensor_names(&expected, &actual);

    assert!(missing.is_empty());
}

#[test]
fn test_list_available_tensors() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");

    create_test_safetensors(
        &file_path,
        &[
            ("layer1.weight", vec![1.0, 2.0], vec![2]),
            ("layer1.bias", vec![0.0], vec![1]),
            ("layer2.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ],
    );

    let tensor_names = list_tensor_names(&file_path).unwrap();

    assert_eq!(tensor_names.len(), 3);
    assert!(tensor_names.contains(&"layer1.weight".to_string()));
    assert!(tensor_names.contains(&"layer1.bias".to_string()));
    assert!(tensor_names.contains(&"layer2.weight".to_string()));
}

// =============================================================================
// Model Loading Builder Pattern Tests
// =============================================================================

#[test]
fn test_model_loader_builder() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32)
        .add_mapping("old_name", "new_name")
        .add_prefix_mapping("model.", "")
        .with_strict_mode(true);

    assert!(loader.is_strict_mode());
}

#[test]
fn test_model_loader_non_strict_mode() {
    let loader = WeightLoader::new(Device::Cpu, DType::F32).with_strict_mode(false);

    assert!(!loader.is_strict_mode());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_invalid_safetensors_file() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("invalid.safetensors");

    fs::write(&file_path, "not a valid safetensors file").unwrap();

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let result = loader.load_single(&file_path);

    assert!(result.is_err());
}

#[test]
fn test_missing_shard_file() {
    let dir = tempdir().unwrap();

    // Create only partial shards
    create_test_safetensors(
        &dir.path().join("model-00001-of-00003.safetensors"),
        &[("layer1.weight", vec![1.0], vec![1])],
    );
    // Missing shard 2 and 3

    let mut weight_map = HashMap::new();
    weight_map.insert(
        "layer1.weight".to_string(),
        "model-00001-of-00003.safetensors".to_string(),
    );
    weight_map.insert(
        "layer2.weight".to_string(),
        "model-00002-of-00003.safetensors".to_string(), // Missing!
    );

    create_test_index_json(
        &dir.path().join("model.safetensors.index.json"),
        &weight_map,
        None,
    );

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let result = loader.load_from_directory(dir.path());

    assert!(result.is_err());
}

// =============================================================================
// Integration with VarBuilder Tests
// =============================================================================

#[test]
fn test_varbuilder_push_prefix() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");

    create_test_safetensors(
        &file_path,
        &[
            (
                "transformer.layer.0.weight",
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2, 2],
            ),
            ("transformer.layer.0.bias", vec![0.1, 0.2], vec![2]),
        ],
    );

    let loader = WeightLoader::new(Device::Cpu, DType::F32);
    let vb = loader.load_single(&file_path).unwrap();

    // Test pushing prefix to access nested weights
    let transformer_vb = vb.push_prefix("transformer");
    let layer_vb = transformer_vb.push_prefix("layer.0");

    let weight = layer_vb.get((2, 2), "weight");
    assert!(weight.is_ok());

    let bias = layer_vb.get(2, "bias");
    assert!(bias.is_ok());
}
