//! Weight format detection and key remapping for LTX-Video models.
//!
//! Supports two formats:
//! - Diffusers: separate files in transformer/, vae/, text_encoder/ directories
//! - Official: single unified safetensors file (e.g., ltx-video-2b-v0.9.5.safetensors)

use std::collections::HashMap;
use std::path::Path;

/// Weight format detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFormat {
    /// Diffusers format: separate files in subdirectories
    Diffusers,
    /// Official LTX-Video format: single unified safetensors file
    Official,
}

/// Detect weight format from path
pub fn detect_format(path: &Path) -> WeightFormat {
    if path.is_file() {
        // Single file = Official format
        WeightFormat::Official
    } else if path.is_dir() {
        // Directory with subdirectories = Diffusers format
        WeightFormat::Diffusers
    } else {
        // Default to Diffusers
        WeightFormat::Diffusers
    }
}

/// Key remapping from Official format to Diffusers format.
/// Based on LTX-Video/ltx_video/utils/diffusers_config_mapping.py
pub struct KeyRemapper {
    transformer_map: HashMap<&'static str, &'static str>,
    vae_map: HashMap<&'static str, &'static str>,
}

impl Default for KeyRemapper {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyRemapper {
    pub fn new() -> Self {
        // Transformer key mappings: Official -> Diffusers
        let transformer_map: HashMap<&'static str, &'static str> = [
            ("patchify_proj", "proj_in"),
            ("adaln_single", "time_embed"),
            ("q_norm", "norm_q"),
            ("k_norm", "norm_k"),
        ]
        .into_iter()
        .collect();

        // VAE key mappings: Official -> Diffusers (complex remapping)
        let vae_map: HashMap<&'static str, &'static str> = [
            // Decoder block remapping
            ("decoder.up_blocks.9", "decoder.up_blocks.3"),
            ("decoder.up_blocks.8", "decoder.up_blocks.3.upsamplers.0"),
            ("decoder.up_blocks.7", "decoder.up_blocks.3.conv_in"),
            ("decoder.up_blocks.6", "decoder.up_blocks.2"),
            ("decoder.up_blocks.5", "decoder.up_blocks.2.upsamplers.0"),
            ("decoder.up_blocks.4", "decoder.up_blocks.2.conv_in"),
            ("decoder.up_blocks.3", "decoder.up_blocks.1"),
            ("decoder.up_blocks.2", "decoder.up_blocks.1.upsamplers.0"),
            ("decoder.up_blocks.1", "decoder.up_blocks.0"),
            ("decoder.up_blocks.0", "decoder.mid_block"),
            // Encoder block remapping
            ("encoder.down_blocks.9", "encoder.mid_block"),
            ("encoder.down_blocks.8", "encoder.down_blocks.3"),
            (
                "encoder.down_blocks.7",
                "encoder.down_blocks.2.downsamplers.0",
            ),
            ("encoder.down_blocks.6", "encoder.down_blocks.2"),
            ("encoder.down_blocks.5", "encoder.down_blocks.1.conv_out"),
            (
                "encoder.down_blocks.4",
                "encoder.down_blocks.1.downsamplers.0",
            ),
            ("encoder.down_blocks.3", "encoder.down_blocks.1"),
            ("encoder.down_blocks.2", "encoder.down_blocks.0.conv_out"),
            (
                "encoder.down_blocks.1",
                "encoder.down_blocks.0.downsamplers.0",
            ),
            // Other mappings
            ("conv_shortcut", "conv_shortcut.conv"),
            ("res_blocks", "resnets"),
            ("norm3.norm", "norm3"),
            ("per_channel_statistics.mean-of-means", "latents_mean"),
            ("per_channel_statistics.std-of-means", "latents_std"),
        ]
        .into_iter()
        .collect();

        Self {
            transformer_map,
            vae_map,
        }
    }

    /// Remap a key from Official format to Diffusers format
    pub fn remap_key(&self, key: &str) -> String {
        let mut result = key.to_string();

        // Apply transformer mappings
        for (from, to) in &self.transformer_map {
            if result.contains(from) {
                result = result.replace(from, to);
            }
        }

        // Apply VAE mappings (order matters - longer patterns first)
        let mut vae_entries: Vec<_> = self.vae_map.iter().collect();
        vae_entries.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // Sort by length desc

        for (from, to) in vae_entries {
            if result.contains(from) {
                result = result.replace(from, to);
            }
        }

        result
    }

    /// Check if a key belongs to the transformer
    pub fn is_transformer_key(key: &str) -> bool {
        key.starts_with("transformer.")
            || key.contains("transformer_blocks")
            || key.contains("patchify_proj")
            || key.contains("proj_in")
            || key.contains("adaln_single")
            || key.contains("time_embed")
    }

    /// Check if a key belongs to the VAE
    pub fn is_vae_key(key: &str) -> bool {
        key.starts_with("vae.")
            || key.starts_with("encoder.")
            || key.starts_with("decoder.")
            || key.contains("per_channel_statistics")
            || key.contains("latents_mean")
            || key.contains("latents_std")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_transformer_key() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("transformer.patchify_proj.weight"),
            "transformer.proj_in.weight"
        );
        assert_eq!(
            remapper.remap_key("transformer.adaln_single.linear.weight"),
            "transformer.time_embed.linear.weight"
        );
    }

    #[test]
    fn test_remap_vae_key() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("decoder.up_blocks.0.res_blocks.0.weight"),
            "decoder.mid_block.resnets.0.weight"
        );
    }
}
