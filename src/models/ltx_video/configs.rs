//! Official LTX-Video configurations and presets.
//! This module provides centralized, up-to-date parameters for various LTX-Video model versions (0.9.*).

use crate::models::ltx_video::ltx_transformer::LtxVideoTransformer3DModelConfig;
use crate::models::ltx_video::vae::AutoencoderKLLtxVideoConfig;
use crate::models::ltx_video::scheduler::FlowMatchEulerDiscreteSchedulerConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXVInferenceConfig {
    pub guidance_scale: f32,
    pub num_inference_steps: usize,
    pub stg_scale: f32,
    pub rescaling_scale: f32,
    pub stochastic_sampling: bool,
    pub skip_block_list: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXVFullConfig {
    pub inference: LTXVInferenceConfig,
    pub transformer: LtxVideoTransformer3DModelConfig,
    pub vae: AutoencoderKLLtxVideoConfig,
    pub scheduler: FlowMatchEulerDiscreteSchedulerConfig,
}

/// Returns the full configuration for a given version string.
pub fn get_config_by_version(version: &str) -> LTXVFullConfig {
    match version {
        "0.9.0" | "0.9.1" => presets::v0_9_1_2b(),
        "0.9.5" => presets::v0_9_5_2b(),
        "0.9.6-dev" => presets::v0_9_6_dev_2b(),
        "0.9.6-distilled" => presets::v0_9_6_distilled_2b(),
        "0.9.7-dev" | "0.9.7-13b-dev" => presets::v0_9_7_dev_13b(),
        "0.9.7-distilled" => presets::v0_9_7_distilled_2b(),
        "0.9.8-2b-dev" => presets::v0_9_8_dev_2b(),
        "0.9.8-13b-dev" => presets::v0_9_8_dev_13b(),
        "0.9.8-2b" | "0.9.8-2b-distilled" | "0.9.8-distilled" => presets::v0_9_8_distilled_2b(),
        "0.9.8-13b" | "0.9.8-13b-distilled" => presets::v0_9_8_distilled_13b(),
        _ => presets::v0_9_5_2b(), // Default to 0.9.5
    }
}

pub mod presets {
    use super::*;

    fn common_vae_config() -> AutoencoderKLLtxVideoConfig {
        AutoencoderKLLtxVideoConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: vec![4, 3, 3, 3, 4],
            latent_channels: 128,
            patch_size: 4,
            timestep_conditioning: true,
            ..Default::default()
        }
    }

    fn common_scheduler_config() -> FlowMatchEulerDiscreteSchedulerConfig {
        FlowMatchEulerDiscreteSchedulerConfig {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: true,
            ..Default::default()
        }
    }

    pub fn v0_9_1_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![19],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 28,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_5_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![19],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 28,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_6_dev_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 28,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_6_distilled_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 20,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: true,
                skip_block_list: vec![],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 28,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_7_dev_13b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 40,
                num_attention_heads: 80,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_7_distilled_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 8,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: true,
                skip_block_list: vec![],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 28,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_8_dev_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 28,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_8_dev_13b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![11, 25, 35, 39],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 40,
                num_attention_heads: 80,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_8_distilled_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 30,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: false,
                skip_block_list: vec![42],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 48,
                num_attention_heads: 32,
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    pub fn v0_9_8_distilled_13b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 30,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: false,
                skip_block_list: vec![42],
            },
            transformer: LtxVideoTransformer3DModelConfig {
                num_layers: 64,
                num_attention_heads: 64, // Hidden = 4096
                attention_head_dim: 64,
                cross_attention_dim: 2048,
                caption_channels: 4096,
                ..Default::default()
            },
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }
}
