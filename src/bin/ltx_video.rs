//! LTX-Video Text-to-Video Generation Example
//!
//! This example demonstrates how to use the candle-video library to generate
//! videos from text prompts using the LTX-Video model.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin ltx-video -- \
//!     --prompt "A beautiful sunset over the ocean" \
//!     --model-path ./ltxv-2b-0.9.8-distilled \
//!     --output output_video
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn};

use candle_video::{
    config::{DitConfig, InferenceConfig, SchedulerConfig, VaeConfig},
    dit::Transformer3DModel,
    latents_bin::write_f32_tensor_with_header,
    pipeline::PipelineConfig,
    rope,  // Use module directly for generate_indices_grid_raw
    scheduler::RectifiedFlowScheduler,
    vae::VaeDecoder,
};

/// Command line arguments for LTX-Video generation
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate videos from text prompts using LTX-Video"
)]
struct Args {
    /// The text prompt describing the video to generate
    #[arg(short, long)]
    prompt: String,

    /// Negative prompt for classifier-free guidance
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Path to the model directory (containing safetensors files)
    #[arg(short, long)]
    model_path: PathBuf,

    /// Path to T5 text encoder (optional, will download if not provided)
    #[arg(long)]
    text_encoder_path: Option<PathBuf>,

    /// Output directory for generated video frames
    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    /// Number of frames to generate (must be 8N+1: 9, 17, 25, 33...)
    #[arg(long, default_value = "17")]
    num_frames: usize,

    /// Video height (must be multiple of 32)
    #[arg(long, default_value = "512")]
    height: usize,

    /// Video width (must be multiple of 32)
    #[arg(long, default_value = "768")]
    width: usize,

    /// Number of inference steps
    #[arg(long, default_value = "20")]
    num_inference_steps: usize,

    /// Classifier-free guidance scale
    #[arg(long, default_value = "7.5")]
    guidance_scale: f64,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Use CPU instead of CUDA
    #[arg(long)]
    cpu: bool,

    /// Use bfloat16 precision (faster on supported GPUs)
    #[arg(long)]
    bf16: bool,

    /// Use float16 precision
    #[arg(long)]
    f16: bool,

    /// Separate VAE model path (optional, uses embedded VAE from main model if not provided)
    #[arg(long)]
    vae_path: Option<PathBuf>,

    /// Enable mock mode (for testing without loading models)
    #[arg(long)]
    mock: bool,
}

/// Main entry point for LTX-Video generation
fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("candle_video=info".parse()?)
                .add_directive("ltx_video=info".parse()?),
        )
        .init();

    let args = Args::parse();

    info!("LTX-Video Text-to-Video Generator");
    info!("==================================");
    info!("Prompt: {}", args.prompt);
    info!(
        "Video: {}x{} @ {} frames",
        args.width, args.height, args.num_frames
    );
    info!("Inference steps: {}", args.num_inference_steps);
    info!("Guidance scale: {}", args.guidance_scale);

    // Select device
    let device = if args.cpu {
        info!("Using CPU");
        Device::Cpu
    } else {
        match Device::cuda_if_available(0) {
            Ok(device) => {
                info!("Using CUDA device 0");
                device
            }
            Err(e) => {
                warn!("CUDA not available: {}, falling back to CPU", e);
                Device::Cpu
            }
        }
    };

    // Select dtype - BF16 is default (LTX models are stored in BF16)
    let dtype = if args.f16 {
        info!("Using Float16 precision");
        DType::F16
    } else if args.bf16 {
        info!("Using BFloat16 precision");
        DType::BF16
    } else {
        // Default to BF16 for LTX models
        info!("Using BFloat16 precision (default)");
        DType::BF16
    };

    // Validate num_frames
    if args.num_frames < 1 {
        anyhow::bail!("num_frames must be at least 1 (got {}).", args.num_frames);
    }

    // Create inference configuration
    let inference_config =
        InferenceConfig::new(args.num_frames, args.height, args.width, args.seed)
            .context("Invalid inference configuration")?
            .with_steps(args.num_inference_steps)
            .with_guidance_scale(args.guidance_scale);

    // Run generation
    if args.mock {
        info!("Running in mock mode (no model loading)");
        run_mock_generation(&args, &device, dtype, &inference_config)?;
    } else {
        run_full_generation(&args, &device, dtype, &inference_config)?;
    }

    info!("Generation complete!");
    Ok(())
}

/// Run mock generation for testing
fn run_mock_generation(
    args: &Args,
    device: &Device,
    _dtype: DType,
    config: &InferenceConfig,
) -> Result<()> {
    use candle_video::pipeline::TextToVideoPipeline;

    info!("Creating mock pipeline...");

    let pipeline_config = PipelineConfig::default();
    let pipeline = TextToVideoPipeline::new(device.clone(), pipeline_config)?;

    // Create mock text embeddings
    let text_emb = Tensor::randn(0f32, 1.0, (1, 77, 4096), device)?;

    info!("Running mock denoising loop...");

    let video = if !args.negative_prompt.is_empty() || args.guidance_scale > 1.0 {
        let neg_emb = Tensor::randn(0f32, 0.1, (1, 77, 4096), device)?;
        pipeline.mock_generate_with_cfg(&text_emb, &neg_emb, config)?
    } else {
        pipeline.mock_generate(&text_emb, config)?
    };

    info!("Generated video tensor shape: {:?}", video.dims());

    // Save frames
    save_video_frames(&video, &args.output)?;

    Ok(())
}

/// Run full generation with loaded models
fn run_full_generation(
    args: &Args,
    device: &Device,
    dtype: DType,
    config: &InferenceConfig,
) -> Result<()> {
    info!("Loading models from {:?}...", args.model_path);

    // Find safetensors file for DiT
    let dit_model_file = find_model_file(&args.model_path)?;
    info!("Using DiT model file: {:?}", dit_model_file);

    // Find VAE file:
    // 1. Use --vae-path if explicitly provided
    // 2. Otherwise use embedded VAE from main model
    let (vae_file, vae_prefix) = if let Some(vae_path) = &args.vae_path {
        info!("Using separate VAE from --vae-path: {:?}", vae_path);
        (vae_path.clone(), "decoder") // Separate file uses "decoder" prefix
    } else {
        info!("Using embedded VAE from main model file");
        (dit_model_file.clone(), "vae.decoder") // Embedded uses "vae.decoder" prefix
    };

    // Create configurations
    let dit_config = create_dit_config();
    let vae_config = create_vae_config();
    let scheduler_config = SchedulerConfig {
        num_inference_steps: args.num_inference_steps,
        guidance_scale: args.guidance_scale,
        timestep_spacing: "linspace".to_string(),
        use_dynamic_shifting: true, // Enable resolution-dependent timestep shifting
        ..SchedulerConfig::default()
    };

    // Load DiT model
    info!("Loading DiT model...");
    let vb_dit = unsafe { VarBuilder::from_mmaped_safetensors(&[&dit_model_file], dtype, device)? };
    let vb_dit = vb_dit.pp("model.diffusion_model");
    let dit_model = Transformer3DModel::new(vb_dit, &dit_config)?;
    info!("DiT model loaded");

    // Load VAE decoder
    info!("Loading VAE decoder...");
    let vb_vae = unsafe { VarBuilder::from_mmaped_safetensors(&[&vae_file], dtype, device)? };
    let vb_vae = vb_vae.pp(vae_prefix);
    let vae_decoder = VaeDecoder::new(vb_vae)?;
    info!("VAE decoder loaded");

    // Load latent statistics for denormalization (from VAE config)
    // These are stored in vae.per_channel_statistics in the main model file
    let vb_stats =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&dit_model_file], dtype, device)? };
    let latents_mean = vb_stats
        .get((128,), "vae.per_channel_statistics.mean-of-means")
        .unwrap_or_else(|_| Tensor::zeros((128,), dtype, device).unwrap());
    let latents_std = vb_stats
        .get((128,), "vae.per_channel_statistics.std-of-means")
        .unwrap_or_else(|_| Tensor::ones((128,), dtype, device).unwrap());
    info!("Loaded latent statistics for denormalization");

    // Create scheduler (will be updated with sample shape after latent dims are computed)
    let mut scheduler = RectifiedFlowScheduler::new(scheduler_config);

    // Text encoding
    info!("Creating text embeddings...");
    let (text_emb, neg_emb) = create_text_embeddings(
        &args.prompt,
        &args.negative_prompt,
        device,
        dtype,
        &args.model_path,
    )?;

    // Initialize noise
    info!("Initializing noise...");
    let (lat_t, lat_h, lat_w) = compute_latent_dims(config, &vae_config);
    let mut latents = Tensor::randn(
        0f32,
        1.0,
        (1, vae_config.latent_channels, lat_t, lat_h, lat_w),
        device,
    )?
    .to_dtype(dtype)?;
    info!(
        "Latent shape: {:?} (T={}, H={}, W={})",
        latents.dims(),
        lat_t,
        lat_h,
        lat_w
    );

    // Update scheduler with sample shape for dynamic timestep shifting
    let sample_shape = (1, vae_config.latent_channels, lat_t, lat_h, lat_w);
    scheduler.set_timesteps_with_shape(scheduler.num_steps(), Some(sample_shape), None, None)?;
    info!(
        "Scheduler timesteps (first 3): {:?}",
        &scheduler.timesteps()[..3.min(scheduler.num_steps())]
    );

    // Generate RoPE indices with standard LTX-Video scaling
    // base_num_frames=128, base_resolution=512x768
    // rope_scales: 8 (temporal), 32 (spatial) -> matches vae compression
    let indices_grid = rope::generate_indices_grid_for_diffusers(
        1, lat_t, lat_h, lat_w,
        128,   // base_num_frames (standard context)
        512,   // base_height
        768,   // base_width
        1,     // patch_size
        1,     // patch_size_t
        8.0,   // rope_scale_t
        32.0,  // rope_scale_h
        32.0,  // rope_scale_w
        device,
    )?;

    // Denoising loop
    info!(
        "Starting denoising loop ({} steps)...",
        scheduler.num_steps()
    );
    let use_cfg = args.guidance_scale > 1.0;

    for step in 0..scheduler.num_steps() {
        let timestep = scheduler.timesteps()[step];
        let progress = (step + 1) as f32 / scheduler.num_steps() as f32 * 100.0;

        info!(
            "Step {}/{} (t={:.4}, progress={:.1}%)",
            step + 1,
            scheduler.num_steps(),
            timestep,
            progress
        );

        // Create timestep tensor - flow matching uses [0, 1] range
        let timestep_tensor = Tensor::new(&[timestep as f32], device)?.to_dtype(dtype)?;

        // Run DiT forward pass
        let model_output = if use_cfg {
            // Run with CFG - two forward passes
            let latents_input = Tensor::cat(&[&latents, &latents], 0)?;
            let text_input = Tensor::cat(&[&neg_emb, &text_emb], 0)?;
            let timestep_input = Tensor::cat(&[&timestep_tensor, &timestep_tensor], 0)?;

            let output = dit_model.forward(
                &latents_input,
                &indices_grid,
                Some(&text_input),
                &timestep_input,
                None,
                None,
                None, // skip_layer_mask
                None, // skip_layer_strategy
            )?;

            // Apply CFG
            let output_uncond = output.i(0)?;
            let output_cond = output.i(1)?;
            scheduler.apply_cfg(&output_cond, &output_uncond)?
        } else {
            dit_model
                .forward(
                    &latents,
                    &indices_grid,
                    Some(&text_emb),
                    &timestep_tensor,
                    None,
                    None,
                    None, // skip_layer_mask
                    None, // skip_layer_strategy
                )?
                .squeeze(0)?
        };

        // Debug: check model_output (convert to F32 for debug)
        let out_flat = model_output.flatten_all()?.to_dtype(DType::F32)?;
        let out_min = out_flat.min(0)?.to_scalar::<f32>()?;
        let out_max = out_flat.max(0)?.to_scalar::<f32>()?;
        info!(
            "  Model output range: min={:.4}, max={:.4}",
            out_min, out_max
        );

        // Euler step
        latents = scheduler.step(&model_output.unsqueeze(0)?, &latents, step)?;

        // Debug: check for NaN after step (convert to F32 for debug)
        let lat_flat = latents.flatten_all()?.to_dtype(DType::F32)?;
        let lat_min = lat_flat.min(0)?.to_scalar::<f32>()?;
        let lat_max = lat_flat.max(0)?.to_scalar::<f32>()?;
        info!("  Latent range: min={:.4}, max={:.4}", lat_min, lat_max);
        if lat_min.is_nan() || lat_max.is_nan() {
            warn!("NaN detected after step {}!", step + 1);
        }
    }

    info!("Denoising complete, decoding to video...");

    // Debug: check latent values (convert to F32 for debug output)
    let lat_flat = latents.flatten_all()?.to_dtype(DType::F32)?;
    let lat_min = lat_flat.min(0)?.to_scalar::<f32>()?;
    let lat_max = lat_flat.max(0)?.to_scalar::<f32>()?;
    info!("Latent range: min={:.4}, max={:.4}", lat_min, lat_max);

    // Denormalize latents before VAE decode
    // Formula: latents = latents * std + mean
    // latents shape: (B, C, T, H, W), mean/std shape: (C,)
    // Reshape mean/std to (1, C, 1, 1, 1) for broadcasting
    let latents_mean_5d = latents_mean.reshape((1, 128, 1, 1, 1))?;
    let latents_std_5d = latents_std.reshape((1, 128, 1, 1, 1))?;
    let latents_denorm = latents
        .broadcast_mul(&latents_std_5d)?
        .broadcast_add(&latents_mean_5d)?;

    // Debug: check denormalized latent values (convert to F32 for debug)
    let denorm_flat = latents_denorm.flatten_all()?.to_dtype(DType::F32)?;
    let denorm_min = denorm_flat.min(0)?.to_scalar::<f32>()?;
    let denorm_max = denorm_flat.max(0)?.to_scalar::<f32>()?;
    info!(
        "Denormalized latent range: min={:.4}, max={:.4}",
        denorm_min, denorm_max
    );

    // Save latents for Python VAE decoding test (use denormalized latents)
    let latents_path = args.output.join("latents.bin");
    info!(
        "Saving latents to {:?} for Python VAE test...",
        latents_path
    );
    write_f32_tensor_with_header(&latents_path, &latents_denorm)?;
    info!("Latents saved");

    // Decode latents to video with timestep conditioning
    // Use small timestep value (near end of diffusion) for VAE decode
    let vae_timestep = Some(0.05);
    let video = vae_decoder.decode(&latents_denorm, vae_timestep)?;
    info!("Video tensor shape (raw): {:?}", video.dims());

    // Trim temporal dimension to match requested num_frames
    // VAE produces: latent_frames * 8 = (num_frames - 1) / 8 * 8 + 8 frames
    // We want: num_frames frames
    let (_b, _c, t, _h, _w) = video.dims5()?;
    let video = if t > config.num_frames {
        info!("Trimming video from {} to {} frames", t, config.num_frames);
        video.narrow(2, 0, config.num_frames)?
    } else {
        video
    };
    info!("Video tensor shape: {:?}", video.dims());

    // Debug: check video values (convert to F32 for debug)
    let vid_flat = video.flatten_all()?.to_dtype(DType::F32)?;
    let vid_min = vid_flat.min(0)?.to_scalar::<f32>()?;
    let vid_max = vid_flat.max(0)?.to_scalar::<f32>()?;
    info!("Video range: min={:.4}, max={:.4}", vid_min, vid_max);

    // Save frames
    save_video_frames(&video, &args.output)?;

    Ok(())
}

/// Find the model safetensors file in the given directory
fn find_model_file(model_path: &PathBuf) -> Result<PathBuf> {
    if model_path.is_file() {
        return Ok(model_path.clone());
    }

    // Look for safetensors files
    let patterns = ["*.safetensors", "model.safetensors", "ltxv*.safetensors"];

    for pattern in patterns {
        let full_pattern = model_path.join(pattern);
        if let Ok(entries) = glob::glob(full_pattern.to_str().unwrap_or(""))
            && let Some(entry) = entries.flatten().next()
        {
            return Ok(entry);
        }
    }

    // Fallback: look for any .safetensors file
    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            return Ok(path);
        }
    }

    anyhow::bail!("No safetensors file found in {:?}", model_path);
}

/// Create DiT configuration for LTX-Video 2B
fn create_dit_config() -> DitConfig {
    DitConfig {
        patch_size: 1,          // From model config
        patch_size_t: Some(1),  // From model config
        in_channels: 128,       // VAE latent channels
        hidden_size: 2048,      // 32 heads * 64 head_dim
        depth: 28,              // num_layers
        num_heads: 32,          // num_attention_heads
        caption_channels: 4096, // T5-XXL output dimension
        mlp_ratio: 4.0,         // 2048 * 4 = 8192 inner dim
        use_flash_attention: false,
        timestep_scale_multiplier: Some(1000.0), // LTX-Video 2B uses 1000x scaling
    }
}

/// Create VAE configuration for LTX-Video
fn create_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 128,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 4,
        temporal_downsample: 8,
        spatial_downsample: 32,
        causal: true,
        latents_mean: None, // Loaded from model file at runtime
        latents_std: None,  // Loaded from model file at runtime
        scaling_factor: 1.0,
        timestep_conditioning: true, // LTX-Video uses timestep conditioning
    }
}

/// Compute latent dimensions from inference config
/// Formula from diffusers: lat_t = (num_frames - 1) // temporal_compression + 1
fn compute_latent_dims(config: &InferenceConfig, vae_config: &VaeConfig) -> (usize, usize, usize) {
    // From diffusers: num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1
    let lat_t = (config.num_frames.saturating_sub(1)) / vae_config.temporal_downsample + 1;
    let lat_h = config.height / vae_config.spatial_downsample;
    let lat_w = config.width / vae_config.spatial_downsample;
    (lat_t.max(1), lat_h.max(1), lat_w.max(1))
}

/// Create text embeddings using T5 encoder
fn create_text_embeddings(
    prompt: &str,
    negative_prompt: &str,
    device: &Device,
    dtype: DType,
    model_path: &std::path::Path,
) -> Result<(Tensor, Tensor)> {
    use tokenizers::Tokenizer;

    // Try to find T5 GGUF file
    let gguf_path = model_path.join("t5-v1_1-xxl-encoder-Q5_K_M.gguf");
    let tokenizer_path = model_path.join("T5-XXL-8bit").join("tokenizer.json");

    if gguf_path.exists() && tokenizer_path.exists() {
        info!("Loading T5 encoder from GGUF...");

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model on CPU (quantized ops don't have CUDA implementations)
        let cpu = Device::Cpu;
        let t5_model = candle_video::QuantizedT5EncoderModel::load(&gguf_path, &cpu)
            .context("Failed to load T5 encoder model")?;

        info!("T5 encoder loaded, encoding prompts...");

        // Tokenize and encode positive prompt (on CPU)
        let pos_encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let pos_ids: Vec<u32> = pos_encoding.get_ids().to_vec();
        let pos_input = Tensor::new(&pos_ids[..], &cpu)?.unsqueeze(0)?;
        let text_emb = t5_model.forward(&pos_input)?;
        // Transfer to target device and convert dtype
        let text_emb = text_emb.to_device(device)?.to_dtype(dtype)?;

        // Tokenize and encode negative prompt (on CPU)
        let neg_prompt = if negative_prompt.is_empty() {
            ""
        } else {
            negative_prompt
        };
        let neg_encoding = tokenizer
            .encode(neg_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let neg_ids: Vec<u32> = neg_encoding.get_ids().to_vec();
        let neg_input = Tensor::new(&neg_ids[..], &cpu)?.unsqueeze(0)?;
        let neg_emb = t5_model.forward(&neg_input)?;
        // Transfer to target device and convert dtype
        let neg_emb = neg_emb.to_device(device)?.to_dtype(dtype)?;

        // Ensure same sequence length by padding
        let pos_len = text_emb.dim(1)?;
        let neg_len = neg_emb.dim(1)?;

        let (text_emb, neg_emb) = if pos_len != neg_len {
            let max_len = pos_len.max(neg_len);
            let text_emb = pad_sequence(&text_emb, max_len, dtype, device)?;
            let neg_emb = pad_sequence(&neg_emb, max_len, dtype, device)?;
            (text_emb, neg_emb)
        } else {
            (text_emb, neg_emb)
        };

        info!("Text embeddings shape: {:?}", text_emb.dims());

        // Debug: show embedding range (convert to F32 for display)
        let emb_flat = text_emb.to_dtype(DType::F32)?.flatten_all()?;
        let emb_min = emb_flat.min(0)?.to_scalar::<f32>()?;
        let emb_max = emb_flat.max(0)?.to_scalar::<f32>()?;
        info!(
            "Text embeddings range: min={:.4}, max={:.4}",
            emb_min, emb_max
        );

        Ok((text_emb, neg_emb))
    } else {
        // Fallback to mock embeddings
        warn!("T5 encoder files not found, using mock embeddings");
        warn!("  Expected GGUF: {:?}", gguf_path);
        warn!("  Expected tokenizer: {:?}", tokenizer_path);

        let seq_len = 77;
        let d_model = 4096;

        let text_emb = Tensor::randn(0f32, 1.0, (1, seq_len, d_model), device)?.to_dtype(dtype)?;
        let neg_emb = Tensor::randn(0f32, 0.1, (1, seq_len, d_model), device)?.to_dtype(dtype)?;

        Ok((text_emb, neg_emb))
    }
}

/// Pad a sequence tensor to target length
fn pad_sequence(
    tensor: &Tensor,
    target_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let current_len = tensor.dim(1)?;
    if current_len >= target_len {
        return Ok(tensor.clone());
    }

    let batch_size = tensor.dim(0)?;
    let d_model = tensor.dim(2)?;
    let pad_len = target_len - current_len;

    let padding = Tensor::zeros((batch_size, pad_len, d_model), dtype, device)?;
    let padded = Tensor::cat(&[tensor, &padding], 1)?;

    Ok(padded)
}

/// Save video tensor to image frames
fn save_video_frames(video: &Tensor, output_dir: &PathBuf) -> Result<()> {
    use image::{ImageBuffer, Rgb};

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Video shape: (B, C, T, H, W)
    let video = video.to_dtype(DType::F32)?;
    let dims = video.dims();

    if dims.len() != 5 {
        anyhow::bail!("Expected 5D video tensor, got {:?}", dims);
    }

    let num_frames = dims[2];
    let height = dims[3];
    let width = dims[4];

    info!("Saving {} frames to {:?}...", num_frames, output_dir);

    for t in 0..num_frames {
        // Extract frame: (B, C, H, W) -> (C, H, W)
        let frame = video.i((0, .., t, .., ..))?;

        // Normalize to [0, 255]
        let frame = ((frame.clamp(-1.0, 1.0)? + 1.0)? * 127.5)?;
        let frame = frame.to_vec3::<f32>()?;

        // Create image
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);

        for (y, row) in frame[0].iter().enumerate().take(height) {
            for (x, _) in row.iter().enumerate().take(width) {
                let r = frame[0][y][x].round().clamp(0.0, 255.0) as u8;
                let g = frame[1][y][x].round().clamp(0.0, 255.0) as u8;
                let b = frame[2][y][x].round().clamp(0.0, 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        let frame_path = output_dir.join(format!("frame_{:04}.png", t));
        img.save(&frame_path)?;
    }

    info!("Frames saved to {:?}", output_dir);
    info!(
        "To create video, run: ffmpeg -framerate 8 -i {:?}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4",
        output_dir
    );

    Ok(())
}
