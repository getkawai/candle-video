use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Deserialize;
use tokenizers::Tokenizer as HfTokenizer;

use crate::models::ltx_video::{
    AutoencoderKLLtxVideo, LtxVideoTransformer3DModel,
    loader::WeightLoader,
    quantized_t5_encoder::QuantizedT5EncoderModel,
    scheduler::FlowMatchEulerDiscreteScheduler,
    t2v_pipeline::{LtxPipeline, LtxVideoProcessor, OutputType, TextEncoder, Tokenizer, VaeConfig},
    text_encoder::{T5EncoderConfig, T5TextEncoderWrapper},
};
use crate::utils::deterministic_rng::Pcg32;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn candle_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => std::ptr::null(),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn candle_binding_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn candle_healthcheck() -> i32 {
    1
}

#[derive(Debug, Deserialize)]
struct VideoGenerateConfig {
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_version")]
    ltxv_version: String,
    #[serde(default = "default_model_id")]
    model_id: String,
    #[serde(default)]
    local_weights: Option<String>,
    #[serde(default)]
    unified_weights: Option<String>,
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_height")]
    height: usize,
    #[serde(default = "default_width")]
    width: usize,
    #[serde(default = "default_num_frames")]
    num_frames: usize,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    cpu: bool,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    gif: bool,
    #[serde(default)]
    frames: bool,
    #[serde(default)]
    vae_tiling: bool,
    #[serde(default)]
    vae_slicing: bool,
    #[serde(default)]
    use_bf16_t5: bool,
}

fn default_version() -> String {
    "0.9.8-2b-distilled".to_string()
}
fn default_model_id() -> String {
    "oxide-lab/LTX-Video-0.9.8-2B-distilled".to_string()
}
fn default_output_dir() -> String {
    "output".to_string()
}
fn default_height() -> usize {
    512
}
fn default_width() -> usize {
    768
}
fn default_num_frames() -> usize {
    97
}

struct TokenizerAdapter {
    tokenizer: HfTokenizer,
    device: Device,
}

impl Tokenizer for TokenizerAdapter {
    fn encode_batch(
        &self,
        prompts: &[String],
        max_length: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let encodings = self
            .tokenizer
            .encode_batch(prompts.to_vec(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut ids_vec = Vec::new();
        let mut mask_vec = Vec::new();
        for e in encodings {
            let mut ids = e.get_ids().to_vec();
            let mut mask = e.get_attention_mask().to_vec();
            if ids.len() > max_length {
                ids.truncate(max_length);
                mask.truncate(max_length);
            } else {
                ids.resize(max_length, 0);
                mask.resize(max_length, 0);
            }
            ids_vec.push(ids);
            mask_vec.push(mask);
        }
        let b = ids_vec.len();
        let flat_ids: Vec<u32> = ids_vec.into_iter().flatten().collect();
        let flat_mask: Vec<u32> = mask_vec.into_iter().flatten().collect();
        let ids_t = Tensor::new(flat_ids, &self.device)?.reshape((b, max_length))?;
        let mask_t = Tensor::new(flat_mask, &self.device)?.reshape((b, max_length))?;
        Ok((ids_t, mask_t))
    }

    fn model_max_length(&self) -> usize {
        128
    }
}

struct DummyTextEncoder;
impl TextEncoder for DummyTextEncoder {
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn forward(&mut self, _: &Tensor) -> candle_core::Result<Tensor> {
        candle_core::bail!("DummyTextEncoder: forward should not be called")
    }
}

struct DummyTokenizer;
impl Tokenizer for DummyTokenizer {
    fn encode_batch(&self, _: &[String], _: usize) -> candle_core::Result<(Tensor, Tensor)> {
        candle_core::bail!("DummyTokenizer: encode_batch should not be called")
    }

    fn model_max_length(&self) -> usize {
        128
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn candle_video_generate(config_json: *const c_char) -> i32 {
    if config_json.is_null() {
        set_last_error("config_json is null".to_string());
        return -1;
    }

    let run = || -> anyhow::Result<()> {
        let c_str = unsafe { CStr::from_ptr(config_json) };
        let json_str = c_str.to_str().context("config_json is not valid UTF-8")?;
        let cfg: VideoGenerateConfig =
            serde_json::from_str(json_str).context("invalid JSON config")?;
        run_generation(cfg)
    };

    match run() {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(format!("{err:#}"));
            -1
        }
    }
}

fn run_generation(cfg: VideoGenerateConfig) -> anyhow::Result<()> {
    if cfg.height % 32 != 0 || cfg.width % 32 != 0 {
        anyhow::bail!("height and width must be divisible by 32")
    }

    let ltxv_config = crate::models::ltx_video::configs::get_config_by_version(&cfg.ltxv_version);
    let num_inference_steps = cfg
        .steps
        .unwrap_or(ltxv_config.inference.num_inference_steps);
    let guidance_scale = cfg
        .guidance_scale
        .unwrap_or(ltxv_config.inference.guidance_scale);
    let rescaling_scale = ltxv_config.inference.rescaling_scale;
    let stochastic_sampling = ltxv_config.inference.stochastic_sampling;
    let stg_scale = ltxv_config.inference.stg_scale;

    let device = if cfg.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };

    let seed = cfg.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        now.as_secs() ^ (now.subsec_nanos() as u64)
    });
    // Some backends (notably CPU) do not support explicit seed configuration.
    let _ = device.set_seed(seed);

    let dtype = DType::BF16;

    let (transformer_file, vae_file, t5_file, tokenizer_file) =
        if let Some(local_path) = &cfg.local_weights {
            let base = PathBuf::from(local_path);

            let transformer = if cfg.unified_weights.is_some() {
                PathBuf::from("dummy")
            } else if base
                .join("transformer/diffusion_pytorch_model.safetensors")
                .exists()
            {
                base.join("transformer/diffusion_pytorch_model.safetensors")
            } else if base.join("ltx_video_transformer_3d.safetensors").exists() {
                base.join("ltx_video_transformer_3d.safetensors")
            } else {
                anyhow::bail!("Transformer weights not found in {:?}", base);
            };

            let vae = if cfg.unified_weights.is_some() {
                PathBuf::from("dummy")
            } else if base
                .join("vae/diffusion_pytorch_model.safetensors")
                .exists()
            {
                base.join("vae/diffusion_pytorch_model.safetensors")
            } else if base.join("vae.safetensors").exists() {
                base.join("vae.safetensors")
            } else {
                anyhow::bail!("VAE weights not found in {:?}", base);
            };

            let t5 = if cfg.use_bf16_t5 {
                let bf16_path = base.join("text_encoder").join("model.safetensors");
                if bf16_path.exists() {
                    bf16_path
                } else {
                    anyhow::bail!("BF16 T5 not found at {:?}", bf16_path);
                }
            } else if base.join("t5-v1_1-xxl-encoder-Q8_0.gguf").exists() {
                base.join("t5-v1_1-xxl-encoder-Q8_0.gguf")
            } else if base
                .join("text_encoder_gguf")
                .join("t5-v1_1-xxl-encoder-Q8_0.gguf")
                .exists()
            {
                base.join("text_encoder_gguf")
                    .join("t5-v1_1-xxl-encoder-Q8_0.gguf")
            } else if base.join("t5-xxl.gguf").exists() {
                base.join("t5-xxl.gguf")
            } else if base
                .join("text_encoder_gguf")
                .join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
                .exists()
            {
                base.join("text_encoder_gguf")
                    .join("t5-v1_1-xxl-encoder-Q5_K_M.gguf")
            } else {
                anyhow::bail!("T5 GGUF not found in {:?}", base);
            };

            let tokenizer = {
                let path1 = base.join("tokenizer").join("tokenizer.json");
                let path2 = base.join("text_encoder").join("tokenizer.json");
                let path3 = base.join("text_encoder_8bit").join("tokenizer.json");
                let path4 = base.join("text_encoder_gguf").join("tokenizer.json");
                if path1.exists() {
                    path1
                } else if path2.exists() {
                    path2
                } else if path3.exists() {
                    path3
                } else if path4.exists() {
                    path4
                } else {
                    Api::new()?
                        .repo(Repo::new(
                            "google-t5/t5-v1_1-xxl".to_string(),
                            RepoType::Model,
                        ))
                        .get("tokenizer.json")?
                }
            };

            (transformer, vae, t5, tokenizer)
        } else {
            let api = Api::new()?;
            let repo = api.repo(Repo::with_revision(
                cfg.model_id.clone(),
                RepoType::Model,
                "main".into(),
            ));

            let is_098 = cfg.ltxv_version.contains("0.9.8");
            let (transformer, vae) = if is_098 {
                let unified = repo.get("ltxv-2b-0.9.8-distilled.safetensors")?;
                (unified.clone(), unified)
            } else {
                let transformer = repo.get("transformer/diffusion_pytorch_model.safetensors")?;
                let vae = repo.get("vae/diffusion_pytorch_model.safetensors")?;
                (transformer, vae)
            };

            let t5 = repo.get("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf")?;
            let tokenizer = repo.get("text_encoder_gguf/tokenizer.json")?;

            (transformer, vae, t5, tokenizer)
        };

    let effective_unified_weights = cfg.unified_weights.clone().or_else(|| {
        if cfg.ltxv_version.contains("0.9.8") && cfg.local_weights.is_none() {
            Some(transformer_file.to_string_lossy().to_string())
        } else {
            None
        }
    });

    let tokenizer_hf = HfTokenizer::from_file(&tokenizer_file).map_err(|e| anyhow::anyhow!(e))?;
    let tokenizer_adapter = TokenizerAdapter {
        tokenizer: tokenizer_hf,
        device: device.clone(),
    };

    let (p_ids, p_mask) = tokenizer_adapter.encode_batch(std::slice::from_ref(&cfg.prompt), 128)?;
    let (n_ids, n_mask) =
        tokenizer_adapter.encode_batch(std::slice::from_ref(&cfg.negative_prompt), 128)?;

    let (prompt_embeds, negative_prompt_embeds) = if cfg.use_bf16_t5 {
        let config = T5EncoderConfig::t5_xxl();
        let mut t5_wrapper = T5TextEncoderWrapper::new(config, device.clone(), DType::BF16)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&t5_file], DType::BF16, &device)? };
        t5_wrapper.load_model(vb)?;

        let p_emb = t5_wrapper.forward(&p_ids)?;
        let n_emb = t5_wrapper.forward(&n_ids)?;
        (p_emb, n_emb)
    } else {
        let t5_model = QuantizedT5EncoderModel::load(&t5_file, &device)?;
        let p_emb = t5_model.forward(&p_ids, Some(&p_mask))?;
        let n_emb = t5_model.forward(&n_ids, Some(&n_mask))?;
        (p_emb, n_emb)
    };

    let prompt_attention_mask = p_mask.to_dtype(DType::F32)?;
    let negative_prompt_attention_mask = n_mask.to_dtype(DType::F32)?;

    let (vae, transformer) = if let Some(ref unified_path) = effective_unified_weights {
        use crate::models::ltx_video::weight_format::KeyRemapper;

        let unified_file = PathBuf::from(unified_path);
        if !unified_file.exists() {
            anyhow::bail!("Unified weights file not found: {:?}", unified_file);
        }

        let all_tensors = candle_core::safetensors::load(&unified_file, &device)?;
        let remapper = KeyRemapper::new();

        let mut vae_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        let mut trans_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (key, tensor) in all_tensors {
            let remapped_key = remapper.remap_key(&key);
            if KeyRemapper::is_vae_key(&key) {
                let clean_key = remapped_key
                    .strip_prefix("vae.")
                    .unwrap_or(&remapped_key)
                    .to_string();
                vae_tensors.insert(clean_key, tensor);
            } else if KeyRemapper::is_transformer_key(&key) {
                let clean_key = remapped_key
                    .strip_prefix("model.diffusion_model.")
                    .or_else(|| remapped_key.strip_prefix("transformer."))
                    .unwrap_or(&remapped_key)
                    .to_string();
                trans_tensors.insert(clean_key, tensor);
            }
        }

        let vae_vb = VarBuilder::from_tensors(vae_tensors, dtype, &device);
        let trans_vb = VarBuilder::from_tensors(trans_tensors, dtype, &device);

        let mut vae_config = ltxv_config.vae.clone();
        vae_config.timestep_conditioning = true;
        let mut vae = AutoencoderKLLtxVideo::new(vae_config, vae_vb)?;
        vae.use_tiling = cfg.vae_tiling;
        vae.use_slicing = cfg.vae_slicing;
        vae.use_framewise_decoding = cfg.vae_tiling && cfg.num_frames > 16;

        let trans_config = ltxv_config.transformer.clone();
        let transformer = LtxVideoTransformer3DModel::new(&trans_config, trans_vb)?;

        (vae, transformer)
    } else {
        let vae_load = WeightLoader::new(device.clone(), dtype);
        let vae_vb = vae_load.load_single(&vae_file)?;
        let vae_dir = vae_file.parent().unwrap_or(Path::new("."));
        let mut vae_config = if vae_dir.join("config.json").exists() {
            let f = std::fs::File::open(vae_dir.join("config.json"))?;
            serde_json::from_reader(f)?
        } else {
            ltxv_config.vae.clone()
        };
        vae_config.timestep_conditioning = true;
        let mut vae = AutoencoderKLLtxVideo::new(vae_config, vae_vb)?;
        vae.use_tiling = cfg.vae_tiling;
        vae.use_slicing = cfg.vae_slicing;
        vae.use_framewise_decoding = cfg.vae_tiling && cfg.num_frames > 16;

        let trans_load = WeightLoader::new(device.clone(), dtype);
        let trans_vb = trans_load.load_single(&transformer_file)?;
        let trans_config = ltxv_config.transformer.clone();
        let transformer = LtxVideoTransformer3DModel::new(&trans_config, trans_vb)?;

        (vae, transformer)
    };

    let mut scheduler_config = ltxv_config.scheduler.clone();
    scheduler_config.stochastic_sampling = stochastic_sampling;
    let scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config)?;

    let mut pipeline = LtxPipeline::new(
        Box::new(scheduler),
        Box::new(vae),
        Box::new(DummyTextEncoder),
        Box::new(DummyTokenizer),
        Box::new(transformer),
        Box::new(LtxVideoProcessor::new(VaeConfig {
            scaling_factor: 1.0,
            timestep_conditioning: true,
        })),
    );

    let mut rng = Pcg32::new(seed, 1442695040888963407);

    let latent_height = cfg.height / 32;
    let latent_width = cfg.width / 32;
    let latent_frames = (cfg.num_frames - 1) / 8 + 1;
    let channels = 128;
    let shape = (1, channels, latent_frames, latent_height, latent_width);
    let latents_unpacked = rng.randn(shape, &device)?;
    let latents_packed = LtxPipeline::pack_latents(&latents_unpacked, 1, 1)?;

    pipeline.guidance_rescale = rescaling_scale;

    let sigmas_from_config = ltxv_config.inference.timesteps.clone();
    let decode_timestep = ltxv_config
        .inference
        .decode_timestep
        .clone()
        .unwrap_or(vec![0.0]);
    let decode_noise_scale = ltxv_config.inference.decode_noise_scale.clone();

    let video_out = pipeline.call(
        None,
        None,
        cfg.height,
        cfg.width,
        cfg.num_frames,
        25,
        num_inference_steps,
        None,
        sigmas_from_config,
        guidance_scale,
        rescaling_scale,
        stg_scale,
        1,
        Some(latents_packed),
        Some(prompt_embeds),
        Some(prompt_attention_mask),
        Some(negative_prompt_embeds),
        Some(negative_prompt_attention_mask),
        decode_timestep,
        decode_noise_scale,
        OutputType::Tensor,
        128,
        Some(ltxv_config.inference.skip_block_list.clone()),
        &device,
    )?;

    if !Path::new(&cfg.output_dir).exists() {
        std::fs::create_dir_all(&cfg.output_dir)?;
    }

    let (b, _c, f, h, w) = video_out.frames.dims5()?;

    let mut frame_data = Vec::new();
    for i in 0..b {
        for j in 0..f {
            let frame = video_out.frames.i((i, .., j, .., ..))?;
            let frame = frame
                .permute((1, 2, 0))?
                .clamp(0.0, 255.0)?
                .to_dtype(DType::U8)?;
            let data = frame.flatten_all()?.to_vec1::<u8>()?;
            frame_data.push(data);
        }
    }

    if cfg.frames {
        for (j, data) in frame_data.iter().enumerate() {
            let filename = format!("{}/frame_{:04}.png", cfg.output_dir, j);
            image::save_buffer(&filename, data, w as u32, h as u32, image::ColorType::Rgb8)?;
        }
    }

    if cfg.gif || !cfg.frames {
        use gif::{Encoder, Frame, Repeat};
        use std::fs::File;

        let gif_path = format!("{}/video.gif", cfg.output_dir);
        let mut image_file = File::create(&gif_path)?;
        let mut encoder = Encoder::new(&mut image_file, w as u16, h as u16, &[])?;
        encoder.set_repeat(Repeat::Infinite)?;

        for data in &frame_data {
            let mut frame = Frame::from_rgb_speed(w as u16, h as u16, data, 30);
            frame.delay = 4;
            encoder.write_frame(&frame)?;
        }
    }

    Ok(())
}
