#!/usr/bin/env python3
"""
Dump tensor shapes from diffusers SVD pipeline for comparison with Rust implementation.
"""
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
import os


def main():
    model_id = "models/svd"
    
    print("=" * 60)
    print("SVD Tensor Shape Analysis")
    print("=" * 60)
    
    # Load pipeline
    print(f"\nLoading pipeline from {model_id}...")
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16",
            use_safetensors=True
        )
    except Exception:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        )
    
    device = "cuda"
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires GPU.")
        return
    pipe.to(device)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Parameters matching Rust
    height = 576
    width = 1024
    num_frames = 14
    num_inference_steps = 2  # Just 2 steps to analyze shapes
    
    # Load image
    image_path = "tp/generative-models/assets/test_image.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    image = load_image(image_path)
    image = image.resize((width, height))
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    print(f"\nParameters:")
    print(f"  height: {height}")
    print(f"  width: {width}")
    print(f"  num_frames: {num_frames}")
    print(f"  num_inference_steps: {num_inference_steps}")
    
    # === Analyze shapes at each stage ===
    print("\n" + "=" * 60)
    print("SHAPE ANALYSIS")
    print("=" * 60)
    
    # 1. Image preprocessing
    batch_size = 1
    num_videos_per_prompt = 1
    
    # CLIP image encoding
    print("\n[1] CLIP Image Encoding")
    dtype = next(pipe.image_encoder.parameters()).dtype
    clip_image = pipe.video_processor.pil_to_numpy(image)
    clip_image = pipe.video_processor.numpy_to_pt(clip_image)
    print(f"  image after numpy_to_pt: {clip_image.shape}")
    
    clip_image = clip_image * 2.0 - 1.0
    # Note: diffusers uses _resize_with_antialiasing internally
    clip_image_resized = torch.nn.functional.interpolate(clip_image, (224, 224), mode='bicubic', align_corners=True)
    clip_image_resized = (clip_image_resized + 1.0) / 2.0
    print(f"  clip_image_resized: {clip_image_resized.shape}")
    
    # CLIP encoding produces embeddings
    clip_image_for_encoder = pipe.feature_extractor(
        images=clip_image_resized,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values.to(device=device, dtype=dtype)
    print(f"  clip_input: {clip_image_for_encoder.shape}")
    
    image_embeddings = pipe.image_encoder(clip_image_for_encoder).image_embeds
    print(f"  image_embeddings (raw): {image_embeddings.shape}")
    
    image_embeddings = image_embeddings.unsqueeze(1)
    print(f"  image_embeddings (unsqueeze): {image_embeddings.shape}")
    
    # For CFG
    do_cfg = True
    if do_cfg:
        negative_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        print(f"  image_embeddings (with CFG): {image_embeddings.shape}")
    
    # 2. VAE encoding
    print("\n[2] VAE Image Latent Encoding")
    image_tensor = pipe.video_processor.preprocess(image, height=height, width=width).to(device, dtype=dtype)
    print(f"  image_tensor: {image_tensor.shape}")
    
    noise = torch.randn(image_tensor.shape, generator=generator, device=device, dtype=image_tensor.dtype)
    noise_aug_strength = 0.02
    image_augmented = image_tensor + noise_aug_strength * noise
    print(f"  image_augmented: {image_augmented.shape}")
    
    image_latents = pipe.vae.encode(image_augmented).latent_dist.mode()
    print(f"  image_latents: {image_latents.shape}")
    
    if do_cfg:
        negative_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([negative_image_latents, image_latents])
        print(f"  image_latents (with CFG): {image_latents.shape}")
    
    # Repeat for frames
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
    print(f"  image_latents (repeated for frames): {image_latents.shape}")
    
    # 3. Initial latents
    print("\n[3] Initial Noise Latents")
    vae_scale_factor = 8
    num_channels_latents = pipe.unet.config.in_channels
    latent_shape = (
        batch_size,
        num_frames,
        num_channels_latents // 2,  # 4 channels for noise
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    print(f"  latent_shape (diffusers): {latent_shape}")
    print(f"    -> batch_size: {batch_size}")
    print(f"    -> num_frames: {num_frames}")
    print(f"    -> channels: {num_channels_latents // 2}")
    print(f"    -> height: {height // vae_scale_factor}")
    print(f"    -> width: {width // vae_scale_factor}")
    
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    print(f"  latents: {latents.shape}")
    
    # Scale by init_noise_sigma
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # 4. Denoising loop shapes
    print("\n[4] Denoising Loop")
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    for i, t in enumerate(pipe.scheduler.timesteps[:2]):  # First 2 steps only
        print(f"\n  Step {i}, t={t}")
        
        # CFG expansion
        if do_cfg:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        print(f"    latent_model_input (after CFG cat): {latent_model_input.shape}")
        
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        print(f"    latent_model_input (after scale): {latent_model_input.shape}")
        
        # Concatenate image latents
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        print(f"    latent_model_input (with image_latents, dim=2): {latent_model_input.shape}")
        
        # Added time IDs shape
        fps = 6  # fps - 1
        motion_bucket_id = 127
        add_time_ids = torch.tensor([[fps, motion_bucket_id, noise_aug_strength]], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        if do_cfg:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        print(f"    added_time_ids: {add_time_ids.shape}")
        
        print(f"    encoder_hidden_states: {image_embeddings.shape}")
        
        # UNet forward - just show input shapes, don't actually run
        print(f"\n    UNet inputs:")
        print(f"      sample: {latent_model_input.shape}")
        print(f"      timestep: scalar {t}")
        print(f"      encoder_hidden_states: {image_embeddings.shape}")
        print(f"      added_time_ids: {add_time_ids.shape}")
        break  # Only analyze first step
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Key Shape Differences")
    print("=" * 60)
    print("""
Diffusers latent shape convention: [B, F, C, H, W]
  - B = batch_size (typically 1)
  - F = num_frames (14)
  - C = latent_channels (4 for noise, 8 with image conditioning)
  - H = height / 8
  - W = width / 8

Key observations:
1. Initial latents: [B, F, C, H, W] = [1, 14, 4, 72, 128]
2. After CFG cat: [2*B, F, C, H, W] = [2, 14, 4, 72, 128]
3. After concat with image_latents (dim=2): [2*B, F, 8, H, W] = [2, 14, 8, 72, 128]
4. image_embeddings: [2*B, 1, embed_dim] = [2, 1, 1024]
5. added_time_ids: [2*B, 3] = [2, 3]

NOTE: Rust implementation uses [B*F, C, H, W] which requires:
- Reshaping before/after UNet
- Different dim for concat operations
""")


if __name__ == "__main__":
    main()
