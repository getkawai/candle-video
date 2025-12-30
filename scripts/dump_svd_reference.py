#!/usr/bin/env python3
"""
Dump reference tensors from diffusers SVD pipeline for comparison with Rust.
Saves tensors as .npy files in output/reference_tensors/
"""
import torch
import numpy as np
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
import os
from pathlib import Path


@torch.inference_mode()
def main():
    output_dir = Path("output/reference_tensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_id = "models/svd"
    
    print("=" * 60)
    print("Dumping SVD Reference Tensors")
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
    
    # Enable CPU offload to save VRAM on 12GB GPUs
    pipe.enable_model_cpu_offload()
    print(f"Device: {device} ({torch.cuda.get_device_name(0)}) with CPU offload")
    
    # Parameters matching Rust
    height = 576
    width = 1024
    num_frames = 14
    num_inference_steps = 2  # Just 2 steps for comparison
    batch_size = 1
    
    # Load image
    image_path = "tp/generative-models/assets/test_image.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    image = load_image(image_path)
    image = image.resize((width, height))
    
    # Use CPU generator with CPU offload
    generator = torch.Generator().manual_seed(42)
    
    print(f"\nParameters:")
    print(f"  height: {height}")
    print(f"  width: {width}")
    print(f"  num_frames: {num_frames}")
    print(f"  num_inference_steps: {num_inference_steps}")
    print(f"  seed: 42")
    
    dtype = next(pipe.image_encoder.parameters()).dtype
    
    # Use PyTorch native SDPA (flash attention) instead of xformers
    # xformers can fail on some configs, SDPA is safer on RTX 3060
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    print("Using PyTorch native SDPA")
    
    # === 1. CLIP Image Encoding ===
    print("\n[1] CLIP Image Encoding...")
    clip_image = pipe.video_processor.pil_to_numpy(image)
    clip_image = pipe.video_processor.numpy_to_pt(clip_image)
    
    # Resize for CLIP (224x224)
    clip_image = clip_image * 2.0 - 1.0
    clip_image_resized = torch.nn.functional.interpolate(
        clip_image, (224, 224), mode='bicubic', align_corners=True
    )
    clip_image_resized = (clip_image_resized + 1.0) / 2.0
    
    # CLIP encoding
    clip_input = pipe.feature_extractor(
        images=clip_image_resized,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values.to(device=device, dtype=dtype)
    
    np.save(output_dir / "clip_input.npy", clip_input.cpu().detach().float().numpy())
    
    image_embeddings = pipe.image_encoder(clip_input).image_embeds
    np.save(output_dir / "image_embeddings_raw.npy", image_embeddings.cpu().detach().float().numpy())
    
    image_embeddings = image_embeddings.unsqueeze(1)  # [B, 1, D]
    
    # For CFG
    do_cfg = True
    if do_cfg:
        negative_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
    
    np.save(output_dir / "image_embeddings_cfg.npy", image_embeddings.cpu().detach().float().numpy())
    print(f"  image_embeddings: {image_embeddings.shape}")
    
    # === 2. VAE Encoding ===
    print("\n[2] VAE Image Latent Encoding...")
    image_tensor = pipe.video_processor.preprocess(image, height=height, width=width).to(device, dtype=dtype)
    
    noise_aug_strength = 0.02
    noise = torch.randn(image_tensor.shape, generator=generator, dtype=image_tensor.dtype).to(device)
    image_augmented = image_tensor + noise_aug_strength * noise
    
    np.save(output_dir / "vae_input.npy", image_augmented.cpu().detach().float().numpy())
    
    image_latents = pipe.vae.encode(image_augmented).latent_dist.mode()
    np.save(output_dir / "image_latents_raw.npy", image_latents.cpu().detach().float().numpy())
    
    if do_cfg:
        negative_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([negative_image_latents, image_latents])
    
    # Repeat for frames
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
    np.save(output_dir / "image_latents_cfg.npy", image_latents.cpu().detach().float().numpy())
    print(f"  image_latents: {image_latents.shape}")
    
    # === 3. Initial Latents ===
    print("\n[3] Initial Noise Latents...")
    vae_scale_factor = 8
    num_channels_latents = pipe.unet.config.in_channels
    latent_shape = (
        batch_size,
        num_frames,
        num_channels_latents // 2,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    
    # Reset generator for reproducible latents
    generator = torch.Generator().manual_seed(42)
    latents = torch.randn(latent_shape, generator=generator, dtype=dtype).to(device)
    np.save(output_dir / "initial_latents.npy", latents.cpu().detach().float().numpy())
    print(f"  latents: {latents.shape}")
    
    # Scale by init_noise_sigma
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # === 4. Denoising Loop ===
    print("\n[4] Denoising Loop...")
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    fps = 6
    motion_bucket_id = 127
    add_time_ids = torch.tensor([[fps, motion_bucket_id, noise_aug_strength]], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])
    
    np.save(output_dir / "added_time_ids.npy", add_time_ids.cpu().detach().float().numpy())
    
    for i, t in enumerate(pipe.scheduler.timesteps[:num_inference_steps]):
        print(f"\n  Step {i}, t={t}")
        
        # CFG expansion
        if do_cfg:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Concatenate image latents
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        
        np.save(output_dir / f"step{i}_unet_input.npy", latent_model_input.cpu().detach().float().numpy())
        print(f"    UNet input: {latent_model_input.shape}")
        
        # UNet forward
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=add_time_ids,
            return_dict=False,
        )[0]
        
        np.save(output_dir / f"step{i}_noise_pred.npy", noise_pred.cpu().detach().float().numpy())
        print(f"    noise_pred: {noise_pred.shape}")
        
        # CFG combine
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guidance_scale = 3.0  # Default for SVD
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        np.save(output_dir / f"step{i}_noise_pred_cfg.npy", noise_pred.cpu().detach().float().numpy())
        
        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        np.save(output_dir / f"step{i}_latents_after.npy", latents.cpu().detach().float().numpy())
        print(f"    latents after: {latents.shape}")
    
    # === 5. Summary ===
    print("\n" + "=" * 60)
    print(f"Tensors saved to: {output_dir.absolute()}")
    print("=" * 60)
    
    # List saved files
    files = sorted(output_dir.glob("*.npy"))
    print(f"\nSaved {len(files)} tensor files:")
    for f in files:
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
