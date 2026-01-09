#!/usr/bin/env python3
"""
Capture reference tensors for full LTX-Video pipeline parity verification.

This script captures:
1. Latents at each denoising step
2. Video coordinates for RoPE
3. CFG intermediate results
4. Final decoded video

Output: gen_pipeline_parity.safetensors

Requirements: 11.1 - Provide Python scripts to capture reference tensors at each pipeline stage
"""

import os
import sys
import math
import numpy as np
import torch
from safetensors.torch import save_file
from typing import Optional, List, Tuple

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate mu for dynamic shifting (matches diffusers implementation)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def capture_video_coords():
    """Capture video coordinates computation for RoPE."""
    print("Capturing video coordinates...")
    
    results = {}
    
    # Test various video dimensions
    test_configs = [
        # (num_frames, height, width, frame_rate)
        (9, 256, 256, 25),    # Small
        (25, 512, 768, 25),   # Medium
        (97, 512, 768, 25),   # Large (typical LTX)
    ]
    
    vae_temporal_compression = 8
    vae_spatial_compression = 32
    
    for idx, (num_frames, height, width, frame_rate) in enumerate(test_configs):
        # Calculate latent dimensions
        latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression
        
        # Create video coordinates grid
        grid_f = torch.arange(latent_num_frames, dtype=torch.float32)
        grid_h = torch.arange(latent_height, dtype=torch.float32)
        grid_w = torch.arange(latent_width, dtype=torch.float32)
        
        # Meshgrid
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        video_coords = torch.stack(grid, dim=0)  # [3, F, H, W]
        video_coords = video_coords.unsqueeze(0)  # [1, 3, F, H, W]
        
        # Flatten to sequence
        video_coords_flat = video_coords.flatten(2, 4)  # [1, 3, seq_len]
        
        # Apply scaling as done in pipeline
        # rope_interpolation_scale = (vae_temporal_compression / frame_rate, vae_spatial_compression, vae_spatial_compression)
        rope_scale_t = vae_temporal_compression / frame_rate
        rope_scale_h = vae_spatial_compression
        rope_scale_w = vae_spatial_compression
        
        # The transformer's RoPE module applies:
        # grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * patch_size_t / base_num_frames
        # grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * patch_size / base_height
        # grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * patch_size / base_width
        
        # With patch_size=1, patch_size_t=1, base_num_frames=20, base_height=2048, base_width=2048
        base_num_frames = 20
        base_height = 2048
        base_width = 2048
        
        video_coords_scaled = video_coords_flat.clone()
        video_coords_scaled[:, 0] = video_coords_flat[:, 0] * rope_scale_t * 1 / base_num_frames
        video_coords_scaled[:, 1] = video_coords_flat[:, 1] * rope_scale_h * 1 / base_height
        video_coords_scaled[:, 2] = video_coords_flat[:, 2] * rope_scale_w * 1 / base_width
        
        key = f"coords_f{num_frames}_h{height}_w{width}"
        results[f"{key}_raw"] = video_coords_flat.clone()
        results[f"{key}_scaled"] = video_coords_scaled.clone()
        results[f"{key}_latent_dims"] = torch.tensor([latent_num_frames, latent_height, latent_width], dtype=torch.int64)
        results[f"{key}_rope_scale"] = torch.tensor([rope_scale_t, rope_scale_h, rope_scale_w], dtype=torch.float32)
        
        print(f"  Config {idx}: frames={num_frames}, h={height}, w={width}")
        print(f"    latent dims: f={latent_num_frames}, h={latent_height}, w={latent_width}")
        print(f"    rope_scale: {(rope_scale_t, rope_scale_h, rope_scale_w)}")
    
    return results


def capture_cfg_computation():
    """Capture CFG computation for various guidance scales."""
    print("\nCapturing CFG computation...")
    
    results = {}
    
    # Test various guidance scales
    guidance_scales = [1.0, 2.0, 3.0, 5.0, 7.5]
    
    torch.manual_seed(42)
    
    # Create test noise predictions
    shape = (1, 1024, 128)  # [batch, seq_len, channels]
    noise_pred_uncond = torch.randn(shape, dtype=torch.float32)
    noise_pred_text = torch.randn(shape, dtype=torch.float32)
    
    results["cfg_noise_pred_uncond"] = noise_pred_uncond.clone()
    results["cfg_noise_pred_text"] = noise_pred_text.clone()
    
    for gs in guidance_scales:
        # CFG formula: noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        noise_pred = noise_pred_uncond + gs * (noise_pred_text - noise_pred_uncond)
        
        results[f"cfg_output_gs{gs}"] = noise_pred.clone()
        
        print(f"  guidance_scale={gs}: output mean={noise_pred.mean().item():.6f}")
    
    return results


def capture_cfg_rescale():
    """Capture CFG rescale computation."""
    print("\nCapturing CFG rescale computation...")
    
    results = {}
    
    torch.manual_seed(42)
    
    shape = (1, 1024, 128)
    noise_pred_text = torch.randn(shape, dtype=torch.float32)
    noise_pred = torch.randn(shape, dtype=torch.float32) * 2  # Different from text
    
    results["rescale_noise_pred"] = noise_pred.clone()
    results["rescale_noise_pred_text"] = noise_pred_text.clone()
    
    # Test various rescale values
    rescale_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for rescale in rescale_values:
        if rescale > 0:
            # rescale_noise_cfg formula from diffusers
            std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
            std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
            
            # Rescale
            noise_pred_rescaled = noise_pred * (std_text / std_cfg)
            noise_pred_rescaled = rescale * noise_pred_rescaled + (1 - rescale) * noise_pred
        else:
            noise_pred_rescaled = noise_pred.clone()
        
        results[f"rescale_output_r{rescale}"] = noise_pred_rescaled.clone()
        
        print(f"  rescale={rescale}: output mean={noise_pred_rescaled.mean().item():.6f}")
    
    return results


def capture_mu_calculation():
    """Capture mu calculation for various sequence lengths."""
    print("\nCapturing mu calculation...")
    
    results = {}
    
    # Test various video configurations
    test_configs = [
        # (num_frames, height, width)
        (9, 256, 256),
        (25, 512, 768),
        (97, 512, 768),
        (161, 512, 768),
    ]
    
    vae_temporal_compression = 8
    vae_spatial_compression = 32
    
    # Default scheduler config values
    base_seq_len = 256
    max_seq_len = 4096
    base_shift = 0.5
    max_shift = 1.15
    
    for num_frames, height, width in test_configs:
        latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression
        
        video_seq_len = latent_num_frames * latent_height * latent_width
        
        mu = calculate_shift(video_seq_len, base_seq_len, max_seq_len, base_shift, max_shift)
        
        key = f"mu_f{num_frames}_h{height}_w{width}"
        results[f"{key}_seq_len"] = torch.tensor([video_seq_len], dtype=torch.int64)
        results[f"{key}_mu"] = torch.tensor([mu], dtype=torch.float32)
        
        print(f"  frames={num_frames}, h={height}, w={width}: seq_len={video_seq_len}, mu={mu:.6f}")
    
    return results


def capture_latent_packing():
    """Capture latent packing/unpacking operations."""
    print("\nCapturing latent packing/unpacking...")
    
    results = {}
    
    # Test various latent shapes
    test_shapes = [
        (1, 128, 2, 8, 8),
        (1, 128, 4, 16, 24),
        (1, 128, 13, 16, 24),
    ]
    
    patch_size = 1
    patch_size_t = 1
    
    for idx, shape in enumerate(test_shapes):
        torch.manual_seed(42 + idx)
        
        latents = torch.randn(shape, dtype=torch.float32)
        
        # Pack latents (as done in pipeline)
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        
        packed = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        packed = packed.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        
        # Unpack latents
        unpacked = packed.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        unpacked = unpacked.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        results[f"pack_input_{idx}"] = latents.clone().contiguous()
        results[f"pack_output_{idx}"] = packed.clone().contiguous()
        results[f"unpack_output_{idx}"] = unpacked.clone().contiguous()
        
        # Verify round-trip
        mse = ((latents - unpacked) ** 2).mean().item()
        print(f"  Shape {shape}: packed shape={packed.shape}, round-trip MSE={mse:.2e}")
    
    return results


def capture_denoising_loop():
    """Capture latents at each step of the denoising loop."""
    print("\nCapturing denoising loop (requires model)...")
    
    results = {}
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use F32 for max precision
    
    print(f"  Loading pipeline from {model_path}...")
    
    # Load components separately for more control
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"), torch_dtype=dtype
    ).to(device)
    transformer.eval()
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(model_path, "scheduler")
    )
    
    # Use small dimensions for speed
    num_frames = 9
    height = 128
    width = 128
    num_inference_steps = 10
    guidance_scale = 3.0
    seed = 42
    
    vae_temporal_compression = 8
    vae_spatial_compression = 32
    
    latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
    latent_height = height // vae_spatial_compression
    latent_width = width // vae_spatial_compression
    video_seq_len = latent_num_frames * latent_height * latent_width
    
    # Calculate mu
    mu = calculate_shift(video_seq_len)
    
    # Set timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas.tolist(), mu=mu)
    timesteps = scheduler.timesteps
    
    results["loop_timesteps"] = timesteps.cpu().clone()
    results["loop_sigmas"] = scheduler.sigmas.cpu().clone()
    results["loop_mu"] = torch.tensor([mu], dtype=torch.float32)
    
    # Create initial latents
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    latents_shape = (1, 128, latent_num_frames, latent_height, latent_width)
    latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
    
    # Pack latents
    latents = latents.reshape(1, -1, latent_num_frames, 1, latent_height, 1, latent_width, 1)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    
    results["loop_initial_latents"] = latents.cpu().clone().contiguous()
    
    # Create dummy prompt embeddings
    torch.manual_seed(seed + 1)
    prompt_embeds = torch.randn(2, 128, 4096, device=device, dtype=dtype)  # [uncond, cond]
    prompt_attention_mask = torch.ones(2, 128, device=device, dtype=dtype)
    
    results["loop_prompt_embeds"] = prompt_embeds.cpu().clone()
    results["loop_prompt_attention_mask"] = prompt_attention_mask.cpu().clone()
    
    # RoPE interpolation scale
    frame_rate = 25
    rope_interpolation_scale = (
        vae_temporal_compression / frame_rate,
        vae_spatial_compression,
        vae_spatial_compression,
    )
    
    print(f"  Running {num_inference_steps} denoising steps...")
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        timestep = t.expand(latent_model_input.shape[0])
        
        # Transformer forward
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=prompt_attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                rope_interpolation_scale=rope_interpolation_scale,
                return_dict=False,
            )[0]
        
        noise_pred = noise_pred.float()
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Save intermediate results
        results[f"loop_step{i}_noise_pred_uncond"] = noise_pred_uncond.cpu().clone()
        results[f"loop_step{i}_noise_pred_text"] = noise_pred_text.cpu().clone()
        results[f"loop_step{i}_noise_pred_cfg"] = noise_pred.cpu().clone()
        
        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        results[f"loop_step{i}_latents"] = latents.cpu().clone()
        
        print(f"    Step {i}: t={t.item():.2f}, latents mean={latents.mean().item():.6f}")
    
    results["loop_final_latents"] = latents.cpu().clone()
    
    return results


def capture_full_pipeline_output():
    """Capture full pipeline output including decoded video."""
    print("\nCapturing full pipeline output (requires model)...")
    
    results = {}
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"  Loading full pipeline from {model_path}...")
    pipeline = LTXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipeline.to(device)
    
    # Small test configuration
    prompt = "A calm river flowing through a forest"
    num_frames = 9
    height = 128
    width = 128
    num_inference_steps = 5
    guidance_scale = 3.0
    seed = 42
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"  Running pipeline with {num_inference_steps} steps...")
    
    # Get latent output
    output_latent = pipeline(
        prompt=prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
    ).frames
    
    results["full_output_latent"] = output_latent.cpu().to(torch.float32).clone()
    
    # Get decoded video output
    generator = torch.Generator(device=device).manual_seed(seed)
    
    output_video = pipeline(
        prompt=prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
    ).frames
    
    results["full_output_video"] = output_video.cpu().to(torch.float32).clone()
    
    print(f"  Latent output shape: {output_latent.shape}")
    print(f"  Video output shape: {output_video.shape}")
    
    return results


def main():
    print("=" * 60)
    print("Capturing LTX-Video Pipeline reference data")
    print("=" * 60)
    
    all_results = {}
    
    # Capture all reference data
    all_results.update(capture_video_coords())
    all_results.update(capture_cfg_computation())
    all_results.update(capture_cfg_rescale())
    all_results.update(capture_mu_calculation())
    all_results.update(capture_latent_packing())
    all_results.update(capture_denoising_loop())
    all_results.update(capture_full_pipeline_output())
    
    # Ensure all tensors are contiguous
    for key in all_results:
        if isinstance(all_results[key], torch.Tensor):
            all_results[key] = all_results[key].contiguous()
    
    # Save metadata
    metadata = {
        "description": "LTX-Video Pipeline parity reference data",
        "diffusers_version": str(getattr(__import__('diffusers'), '__version__', 'unknown')),
        "torch_version": torch.__version__,
        "num_tensors": str(len(all_results)),
    }
    
    # Save to safetensors
    output_path = "gen_pipeline_parity.safetensors"
    save_file(all_results, output_path, metadata=metadata)
    
    print("\n" + "=" * 60)
    print(f"Saved {len(all_results)} tensors to {output_path}")
    print("=" * 60)
    
    # Print summary
    print("\nTensor summary:")
    for key in sorted(all_results.keys())[:20]:
        tensor = all_results[key]
        print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
    if len(all_results) > 20:
        print(f"  ... and {len(all_results) - 20} more tensors")


if __name__ == "__main__":
    main()
