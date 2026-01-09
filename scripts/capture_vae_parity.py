#!/usr/bin/env python3
"""
Capture reference tensors for AutoencoderKLLTXVideo parity verification.

This script captures:
1. Decode outputs for various latent shapes
2. Timestep embedding values
3. Latent normalization/denormalization values
4. Intermediate layer outputs

Output: gen_vae_parity.safetensors

Requirements: 11.1 - Provide Python scripts to capture reference tensors at each pipeline stage
"""

import os
import sys
import math
import torch
from safetensors.torch import save_file
from typing import Optional

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from diffusers import AutoencoderKLLTXVideo


def capture_latent_normalization():
    """Capture latent normalization/denormalization values."""
    print("Capturing latent normalization values...")
    
    results = {}
    
    # Load VAE to get latents_mean and latents_std
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=torch.float32)
    
    # Get normalization parameters
    latents_mean = vae.latents_mean
    latents_std = vae.latents_std
    scaling_factor = vae.config.scaling_factor
    
    results["latents_mean"] = latents_mean.clone()
    results["latents_std"] = latents_std.clone()
    results["scaling_factor"] = torch.tensor([scaling_factor], dtype=torch.float32)
    
    print(f"  latents_mean shape: {latents_mean.shape}")
    print(f"  latents_std shape: {latents_std.shape}")
    print(f"  scaling_factor: {scaling_factor}")
    
    # Test normalization/denormalization round-trip
    torch.manual_seed(42)
    test_latents = torch.randn(1, 128, 2, 8, 8, dtype=torch.float32)
    
    # Normalize: (latents - mean) * scaling_factor / std
    # Reshape mean/std for broadcasting: [C] -> [1, C, 1, 1, 1]
    mean_broadcast = latents_mean.view(1, -1, 1, 1, 1)
    std_broadcast = latents_std.view(1, -1, 1, 1, 1)
    
    normalized = (test_latents - mean_broadcast) * scaling_factor / std_broadcast
    
    # Denormalize: latents * std / scaling_factor + mean
    denormalized = normalized * std_broadcast / scaling_factor + mean_broadcast
    
    results["test_latents"] = test_latents.clone()
    results["normalized_latents"] = normalized.clone()
    results["denormalized_latents"] = denormalized.clone()
    
    # Verify round-trip
    mse = ((test_latents - denormalized) ** 2).mean().item()
    print(f"  Round-trip MSE: {mse:.2e}")
    
    return results


def capture_timestep_embedding():
    """Capture timestep embedding values for VAE decoder."""
    print("\nCapturing timestep embedding values...")
    
    results = {}
    
    # Test various timestep values
    timesteps = [0.0, 0.05, 0.1, 0.5, 1.0]
    
    for t in timesteps:
        # Create timestep tensor as done in pipeline
        temb = torch.tensor([t], dtype=torch.float32)
        
        results[f"timestep_{t}"] = temb.clone()
        print(f"  t={t}: temb={temb.item()}")
    
    return results


def capture_vae_decode_small():
    """Capture VAE decode outputs with small latent shapes."""
    print("\nCapturing VAE decode (small shapes)...")
    
    results = {}
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    print(f"  Loading VAE from {model_path}...")
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    # Test various latent shapes
    test_shapes = [
        (1, 128, 2, 4, 4),    # Very small
        (1, 128, 2, 8, 8),    # Small
        (1, 128, 4, 8, 8),    # Medium frames
    ]
    
    for idx, shape in enumerate(test_shapes):
        torch.manual_seed(42 + idx)
        
        latents = torch.randn(shape, device=device, dtype=dtype)
        batch_size = shape[0]
        
        # Test with different timestep values
        for t in [0.0, 0.05]:
            temb = torch.tensor([t] * batch_size, device=device, dtype=dtype)
            
            with torch.no_grad():
                output = vae.decode(latents, temb=temb, return_dict=False)[0]
            
            key = f"decode_shape{idx}_t{t}"
            results[f"{key}_latents"] = latents.cpu().clone()
            results[f"{key}_temb"] = temb.cpu().clone()
            results[f"{key}_output"] = output.cpu().clone()
            
            print(f"  Shape {shape}, t={t}: output shape={output.shape}, "
                  f"mean={output.mean().item():.6f}")
    
    return results


def capture_vae_decode_with_hooks():
    """Capture VAE decode with intermediate layer outputs."""
    print("\nCapturing VAE decode with intermediate outputs...")
    
    results = {}
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    # Storage for intermediate outputs
    intermediate_outputs = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediate_outputs[name] = output[0].detach().cpu()
            else:
                intermediate_outputs[name] = output.detach().cpu()
        return hook
    
    # Register hooks on key layers
    hooks = []
    
    # Decoder layers
    if hasattr(vae.decoder, 'conv_in'):
        hooks.append(vae.decoder.conv_in.register_forward_hook(make_hook('decoder_conv_in')))
    if hasattr(vae.decoder, 'mid_block'):
        hooks.append(vae.decoder.mid_block.register_forward_hook(make_hook('decoder_mid_block')))
    if hasattr(vae.decoder, 'up_blocks'):
        for i, block in enumerate(vae.decoder.up_blocks):
            hooks.append(block.register_forward_hook(make_hook(f'decoder_up_block_{i}')))
    if hasattr(vae.decoder, 'conv_out'):
        hooks.append(vae.decoder.conv_out.register_forward_hook(make_hook('decoder_conv_out')))
    
    # Run decode
    torch.manual_seed(42)
    latents = torch.randn(1, 128, 2, 8, 8, device=device, dtype=dtype)
    temb = torch.tensor([0.05], device=device, dtype=dtype)
    
    with torch.no_grad():
        output = vae.decode(latents, temb=temb, return_dict=False)[0]
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Save results
    results["hooks_latents"] = latents.cpu().clone()
    results["hooks_temb"] = temb.cpu().clone()
    results["hooks_output"] = output.cpu().clone()
    
    for name, tensor in intermediate_outputs.items():
        results[f"hooks_{name}"] = tensor.clone()
        print(f"  {name}: shape={tensor.shape}")
    
    return results


def capture_sinusoidal_embedding():
    """Capture sinusoidal timestep embedding computation."""
    print("\nCapturing sinusoidal embedding computation...")
    
    results = {}
    
    # LTX-Video uses sinusoidal embedding for timestep conditioning
    # The formula is: [cos(t * freqs), sin(t * freqs)]
    # with flip_sin_to_cos=True and downscale_freq_shift=0
    
    embedding_dim = 256  # Typical embedding dimension
    
    # Compute frequencies
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    
    results["sinusoidal_freqs"] = freqs.clone()
    results["sinusoidal_embedding_dim"] = torch.tensor([embedding_dim], dtype=torch.int64)
    
    print(f"  embedding_dim={embedding_dim}, half_dim={half_dim}")
    print(f"  freqs shape: {freqs.shape}")
    print(f"  freqs range: [{freqs.min().item():.6f}, {freqs.max().item():.6f}]")
    
    # Test with various timestep values
    timesteps = [0.0, 0.05, 0.1, 0.5, 1.0]
    
    for t in timesteps:
        timestep = torch.tensor([t], dtype=torch.float32)
        
        # Compute embedding: [cos, sin] order (flip_sin_to_cos=True)
        args = timestep[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        results[f"sinusoidal_t{t}_embedding"] = embedding.clone()
        
        print(f"  t={t}: embedding shape={embedding.shape}, "
              f"mean={embedding.mean().item():.6f}")
    
    return results


def capture_denormalization_formula():
    """Capture exact denormalization formula verification."""
    print("\nCapturing denormalization formula verification...")
    
    results = {}
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=torch.float32)
    
    latents_mean = vae.latents_mean
    latents_std = vae.latents_std
    scaling_factor = vae.config.scaling_factor
    
    # Test various latent values
    torch.manual_seed(123)
    
    test_cases = [
        torch.zeros(1, 128, 2, 4, 4),
        torch.ones(1, 128, 2, 4, 4),
        torch.randn(1, 128, 2, 4, 4),
        torch.randn(1, 128, 2, 4, 4) * 2,  # Larger values
    ]
    
    for idx, latents in enumerate(test_cases):
        # Denormalize: latents * std / scaling_factor + mean
        mean_broadcast = latents_mean.view(1, -1, 1, 1, 1)
        std_broadcast = latents_std.view(1, -1, 1, 1, 1)
        
        denormalized = latents * std_broadcast / scaling_factor + mean_broadcast
        
        results[f"denorm_input_{idx}"] = latents.clone()
        results[f"denorm_output_{idx}"] = denormalized.clone()
        
        print(f"  Case {idx}: input mean={latents.mean().item():.6f}, "
              f"output mean={denormalized.mean().item():.6f}")
    
    return results


def capture_vae_config():
    """Capture VAE configuration values."""
    print("\nCapturing VAE configuration...")
    
    results = {}
    
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/vae"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=torch.float32)
    
    # Extract key config values
    config = vae.config
    
    results["config_in_channels"] = torch.tensor([config.in_channels], dtype=torch.int64)
    results["config_out_channels"] = torch.tensor([config.out_channels], dtype=torch.int64)
    results["config_latent_channels"] = torch.tensor([config.latent_channels], dtype=torch.int64)
    results["config_scaling_factor"] = torch.tensor([config.scaling_factor], dtype=torch.float32)
    results["config_spatial_compression"] = torch.tensor([vae.spatial_compression_ratio], dtype=torch.int64)
    results["config_temporal_compression"] = torch.tensor([vae.temporal_compression_ratio], dtype=torch.int64)
    
    print(f"  in_channels: {config.in_channels}")
    print(f"  out_channels: {config.out_channels}")
    print(f"  latent_channels: {config.latent_channels}")
    print(f"  scaling_factor: {config.scaling_factor}")
    print(f"  spatial_compression_ratio: {vae.spatial_compression_ratio}")
    print(f"  temporal_compression_ratio: {vae.temporal_compression_ratio}")
    
    return results


def main():
    print("=" * 60)
    print("Capturing AutoencoderKLLTXVideo reference data")
    print("=" * 60)
    
    all_results = {}
    
    # Capture all reference data
    all_results.update(capture_latent_normalization())
    all_results.update(capture_timestep_embedding())
    all_results.update(capture_vae_decode_small())
    all_results.update(capture_vae_decode_with_hooks())
    all_results.update(capture_sinusoidal_embedding())
    all_results.update(capture_denormalization_formula())
    all_results.update(capture_vae_config())
    
    # Save metadata
    metadata = {
        "description": "AutoencoderKLLTXVideo parity reference data",
        "diffusers_version": str(getattr(__import__('diffusers'), '__version__', 'unknown')),
        "torch_version": torch.__version__,
        "num_tensors": str(len(all_results)),
    }
    
    # Save to safetensors
    output_path = "gen_vae_parity.safetensors"
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
