#!/usr/bin/env python3
"""
Generate reference VAE outputs for Rust comparison.
Loads LTX-Video VAE and runs decode with deterministic inputs.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add diffusers to path
sys.path.insert(0, "tp/diffusers/src")

def main():
    print("=" * 60)
    print("LTX-Video VAE Reference Generator")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = Path("output/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VAE model
    print("\n[1/4] Loading VAE model...")
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    
    model_path = "ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors"
    vae = AutoencoderKLLTXVideo.from_single_file(model_path, torch_dtype=torch.float32)
    vae.eval()
    
    # Print model config
    print(f"  latent_channels: {vae.config.latent_channels}")
    print(f"  spatial_compression_ratio: {vae.spatial_compression_ratio}")
    print(f"  temporal_compression_ratio: {vae.temporal_compression_ratio}")
    print(f"  decoder_block_out_channels: {vae.config.decoder_block_out_channels}")
    print(f"  decoder_layers_per_block: {vae.config.decoder_layers_per_block}")
    print(f"  decoder_spatio_temporal_scaling: {vae.config.decoder_spatio_temporal_scaling}")
    print(f"  timestep_conditioning: {vae.config.timestep_conditioning}")
    
    # Generate deterministic latent input
    # Small size for fast testing: (B, C, T, H, W)
    print("\n[2/4] Creating deterministic latent input...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    latent_channels = 128
    latent_t = 1  # 1 latent frame -> 8 video frames (temporal_compression=8)
    latent_h = 4  # 4 latent spatial -> 128 pixels (spatial_compression=32)
    latent_w = 6  # 6 latent spatial -> 192 pixels
    
    latents = torch.randn(batch_size, latent_channels, latent_t, latent_h, latent_w, dtype=torch.float32)
    
    print(f"  Latents shape: {latents.shape}")
    print(f"  Latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
    
    # Run VAE decode
    print("\n[3/4] Running VAE decode...")
    with torch.no_grad():
        # Check if timestep_conditioning is enabled
        temb = None
        if vae.config.timestep_conditioning:
            temb = torch.tensor([0.05], dtype=torch.float32)  # Small timestep
            print(f"  Using timestep: {temb.item()}")
        
        # Direct decode (no tiling)
        output = vae.decoder(latents, temb)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Expected output dimensions:
    # T_out = T_lat (1) * temporal_compression (8) = 8? Actually depends on patch_size_t
    # H_out = H_lat (4) * spatial_compression (32) = 128
    # W_out = W_lat (6) * spatial_compression (32) = 192
    
    # Save data for Rust
    print("\n[4/4] Saving reference data...")
    
    # Save as binary files
    def save_tensor(tensor, name):
        path = output_dir / f"vae_{name}.bin"
        arr = tensor.cpu().numpy().astype(np.float32)
        arr.tofile(path)
        print(f"  Saved {name}: {path} ({arr.shape})")
    
    save_tensor(latents, "latents")
    save_tensor(output, "output_ref")
    
    # Also save as .npy for easy inspection
    np.save(output_dir / "vae_latents.npy", latents.cpu().numpy())
    np.save(output_dir / "vae_output_ref.npy", output.cpu().numpy())
    
    print("\n" + "=" * 60)
    print("Reference generation complete!")
    print("=" * 60)
    
    # Print summary for Rust test
    print("\nFor Rust test:")
    print(f"  Input:  latents shape = [{batch_size}, {latent_channels}, {latent_t}, {latent_h}, {latent_w}]")
    print(f"  Output: output shape = {list(output.shape)}")

if __name__ == "__main__":
    main()
