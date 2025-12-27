#!/usr/bin/env python3
"""Decode latents using diffusers VAE and compare with Rust output."""

import numpy as np
import torch
from PIL import Image
import os
import sys

# Force use local diffusers
sys.path.insert(0, os.path.abspath('tp/diffusers/src'))
import diffusers
print(f"Using diffusers from: {diffusers.__file__}")

from diffusers import AutoencoderKLLTXVideo

def main():
    # Load latents saved by Rust (binary format with header)
    with open('output/latents.bin', 'rb') as f:
        import struct
        ndims = struct.unpack('<Q', f.read(8))[0]
        dims = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(ndims))
        num_elements = 1
        for d in dims:
            num_elements *= d
        data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
    
    latents = data.reshape(dims)
    print(f"Loaded latents: shape={latents.shape}, range=[{latents.min():.4f}, {latents.max():.4f}]")
    
    # Convert to torch
    latents_torch = torch.from_numpy(latents.copy()).float()
    
    # Load VAE from safetensors
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    print(f"Loading VAE from {model_path}...")
    
    # Use from_single_file
    vae = AutoencoderKLLTXVideo.from_single_file(
        model_path,
        torch_dtype=torch.float32,
    )
    vae.eval()
    
    print("VAE loaded!")
    print(f"VAE config: timestep_conditioning={getattr(vae.config, 'timestep_conditioning', None)}")
    
    # Decode with timestep
    timestep = torch.tensor([0.05])
    with torch.no_grad():
        # VAE decode expects (B, C, T, H, W)
        decoded = vae.decode(latents_torch, temb=timestep, return_dict=False)[0]
    
    print(f"Decoded shape: {decoded.shape}")
    print(f"Decoded range: [{decoded.min():.4f}, {decoded.max():.4f}]")
    
    # Save first frame
    video = decoded[0].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    frame = video[0]  # First frame
    frame = ((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(frame).save('output/python_vae_frame.png')
    print("Saved output/python_vae_frame.png")

if __name__ == '__main__':
    main()
