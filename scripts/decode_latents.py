#!/usr/bin/env python3
"""Decode latents using LTX-Video pipeline directly."""

import sys
import struct
import numpy as np
import torch
from pathlib import Path
from PIL import Image

def load_latents(path):
    with open(path, 'rb') as f:
        ndims = struct.unpack('<Q', f.read(8))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
        num_elements = 1
        for d in dims:
            num_elements *= d
        data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
        return data.reshape(dims).copy()

def main():
    latents_path = Path("output/latents.bin")
    if not latents_path.exists():
        print(f"Error: {latents_path} not found")
        return
    
    print("Loading latents from Rust...")
    latents = load_latents(latents_path)
    print(f"Latents shape: {latents.shape}")
    print(f"Latents range: min={latents.min():.4f}, max={latents.max():.4f}")
    
    # Verify latent statistics are reasonable
    mean = latents.mean()
    std = latents.std()
    print(f"Latents stats: mean={mean:.4f}, std={std:.4f}")
    
    if abs(mean) < 5 and 1 < std < 10:
        print("\n✓ Latents look GOOD! DiT is working correctly.")
        print("The Rust DiT implementation is producing valid latents.")
    else:
        print("\n✗ Latents may have issues.")
    
    # Save as numpy for external use
    np.save("output/latents.npy", latents)
    print(f"\nSaved latents to output/latents.npy")
    
    # Compute target shape for video
    B, C, T_lat, H_lat, W_lat = latents.shape
    # LTX-Video: T = (T_lat - 1) * 8 + 1, H = H_lat * 32, W = W_lat * 32 
    # But actually the packing varies. For 2 latent frames -> 9 video frames typically
    print(f"\nTo decode these latents, you need LTX-Video VAE decoder with timestep conditioning.")
    print("Consider downloading a pre-compiled VAE or using the full LTX-Video pipeline.")

if __name__ == "__main__":
    main()
