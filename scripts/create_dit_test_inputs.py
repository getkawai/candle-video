#!/usr/bin/env python3
"""Create deterministic inputs for Rust DiT test."""

import numpy as np
import torch
import os

def main():
    print("=== Creating Deterministic Inputs for DiT Comparison ===\n")
    
    # Create deterministic inputs (same seed as compare_dit_forward.py)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Input dimensions
    B, C, T, H, W = 1, 128, 1, 16, 24
    
    # Random latent input
    latents = torch.randn(B, C, T, H, W).numpy().astype(np.float32)
    
    # Random text embeddings  
    text_emb = torch.randn(B, 6, 4096).numpy().astype(np.float32)
    
    # Timestep = 1.0
    timestep = np.array([1.0], dtype=np.float32)
    
    print(f"Latents: {latents.shape}, range [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"Text embeddings: {text_emb.shape}")
    print(f"Timestep: {timestep}")
    
    # Save as binary files for Rust
    os.makedirs('output', exist_ok=True)
    
    # Save with simple format: just raw f32 bytes
    latents.tofile('output/dit_test_latents.bin')
    text_emb.tofile('output/dit_test_text_emb.bin')
    timestep.tofile('output/dit_test_timestep.bin')
    
    print(f"\nSaved binary files to output/")
    print(f"  dit_test_latents.bin: {latents.size * 4} bytes")
    print(f"  dit_test_text_emb.bin: {text_emb.size * 4} bytes")
    print(f"  dit_test_timestep.bin: {timestep.size * 4} bytes")
    
    # Also save expected output from reference
    ref_output = np.load('output/dit_output_ref.npy')
    print(f"\nReference output: {ref_output.shape}, range [{ref_output.min():.4f}, {ref_output.max():.4f}]")

if __name__ == '__main__':
    main()
