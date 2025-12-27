#!/usr/bin/env python3
"""
Generate test inputs for Rust DiT validation.

Creates .bin files with deterministic inputs that can be loaded by Rust tests.
"""

import numpy as np
import struct
import os

def save_tensor_bin(tensor: np.ndarray, path: str):
    """Save tensor in a format Rust can easily read."""
    with open(path, 'wb') as f:
        # Header: ndims, dim0, dim1, ...
        f.write(struct.pack('<Q', len(tensor.shape)))
        for d in tensor.shape:
            f.write(struct.pack('<Q', d))
        # Data as f32
        tensor.astype(np.float32).tofile(f)
    print(f"Saved {path}: shape={tensor.shape}, size={tensor.nbytes} bytes")

def main():
    np.random.seed(42)
    
    os.makedirs('output', exist_ok=True)
    
    # Load reference outputs from diffusers run
    if os.path.exists('output/dit_latents.npy'):
        latents = np.load('output/dit_latents.npy')
        text_emb = np.load('output/dit_text_emb.npy')
        output_ref = np.load('output/dit_output_ref.npy')
        
        # Save as .bin for Rust
        save_tensor_bin(latents, 'output/dit_latents.bin')
        save_tensor_bin(text_emb, 'output/dit_text_emb.bin')
        save_tensor_bin(output_ref, 'output/dit_output_ref.bin')
        
        print("\n✓ Created .bin files for Rust comparison")
        print(f"\nReference stats:")
        print(f"  Output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
        print(f"  Output mean: {output_ref.mean():.4f}, std: {output_ref.std():.4f}")
    else:
        print("Reference outputs not found. Run compare_dit.py --reference first.")

if __name__ == '__main__':
    main()
