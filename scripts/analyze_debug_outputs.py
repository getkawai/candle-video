#!/usr/bin/env python3
"""
Quick comparison of diffusers debug outputs with Rust expected behavior.

Checks specific values to identify the source of discrepancy.
"""

import os
import numpy as np
import struct

def load_tensor_bin(path):
    """Load tensor from .bin file."""
    with open(path, 'rb') as f:
        ndims = struct.unpack('<Q', f.read(8))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape(dims)

def main():
    print("=" * 60)
    print("Comparing Diffusers Intermediate Outputs")
    print("=" * 60)
    
    debug_dir = 'output/debug'
    if not os.path.exists(debug_dir):
        print("No debug outputs found. Run debug_dit_layers.py first.")
        return
    
    # Load and analyze each layer
    for name in ['rope', 'proj_in', 'time_embed', 'caption_proj', 
                 'block0_attn1', 'block0_attn2', 'block0_ff', 'block0_out']:
        path = f'{debug_dir}/{name}.bin'
        if os.path.exists(path):
            t = load_tensor_bin(path)
            print(f"\n{name}:")
            print(f"  Shape: {t.shape}")
            print(f"  First 5: {t.flatten()[:5]}")
    
    # Load final outputs
    ref_path = 'output/dit_output_ref.bin'
    if os.path.exists(ref_path):
        ref = load_tensor_bin(ref_path)
        print(f"\nFinal Reference Output:")
        print(f"  Shape: {ref.shape}")
        print(f"  First 5: {ref.flatten()[:5]}")
        print(f"  Range: [{ref.min():.4f}, {ref.max():.4f}]")
    
    # Key insight: Check if there's an offset or scaling issue
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    
    # RoPE should have cos/sin values in [-1, 1]
    rope = load_tensor_bin(f'{debug_dir}/rope.bin')
    print(f"\n1. RoPE Analysis:")
    print(f"   Shape: {rope.shape} (expected: [B, seq_len, hidden_size])")
    print(f"   This is cos_freqs concatenated - check if Rust produces same")
    
    # Time embedding is crucial
    time_embed = load_tensor_bin(f'{debug_dir}/time_embed.bin')
    print(f"\n2. Time Embedding:")
    print(f"   Shape: {time_embed.shape} (expected: [B, 6*hidden_size])")
    print(f"   First 10: {time_embed.flatten()[:10]}")
    
    # Block 0 output shows cumulative error
    block0 = load_tensor_bin(f'{debug_dir}/block0_out.bin')
    print(f"\n3. Block 0 Output:")
    print(f"   Range: [{block0.min():.4f}, {block0.max():.4f}]")
    print(f"   Std: {block0.std():.4f}")
    print(f"   If Rust has similar block0 output, error is earlier")

if __name__ == '__main__':
    main()
