#!/usr/bin/env python3
"""
Compare RoPE grid coordinate computation between diffusers and expected Rust behavior.
"""

import numpy as np
import torch
import math

def diffusers_rope_grid(batch_size, num_frames, height, width, device='cpu'):
    """Exactly as diffusers does it."""
    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width, dtype=torch.float32, device=device)
    grid_f = torch.arange(num_frames, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    
    # Flatten spatial dimensions
    grid = grid.flatten(2, 4).transpose(1, 2)  # (B, seq_len, 3)
    
    return grid

def rust_expected_grid(batch_size, num_frames, height, width, device='cpu'):
    """How Rust should compute it based on generate_indices_grid."""
    seq_len = num_frames * height * width
    
    # Rust uses fractional positions in [0, 1]
    indices = []
    for b in range(batch_size):
        t_idx = []
        h_idx = []
        w_idx = []
        for t in range(num_frames):
            for h in range(height):
                for w in range(width):
                    t_idx.append(t / max(num_frames - 1, 1))
                    h_idx.append(h / max(height - 1, 1))
                    w_idx.append(w / max(width - 1, 1))
        indices.append([t_idx, h_idx, w_idx])
    
    grid = torch.tensor(indices, dtype=torch.float32, device=device)
    grid = grid.permute(0, 2, 1)  # (B, seq_len, 3)
    return grid

def main():
    print("=" * 60)
    print("RoPE Grid Comparison")
    print("=" * 60)
    
    B, T, H, W = 1, 1, 16, 24
    
    diffusers_grid = diffusers_rope_grid(B, T, H, W)
    rust_grid = rust_expected_grid(B, T, H, W)
    
    print(f"\nDiffusers grid shape: {diffusers_grid.shape}")
    print(f"Rust expected grid shape: {rust_grid.shape}")
    
    print(f"\nDiffusers grid[0,:5,:]:")
    print(diffusers_grid[0, :5, :])
    
    print(f"\nRust expected grid[0,:5,:]:")
    print(rust_grid[0, :5, :])
    
    # Note: Diffusers uses raw indices, Rust uses normalized [0,1]
    print("\n" + "=" * 60)
    print("KEY DIFFERENCE FOUND!")
    print("=" * 60)
    print("\nDiffusers uses raw indices: 0, 1, 2, 3...")
    print("Rust uses normalized [0,1]: 0.0, 0.066, 0.133...")
    print("\nBut diffusers ALSO normalizes in RoPE with base_height/width!")
    print("So we need to check if the final frequencies match.")
    
    # Compute actual frequencies
    theta = 10000.0
    dim = 2048
    freq_dim = dim // 6
    
    # Diffusers way
    freqs_base = theta ** torch.linspace(
        math.log(1.0, theta),
        math.log(theta, theta),
        freq_dim,
        dtype=torch.float32,
    ) * math.pi / 2.0
    
    # For diffusers, grid values are scaled in _prepare_video_coords
    # by rope_interpolation_scale * patch_size / base_dim
    # For testing, assume no interpolation
    base_num_frames, base_height, base_width = 9, 512, 512
    patch_size, patch_size_t = 1, 1
    
    # Diffusers scaling for T=1, H=16, W=24 latent
    diffusers_scaled = diffusers_grid.clone()
    diffusers_scaled[:, :, 0] = diffusers_scaled[:, :, 0] * patch_size_t / base_num_frames
    diffusers_scaled[:, :, 1] = diffusers_scaled[:, :, 1] * patch_size / base_height
    diffusers_scaled[:, :, 2] = diffusers_scaled[:, :, 2] * patch_size / base_width
    
    print(f"\nDiffusers scaled grid[0,:5,:]:")
    print(diffusers_scaled[0, :5, :])
    
    # Apply the 2x - 1 transformation
    diffusers_final = diffusers_scaled * 2 - 1
    print(f"\nDiffusers final (2x-1) grid[0,:5,:]:")
    print(diffusers_final[0, :5, :])
    
    rust_final = rust_grid * 2 - 1
    print(f"\nRust final (2x-1) grid[0,:5,:]:")
    print(rust_final[0, :5, :])
    
    # The difference
    diff = (diffusers_final - rust_final).abs()
    print(f"\nMax diff in grid: {diff.max().item():.6f}")
    print(f"Mean diff in grid: {diff.mean().item():.6f}")
    
    if diff.max() > 0.01:
        print("\n⚠️  Significant grid difference found!")
        print("This explains the numerical discrepancy.")

if __name__ == '__main__':
    main()
