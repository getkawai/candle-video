#!/usr/bin/env python3
"""
Compare actual RoPE freqs_cis output from diffusers vs simulated Rust.

Uses the exact same grid and compares the resulting cos/sin frequencies.
"""

import numpy as np
import torch
import math
import struct
import sys
import os

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def rust_compute_freqs_cis(grid, dim, theta=10000.0):
    """Simulates Rust FractionalRoPE.compute_freqs_cis"""
    # grid: (B, 3, seq_len) - will transpose to (B, seq_len, 3)
    B = grid.shape[0]
    seq_len = grid.shape[2]
    
    # Transpose grid to (B, seq_len, 3)
    grid = grid.permute(0, 2, 1)
    
    freq_dim = dim // 6
    
    # Compute indices (matching Rust)
    indices = []
    for i in range(freq_dim):
        t = i / max(freq_dim - 1, 1)
        log_start = 0  # log_theta(1)
        log_end = 1    # log_theta(theta)
        val = theta ** (log_start + t * (log_end - log_start))
        indices.append(val * math.pi / 2.0)
    indices = torch.tensor(indices, dtype=torch.float32)
    
    # scaled_positions: grid * 2 - 1
    scaled_positions = grid * 2 - 1
    
    # (B, seq, 3, 1) * (1, 1, 1, freq_dim) -> (B, seq, 3, freq_dim)
    scaled_positions = scaled_positions.unsqueeze(-1)
    indices_expanded = indices.view(1, 1, 1, -1)
    freqs = scaled_positions * indices_expanded
    
    # Transpose and flatten (B, seq, 3, freq_dim) -> (B, seq, freq_dim, 3) -> (B, seq, 3*freq_dim)
    freqs = freqs.transpose(-1, -2).flatten(2)
    
    # cos/sin
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()
    
    # Repeat interleave by 2
    cos_freq = cos_freq.repeat_interleave(2, dim=-1)
    sin_freq = sin_freq.repeat_interleave(2, dim=-1)
    
    # Padding
    current_dim = 3 * freq_dim * 2
    if current_dim < dim:
        pad_size = dim - current_dim
        cos_pad = torch.ones(B, seq_len, pad_size)
        sin_pad = torch.zeros(B, seq_len, pad_size)
        cos_freq = torch.cat([cos_pad, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_pad, sin_freq], dim=-1)
    
    return cos_freq, sin_freq

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoRotaryPosEmbed
    
    print("=" * 60)
    print("RoPE Frequencies Comparison")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    B, T, H, W = 1, 1, 16, 24
    dim = 2048
    seq_len = T * H * W
    
    # Create diffusers RoPE
    rope = LTXVideoRotaryPosEmbed(
        dim=dim,
        base_num_frames=9,
        base_height=512,
        base_width=512,
        patch_size=1,
        patch_size_t=1,
        theta=10000.0,
    )
    
    # Create dummy hidden_states for the forward call
    hidden_states = torch.randn(B, seq_len, dim)
    
    # Get diffusers cos/sin
    cos_diff, sin_diff = rope(
        hidden_states,
        num_frames=T,
        height=H,
        width=W,
        rope_interpolation_scale=(torch.tensor(1.0), 1.0, 1.0),
    )
    
    print(f"\nDiffusers cos shape: {cos_diff.shape}")
    print(f"Diffusers cos[0,0,:10]: {cos_diff[0,0,:10].tolist()}")
    print(f"Diffusers sin[0,0,:10]: {sin_diff[0,0,:10].tolist()}")
    
    # Create same grid for Rust simulation
    # Rust uses generate_indices_grid_for_diffusers which produces (B, 3, seq_len)
    base_num_frames, base_height, base_width = 9, 512, 512
    patch_size, patch_size_t = 1, 1
    
    indices = []
    for ti in range(T):
        t_coord = ti * patch_size_t / base_num_frames
        for hi in range(H):
            h_coord = hi * patch_size / base_height
            for wi in range(W):
                w_coord = wi * patch_size / base_width
                indices.append([t_coord, h_coord, w_coord])
    
    grid = torch.tensor([indices], dtype=torch.float32)  # (1, seq_len, 3)
    grid = grid.permute(0, 2, 1)  # (1, 3, seq_len) - Rust format
    
    # Compute Rust-style freqs
    cos_rust, sin_rust = rust_compute_freqs_cis(grid, dim)
    
    print(f"\nRust cos shape: {cos_rust.shape}")
    print(f"Rust cos[0,0,:10]: {cos_rust[0,0,:10].tolist()}")
    print(f"Rust sin[0,0,:10]: {sin_rust[0,0,:10].tolist()}")
    
    # Compare
    cos_diff_val = (cos_diff - cos_rust).abs()
    sin_diff_val = (sin_diff - sin_rust).abs()
    
    print(f"\nMax cos diff: {cos_diff_val.max().item():.6f}")
    print(f"Max sin diff: {sin_diff_val.max().item():.6f}")
    print(f"Mean cos diff: {cos_diff_val.mean().item():.6f}")
    print(f"Mean sin diff: {sin_diff_val.mean().item():.6f}")
    
    if cos_diff_val.max() < 0.01 and sin_diff_val.max() < 0.01:
        print("\n✓ Frequencies match!")
    else:
        print("\n✗ Frequencies DIFFER!")
        
        # Find where they differ
        print("\n--- Debug ---")
        # Check padding
        print(f"Diffusers cos[:,:,:5] (padding): {cos_diff[0,0,:5].tolist()}")
        print(f"Rust cos[:,:,:5] (padding): {cos_rust[0,0,:5].tolist()}")

if __name__ == '__main__':
    main()
