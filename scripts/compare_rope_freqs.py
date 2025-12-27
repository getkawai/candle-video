#!/usr/bin/env python3
"""
Compare exact RoPE frequency computation between diffusers and expected Rust.
"""

import numpy as np
import torch
import math

def diffusers_compute_freqs(dim, theta, grid, device='cpu'):
    """Exact diffusers RoPE frequency computation."""
    start = 1.0
    end = theta
    freqs = theta ** torch.linspace(
        math.log(start, theta),
        math.log(end, theta),
        dim // 6,
        device=device,
        dtype=torch.float32,
    )
    freqs = freqs * math.pi / 2.0  # [freq_dim]
    
    # grid: (B, seq_len, 3) - each row is (t, h, w) coordinates
    # Apply: freqs * (grid.unsqueeze(-1) * 2 - 1)
    scaled_grid = grid.unsqueeze(-1) * 2 - 1  # (B, seq, 3, 1)
    
    # freqs: (freq_dim,) -> (1, 1, 1, freq_dim)
    # Result: (B, seq, 3, freq_dim)
    freqs_applied = scaled_grid * freqs.view(1, 1, 1, -1)
    
    # Transpose and flatten: (B, seq, freq_dim, 3) -> (B, seq, 3*freq_dim)
    freqs_applied = freqs_applied.transpose(-1, -2).flatten(2)
    
    cos_freqs = freqs_applied.cos().repeat_interleave(2, dim=-1)
    sin_freqs = freqs_applied.sin().repeat_interleave(2, dim=-1)
    
    # Handle padding if dim % 6 != 0
    if dim % 6 != 0:
        cos_padding = torch.ones_like(cos_freqs[:, :, : dim % 6])
        sin_padding = torch.zeros_like(cos_freqs[:, :, : dim % 6])
        cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
        sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
    
    return cos_freqs, sin_freqs

def rust_compute_freqs(dim, theta, grid, device='cpu'):
    """How Rust computes RoPE frequencies (current implementation)."""
    freq_dim = dim // 6
    
    # indices = theta^(linspace(log_theta(1), log_theta(theta), freq_dim))
    indices = []
    for i in range(freq_dim):
        t = i / max(freq_dim - 1, 1)
        log_start = math.log(1.0, theta) if theta != 1 else 0
        log_end = math.log(theta, theta) if theta != 1 else 1
        val = theta ** (log_start + t * (log_end - log_start))
        indices.append(val * math.pi / 2.0)
    
    indices = torch.tensor(indices, dtype=torch.float32, device=device)
    
    # scaled_positions: grid * 2 - 1
    scaled_positions = grid * 2 - 1  # (B, seq, 3)
    
    # (B, seq, 3, 1) * (1, 1, 1, freq_dim) -> (B, seq, 3, freq_dim)
    scaled_positions = scaled_positions.unsqueeze(-1)
    indices_expanded = indices.view(1, 1, 1, -1)
    freqs = scaled_positions * indices_expanded
    
    # AFTER FIX: transpose then flatten
    freqs = freqs.transpose(-1, -2).flatten(2)  # (B, seq, 3*freq_dim)
    
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()
    
    # Repeat interleave by 2
    cos_freq = cos_freq.repeat_interleave(2, dim=-1)
    sin_freq = sin_freq.repeat_interleave(2, dim=-1)
    
    # Handle padding if dim % 6 != 0
    current_dim = 3 * freq_dim * 2
    if current_dim < dim:
        pad_size = dim - current_dim
        cos_pad = torch.ones(cos_freq.shape[0], cos_freq.shape[1], pad_size, device=device)
        sin_pad = torch.zeros(sin_freq.shape[0], sin_freq.shape[1], pad_size, device=device)
        cos_freq = torch.cat([cos_pad, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_pad, sin_freq], dim=-1)
    
    return cos_freq, sin_freq

def main():
    print("=" * 60)
    print("RoPE Frequency Computation Comparison")
    print("=" * 60)
    
    dim = 2048
    theta = 10000.0
    
    # Simple test grid
    B, seq = 1, 10
    grid = torch.rand(B, seq, 3)  # Random positions in [0, 1]
    
    print(f"\nParameters: dim={dim}, theta={theta}, seq={seq}")
    
    # Compute frequencies
    cos_diff, sin_diff = diffusers_compute_freqs(dim, theta, grid)
    cos_rust, sin_rust = rust_compute_freqs(dim, theta, grid)
    
    print(f"\nDiffusers cos_freqs: {cos_diff.shape}")
    print(f"Rust cos_freqs: {cos_rust.shape}")
    
    print(f"\nDiffusers cos_freqs[0,0,:10]: {cos_diff[0,0,:10].tolist()}")
    print(f"Rust cos_freqs[0,0,:10]: {cos_rust[0,0,:10].tolist()}")
    
    # Compare
    cos_max_diff = (cos_diff - cos_rust).abs().max().item()
    sin_max_diff = (sin_diff - sin_rust).abs().max().item()
    
    print(f"\nMax diff in cos_freqs: {cos_max_diff:.6f}")
    print(f"Max diff in sin_freqs: {sin_max_diff:.6f}")
    
    if cos_max_diff < 1e-5 and sin_max_diff < 1e-5:
        print("\n✓ RoPE frequency computation matches!")
    else:
        print("\n✗ RoPE frequency computation DIFFERS!")
        
        # Debug: print base frequency computation
        print("\n--- Debug: Base frequency values ---")
        freq_dim = dim // 6
        
        # Diffusers way
        diff_freqs = theta ** torch.linspace(
            math.log(1.0, theta),
            math.log(theta, theta),
            freq_dim,
        ) * math.pi / 2.0
        
        # Rust way
        rust_freqs = []
        for i in range(freq_dim):
            t = i / max(freq_dim - 1, 1)
            log_start = 0  # log_theta(1) = 0
            log_end = 1    # log_theta(theta) = 1
            val = theta ** (log_start + t * (log_end - log_start))
            rust_freqs.append(val * math.pi / 2.0)
        rust_freqs = torch.tensor(rust_freqs)
        
        print(f"Diffusers base freqs[:5]: {diff_freqs[:5].tolist()}")
        print(f"Rust base freqs[:5]: {rust_freqs[:5].tolist()}")
        
        freq_diff = (diff_freqs - rust_freqs).abs().max().item()
        print(f"Max diff in base freqs: {freq_diff:.6f}")

if __name__ == '__main__':
    main()
