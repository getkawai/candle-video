#!/usr/bin/env python3
"""Debug RoPE grid computation to understand differences."""

import os
import sys
import math
import torch

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from diffusers.models.transformers.transformer_ltx import LTXVideoRotaryPosEmbed


def debug_grid_computation():
    """Debug the grid computation step by step."""
    print("=== Debug Grid Computation ===")
    
    rope = LTXVideoRotaryPosEmbed(
        dim=2048,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        patch_size=1,
        patch_size_t=1,
        theta=10000.0,
    )
    
    # Test with f=4, h=4, w=4
    num_frames = 4
    height = 4
    width = 4
    batch_size = 1
    seq_len = num_frames * height * width
    
    # Step 1: Create grid (as in _prepare_video_coords)
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)
    grid_f = torch.arange(num_frames, dtype=torch.float32)
    
    grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # [3, F, H, W]
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, 3, F, H, W]
    
    print(f"Grid shape before flatten: {grid.shape}")
    print(f"Grid[0, 0, :, 0, 0] (frame coords): {grid[0, 0, :, 0, 0].tolist()}")
    print(f"Grid[0, 1, 0, :, 0] (height coords): {grid[0, 1, 0, :, 0].tolist()}")
    print(f"Grid[0, 2, 0, 0, :] (width coords): {grid[0, 2, 0, 0, :].tolist()}")
    
    # No rope_interpolation_scale, so no scaling
    
    # Flatten and transpose
    grid = grid.flatten(2, 4).transpose(1, 2)  # [B, seq, 3]
    print(f"\nGrid shape after flatten: {grid.shape}")
    print(f"Grid[0, 0, :] (first position): {grid[0, 0, :].tolist()}")
    print(f"Grid[0, 1, :] (second position): {grid[0, 1, :].tolist()}")
    print(f"Grid[0, -1, :] (last position): {grid[0, -1, :].tolist()}")
    
    # Step 2: Compute frequencies
    dim = 2048
    theta = 10000.0
    steps = dim // 6
    
    start = 1.0
    end = theta
    freqs = theta ** torch.linspace(
        math.log(start, theta),
        math.log(end, theta),
        steps,
        dtype=torch.float32,
    )
    freqs = freqs * math.pi / 2.0
    
    print(f"\nFreqs shape: {freqs.shape}")
    print(f"Freqs first 5: {freqs[:5].tolist()}")
    print(f"Freqs last 5: {freqs[-5:].tolist()}")
    
    # Step 3: Apply grid scaling
    # freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
    grid_scaled = grid.unsqueeze(-1) * 2 - 1  # [B, seq, 3, 1]
    print(f"\nGrid scaled shape: {grid_scaled.shape}")
    print(f"Grid scaled[0, 0, :, 0] (first position): {grid_scaled[0, 0, :, 0].tolist()}")
    print(f"Grid scaled[0, 1, :, 0] (second position): {grid_scaled[0, 1, :, 0].tolist()}")
    
    # Multiply with freqs
    freqs_grid = freqs * grid_scaled  # [B, seq, 3, steps]
    print(f"\nFreqs_grid shape: {freqs_grid.shape}")
    
    # Transpose and flatten
    freqs_grid = freqs_grid.transpose(-1, -2).flatten(2)  # [B, seq, 3*steps]
    print(f"Freqs_grid after flatten shape: {freqs_grid.shape}")
    
    # Compute cos/sin
    cos_freqs = freqs_grid.cos().repeat_interleave(2, dim=-1)
    sin_freqs = freqs_grid.sin().repeat_interleave(2, dim=-1)
    
    print(f"\nCos shape: {cos_freqs.shape}")
    print(f"Sin shape: {sin_freqs.shape}")
    
    # Check padding
    if dim % 6 != 0:
        cos_padding = torch.ones_like(cos_freqs[:, :, : dim % 6])
        sin_padding = torch.zeros_like(cos_freqs[:, :, : dim % 6])
        cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
        sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        print(f"After padding - Cos shape: {cos_freqs.shape}")
    
    # Now compare with actual rope output
    hidden_states = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)
    actual_cos, actual_sin = rope(
        hidden_states,
        num_frames=num_frames,
        height=height,
        width=width,
        rope_interpolation_scale=None,
    )
    
    print(f"\n=== Comparison ===")
    print(f"Manual cos shape: {cos_freqs.shape}")
    print(f"Actual cos shape: {actual_cos.shape}")
    
    diff = (cos_freqs - actual_cos).abs()
    print(f"Max diff: {diff.max().item()}")
    print(f"Mean diff: {diff.mean().item()}")
    
    if diff.max().item() < 1e-6:
        print("✓ Manual computation matches actual!")
    else:
        print("✗ Mismatch detected!")
        # Find where the difference is
        max_idx = diff.argmax()
        print(f"Max diff at index: {max_idx}")


if __name__ == "__main__":
    debug_grid_computation()
