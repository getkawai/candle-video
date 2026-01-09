#!/usr/bin/env python3
"""
Verify RoPE frequency computation for LTX-Video transformer.

This script computes the exact frequency values and saves them for Rust verification.
Task 5.1: Verify RoPE frequency computation
- Check: theta ** linspace(0, 1, dim//6) * pi/2
- Compare cos/sin with Python
- Requirements: 3.2
"""

import os
import sys
import math
import torch
from safetensors.torch import save_file

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from diffusers.models.transformers.transformer_ltx import LTXVideoRotaryPosEmbed


def compute_frequencies_step_by_step(dim=2048, theta=10000.0):
    """Compute frequencies step by step for verification."""
    print(f"\n=== Step-by-step frequency computation ===")
    print(f"dim={dim}, theta={theta}")
    
    steps = dim // 6
    print(f"steps (dim // 6) = {steps}")
    
    # Python linspace computation
    start = 1.0
    end = theta
    
    log_start = math.log(start, theta)  # log_theta(1) = 0
    log_end = math.log(end, theta)      # log_theta(theta) = 1
    print(f"log_theta(start={start}) = {log_start}")
    print(f"log_theta(end={end}) = {log_end}")
    
    # linspace from 0 to 1
    lin = torch.linspace(log_start, log_end, steps, dtype=torch.float32)
    print(f"linspace shape: {lin.shape}")
    print(f"linspace first 5: {lin[:5].tolist()}")
    print(f"linspace last 5: {lin[-5:].tolist()}")
    
    # theta ** linspace
    freqs_base = theta ** lin
    print(f"theta ** linspace first 5: {freqs_base[:5].tolist()}")
    print(f"theta ** linspace last 5: {freqs_base[-5:].tolist()}")
    
    # * pi/2
    freqs = freqs_base * (math.pi / 2.0)
    print(f"freqs (* pi/2) first 5: {freqs[:5].tolist()}")
    print(f"freqs (* pi/2) last 5: {freqs[-5:].tolist()}")
    print(f"freqs range: [{freqs.min().item():.6f}, {freqs.max().item():.6f}]")
    
    return {
        "linspace": lin,
        "freqs_base": freqs_base,
        "freqs": freqs,
    }


def verify_rope_module():
    """Verify RoPE module produces expected frequencies."""
    print("\n=== Verifying LTXVideoRotaryPosEmbed ===")
    
    rope = LTXVideoRotaryPosEmbed(
        dim=2048,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        patch_size=1,
        patch_size_t=1,
        theta=10000.0,
    )
    
    # Test with simple coordinates
    batch_size = 1
    num_frames = 2
    height = 2
    width = 2
    seq_len = num_frames * height * width
    
    hidden_states = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)
    
    cos_freqs, sin_freqs = rope(
        hidden_states,
        num_frames=num_frames,
        height=height,
        width=width,
        rope_interpolation_scale=None,
    )
    
    print(f"cos_freqs shape: {cos_freqs.shape}")
    print(f"sin_freqs shape: {sin_freqs.shape}")
    print(f"cos_freqs[0,0,:10]: {cos_freqs[0,0,:10].tolist()}")
    print(f"sin_freqs[0,0,:10]: {sin_freqs[0,0,:10].tolist()}")
    
    return cos_freqs, sin_freqs


def generate_rope_reference_data():
    """Generate reference data for Rust verification."""
    print("\n=== Generating RoPE reference data ===")
    
    results = {}
    
    # 1. Basic frequency computation
    freq_data = compute_frequencies_step_by_step(dim=2048, theta=10000.0)
    results["freq_linspace"] = freq_data["linspace"]
    results["freq_base"] = freq_data["freqs_base"]
    results["freq_final"] = freq_data["freqs"]
    
    # 2. RoPE with various configurations
    rope = LTXVideoRotaryPosEmbed(
        dim=2048,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        patch_size=1,
        patch_size_t=1,
        theta=10000.0,
    )
    
    test_configs = [
        # (num_frames, height, width, batch_size)
        (2, 2, 2, 1),
        (4, 4, 4, 1),
        (8, 16, 24, 1),
    ]
    
    for num_frames, height, width, batch_size in test_configs:
        seq_len = num_frames * height * width
        hidden_states = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)
        
        cos_freqs, sin_freqs = rope(
            hidden_states,
            num_frames=num_frames,
            height=height,
            width=width,
            rope_interpolation_scale=None,
        )
        
        key = f"rope_f{num_frames}_h{height}_w{width}"
        results[f"{key}_cos"] = cos_freqs
        results[f"{key}_sin"] = sin_freqs
        
        print(f"  {key}: cos shape={cos_freqs.shape}")
    
    # 3. RoPE with video_coords (as used in pipeline)
    for num_frames, height, width, batch_size in test_configs:
        seq_len = num_frames * height * width
        
        # Create video_coords as done in pipeline
        grid_f = torch.arange(num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)
        
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        video_coords = torch.stack(grid, dim=0)  # [3, F, H, W]
        video_coords = video_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, 3, F, H, W]
        video_coords = video_coords.flatten(2, 4)  # [B, 3, seq_len]
        
        hidden_states = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)
        
        cos_freqs, sin_freqs = rope(
            hidden_states,
            video_coords=video_coords,
        )
        
        key = f"rope_coords_f{num_frames}_h{height}_w{width}"
        results[f"{key}_video_coords"] = video_coords
        results[f"{key}_cos"] = cos_freqs
        results[f"{key}_sin"] = sin_freqs
        
        print(f"  {key}: cos shape={cos_freqs.shape}")
    
    # Save to safetensors
    output_path = "gen_rope_parity.safetensors"
    save_file(results, output_path)
    print(f"\nSaved {len(results)} tensors to {output_path}")
    
    return results


def main():
    print("=" * 60)
    print("RoPE Frequency Verification for LTX-Video")
    print("=" * 60)
    
    # Step-by-step verification
    compute_frequencies_step_by_step()
    
    # Verify RoPE module
    verify_rope_module()
    
    # Generate reference data
    generate_rope_reference_data()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
