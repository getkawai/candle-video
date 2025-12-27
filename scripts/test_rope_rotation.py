#!/usr/bin/env python3
"""
Isolated RoPE rotation comparison between diffusers and expected Rust.

This tests JUST the rotation formula without any model loading.
"""

import numpy as np
import torch
import struct

def diffusers_apply_rotary_emb(x, freqs):
    """Exact diffusers apply_rotary_emb function."""
    cos, sin = freqs
    # x: (B, S, C)
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out

def rust_apply_rotary_emb_linear(x, freqs):
    """What Rust apply_rotary_emb_linear should do."""
    cos, sin = freqs
    # x: (B, S, C) where C = inner_dim
    B, S, C = x.shape
    
    # Reshape to (B, S, C//2, 2)
    x_reshaped = x.reshape(B, S, C // 2, 2)
    
    # Get real and imag
    x_real = x_reshaped[..., 0]  # (B, S, C//2)
    x_imag = x_reshaped[..., 1]
    
    # Create rotated: (-imag, real) then flatten
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(2)  # (B, S, C)
    
    # Apply rotation
    out = x * cos + x_rot * sin
    return out

def main():
    print("=" * 60)
    print("Isolated RoPE Rotation Test")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    B, S, C = 1, 384, 2048
    
    # Random input tensor
    x = torch.randn(B, S, C)
    
    # Random freqs (cos, sin)
    cos = torch.randn(B, S, C)
    sin = torch.randn(B, S, C)
    freqs = (cos, sin)
    
    # Apply both
    out_diff = diffusers_apply_rotary_emb(x, freqs)
    out_rust = rust_apply_rotary_emb_linear(x, freqs)
    
    # Compare
    diff = (out_diff - out_rust).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nDiffusers output shape: {out_diff.shape}")
    print(f"Rust output shape: {out_rust.shape}")
    print(f"\nMax diff: {max_diff:.10f}")
    print(f"Mean diff: {mean_diff:.10f}")
    
    if max_diff < 1e-6:
        print("\n✓ RoPE rotation formula MATCHES!")
    else:
        print("\n✗ RoPE rotation formula DIFFERS!")
        
        # Debug
        print("\n--- Debug ---")
        print(f"Diffusers out[0,0,:10]: {out_diff[0,0,:10].tolist()}")
        print(f"Rust out[0,0,:10]: {out_rust[0,0,:10].tolist()}")
    
    # Now test with actual RoPE freqs (cos/sin in [-1,1])
    print("\n" + "=" * 60)
    print("Test with realistic cos/sin values")
    print("=" * 60)
    
    cos = torch.rand(B, S, C) * 2 - 1  # [-1, 1]
    sin = torch.rand(B, S, C) * 2 - 1
    freqs = (cos, sin)
    
    out_diff = diffusers_apply_rotary_emb(x, freqs)
    out_rust = rust_apply_rotary_emb_linear(x, freqs)
    
    diff = (out_diff - out_rust).abs()
    max_diff = diff.max().item()
    
    print(f"Max diff: {max_diff:.10f}")
    
    if max_diff < 1e-6:
        print("✓ Matches with realistic freqs!")
    else:
        print("✗ Differs with realistic freqs!")

if __name__ == '__main__':
    main()
