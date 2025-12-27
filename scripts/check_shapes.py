#!/usr/bin/env python3
"""
Script to trace tensor shapes through LTX-Video VAE decoder pipeline.
This helps identify shape mismatches before runtime.
"""

import torch
from safetensors import safe_open

def analyze_vae_shapes():
    """Analyze VAE decoder shapes to identify mismatches."""
    
    # Load VAE weights to get actual shapes
    f = safe_open('ltxv-2b-0.9.8-distilled/vae/vae.safetensors', framework='pt')
    
    # Example input dimensions
    batch = 1
    latent_channels = 128
    latent_t, latent_h, latent_w = 1, 16, 24
    
    print("=" * 60)
    print("LTX-Video VAE Decoder Shape Analysis")
    print("=" * 60)
    print(f"\nInput latent: [{batch}, {latent_channels}, {latent_t}, {latent_h}, {latent_w}]")
    
    # conv_in
    conv_in_w = f.get_tensor('decoder.conv_in.conv.weight')
    print(f"\nconv_in.weight: {conv_in_w.shape}")
    # [out_ch, in_ch, kt, kh, kw]
    out_ch = conv_in_w.shape[0]  # 512
    print(f"  After conv_in: [{batch}, {out_ch}, {latent_t}, {latent_h}, {latent_w}]")
    
    current_ch = out_ch
    current_t, current_h, current_w = latent_t, latent_h, latent_w
    
    # mid_block
    print(f"\nmid_block (4 resnets, {current_ch} channels):")
    for i in range(4):
        w = f.get_tensor(f'decoder.mid_block.resnets.{i}.conv1.conv.weight')
        print(f"  resnets.{i}.conv1: {w.shape}")
    print(f"  After mid_block: [{batch}, {current_ch}, {current_t}, {current_h}, {current_w}]")
    
    # up_blocks
    up_configs = [
        (0, 512, 512, 3, False, False),  # block, in_ch, out_ch, num_resnets, has_upsampler, has_conv_in
        (1, 512, 512, 3, True, False),
        (2, 512, 256, 3, True, True),
        (3, 256, 128, 4, True, True),
    ]
    
    for block_idx, in_ch, out_ch, num_resnets, has_upsampler, has_conv_in in up_configs:
        print(f"\nup_blocks.{block_idx}:")
        print(f"  Input: [{batch}, {current_ch}, {current_t}, {current_h}, {current_w}]")
        
        if has_conv_in:
            key = f'decoder.up_blocks.{block_idx}.conv_in.conv1.conv.weight'
            w = f.get_tensor(key)
            print(f"  conv_in.conv1: {w.shape} (in={w.shape[1]}, out={w.shape[0]})")
            current_ch = out_ch
            print(f"  After conv_in: [{batch}, {current_ch}, {current_t}, {current_h}, {current_w}]")
        
        for i in range(num_resnets):
            key = f'decoder.up_blocks.{block_idx}.resnets.{i}.conv1.conv.weight'
            w = f.get_tensor(key)
            print(f"  resnets.{i}.conv1: {w.shape}")
        
        if has_upsampler:
            key = f'decoder.up_blocks.{block_idx}.upsamplers.0.conv.conv.weight'
            w = f.get_tensor(key)
            print(f"  upsampler: {w.shape}")
            # 8x channels for 2x2x2 spatial-temporal upsample
            # After unpatchify: divide by 8 and multiply spatial by 2
            current_t *= 2
            current_h *= 2
            current_w *= 2
            print(f"  After upsampler+unpatchify: [{batch}, {current_ch}, {current_t}, {current_h}, {current_w}]")
        else:
            print(f"  No upsampler")
    
    # conv_out
    conv_out_w = f.get_tensor('decoder.conv_out.conv.weight')
    print(f"\nconv_out.weight: {conv_out_w.shape}")
    final_ch = conv_out_w.shape[0]  # 48 = 3 * 4 * 4 = RGB * patch_size^2
    print(f"  After conv_out: [{batch}, {final_ch}, {current_t}, {current_h}, {current_w}]")
    print(f"  After unpatchify(4): [{batch}, 3, {current_t}, {current_h * 4}, {current_w * 4}]")
    
    print("\n" + "=" * 60)
    print("CHECKING FOR POTENTIAL SHAPE MISMATCHES:")
    print("=" * 60)
    
    # Check conv channel progressions
    checks = [
        ('conv_in', 128, 512),
        ('mid_block.resnets.0', 512, 512),
        ('up_blocks.0.resnets.0', 512, 512),
        ('up_blocks.1.resnets.0', 512, 512),
        ('up_blocks.2.conv_in', 512, 256),
        ('up_blocks.2.resnets.0', 256, 256),
        ('up_blocks.3.conv_in', 256, 128),
        ('up_blocks.3.resnets.0', 128, 128),
        ('conv_out', 128, 48),
    ]
    
    errors = []
    for name, expected_in, expected_out in checks:
        try:
            if 'conv_in' in name and 'up_blocks' in name:
                key = f'decoder.{name}.conv1.conv.weight'
            elif 'resnets' in name:
                key = f'decoder.{name}.conv1.conv.weight'
            else:
                key = f'decoder.{name}.conv.weight'
            
            w = f.get_tensor(key)
            actual_out, actual_in = w.shape[0], w.shape[1]
            
            if actual_in != expected_in or actual_out != expected_out:
                errors.append(f"  {name}: expected ({expected_in}->{expected_out}), got ({actual_in}->{actual_out})")
            else:
                print(f"  ✓ {name}: {actual_in} -> {actual_out}")
        except Exception as e:
            print(f"  ? {name}: {e}")
    
    if errors:
        print("\nERRORS FOUND:")
        for e in errors:
            print(e)
    else:
        print("\n✓ All channel progressions match!")
    
    print("\n" + "=" * 60)
    print("CONV3D MATMUL SHAPE CHECK:")
    print("=" * 60)
    # For Conv3dVia2d, the matmul is:
    # patches: [B, h_out*w_out, c_in*kh*kw]
    # weight: [c_out, c_in*kh*kw] -> transposed to [c_in*kh*kw, c_out]
    # Result: [B, h_out*w_out, c_out]
    #
    # Issue: 3D @ 2D fails - need to use proper broadcasting or batch matmul
    print("Conv2d manual matmul:")
    print("  patches_stacked: [B, h*w, c_in*k*k] = [1, 384, 1152]")  
    print("  weight_flat.t(): [c_in*k*k, c_out] = [1152, 512]")
    print("  PROBLEM: 3D @ 2D - need weight to be [1, 1152, 512] for broadcast")
    print("  FIX: Add unsqueeze(0) before matmul")

if __name__ == '__main__':
    analyze_vae_shapes()
