#!/usr/bin/env python3
"""
Compare single block output - run just 1 transformer block.

This creates a modified forward that stops after block 0.
"""

import numpy as np
import torch
import struct
import sys
import os

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def save_tensor_bin(tensor, path):
    """Save in Rust-loadable binary format."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(tensor.shape)))
        for d in tensor.shape:
            f.write(struct.pack('<Q', d))
        tensor.astype(np.float32).tofile(f)

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    
    print("=" * 60)
    print("Single Block Output Comparison")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    transformer = LTXVideoTransformer3DModel.from_single_file(
        'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors',
        torch_dtype=torch.float32,
    )
    
    B, C, T, H, W = 1, 128, 1, 16, 24
    latents = torch.randn(B, C, T, H, W)
    text_emb = torch.randn(B, 8, 4096)
    timestep = torch.tensor([1.0])
    
    def pack_latents(x, ps, ps_t):
        B, C, T, H, W = x.shape
        pT, pH, pW = T // ps_t, H // ps, W // ps
        x = x.reshape(B, C, pT, ps_t, pH, ps, pW, ps)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return x
    
    packed = pack_latents(latents, 1, 1)
    seq_len = packed.shape[1]
    
    with torch.no_grad():
        # Manual forward similar to diffusers but stopping early
        hidden_states = packed
        
        # proj_in
        hidden_states = transformer.proj_in(hidden_states)
        print(f"After proj_in: shape={hidden_states.shape}")
        print(f"  first 5: {hidden_states[0,0,:5].tolist()}")
        
        # time_embed (AdaLN)
        temb, embedded_timestep = transformer.time_embed(
            timestep, 
            hidden_dtype=hidden_states.dtype
        )
        print(f"\nTime emb: shape={temb.shape}")
        print(f"  first 5: {temb[0,:5].tolist()}")
        
        # caption projection
        caption_proj = transformer.caption_projection(text_emb)
        print(f"\nCaption proj: shape={caption_proj.shape}")
        
        # RoPE
        cos, sin = transformer.rope(
            hidden_states,
            num_frames=T,
            height=H,
            width=W,
            rope_interpolation_scale=(torch.tensor(1.0), 1.0, 1.0),
        )
        image_rotary_emb = (cos, sin)
        print(f"\nRoPE cos: shape={cos.shape}")
        print(f"  cos first 5: {cos[0,0,:5].tolist()}")
        
        # Just block 0
        block = transformer.transformer_blocks[0]
        block_output = block(
            hidden_states,
            encoder_hidden_states=caption_proj,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            encoder_attention_mask=None,
        )
        
        print(f"\nBlock 0 output: shape={block_output.shape}")
        print(f"  first 5: {block_output[0,0,:5].tolist()}")
        print(f"  range: [{block_output.min():.4f}, {block_output.max():.4f}]")
        print(f"  mean: {block_output.mean():.4f}, std: {block_output.std():.4f}")
        
        # Save for Rust comparison
        os.makedirs('output/debug', exist_ok=True)
        np.save('output/debug/block0_input.npy', hidden_states.numpy())
        np.save('output/debug/block0_output.npy', block_output.numpy())
        np.save('output/debug/temb.npy', temb.numpy())
        save_tensor_bin(block_output.numpy(), 'output/debug/block0_output.bin')
        
        print("\n✓ Saved block 0 input/output to output/debug/")

if __name__ == '__main__':
    main()
