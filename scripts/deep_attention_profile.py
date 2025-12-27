#!/usr/bin/env python3
"""
Deep attention profiling - compare every step of attention computation.

This captures:
1. Q/K/V after linear projection
2. Q/K after norm
3. Q/K after RoPE
4. Attention weights (Q @ K.T * scale)
5. Attention output
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    
    print("=" * 60)
    print("Deep Attention Profiling")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        torch_dtype=torch.float32,
    )
    
    # Get first block's attention
    block0 = transformer.transformer_blocks[0]
    attn1 = block0.attn1
    
    # Capture every attention step
    attn_debug = {}
    
    # Register hooks on attn1 internals
    def hook_output(name):
        def hook(module, input, output):
            attn_debug[name] = output.detach().clone() if not isinstance(output, tuple) else output[0].detach().clone()
        return hook
    
    # Hook linear layers
    attn1.to_q.register_forward_hook(hook_output('to_q'))
    attn1.to_k.register_forward_hook(hook_output('to_k'))
    attn1.to_v.register_forward_hook(hook_output('to_v'))
    attn1.norm_q.register_forward_hook(hook_output('norm_q'))
    attn1.norm_k.register_forward_hook(hook_output('norm_k'))
    attn1.to_out[0].register_forward_hook(hook_output('to_out'))
    
    # Simple forward
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
    
    with torch.no_grad():
        _ = transformer(
            hidden_states=packed,
            encoder_hidden_states=text_emb,
            timestep=timestep,
            encoder_attention_mask=torch.ones(B, text_emb.shape[1]),
            num_frames=T,
            height=H,
            width=W,
            return_dict=False,
        )
    
    print("\n" + "=" * 60)
    print("Attention Step-by-Step Values")
    print("=" * 60)
    
    for name, tensor in attn_debug.items():
        t = tensor.cpu().numpy()
        print(f"\n{name}:")
        print(f"  Shape: {t.shape}")
        print(f"  Range: [{t.min():.6f}, {t.max():.6f}]")
        print(f"  Mean: {t.mean():.6f}, Std: {t.std():.6f}")
        print(f"  First 5: {t.flatten()[:5]}")
        np.save(f'output/debug/attn_{name}.npy', t)
    
    # Key comparison values
    print("\n" + "=" * 60)
    print("Key Values for Rust Comparison")
    print("=" * 60)
    
    if 'to_q' in attn_debug and 'norm_q' in attn_debug:
        q_before = attn_debug['to_q']
        q_after = attn_debug['norm_q']
        print(f"\nQ before norm - mean: {q_before.mean():.6f}, std: {q_before.std():.6f}")
        print(f"Q after norm - mean: {q_after.mean():.6f}, std: {q_after.std():.6f}")
        
    if 'to_out' in attn_debug:
        out = attn_debug['to_out']
        print(f"\nAttention output - mean: {out.mean():.6f}, std: {out.std():.6f}")
    
    print("\n✓ Saved attention debug data to output/debug/attn_*.npy")

if __name__ == '__main__':
    main()
