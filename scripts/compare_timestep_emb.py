#!/usr/bin/env python3
"""
Compare timestep embedding and AdaLN flow between diffusers and Rust.
"""

import numpy as np
import torch
import struct
import sys
import os

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    
    print("=" * 60)
    print("Timestep Embedding Comparison")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        torch_dtype=torch.float32,
    )
    
    # Hook into adaln_single
    adaln_outputs = {}
    
    def hook_adaln(module, input, output):
        if isinstance(output, tuple):
            adaln_outputs['temb'] = output[0].detach().clone()
            adaln_outputs['temb_proj'] = output[1].detach().clone() if len(output) > 1 else None
        else:
            adaln_outputs['temb'] = output.detach().clone()
    
    transformer.adaln_single.register_forward_hook(hook_adaln)
    
    # Simple forward
    B, C, T, H, W = 1, 128, 1, 16, 24
    seq_len = T * H * W
    
    latents = torch.randn(B, C, T, H, W)
    text_emb = torch.randn(B, 8, 4096)
    timestep = torch.tensor([1.0])
    
    # Pack latents
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
    
    print("\nAdaLN Single output:")
    print(f"  temb shape: {adaln_outputs['temb'].shape}")
    print(f"  temb[0,:20]: {adaln_outputs['temb'][0,:20].tolist()}")
    print(f"  temb range: [{adaln_outputs['temb'].min():.4f}, {adaln_outputs['temb'].max():.4f}]")
    
    # Check how temb is reshaped for blocks
    temb = adaln_outputs['temb']
    hidden_size = 2048
    
    # Diffusers reshapes: temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
    # But first check temb shape
    print(f"\n  temb.size(1) = {temb.size(1)}")
    
    # In LTXVideoTransformerBlock, temb comes already reshaped
    # Let's see what the block expects
    print("\n" + "=" * 60)
    print("Expected flow in Rust")
    print("=" * 60)
    print("""
Rust AdaLayerNormSingle.forward() returns:
  - out: (batch, 1, 6*hidden_size) = (1, 1, 12288)
  - embedded_timestep: (batch, hidden_size) = (1, 2048)

But diffusers seems to have different shape.
""")
    
    # Check the actual diffusers adaln_single structure
    print("\nDiffusers adaln_single structure:")
    print(f"  Type: {type(transformer.adaln_single)}")

if __name__ == '__main__':
    main()
