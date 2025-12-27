#!/usr/bin/env python3
"""Debug DiT intermediate values."""

import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

def main():
    print("=== DiT Debug - Intermediate Values ===\n")
    
    torch.manual_seed(42)
    
    B, C, T, H, W = 1, 128, 1, 16, 24
    latents = torch.randn(B, C, T, H, W)
    text_emb = torch.randn(B, 6, 4096)
    timestep = torch.tensor([1.0])
    
    # Load model
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        torch_dtype=torch.float32,
    )
    
    # Pack latents
    def pack_latents(latents, patch_size=1, patch_size_t=1):
        batch_size, num_channels, num_frames, height, width = latents.shape
        latents = latents.reshape(
            batch_size, -1,
            num_frames // patch_size_t, patch_size_t,
            height // patch_size, patch_size,
            width // patch_size, patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents
    
    packed_latents = pack_latents(latents)
    print(f"Packed latents: {packed_latents.shape}, range [{packed_latents.min():.4f}, {packed_latents.max():.4f}]")
    
    # manual forward with debug
    batch_size = packed_latents.size(0)
    
    # proj_in
    hidden = transformer.proj_in(packed_latents)
    print(f"\nAfter proj_in: {hidden.shape}")
    print(f"  range: [{hidden.min():.4f}, {hidden.max():.4f}]")
    print(f"  mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
    
    # time_embed
    temb, emb_ts = transformer.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden.dtype,
    )
    temb = temb.view(batch_size, -1, temb.size(-1))
    print(f"\ntemb: {temb.shape}")
    print(f"  range: [{temb.min():.4f}, {temb.max():.4f}]")
    print(f"  mean: {temb.mean():.6f}, std: {temb.std():.6f}")
    
    # caption_projection
    caption = transformer.caption_projection(text_emb)
    caption = caption.view(batch_size, -1, hidden.size(-1))
    print(f"\ncaption: {caption.shape}")
    print(f"  range: [{caption.min():.4f}, {caption.max():.4f}]")
    
    # Check scale_shift_table
    print(f"\nscale_shift_table: {transformer.scale_shift_table.shape}")
    print(f"  range: [{transformer.scale_shift_table.min():.4f}, {transformer.scale_shift_table.max():.4f}]")
    
    # First block check
    block = transformer.transformer_blocks[0]
    print(f"\nFirst block scale_shift_table: {block.scale_shift_table.shape}")
    print(f"  range: [{block.scale_shift_table.min():.4f}, {block.scale_shift_table.max():.4f}]")
    
    # Check ada_values computation
    num_ada_params = block.scale_shift_table.shape[0]  # 6
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch_size, temb.size(1), num_ada_params, -1
    )
    print(f"\nada_values: {ada_values.shape}")
    print(f"  range: [{ada_values.min():.4f}, {ada_values.max():.4f}]")
    
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
    print(f"\nshift_msa: {shift_msa.shape}, range [{shift_msa.min():.4f}, {shift_msa.max():.4f}]")
    print(f"scale_msa: {scale_msa.shape}, range [{scale_msa.min():.4f}, {scale_msa.max():.4f}]")
    print(f"gate_msa: {gate_msa.shape}, range [{gate_msa.min():.4f}, {gate_msa.max():.4f}]")
    
    # First block forward
    norm_hidden = block.norm1(hidden)
    print(f"\nnorm1 output: {norm_hidden.shape}")
    print(f"  range: [{norm_hidden.min():.4f}, {norm_hidden.max():.4f}]")
    print(f"  mean: {norm_hidden.mean():.6f}, std: {norm_hidden.std():.6f}")
    
    # Modulation
    modulated = norm_hidden * (1 + scale_msa) + shift_msa
    print(f"\nAfter modulation: {modulated.shape}")
    print(f"  range: [{modulated.min():.4f}, {modulated.max():.4f}]")
    print(f"  mean: {modulated.mean():.6f}, std: {modulated.std():.6f}")

if __name__ == '__main__':
    main()
