#!/usr/bin/env python3
"""Compare DiT single forward pass between Rust and Python reference."""

import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
from safetensors.torch import load_file

def main():
    print("=== DiT Forward Pass Comparison ===\n")
    
    # Create deterministic inputs
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Input dimensions matching Rust
    B, C, T, H, W = 1, 128, 1, 16, 24
    
    # Random latent input
    latents = torch.randn(B, C, T, H, W)
    
    # Random text embeddings  
    text_emb = torch.randn(B, 6, 4096)
    
    # Timestep = 1.0 (first step)
    timestep = torch.tensor([1.0])
    
    print(f"Input latents: {latents.shape}, range [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"Text embeddings: {text_emb.shape}")
    print(f"Timestep: {timestep.item()}")
    
    # Load transformer from safetensors
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    print(f"\nLoading transformer from {model_path}...")
    
    try:
        # Load from single file
        transformer = LTXVideoTransformer3DModel.from_single_file(
            model_path,
            torch_dtype=torch.float32,
        )
        print("Transformer loaded!")
        print(f"Config: hidden={getattr(transformer.config, 'num_attention_heads', 'N/A')} heads")
    except Exception as e:
        print(f"Failed to load: {e}")
        
        # Try manual loading
        print("\nTrying manual weight analysis...")
        weights = load_file(model_path)
        
        # Find DiT weights
        dit_weights = {k: v for k, v in weights.items() if 'diffusion_model' in k}
        print(f"Found {len(dit_weights)} DiT weights")
        
        # Check patchify_proj
        key = 'model.diffusion_model.patchify_proj.weight'
        if key in weights:
            w = weights[key]
            print(f"\npatchify_proj weight: {w.shape}")
            print(f"  Expected in_channels: 128 (latent channels)")
            print(f"  Actual in_features: {w.shape[1]}")
            
        # Check a few attention weights
        for k in list(dit_weights.keys())[:5]:
            print(f"  {k}: {dit_weights[k].shape}")
        
        return
    
    # Pack latents (patchify) - transformer expects [B, seq_len, C*patch^3]
    def pack_latents(latents, patch_size=1, patch_size_t=1):
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size, -1,
            post_patch_num_frames, patch_size_t,
            post_patch_height, patch_size,
            post_patch_width, patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents
    
    # Get patch sizes from config
    patch_size = transformer.config.patch_size
    patch_size_t = transformer.config.patch_size_t
    print(f"Patch sizes: spatial={patch_size}, temporal={patch_size_t}")
    
    packed_latents = pack_latents(latents, patch_size, patch_size_t)
    print(f"Packed latents: {packed_latents.shape}")
    
    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        # Need to specify dimensions for RoPE
        output = transformer(
            hidden_states=packed_latents,
            encoder_hidden_states=text_emb,
            timestep=timestep,
            encoder_attention_mask=torch.ones(B, 6),
            num_frames=T,
            height=H * patch_size,  # Original pixel dimensions
            width=W * patch_size,
            return_dict=False,
        )[0]
    
    print(f"\nOutput: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Save for Rust comparison
    np.save('output/dit_input_latents.npy', latents.numpy())
    np.save('output/dit_input_text.npy', text_emb.numpy())
    np.save('output/dit_output_ref.npy', output.numpy())
    print("\nSaved reference inputs/outputs to output/")

if __name__ == '__main__':
    main()
