#!/usr/bin/env python3
"""
Debug DiT layer-by-layer to find the source of numerical discrepancy.

This script hooks into intermediate layers of diffusers transformer
and saves outputs for comparison with Rust.
"""

import os
import sys
import numpy as np
import torch
import struct

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def save_tensor_bin(tensor: np.ndarray, path: str):
    """Save tensor in binary format for Rust."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(tensor.shape)))
        for d in tensor.shape:
            f.write(struct.pack('<Q', d))
        tensor.astype(np.float32).tofile(f)

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    
    print("=" * 60)
    print("DiT Debug: Layer-by-Layer Analysis")
    print("=" * 60)
    
    # Deterministic inputs
    torch.manual_seed(42)
    np.random.seed(42)
    
    B, C, T, H, W = 1, 128, 1, 16, 24
    latents = torch.randn(B, C, T, H, W)
    text_emb = torch.randn(B, 8, 4096)
    timestep = torch.tensor([1.0])
    
    # Load model
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    print(f"\nLoading model from: {model_path}")
    
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        torch_dtype=torch.float32,
    )
    print("✓ Model loaded")
    
    # Pack latents
    patch_size = transformer.config.patch_size
    patch_size_t = transformer.config.patch_size_t
    
    def pack_latents(x, ps, ps_t):
        B, C, T, H, W = x.shape
        pT, pH, pW = T // ps_t, H // ps, W // ps
        x = x.reshape(B, C, pT, ps_t, pH, ps, pW, ps)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return x
    
    packed = pack_latents(latents, patch_size, patch_size_t)
    
    # Collect intermediate outputs
    intermediates = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            intermediates[name] = out.detach().clone()
        return hook
    
    # Register hooks on key components
    hooks = []
    
    # 1. After patchify projection
    hooks.append(transformer.proj_in.register_forward_hook(hook_fn('proj_in')))
    
    # 2. After RoPE
    hooks.append(transformer.rope.register_forward_hook(hook_fn('rope')))
    
    # 3. After timestep embedding
    hooks.append(transformer.time_embed.register_forward_hook(hook_fn('time_embed')))
    
    # 4. After caption projection
    hooks.append(transformer.caption_projection.register_forward_hook(hook_fn('caption_proj')))
    
    # 5. First transformer block
    if hasattr(transformer, 'transformer_blocks') and len(transformer.transformer_blocks) > 0:
        block0 = transformer.transformer_blocks[0]
        hooks.append(block0.attn1.register_forward_hook(hook_fn('block0_attn1')))
        hooks.append(block0.attn2.register_forward_hook(hook_fn('block0_attn2')))
        hooks.append(block0.ff.register_forward_hook(hook_fn('block0_ff')))
        hooks.append(block0.register_forward_hook(hook_fn('block0_out')))
    
    # Run forward pass
    print("\nRunning forward pass with hooks...")
    with torch.no_grad():
        output = transformer(
            hidden_states=packed,
            encoder_hidden_states=text_emb,
            timestep=timestep,
            encoder_attention_mask=torch.ones(B, text_emb.shape[1]),
            num_frames=T,
            height=H,
            width=W,
            return_dict=False,
        )[0]
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Save and print intermediate values
    os.makedirs('output/debug', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Intermediate Layer Statistics")
    print("=" * 60)
    
    for name, tensor in intermediates.items():
        t = tensor.cpu().numpy()
        print(f"\n{name}:")
        print(f"  Shape: {t.shape}")
        print(f"  Range: [{t.min():.4f}, {t.max():.4f}]")
        print(f"  Mean: {t.mean():.4f}, Std: {t.std():.4f}")
        
        save_tensor_bin(t, f'output/debug/{name}.bin')
    
    # Final output
    print(f"\nFinal output:")
    print(f"  Shape: {output.shape}")
    print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    # Save final output
    np.save('output/debug/final_output.npy', output.numpy())
    
    # Check specific values for debugging
    print("\n" + "=" * 60)
    print("Key Values to Compare with Rust")
    print("=" * 60)
    
    # RoPE frequencies
    if 'rope' in intermediates:
        rope_out = intermediates['rope']
        if isinstance(rope_out, tuple):
            cos_freqs, sin_freqs = rope_out
            print(f"\nRoPE cos_freqs[0,0,:10]: {cos_freqs[0,0,:10].tolist()}")
            print(f"RoPE sin_freqs[0,0,:10]: {sin_freqs[0,0,:10].tolist()}")
    
    # Timestep embedding
    if 'time_embed' in intermediates:
        temb = intermediates['time_embed']
        print(f"\nTimestep embed[0,:10]: {temb[0,:10].tolist()}")
    
    print("\n✓ Debug data saved to output/debug/")

if __name__ == '__main__':
    main()
