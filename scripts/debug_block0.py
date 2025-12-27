#!/usr/bin/env python3
"""
Compare first transformer block output to isolate discrepancy.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    
    print("=" * 60)
    print("First Block Output Comparison")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    B, C, T, H, W = 1, 128, 1, 16, 24
    latents = torch.randn(B, C, T, H, W)
    text_emb = torch.randn(B, 8, 4096)
    timestep = torch.tensor([1.0])
    
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    transformer = LTXVideoTransformer3DModel.from_single_file(
        model_path,
        torch_dtype=torch.float32,
    )
    
    # Register detailed hooks
    block0_inputs = {}
    block0_outputs = {}
    
    def hook_input(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                block0_inputs[name] = input[0].detach().clone()
            elif not isinstance(input, tuple):
                block0_inputs[name] = input.detach().clone()
        return hook
    
    def hook_output(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                block0_outputs[name] = output[0].detach().clone()
            else:
                block0_outputs[name] = output.detach().clone()
        return hook
    
    hooks = []
    block0 = transformer.transformer_blocks[0]
    
    # Hook norm layers
    hooks.append(block0.norm1.register_forward_hook(hook_output('norm1')))
    hooks.append(block0.norm2.register_forward_hook(hook_output('norm2')))
    
    # Hook attention
    hooks.append(block0.attn1.to_q.register_forward_hook(hook_output('attn1_q')))
    hooks.append(block0.attn1.to_k.register_forward_hook(hook_output('attn1_k')))
    hooks.append(block0.attn1.norm_q.register_forward_hook(hook_output('attn1_norm_q')))
    hooks.append(block0.attn1.norm_k.register_forward_hook(hook_output('attn1_norm_k')))
    
    # Hook the full attention output
    hooks.append(block0.attn1.register_forward_hook(hook_output('attn1_out')))
    hooks.append(block0.attn2.register_forward_hook(hook_output('attn2_out')))
    hooks.append(block0.ff.register_forward_hook(hook_output('ff_out')))
    hooks.append(block0.register_forward_hook(hook_output('block0_out')))
    
    # Also hook the input to the first block
    hooks.append(block0.register_forward_hook(hook_input('block0_in')))
    
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
    
    # Run forward pass
    print("\nRunning forward pass...")
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
    
    # Print and save intermediate outputs
    os.makedirs('output/debug', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Block 0 Intermediate Values")
    print("=" * 60)
    
    for name, tensor in block0_outputs.items():
        t = tensor.cpu().numpy()
        print(f"\n{name}:")
        print(f"  Shape: {t.shape}")
        print(f"  First 5: {t.flatten()[:5]}")
        print(f"  Range: [{t.min():.4f}, {t.max():.4f}]")
        print(f"  Mean: {t.mean():.4f}, Std: {t.std():.4f}")
        np.save(f'output/debug/{name}.npy', t)
    
    print("\n✓ Saved to output/debug/")
    
    # Key values for comparison
    print("\n" + "=" * 60)
    print("Key Values for Rust Comparison")
    print("=" * 60)
    
    if 'attn1_norm_q' in block0_outputs:
        q = block0_outputs['attn1_norm_q']
        print(f"\nattn1 Q after norm [0,0,:5]: {q[0,0,:5].tolist()}")
    
    if 'attn1_out' in block0_outputs:
        a = block0_outputs['attn1_out']
        print(f"attn1 output [0,0,:5]: {a[0,0,:5].tolist()}")

if __name__ == '__main__':
    main()
