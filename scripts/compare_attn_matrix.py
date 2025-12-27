#!/usr/bin/env python3
"""
Deep matrix operation comparison - compare Q@K.T attention scores.

Hooks into the attention processor to capture exact attention weights.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def main():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    
    print("=" * 60)
    print("Deep Attention Matrix Comparison")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    transformer = LTXVideoTransformer3DModel.from_single_file(
        'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors',
        torch_dtype=torch.float32,
    )
    
    # We need to patch the attention processor to capture intermediate values
    block0_attn1 = transformer.transformer_blocks[0].attn1
    
    # Store original processor
    original_processor = block0_attn1.processor
    
    # Create a capturing processor
    captured = {}
    
    class CapturingProcessor:
        def __init__(self, original):
            self.original = original
            
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, 
                     attention_mask=None, image_rotary_emb=None, **kwargs):
            from diffusers.models.transformers.transformer_ltx import apply_rotary_emb
            
            batch_size, seq_len, _ = hidden_states.shape
            
            # Get Q, K, V
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            
            # Normalize
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            
            captured['query_after_norm'] = query.detach().clone()
            captured['key_after_norm'] = key.detach().clone()
            
            # Apply RoPE
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            
            captured['query_after_rope'] = query.detach().clone()
            captured['key_after_rope'] = key.detach().clone()
            
            # Reshape for multi-head attention
            query = query.unflatten(2, (attn.heads, -1))  # (B, S, H, D)
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            
            captured['query_multihead'] = query.detach().clone()
            captured['key_multihead'] = key.detach().clone()
            
            # Transpose for attention: (B, H, S, D)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Scaled dot-product attention
            scale = attn.head_dim ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            captured['attn_weights_raw'] = attn_weights.detach().clone()
            
            # Softmax
            attn_probs = torch.softmax(attn_weights, dim=-1)
            captured['attn_probs'] = attn_probs.detach().clone()
            
            # Apply to values
            out = torch.matmul(attn_probs, value)
            captured['attn_output_before_proj'] = out.detach().clone()
            
            # Reshape back
            out = out.transpose(1, 2).flatten(2, 3)
            
            # Output projection
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            
            captured['attn_output_final'] = out.detach().clone()
            
            return out
    
    # Replace processor
    block0_attn1.processor = CapturingProcessor(original_processor)
    
    # Run forward
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
    
    # Restore
    block0_attn1.processor = original_processor
    
    # Print results
    print("\n" + "=" * 60)
    print("Captured Attention Intermediate Values")
    print("=" * 60)
    
    for name, tensor in captured.items():
        t = tensor.cpu().numpy()
        print(f"\n{name}:")
        print(f"  Shape: {t.shape}")
        print(f"  Range: [{t.min():.6f}, {t.max():.6f}]")
        print(f"  Mean: {t.mean():.6f}, Std: {t.std():.6f}")
        if len(t.shape) <= 3:
            print(f"  First 5: {t.flatten()[:5]}")
        else:
            print(f"  [0,0,0,:5]: {t[0,0,0,:5] if t.shape[3] >= 5 else t[0,0,0,:]}")
        
        # Save
        os.makedirs('output/debug', exist_ok=True)
        np.save(f'output/debug/attn_matrix_{name}.npy', t)
    
    print("\n✓ Saved all attention matrices to output/debug/attn_matrix_*.npy")
    
    # Key check - attention weights should sum to 1 along last dim after softmax
    probs = captured['attn_probs']
    sum_check = probs.sum(dim=-1).mean().item()
    print(f"\nAttention probs sum check (should be 1.0): {sum_check:.6f}")

if __name__ == '__main__':
    main()
