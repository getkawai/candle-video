#!/usr/bin/env python3
"""
Compare LTX-Video DiT outputs: Rust vs diffusers reference.

This script:
1. Loads the diffusers LTXVideoTransformer3DModel
2. Runs a forward pass with deterministic inputs
3. Saves tensors for Rust comparison
4. Optionally compares with existing Rust outputs
"""

import os
import sys
import numpy as np
import torch

# Add diffusers source to path
sys.path.insert(0, os.path.abspath('tp/diffusers/src'))

def create_deterministic_inputs(device='cpu', dtype=torch.float32):
    """Create deterministic inputs for reproducible comparison."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Latent space dimensions (small for testing)
    B, C, T, H, W = 1, 128, 1, 16, 24
    
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    text_emb = torch.randn(B, 8, 4096, device=device, dtype=dtype)  # 8 tokens
    timestep = torch.tensor([1.0], device=device, dtype=dtype)
    
    return latents, text_emb, timestep, (B, C, T, H, W)

def run_diffusers_forward(model_path: str):
    """Run forward pass through diffusers model."""
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
    from safetensors.torch import load_file
    
    print("=" * 60)
    print("LTX-Video DiT Comparison: diffusers Reference")
    print("=" * 60)
    
    latents, text_emb, timestep, dims = create_deterministic_inputs()
    B, C, T, H, W = dims
    
    print(f"\nInputs:")
    print(f"  Latents: {latents.shape}")
    print(f"  Text: {text_emb.shape}")
    print(f"  Timestep: {timestep.item()}")
    
    # Try to load the transformer
    print(f"\nLoading model from: {model_path}")
    
    try:
        transformer = LTXVideoTransformer3DModel.from_single_file(
            model_path,
            torch_dtype=torch.float32,
        )
        print(f"✓ Model loaded successfully")
        
        patch_size = transformer.config.patch_size
        patch_size_t = transformer.config.patch_size_t
        print(f"  Patch size: spatial={patch_size}, temporal={patch_size_t}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nFalling back to weight analysis...")
        analyze_weights(model_path)
        return None
    
    # Pack latents
    def pack_latents(x, ps, ps_t):
        B, C, T, H, W = x.shape
        pT, pH, pW = T // ps_t, H // ps, W // ps
        x = x.reshape(B, C, pT, ps_t, pH, ps, pW, ps)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return x
    
    packed = pack_latents(latents, patch_size, patch_size_t)
    print(f"\nPacked latents: {packed.shape}")
    
    # Forward pass
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
    
    print(f"\nOutput: {output.shape}")
    print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    # Save outputs
    os.makedirs('output', exist_ok=True)
    np.save('output/dit_latents.npy', latents.numpy())
    np.save('output/dit_text_emb.npy', text_emb.numpy())
    np.save('output/dit_output_ref.npy', output.numpy())
    print("\n✓ Saved reference outputs to output/")
    
    return output

def analyze_weights(model_path: str):
    """Analyze model weights without loading full model."""
    from safetensors.torch import load_file
    
    weights = load_file(model_path)
    
    dit_weights = {k: v for k, v in weights.items() if 'diffusion_model' in k}
    vae_weights = {k: v for k, v in weights.items() if 'vae' in k}
    
    print(f"\nWeight analysis:")
    print(f"  DiT weights: {len(dit_weights)}")
    print(f"  VAE weights: {len(vae_weights)}")
    
    key = 'model.diffusion_model.patchify_proj.weight'
    if key in weights:
        w = weights[key]
        print(f"\n  patchify_proj.weight: {w.shape}")
        print(f"    in_channels={w.shape[1]}, hidden_size={w.shape[0]}")

def compare_with_rust(rust_output_path: str):
    """Compare diffusers output with Rust output."""
    if not os.path.exists('output/dit_output_ref.npy'):
        print("No reference output found. Run with --reference first.")
        return
    
    if not os.path.exists(rust_output_path):
        print(f"Rust output not found: {rust_output_path}")
        return
    
    ref = np.load('output/dit_output_ref.npy')
    rust = np.load(rust_output_path)
    
    print("\n" + "=" * 60)
    print("Comparison: diffusers vs Rust")
    print("=" * 60)
    
    print(f"\nShapes:")
    print(f"  Reference: {ref.shape}")
    print(f"  Rust: {rust.shape}")
    
    if ref.shape != rust.shape:
        print("✗ Shape mismatch!")
        return
    
    diff = np.abs(ref - rust)
    rel_diff = diff / (np.abs(ref) + 1e-8)
    
    print(f"\nAbsolute difference:")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std: {diff.std():.6f}")
    
    print(f"\nRelative difference:")
    print(f"  Max: {rel_diff.max():.6f}")
    print(f"  Mean: {rel_diff.mean():.6f}")
    
    threshold = 1e-4
    if diff.max() < threshold:
        print(f"\n✓ PASS: Max diff {diff.max():.2e} < {threshold}")
    else:
        print(f"\n✗ FAIL: Max diff {diff.max():.2e} >= {threshold}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare DiT outputs')
    parser.add_argument('--model', default='ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors',
                       help='Path to model safetensors')
    parser.add_argument('--rust-output', help='Path to Rust output .npy for comparison')
    parser.add_argument('--reference', action='store_true', help='Generate reference outputs')
    args = parser.parse_args()
    
    if args.reference or not args.rust_output:
        run_diffusers_forward(args.model)
    
    if args.rust_output:
        compare_with_rust(args.rust_output)

if __name__ == '__main__':
    main()
