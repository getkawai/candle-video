#!/usr/bin/env python3
"""Debug DiT - compare single forward pass with reference."""

import numpy as np
import torch
import os, sys

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
from safetensors.torch import load_file

def main():
    print("Loading LTX transformer...")
    
    # Load weights
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    all_weights = load_file(model_path)
    
    # Extract transformer weights
    transformer_weights = {k.replace('model.diffusion_model.', ''): v 
                          for k, v in all_weights.items() 
                          if k.startswith('model.diffusion_model.')}
    
    print(f"Found {len(transformer_weights)} transformer weights")
    
    # Create random input latents like Rust does
    torch.manual_seed(42)
    latents = torch.randn(1, 128, 2, 16, 24)
    
    # Create dummy text embeddings
    encoder_hidden_states = torch.randn(1, 6, 4096)
    encoder_attention_mask = torch.ones(1, 6)
    
    # Timestep
    timestep = torch.tensor([1.0])  # First step
    
    print(f"Latent shape: {latents.shape}")
    print(f"Latent range: [{latents.min():.4f}, {latents.max():.4f}]")
    
    # Try to create and run transformer
    # First check what config the model needs
    print("\nSample weight keys:")
    for k in list(transformer_weights.keys())[:10]:
        print(f"  {k}: {transformer_weights[k].shape}")

if __name__ == '__main__':
    main()
