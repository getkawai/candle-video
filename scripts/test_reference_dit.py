#!/usr/bin/env python3
"""Test generating latents using reference LTX-Video transformer and compare with Rust."""

import numpy as np
import torch
import os
import sys

# Force use local diffusers
sys.path.insert(0, os.path.abspath('tp/diffusers/src'))
import diffusers
print(f"Using diffusers from: {diffusers.__file__}")

from diffusers import LTXPipeline
from safetensors.torch import load_file

def main():
    print("Loading LTX-Video pipeline...")
    
    # Try to load the full pipeline
    model_path = 'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors'
    
    try:
        pipe = LTXPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float32,
        )
        print("Pipeline loaded!")
        
        # Generate with same parameters as Rust
        prompt = "A beautiful sunset over mountains"
        
        result = pipe(
            prompt=prompt,
            num_frames=9,
            height=512,
            width=768,
            num_inference_steps=8,
            guidance_scale=7.5,
            output_type="latent",
        )
        
        latents = result.frames
        print(f"Generated latents: shape={latents.shape}, range=[{latents.min():.4f}, {latents.max():.4f}]")
        
        # Compare with Rust latents
        with open('output/latents.bin', 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        rust_latents = data[12:].reshape(1, 128, 2, 16, 24)
        print(f"Rust latents: range=[{rust_latents.min():.4f}, {rust_latents.max():.4f}]")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
