#!/usr/bin/env python3
"""Run full LTX pipeline to get reference latents."""

import numpy as np
import torch
import os, sys

sys.path.insert(0, os.path.abspath('tp/diffusers/src'))
from diffusers import LTXPipeline

def main():
    print("Loading LTX-Video pipeline...")
    
    try:
        pipe = LTXPipeline.from_single_file(
            'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors',
            torch_dtype=torch.float32,
        )
        print("Pipeline loaded!")
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        print("\nTrying alternative approach - loading just transformer and VAE...")
        
        from diffusers import AutoencoderKLLTXVideo
        from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel
        
        # Load VAE  
        vae = AutoencoderKLLTXVideo.from_single_file(
            'ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors',
            torch_dtype=torch.float32,
        )
        print("VAE loaded!")
        
        # Create simple test - encode a frame then decode
        print("\nTesting VAE encode/decode cycle...")
        
        # Create random "video" frame
        torch.manual_seed(42)
        video = torch.randn(1, 3, 9, 512, 768) * 0.5  # Small random values
        video = video.clamp(-1, 1)
        
        print(f"Input video: shape={video.shape}, range=[{video.min():.4f}, {video.max():.4f}]")
        
        with torch.no_grad():
            # Encode
            latents = vae.encode(video).latent_dist.sample()
            print(f"Encoded latents: shape={latents.shape}, range=[{latents.min():.4f}, {latents.max():.4f}]")
            
            # Decode back
            decoded = vae.decode(latents, return_dict=False)[0]
            print(f"Decoded video: shape={decoded.shape}, range=[{decoded.min():.4f}, {decoded.max():.4f}]")
        
        # Save decoded frame
        from PIL import Image
        frame = decoded[0].permute(1, 2, 3, 0).numpy()[0]
        frame = ((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).save('output/vae_roundtrip.png')
        print("Saved output/vae_roundtrip.png")
        
        return
    
    # If pipeline loaded, generate
    prompt = "A beautiful sunset over mountains"
    
    # Get just latents (output_type="latent")
    result = pipe(
        prompt=prompt,
        num_frames=9,
        height=512,
        width=768,
        num_inference_steps=8,
        guidance_scale=7.5,
        generator=torch.Generator().manual_seed(42),
        output_type="latent",
    )
    
    latents = result.frames
    print(f"Reference latents: shape={latents.shape}, range=[{latents.min():.4f}, {latents.max():.4f}]")
    
    # Save for comparison
    np.save('output/reference_latents.npy', latents.numpy())
    print("Saved output/reference_latents.npy")

if __name__ == '__main__':
    main()
