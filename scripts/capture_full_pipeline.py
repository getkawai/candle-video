"""
Capture full pipeline outputs from diffusers for direct comparison.
This captures:
1. Initial latents (with known seed)
2. Final latents after denoising
3. All intermediate states
"""
import torch
from diffusers import LTXPipeline, FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
import os

def capture_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5"
    
    print(f"Loading pipeline from {model_path}...")
    pipeline = LTXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    pipeline.enable_model_cpu_offload()

    prompt = "A woman with blood on her face"
    seed = 42
    
    print(f"Generating with seed {seed}...")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Capture initial latents
    height, width, num_frames = 512, 768, 25
    
    # Prepare latents manually
    vae_scale_factor_spatial = 32
    vae_scale_factor_temporal = 8
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    
    initial_latents = torch.randn(
        (1, 128, latent_frames, latent_height, latent_width),
        generator=generator,
        device="cpu",
        dtype=dtype
    )
    print(f"Initial latents shape: {initial_latents.shape}")
    print(f"Initial latents mean: {initial_latents.mean().item():.6f}")
    # Run pipeline normally
    video = pipeline(
        prompt=prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=3.0,
        generator=generator,
        output_type="latent"  # Get latents, not decoded video
    ).frames
    
    print(f"Final latents shape: {video.shape}")
    print(f"Final latents mean: {video.mean().item():.6f}")
    
    os.makedirs("reference_output", exist_ok=True)
    save_file({
        "final_latents": video.cpu().to(torch.float32).contiguous(),
    }, "reference_output/full_pipeline.safetensors")
    print("Saved reference_output/full_pipeline.safetensors")

if __name__ == "__main__":
    capture_pipeline()
