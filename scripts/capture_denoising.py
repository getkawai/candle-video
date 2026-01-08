"""
Capture full denoising state from Diffusers for comparison with Rust.
Saves: initial_latents, final_latents (before VAE decode), and embeddings.
"""
import torch
from diffusers import LTXPipeline
from safetensors.torch import save_file
import os
import argparse

def capture_denoising(prompt: str, output_dir: str = "reference_output"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5"
    
    print(f"Loading pipeline from {model_path}...")
    pipeline = LTXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    pipeline.enable_model_cpu_offload()

    seed = 42
    height, width, num_frames = 512, 768, 25
    steps = 30
    guidance_scale = 3.0
    
    print(f"Running full pipeline with seed {seed}...")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Run full pipeline but get latents output
    result = pipeline(
        prompt=prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",  # Get final latents before VAE decode
    )
    
    final_latents = result.frames  # This is the final latents in packed format
    
    print(f"Final latents shape: {final_latents.shape}")
    print(f"Final latents mean: {final_latents.mean().item():.6f}")
    print(f"Final latents std: {final_latents.std().item():.6f}")
    print(f"Final latents min/max: {final_latents.min().item():.4f} / {final_latents.max().item():.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    save_file({
        "final_latents": final_latents.cpu().to(torch.float32).contiguous(),
    }, f"{output_dir}/denoising_output.safetensors")
    print(f"Saved {output_dir}/denoising_output.safetensors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reference_output")
    args = parser.parse_args()
    capture_denoising(args.prompt, args.output_dir)
