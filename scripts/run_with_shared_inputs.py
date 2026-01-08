"""
Run Diffusers inference with the same embeddings and initial latents as Rust.
This allows direct comparison of the final results.
"""
import torch
from diffusers import LTXPipeline
from safetensors.torch import load_file
from PIL import Image
import os
import argparse

def run_with_shared_inputs(
    embeddings_file: str,
    latents_file: str,
    output_path: str = "output_diffusers.gif"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5"
    
    # Load shared inputs
    print(f"Loading embeddings from {embeddings_file}...")
    emb_tensors = load_file(embeddings_file)
    prompt_embeds = emb_tensors["prompt_embeds"].to(device).to(dtype)
    prompt_attention_mask = emb_tensors["prompt_attention_mask"].to(device).to(torch.bool)
    negative_prompt_embeds = emb_tensors["negative_prompt_embeds"].to(device).to(dtype)
    negative_prompt_attention_mask = emb_tensors["negative_prompt_attention_mask"].to(device).to(torch.bool)
    
    print(f"Loading initial latents from {latents_file}...")
    lat_tensors = load_file(latents_file)
    # Use UNPACKED format [B, C, F, H, W] - that's what Diffusers expects
    if "initial_latents_unpacked" in lat_tensors:
        initial_latents = lat_tensors["initial_latents_unpacked"].to(device).to(dtype)
        print(f"Using unpacked latents: {initial_latents.shape}")
    else:
        initial_latents = lat_tensors["initial_latents"].to(device).to(dtype)
        print(f"Using packed latents: {initial_latents.shape}")
    
    print(f"Loading pipeline from {model_path}...")
    pipeline = LTXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    pipeline.enable_model_cpu_offload()

    height, width, num_frames = 512, 768, 25
    steps = 30
    guidance_scale = 3.0
    seed = 42
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    print(f"Running inference with shared embeddings and seed {seed}...")
    # Use prompt_embeds instead of prompt string
    result = pipeline(
        prompt=None,  # Not needed when prompt_embeds is provided
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,  # Use generator instead of custom latents
        output_type="pil",  # Get PIL images
    )
    
    frames = result.frames[0]  # List of PIL images
    print(f"Generated {len(frames)} frames")
    
    # Save as GIF
    print(f"Saving to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="reference_output/embeddings.safetensors")
    parser.add_argument("--latents", type=str, default="reference_output/initial_latents.safetensors")
    parser.add_argument("--output", type=str, default="output/diffusers_video.gif")
    args = parser.parse_args()
    run_with_shared_inputs(args.embeddings, args.latents, args.output)
