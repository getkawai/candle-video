"""
Capture prompt embeddings from Diffusers for use in Rust inference.
This allows testing the Rust pipeline with "known-good" embeddings.
"""
import torch
from diffusers import LTXPipeline
from safetensors.torch import save_file
import os
import argparse

def capture_embeddings(prompt: str, negative_prompt: str = "", output_path: str = "embeddings.safetensors"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5"
    
    print(f"Loading pipeline from {model_path}...")
    pipeline = LTXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    
    # Move text encoder to device for encoding
    pipeline.text_encoder.to(device)
    
    print(f"Encoding prompt: '{prompt[:50]}...'")
    
    # Use the pipeline's encode_prompt method
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=128,
        device=device,
        dtype=dtype,
    )
    
    print(f"Prompt embeds shape: {prompt_embeds.shape}")
    print(f"Prompt mask shape: {prompt_attention_mask.shape}")
    print(f"Negative embeds shape: {negative_prompt_embeds.shape}")
    print(f"Negative mask shape: {negative_prompt_attention_mask.shape}")
    
    # Save to safetensors
    save_data = {
        "prompt_embeds": prompt_embeds.cpu().to(torch.float32).contiguous(),
        "prompt_attention_mask": prompt_attention_mask.cpu().to(torch.float32).contiguous(),
        "negative_prompt_embeds": negative_prompt_embeds.cpu().to(torch.float32).contiguous(),
        "negative_prompt_attention_mask": negative_prompt_attention_mask.cpu().to(torch.float32).contiguous(),
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    save_file(save_data, output_path)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture prompt embeddings from Diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to encode")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, default="reference_output/embeddings.safetensors", help="Output file path")
    
    args = parser.parse_args()
    capture_embeddings(args.prompt, args.negative_prompt, args.output)
