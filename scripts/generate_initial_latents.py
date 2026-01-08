"""
Generate initial latents and save them for both Python and Rust pipelines.
This ensures both pipelines start with identical noise.
"""
import torch
from safetensors.torch import save_file
import os
import argparse

def generate_initial_latents(
    seed: int = 42,
    height: int = 512,
    width: int = 768,
    num_frames: int = 25,
    output_path: str = "reference_output/initial_latents.safetensors"
):
    # LTX Video config
    vae_spatial_compression = 32
    vae_temporal_compression = 8
    in_channels = 128  # Transformer in_channels

    latent_height = height // vae_spatial_compression
    latent_width = width // vae_spatial_compression
    latent_frames = (num_frames - 1) // vae_temporal_compression + 1
    
    print(f"Generating latents with seed {seed}")
    print(f"Shape: [1, {in_channels}, {latent_frames}, {latent_height}, {latent_width}]")
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # [B, C, F, H, W] - unpacked format
    latents_unpacked = torch.randn(
        (1, in_channels, latent_frames, latent_height, latent_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32
    )
    
    # Pack latents like LTX pipeline does
    # [B, C, F, H, W] -> [B, S, D] where S = F*H*W, D = C
    b, c, f, h, w = latents_unpacked.shape
    patch_size = 1
    patch_size_t = 1
    
    # For patch_size=1, packing is just a reshape
    # [B, C, F, H, W] -> [B, F*H*W, C]
    f2 = f // patch_size_t
    h2 = h // patch_size
    w2 = w // patch_size
    
    # Reshape and permute similar to diffusers
    x = latents_unpacked.reshape(b, c, f2, patch_size_t, h2, patch_size, w2, patch_size)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B, F2, H2, W2, C, pt, p, p]
    x = x.flatten(4)  # [B, F2, H2, W2, D]
    s = f2 * h2 * w2
    d = x.shape[-1]
    latents_packed = x.reshape(b, s, d)  # [B, S, D]
    
    print(f"Unpacked shape: {latents_unpacked.shape}")
    print(f"Packed shape: {latents_packed.shape}")
    print(f"Latents mean: {latents_packed.mean().item():.6f}")
    print(f"Latents std: {latents_packed.std().item():.6f}")
    print(f"Latents[0,0,:5]: {latents_packed[0,0,:5]}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    save_file({
        "initial_latents": latents_packed.contiguous(),  # Packed format for transformer
        "initial_latents_unpacked": latents_unpacked.contiguous(),  # Unpacked for VAE
    }, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--output", type=str, default="reference_output/initial_latents.safetensors")
    args = parser.parse_args()
    generate_initial_latents(args.seed, args.height, args.width, args.num_frames, args.output)
