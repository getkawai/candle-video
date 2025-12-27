"""
Test Rust latents with Python VAE

This script loads the latents saved by Rust DiT and decodes them with
Python diffusers VAE to isolate if the problem is in DiT or VAE.
"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Load Rust latents from binary file
latents_path = Path("output/latents.bin")
if not latents_path.exists():
    print(f"Error: {latents_path} not found. Run Rust first.")
    exit(1)

# Read binary with header: ndims (u64), dim0 (u64), dim1 (u64), ..., then f32 data
import struct
with open(latents_path, "rb") as f:
    # Read ndims
    ndims = struct.unpack('<Q', f.read(8))[0]
    print(f"ndims: {ndims}")
    
    # Read dims
    dims = []
    for _ in range(ndims):
        d = struct.unpack('<Q', f.read(8))[0]
        dims.append(d)
    print(f"dims: {dims}")
    
    # Read rest as f32 data
    latents_data = np.frombuffer(f.read(), dtype=np.float32)

print(f"Loaded {len(latents_data)} latent elements")

# Reshape to expected shape
latents = latents_data.reshape(dims)
print(f"Latent shape: {latents.shape}")
print(f"Latent shape: {latents.shape}")
print(f"Latent range: min={latents.min():.4f}, max={latents.max():.4f}")

# Try to load diffusers VAE
try:
    from diffusers import AutoencoderKLLTXVideo
    print("Loading diffusers VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    vae = vae.to("cuda")
    vae.eval()
    print("VAE loaded!")
except Exception as e:
    print(f"Could not load diffusers VAE: {e}")
    print("\nInstall with: pip install diffusers transformers accelerate")
    exit(1)

# Decode latents
latents_tensor = torch.from_numpy(latents).to("cuda").to(torch.bfloat16)
print(f"Tensor on device: {latents_tensor.device}, dtype: {latents_tensor.dtype}")

with torch.no_grad():
    # Denormalize latents (reverse of what diffusers does during encoding)
    # The Rust code already saves normalized latents, so we decode directly
    video = vae.decode(latents_tensor, return_dict=False)[0]
    
print(f"Video shape: {video.shape}")
print(f"Video range: min={video.min().item():.4f}, max={video.max().item():.4f}")

# Convert to images
video = video.float().cpu().numpy()
video = np.clip((video + 1) / 2 * 255, 0, 255).astype(np.uint8)

# Save frames
output_dir = Path("output/python_vae_test")
output_dir.mkdir(exist_ok=True)

num_frames = video.shape[2]
for i in range(num_frames):
    frame = video[0, :, i, :, :]  # [C, H, W]
    frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
    img = Image.fromarray(frame)
    img.save(output_dir / f"frame_{i:04d}.png")
    print(f"Saved frame {i}")

print(f"\nFrames saved to {output_dir}")
print("If these frames look good, the problem is in Rust VAE.")
print("If these frames are also noise, the problem is in Rust DiT.")
