"""
Compare single step of VAE decode between Python and saved latents
This will help identify if the issue is in conv3d, unpatchify, or weights
"""
import torch
import numpy as np
from pathlib import Path

device = "cuda"
dtype = torch.bfloat16

# Load Rust latents
latents_path = Path("output/latents.bin")
import struct
with open(latents_path, "rb") as f:
    ndims = struct.unpack('<Q', f.read(8))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
    latents_data = np.frombuffer(f.read(), dtype=np.float32)

latents = latents_data.reshape(dims).copy()  # Make writeable
print(f"Latent shape: {latents.shape}")
print(f"Latent range: {latents.min():.4f} to {latents.max():.4f}")

# Load Python VAE
from diffusers import AutoencoderKLLTXVideo
vae = AutoencoderKLLTXVideo.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="vae",
    torch_dtype=dtype
).to(device)
vae.eval()
print("VAE loaded!")

# Print VAE config
print(f"\nVAE Decoder config:")
print(f"  patch_size: {vae.decoder.patch_size}")
print(f"  patch_size_t: {vae.decoder.patch_size_t}")
print(f"  in_channels: {vae.config.in_channels}")
print(f"  out_channels: {vae.config.out_channels}")
print(f"  block_out_channels: {vae.config.block_out_channels}")

# Convert latents
latents_tensor = torch.from_numpy(latents).to(device).to(dtype)
print(f"\nInput to decoder: {latents_tensor.shape}")

# Run decoder forward pass with hooks
intermediates = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().clone()
        else:
            intermediates[name] = output.detach().clone()
    return hook

# Register hooks
vae.decoder.conv_in.register_forward_hook(make_hook("conv_in"))
vae.decoder.conv_out.register_forward_hook(make_hook("conv_out"))
vae.decoder.norm_out.register_forward_hook(make_hook("norm_out"))

# Hook up_blocks
for i, block in enumerate(vae.decoder.up_blocks):
    block.register_forward_hook(make_hook(f"up_block_{i}"))

print("\nRunning VAE decode with hooks...")
with torch.no_grad():
    # Scale latents (VAE expects scaled input)
    # Check if VAE has scaling factors
    if hasattr(vae, 'latents_mean'):
        print(f"VAE has latents_mean, denormalizing...")
        latents_mean = torch.tensor(vae.latents_mean).to(device).to(dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.latents_std).to(device).to(dtype).view(1, -1, 1, 1, 1)
        latents_scaled = latents_tensor * latents_std / vae.config.scaling_factor + latents_mean
    else:
        latents_scaled = latents_tensor / vae.config.scaling_factor
    
    print(f"Scaled latents range: {latents_scaled.float().min():.4f} to {latents_scaled.float().max():.4f}")
    
    # Decode
    video = vae.decode(latents_scaled, return_dict=False)[0]

print(f"\nOutput video: {video.shape}")
print(f"Video range: {video.float().min():.4f} to {video.float().max():.4f}")

# Print intermediate values
print("\n=== Intermediate Values ===")
for name, tensor in sorted(intermediates.items()):
    t = tensor.float()
    print(f"{name}: shape={list(tensor.shape)}, range=[{t.min():.4f}, {t.max():.4f}]")

# Save key intermediates
save_dir = Path("output/vae_debug")
save_dir.mkdir(exist_ok=True)

for name, tensor in intermediates.items():
    np.save(save_dir / f"{name}.npy", tensor.float().cpu().numpy())

print(f"\nSaved intermediates to {save_dir}")
