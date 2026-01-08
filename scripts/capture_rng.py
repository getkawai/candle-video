import torch
import numpy as np
from safetensors.torch import save_file
import os

def capture_rng():
    seed = 42
    device = "cpu"
    shape = (1, 128, 13, 16, 24) # Typical latent shape for 512x768x97
    
    # PyTorch Generator
    generator = torch.Generator(device=device).manual_seed(seed)
    torch_noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    
    print(f"Torch noise shape: {torch_noise.shape}")
    print(f"Torch noise mean: {torch_noise.mean().item():.6f}")
    print(f"Torch noise std: {torch_noise.std().item():.6f}")
    print(f"Torch noise [0,0,0,0,:5]: {torch_noise[0,0,0,0,:5]}")
    
    os.makedirs("reference_output", exist_ok=True)
    save_file({"noise": torch_noise}, "reference_output/rng_ref.safetensors")
    print("Saved reference_output/rng_ref.safetensors")

if __name__ == "__main__":
    capture_rng()
