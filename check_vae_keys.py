import torch
from safetensors.torch import load_file
import sys

path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae/diffusion_pytorch_model.safetensors"
try:
    with torch.no_grad():
        sd = load_file(path, device="cpu")
        keys = list(sd.keys())
        print("Keys in VAE safetensors:")
        for k in sorted(keys):
            if "latent" in k:
                print(f"  {k}: {sd[k].shape}")
except Exception as e:
    print(f"Error: {e}")
