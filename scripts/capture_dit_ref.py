import os
import sys
import torch
from safetensors.torch import save_file, load_file
import numpy as np

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.append(tp_path)

from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

def capture():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # Use float32 for max precision comparison
    
    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/transformer"
    print(f"Loading model from {model_path}...")
    
    model = LTXVideoTransformer3DModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()
    
    batch_size = 1
    num_frames = 9
    height = 64
    width = 64
    
    # LTX-Video VAE compression is 8x temporal, 32x spatial.
    # But Transformer patch size is 1x1x1 in config.
    # Sequence length = (F/1) * (H/1) * (W/1) ??? 
    # Wait, the pipeline packs latents before sending to transformer.
    # Latents from VAE are already compressed.
    # F_lat = (161-1)/8 + 1 = 21. 
    # For num_frames=9, F_lat = (9-1)/8 + 1 = 2.
    # H_lat = 64/32 = 2.
    # W_lat = 64/32 = 2.
    # seq_len = 2 * 2 * 2 = 8.
    
    f_lat = (num_frames - 1) // 8 + 1
    h_lat = height // 32
    w_lat = width // 32
    seq_len = f_lat * h_lat * w_lat
    in_channels = 128
    
    # Inputs
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, in_channels, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, 128, 4096, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    encoder_attention_mask = torch.ones(batch_size, 128, device=device, dtype=dtype)
    
    rope_interpolation_scale = (20.0 / 25.0, 32.0, 32.0)
    
    print("Running forward...")
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            num_frames=f_lat,
            height=h_lat,
            width=w_lat,
            rope_interpolation_scale=rope_interpolation_scale,
            return_dict=False
        )[0]
    
    print("Saving references...")
    ref_data = {
        "hidden_states": hidden_states.cpu(),
        "encoder_hidden_states": encoder_hidden_states.cpu(),
        "timestep": timestep.cpu(),
        "encoder_attention_mask": encoder_attention_mask.cpu(),
        "output": output.cpu(),
        "num_frames": torch.tensor([f_lat]),
        "height": torch.tensor([h_lat]),
        "width": torch.tensor([w_lat]),
    }
    save_file(ref_data, "dit_ref.safetensors")
    print("Done! Saved dit_ref.safetensors")

if __name__ == "__main__":
    capture()
