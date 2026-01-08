import torch
from diffusers import AutoencoderKLLTXVideo
from safetensors.torch import save_file
import os

def capture_vae_tiling():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 
    
    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae"
    
    print(f"Loading VAE from {model_path}...")
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    # Enable tiling with small thresholds to force usage
    # Default compression: 32 spatial, 8 temporal.
    # Latent 8x8 -> Image 256x256.
    # Set threshold < 256. E.g. 128.
    vae.enable_tiling(
        tile_sample_min_height=128,
        tile_sample_min_width=128,
        tile_sample_min_num_frames=16, # Latent 2 -> Video 9. Set 8.
        tile_sample_stride_height=96,
        tile_sample_stride_width=96,
        tile_sample_stride_num_frames=4
    )
    
    # Latents: [B, C, F, H, W]
    # Make latents large enough to trigger tiling.
    # H=8 -> 256px. Tile min 128. It will tile.
    latents = torch.randn(1, 128, 2, 8, 8, device=device, dtype=dtype)
    
    batch_size = latents.shape[0]
    timestep = torch.tensor([0.0] * batch_size, device=device, dtype=dtype)
    
    print(f"Decoding latents {latents.shape} with tiling...")
    with torch.no_grad():
        output = vae.decode(latents, temb=timestep, return_dict=False)[0]
        
    print(f"Output: {output.shape}")
    
    save_data = {
        "latents": latents.cpu().contiguous(),
        "video": output.cpu().contiguous()
    }
    
    os.makedirs("reference_output", exist_ok=True)
    save_file(save_data, "reference_output/vae_tiling.safetensors")
    print("Saved reference_output/vae_tiling.safetensors")

if __name__ == "__main__":
    capture_vae_tiling()
