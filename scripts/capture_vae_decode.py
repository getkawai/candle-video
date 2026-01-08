import torch
from diffusers import AutoencoderKLLTXVideo
from safetensors.torch import save_file
import os

def capture_vae():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use standard float32 for comparison
    dtype = torch.float32 
    
    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/vae"
    
    print(f"Loading VAE from {model_path}...")
    vae = AutoencoderKLLTXVideo.from_pretrained(model_path, torch_dtype=dtype).to(device)
    vae.eval()
    
    # Create synthetic latents
    # VAE spatial compression 32, temporal 8.
    # Latents: [B, C, F, H, W]
    # Let's use small size: 1 frame latent -> 8 frames video?
    # No, latent_num_frames = (num_frames - 1) // 8 + 1
    # If we want output 9 frames, latent F=2. (9-1)//8 + 1 = 2.
    # 32x128x128 output -> 1x4x4 latent?
    
    # Let's generate [1, 128, 2, 8, 8] latents
    # Output roughly: [1, 3, 9, 256, 256] ? 
    # 8*32 = 256.
    
    # Real inputs
    latents = torch.randn(1, 128, 2, 8, 8, device=device, dtype=dtype)
    
    # Important: LTX pipeline denormalizes BEFORE decode.
    # But here verify strict decode.
    # We will save latents AS IS.
    
    # But wait, does Rust VAE include denormalization?
    # No, pipeline does it.
    
    print(f"Decoding latents {latents.shape}...")
    with torch.no_grad():
        # Pass timestep 0.0 (as tensor or list? Pipeline passes list if list, else float/int)
        # Model expects tensor usually if timestep_conditioning=True
        # decode signature: z, temb=None.
        # But wait, AutoencoderKLLTXVideo logic:
        # if self.timestep_scale_multiplier is not None: temb = temb * ...
        # If temb is None, it fails.
        # So we MUST pass proper temb if timestep_conditioning=True.
        
        # Pipeline logic:
        # decode_timestep = 0.0
        # timestep = torch.tensor([decode_timestep] * batch_size, device=device)
        
        batch_size = latents.shape[0]
        timestep = torch.tensor([0.0] * batch_size, device=device, dtype=dtype)
        
        output = vae.decode(latents, temb=timestep, return_dict=False)[0]
        
    print(f"Output: {output.shape}")
    
    save_data = {
        "latents": latents.cpu().contiguous(),
        "video": output.cpu().contiguous()
    }
    
    os.makedirs("reference_output", exist_ok=True)
    save_file(save_data, "reference_output/vae_debug.safetensors")
    print("Saved reference_output/vae_debug.safetensors")

if __name__ == "__main__":
    capture_vae()
