import os
import sys
import torch
from safetensors.torch import save_file
import numpy as np

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.append(tp_path)

from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

def capture_step0():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # Use float32 for comparison
    
    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5"
    
    print(f"Loading LTXPipeline from {model_path}...")
    # Load pipeline components
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"), torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(model_path, "scheduler")
    )
    pipeline = LTXPipeline.from_pretrained(
        model_path, 
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=dtype,
        variant=None
    ).to(device)
    
    pipeline.set_progress_bar_config(disable=True)
    
    # Parameters
    prompt = "A cinematic shot of a forest."
    negative_prompt = "low quality, worst quality"
    height = 128 # Small size for speed
    width = 128
    num_frames = 9 # Small frames
    num_inference_steps = 2 # Single step needed, but set 2 to init scheduler
    guidance_scale = 1.0
    seed = 42
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Preparing inputs...")
    
    # 1. Encode prompt
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        num_videos_per_prompt=1
    )
    
    if guidance_scale > 1.0:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
        
    print(f"Prompt Embeds: {prompt_embeds.shape}")
    
    # 2. Prepare Timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # 3. Prepare Latents
    latents = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    
    print(f"Initial Latents: {latents.shape}")
    
    # 4. Run ONE Step
    t = timesteps[0]
    
    # Expand latents for CFG
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
    
    # Broadcast current timestep
    # pipeline_ltx.py logic:
    # timestep = t.expand(latent_model_input.shape[0])
    timestep = t.expand(latent_model_input.shape[0])
    
    print(f"Running Transformer Forward at t={t}...")
    
    # Need to calculate rope_interpolation_scale as in pipeline
    # The pipeline does it internally in check_inputs or prepare_latents... wait.
    # Actually LTXPipeline calculates it for the transformer call.
    # Let's see how pipeline calls transformer.
    
    # From source:
    # rope_interpolation_scale = (1/vae_scale_factor_spatial, 32, 32)
    # usually (1/32, 32, 32) -> wait, no.
    # Default config has patch_size 1.
    # The pipeline calculates it based on original resolution.
    
    # LTXPipeline 0.9.5 logic:
    # default_height = 480
    # default_width = 704
    # default_num_frames = 161
    # ...
    # rope_interpolation_scale = [
    #     default_num_frames / num_frames,
    #     default_height / height,
    #     default_width / width
    # ]
    # Wait, check source.
    
    # Let's inspect the installed pipeline code or assume logic.
    # Better: just let the pipeline run 1 step by hacking the loop?
    # No, I want explicit control.
    
    # In 0.9.5:
    # self.transformer(..., rope_interpolation_scale=rope_interpolation_scale)
    
    # Calculation:
    # default_num_frames = 161
    # default_height = 480
    # default_width = 704
    
    # interpolation_scale_t = default_num_frames / num_frames
    # interpolation_scale_h = default_height / height
    # interpolation_scale_w = default_width / width
    # rope_interpolation_scale = (interpolation_scale_t, interpolation_scale_h, interpolation_scale_w)
    
    rope_scale = (
        161 / num_frames,
        480 / height,
        704 / width
    )
    print(f"RoPE Scale: {rope_scale}")

    # Forward
    # Calculate latent dimensions for transformer
    vae_scale_spatial = 32
    vae_scale_temporal = 8
    
    latent_height = height // vae_scale_spatial
    latent_width = width // vae_scale_spatial
    latent_num_frames = (num_frames - 1) // vae_scale_temporal + 1
    
    noise_pred = pipeline.transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        encoder_attention_mask=prompt_attention_mask,
        num_frames=latent_num_frames,
        height=latent_height,
        width=latent_width,
        rope_interpolation_scale=rope_scale,
        return_dict=False,
    )[0]
    
    # Perform CFG
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
    print(f"Noise Pred (after CFG): {noise_pred.shape}")
    
    # Scheduler Step
    # FlowMatchEulerDiscreteScheduler step
    # step(model_output, timestep, sample)
    
    # Python scheduler often wants output in float32
    # latents is float32
    
    step_output = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    print(f"Next Latents: {step_output.shape}")
    
    # Save everything
    save_data = {
        "prompt_embeds": prompt_embeds.cpu().contiguous(), 
        "prompt_attention_mask": prompt_attention_mask.to(torch.float32).cpu().contiguous(),
        "initial_latents": latents.cpu().contiguous(),
        "timestep": torch.tensor([float(t)]).contiguous(),
        "transformer_output": noise_pred.cpu().contiguous(), 
        "next_latents": step_output.cpu().contiguous(),
        "scheduler_sigmas": pipeline.scheduler.sigmas.cpu().contiguous(),
        "scheduler_timesteps": pipeline.scheduler.timesteps.cpu().contiguous(),
    }
    
    os.makedirs("reference_output", exist_ok=True)
    save_file(save_data, "reference_output/pipeline_step0.safetensors")
    print("Saved reference_output/pipeline_step0.safetensors")

if __name__ == "__main__":
    capture_step0()
