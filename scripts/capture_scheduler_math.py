from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import torch
from safetensors.torch import save_file
import os

def capture_scheduler():
    # Load scheduler with default config (LTX compatible)
    # LTX uses standard FlowMatchEulerDiscreteScheduler
    # Config values from 0.9.5:
    # shift = 1.0 (default in init, but dynamic in pipeline?)
    # pipeline_ltx.py sets timesteps with defaults? 
    # Let's check model config. 
    # LTX model card says "sigma_shift=1.0".
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False
    )
    
    num_inference_steps = 50
    device = "cpu"
    
    # Run set_timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    
    print(f"Timesteps: {timesteps[:5]}...{timesteps[-5:]}")
    print(f"Sigmas: {sigmas[:5]}...{sigmas[-5:]}")
    
    # Save
    save_data = {
        "timesteps": timesteps,
        "sigmas": sigmas
    }
    
    os.makedirs("reference_output", exist_ok=True)
    save_file(save_data, "reference_output/scheduler_ref.safetensors")
    print("Saved reference_output/scheduler_ref.safetensors")

if __name__ == "__main__":
    capture_scheduler()
