import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
import os

def capture_scheduler_step():
    device = "cpu"
    dtype = torch.float32 # Use F32 for math verification

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0, # LTX default? In pipeline code it calculates shift. 
             # Wait, pipeline uses custom sigmas?
             # pipeline_ltx.py: "scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)"
             # But LTX does custom timesteps logic in pipeline_ltx.py 'retrieve_timesteps' if mu is passed.
             # Wait. LTX uses flow matching.
             # Does it use standard FlowMatchEulerDiscreteScheduler 'step'?
             # Yes.
             
        use_dynamic_shifting=False # Default
    )
    
    # 1. Simulate inputs
    sample = torch.randn(1, 128, 2, 8, 8, dtype=dtype) # Latents
    model_output = torch.randn(1, 128, 2, 8, 8, dtype=dtype) # Predicted noise/velocity
    # Set timesteps first
    scheduler.set_timesteps(30, device=device)
    
    # Use the first timestep from the schedule
    timestep = scheduler.timesteps[0]
    print(f"Using timestep: {timestep}")
    
    # 2. Run step
    # step(model_output, timestep, sample)
    output = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
    
    print(f"Prev sample shape: {output.shape}")
    
    save_data = {
        "sample": sample,
        "model_output": model_output,
        "timestep": timestep, # float tensor
        "prev_sample": output
    }
    
    os.makedirs("reference_output", exist_ok=True)
    save_file(save_data, "reference_output/scheduler_step.safetensors")
    print("Saved reference_output/scheduler_step.safetensors")

if __name__ == "__main__":
    capture_scheduler_step()
