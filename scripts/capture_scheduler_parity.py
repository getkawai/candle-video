#!/usr/bin/env python3
"""
Capture reference tensors for FlowMatchEulerDiscreteScheduler parity verification.

This script captures:
1. Timesteps and sigmas for various num_inference_steps and mu values
2. Step outputs for various inputs
3. Time shift scalar values for verification

Output: gen_scheduler_parity.safetensors

Requirements: 11.1 - Provide Python scripts to capture reference tensors at each pipeline stage
"""

import torch
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
import os
import json


def calculate_mu(latent_seq_len: int, base_seq_len: int = 1024, max_seq_len: int = 4096,
                 base_shift: float = 0.95, max_shift: float = 2.05) -> float:
    """Calculate mu for dynamic shifting (matches LTX official implementation)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = m * latent_seq_len + b
    return mu


def time_shift_exponential(mu: float, sigma: float, t: float) -> float:
    """Exponential time shift formula: exp(mu) / (exp(mu) + (1/t - 1)^sigma)"""
    import math
    emu = math.exp(mu)
    base = (1.0 / t - 1.0) ** sigma
    return emu / (emu + base)


def time_shift_linear(mu: float, sigma: float, t: float) -> float:
    """Linear time shift formula: mu / (mu + (1/t - 1)^sigma)"""
    base = (1.0 / t - 1.0) ** sigma
    return mu / (mu + base)


def capture_time_shift_values():
    """Capture time_shift_scalar values for various mu and t combinations."""
    print("Capturing time_shift_scalar values...")
    
    # Test various mu values (typical range for LTX-Video)
    mu_values = [0.5, 0.95, 1.0, 1.5, 2.0, 2.05]
    # Test various t values (sigma values from 0 to 1, avoiding 0 and 1 exactly)
    t_values = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    
    results = {}
    for mu in mu_values:
        for t in t_values:
            key = f"time_shift_exp_mu{mu}_t{t}"
            results[key] = torch.tensor([time_shift_exponential(mu, 1.0, t)], dtype=torch.float32)
            
            key_linear = f"time_shift_lin_mu{mu}_t{t}"
            results[key_linear] = torch.tensor([time_shift_linear(mu, 1.0, t)], dtype=torch.float32)
    
    return results


def capture_scheduler_basic():
    """Capture basic scheduler outputs without dynamic shifting."""
    print("\nCapturing basic scheduler (no dynamic shifting)...")
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    
    results = {}
    
    # Test various num_inference_steps
    for num_steps in [10, 20, 30, 40, 50]:
        scheduler.set_timesteps(num_steps, device="cpu")
        
        results[f"basic_timesteps_{num_steps}"] = scheduler.timesteps.float()
        results[f"basic_sigmas_{num_steps}"] = scheduler.sigmas.float()
        
        print(f"  Steps={num_steps}: timesteps shape={scheduler.timesteps.shape}, "
              f"sigmas shape={scheduler.sigmas.shape}")
    
    return results


def capture_scheduler_dynamic_shifting():
    """Capture scheduler outputs with dynamic shifting enabled."""
    print("\nCapturing scheduler with dynamic shifting...")
    
    # LTX-Video 0.9.5+ config
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        time_shift_type="exponential",
    )
    
    results = {}
    
    # Test various mu values and step counts
    mu_values = [0.95, 1.0, 1.5, 2.0, 2.05]
    step_counts = [20, 30, 40, 50]
    
    for mu in mu_values:
        for num_steps in step_counts:
            scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
            
            key_ts = f"dynamic_timesteps_mu{mu}_steps{num_steps}"
            key_sig = f"dynamic_sigmas_mu{mu}_steps{num_steps}"
            
            results[key_ts] = scheduler.timesteps.float()
            results[key_sig] = scheduler.sigmas.float()
            
            print(f"  mu={mu}, steps={num_steps}: timesteps[0]={scheduler.timesteps[0].item():.4f}, "
                  f"timesteps[-1]={scheduler.timesteps[-1].item():.4f}")
    
    return results


def capture_scheduler_with_shift_terminal():
    """Capture scheduler outputs with shift_terminal enabled (LTX-Video style)."""
    print("\nCapturing scheduler with shift_terminal...")
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        shift_terminal=0.1,  # LTX-Video default
        time_shift_type="exponential",
    )
    
    results = {}
    
    # Test with typical LTX-Video parameters
    mu_values = [1.0, 1.5, 2.0]
    step_counts = [30, 40, 50]
    
    for mu in mu_values:
        for num_steps in step_counts:
            scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
            
            key_ts = f"terminal_timesteps_mu{mu}_steps{num_steps}"
            key_sig = f"terminal_sigmas_mu{mu}_steps{num_steps}"
            
            results[key_ts] = scheduler.timesteps.float()
            results[key_sig] = scheduler.sigmas.float()
            
            print(f"  mu={mu}, steps={num_steps}: sigmas[-2]={scheduler.sigmas[-2].item():.6f}, "
                  f"sigmas[-1]={scheduler.sigmas[-1].item():.6f}")
    
    return results


def capture_scheduler_step():
    """Capture scheduler step outputs for various inputs."""
    print("\nCapturing scheduler step outputs...")
    
    # Use dynamic shifting config (typical LTX-Video setup)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        time_shift_type="exponential",
    )
    
    results = {}
    
    # Set up scheduler with typical parameters
    mu = 1.5
    num_steps = 40
    scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
    
    # Create test inputs with fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Test various latent shapes
    test_shapes = [
        (1, 128, 2, 8, 8),    # Small test
        (1, 128, 4, 16, 16),  # Medium test
        (1, 128, 8, 16, 24),  # Typical LTX-Video shape
    ]
    
    for shape_idx, latent_shape in enumerate(test_shapes):
        torch.manual_seed(42 + shape_idx)
        sample = torch.randn(latent_shape, dtype=torch.float32)
        model_output = torch.randn(latent_shape, dtype=torch.float32)
        
        # Reset scheduler for each test
        scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
        
        # Test first step
        t0 = scheduler.timesteps[0]
        step_output_0 = scheduler.step(model_output, t0, sample, return_dict=False)[0]
        
        results[f"step_sample_shape{shape_idx}"] = sample
        results[f"step_model_output_shape{shape_idx}"] = model_output
        results[f"step_timestep_shape{shape_idx}"] = torch.tensor([t0.item()], dtype=torch.float32)
        results[f"step_output_0_shape{shape_idx}"] = step_output_0
        
        print(f"  Shape {latent_shape}: step 0 output mean={step_output_0.mean().item():.6f}")
        
        # Test middle step (step 20)
        scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
        for i in range(20):
            t = scheduler.timesteps[i]
            sample = scheduler.step(model_output, t, sample, return_dict=False)[0]
        
        t20 = scheduler.timesteps[20]
        step_output_20 = scheduler.step(model_output, t20, sample, return_dict=False)[0]
        
        results[f"step_output_20_shape{shape_idx}"] = step_output_20
        results[f"step_sample_at_20_shape{shape_idx}"] = sample
        
        print(f"  Shape {latent_shape}: step 20 output mean={step_output_20.mean().item():.6f}")
    
    return results


def capture_step_formula_verification():
    """Capture data to verify the step formula: prev_sample = sample + dt * model_output"""
    print("\nCapturing step formula verification data...")
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
        time_shift_type="exponential",
    )
    
    results = {}
    
    mu = 1.5
    num_steps = 40
    scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
    
    torch.manual_seed(123)
    sample = torch.randn(1, 128, 2, 8, 8, dtype=torch.float32)
    model_output = torch.randn(1, 128, 2, 8, 8, dtype=torch.float32)
    
    # Capture sigma values for manual verification
    for step_idx in [0, 10, 20, 30, 39]:
        scheduler.set_timesteps(num_steps, device="cpu", mu=mu)
        
        # Run steps up to step_idx
        current_sample = sample.clone()
        for i in range(step_idx):
            t = scheduler.timesteps[i]
            current_sample = scheduler.step(model_output, t, current_sample, return_dict=False)[0]
        
        # Get sigma values at this step
        sigma_current = scheduler.sigmas[step_idx].item()
        sigma_next = scheduler.sigmas[step_idx + 1].item()
        dt = sigma_next - sigma_current
        
        t = scheduler.timesteps[step_idx]
        step_output = scheduler.step(model_output, t, current_sample, return_dict=False)[0]
        
        # Clone tensors to avoid shared memory issues with safetensors
        results[f"formula_sample_step{step_idx}"] = current_sample.clone()
        results[f"formula_model_output_step{step_idx}"] = model_output.clone()
        results[f"formula_sigma_current_step{step_idx}"] = torch.tensor([sigma_current], dtype=torch.float32)
        results[f"formula_sigma_next_step{step_idx}"] = torch.tensor([sigma_next], dtype=torch.float32)
        results[f"formula_dt_step{step_idx}"] = torch.tensor([dt], dtype=torch.float32)
        results[f"formula_output_step{step_idx}"] = step_output.clone()
        
        # Manual calculation for verification
        expected = current_sample + dt * model_output
        mse = ((step_output - expected) ** 2).mean().item()
        
        print(f"  Step {step_idx}: sigma={sigma_current:.6f}, sigma_next={sigma_next:.6f}, "
              f"dt={dt:.6f}, formula MSE={mse:.2e}")
    
    return results


def capture_mu_calculation():
    """Capture mu calculation for various sequence lengths."""
    print("\nCapturing mu calculation values...")
    
    results = {}
    
    # Test various sequence lengths (typical for different video resolutions)
    seq_lens = [
        256,    # 256x256 image
        1024,   # 512x512 image (base)
        2048,   # 768x512 video
        4096,   # 1024x768 video (max)
        6144,   # Larger than max
    ]
    
    for seq_len in seq_lens:
        mu = calculate_mu(seq_len)
        results[f"mu_seq_len_{seq_len}"] = torch.tensor([mu], dtype=torch.float32)
        print(f"  seq_len={seq_len}: mu={mu:.6f}")
    
    return results


def main():
    print("=" * 60)
    print("Capturing FlowMatchEulerDiscreteScheduler reference data")
    print("=" * 60)
    
    all_results = {}
    
    # Capture all reference data
    all_results.update(capture_time_shift_values())
    all_results.update(capture_scheduler_basic())
    all_results.update(capture_scheduler_dynamic_shifting())
    all_results.update(capture_scheduler_with_shift_terminal())
    all_results.update(capture_scheduler_step())
    all_results.update(capture_step_formula_verification())
    all_results.update(capture_mu_calculation())
    
    # Save metadata
    metadata = {
        "description": "FlowMatchEulerDiscreteScheduler parity reference data",
        "diffusers_version": str(getattr(__import__('diffusers'), '__version__', 'unknown')),
        "torch_version": torch.__version__,
        "num_tensors": str(len(all_results)),
    }
    
    # Save to safetensors
    output_path = "gen_scheduler_parity.safetensors"
    save_file(all_results, output_path, metadata=metadata)
    
    print("\n" + "=" * 60)
    print(f"Saved {len(all_results)} tensors to {output_path}")
    print("=" * 60)
    
    # Print summary
    print("\nTensor summary:")
    for key in sorted(all_results.keys())[:20]:
        tensor = all_results[key]
        print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
    if len(all_results) > 20:
        print(f"  ... and {len(all_results) - 20} more tensors")


if __name__ == "__main__":
    main()
