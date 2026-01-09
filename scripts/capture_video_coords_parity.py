#!/usr/bin/env python3
"""
Capture video coordinates reference data for parity testing.

This script captures the exact video_coords computation as done in diffusers
pipeline_ltx_condition.py for comparison with Rust implementation.
"""

import torch
from safetensors.torch import save_file


def prepare_video_ids(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare video IDs (latent coordinates) - matches diffusers _prepare_video_ids."""
    latent_sample_coords = torch.meshgrid(
        torch.arange(0, num_frames, patch_size_t, device=device),
        torch.arange(0, height, patch_size, device=device),
        torch.arange(0, width, patch_size, device=device),
        indexing="ij",
    )
    latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
    latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)
    
    return latent_coords


def scale_video_ids(
    video_ids: torch.Tensor,
    scale_factor: int = 32,
    scale_factor_t: int = 8,
    frame_index: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """Scale video IDs - matches diffusers _scale_video_ids with causal fix."""
    scaled_latent_coords = (
        video_ids
        * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
    )
    # Causal fix: (L * scale_factor_t + 1 - scale_factor_t).clamp(min=0)
    scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
    scaled_latent_coords[:, 0] += frame_index
    
    return scaled_latent_coords


def compute_video_coords_diffusers_style(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    frame_rate: int,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    """
    Compute video_coords exactly as done in diffusers pipeline_ltx_condition.py.
    
    Steps:
    1. Compute latent dimensions
    2. Create video_ids grid
    3. Scale video_ids (with causal fix)
    4. Divide temporal by frame_rate
    5. Transpose to [batch, seq, 3] format
    """
    # Compute latent dimensions
    latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
    latent_height = height // vae_spatial_compression
    latent_width = width // vae_spatial_compression
    
    # Step 1: Prepare video IDs [batch, 3, seq_len]
    video_ids = prepare_video_ids(
        batch_size=batch_size,
        num_frames=latent_num_frames,
        height=latent_height,
        width=latent_width,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
    )
    
    # Step 2: Scale video IDs (with causal fix)
    video_coords = scale_video_ids(
        video_ids,
        scale_factor=vae_spatial_compression,
        scale_factor_t=vae_temporal_compression,
        frame_index=0,
    )
    
    # Step 3: Convert to float and apply frame_rate scaling
    video_coords = video_coords.float()
    video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)
    
    # Step 4: Transpose to [batch, seq, 3] format (as expected by transformer)
    video_coords = video_coords.transpose(1, 2)  # [batch, seq, 3]
    
    return video_coords


def compute_video_coords_rust_style(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    frame_rate: int,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
) -> torch.Tensor:
    """
    Compute video_coords as done in Rust t2v_pipeline.rs.
    
    This should produce identical results to compute_video_coords_diffusers_style.
    """
    ts_ratio = float(vae_temporal_compression)
    sp_ratio = float(vae_spatial_compression)
    
    # Compute latent dimensions
    latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
    latent_height = height // vae_spatial_compression
    latent_width = width // vae_spatial_compression
    
    # Create grids
    grid_f = torch.arange(latent_num_frames, dtype=torch.float32)
    grid_h = torch.arange(latent_height, dtype=torch.float32)
    grid_w = torch.arange(latent_width, dtype=torch.float32)
    
    # Broadcast to [F, H, W]
    f = grid_f.reshape(latent_num_frames, 1, 1).expand(latent_num_frames, latent_height, latent_width)
    h = grid_h.reshape(1, latent_height, 1).expand(latent_num_frames, latent_height, latent_width)
    w = grid_w.reshape(1, 1, latent_width).expand(latent_num_frames, latent_height, latent_width)
    
    # Stack and flatten: [3, F, H, W] -> [3, seq] -> [seq, 3] -> [1, seq, 3]
    video_coords = torch.stack([f, h, w], dim=0)  # [3, F, H, W]
    video_coords = video_coords.flatten(1)  # [3, seq]
    video_coords = video_coords.transpose(0, 1)  # [seq, 3]
    video_coords = video_coords.unsqueeze(0)  # [1, seq, 3]
    
    vf = video_coords[:, :, 0]
    vh = video_coords[:, :, 1]
    vw = video_coords[:, :, 2]
    
    # CAUSAL FIX: (L * 8 + 1 - 8).clamp(0) / frame_rate
    # affine(ts_ratio, 1.0 - ts_ratio) = L * ts_ratio + (1 - ts_ratio)
    vf = vf * ts_ratio + (1.0 - ts_ratio)
    vf = vf.clamp(min=0.0)
    vf = vf / frame_rate
    
    # SPATIAL SCALE: L * 32
    vh = vh * sp_ratio
    vw = vw * sp_ratio
    
    # Stack back
    video_coords = torch.stack([vf, vh, vw], dim=-1)  # [1, seq, 3]
    
    # Broadcast to batch
    video_coords = video_coords.expand(batch_size, -1, -1)
    
    return video_coords


def capture_video_coords_parity():
    """Capture video coordinates for parity testing."""
    print("Capturing video coordinates parity data...")
    
    results = {}
    
    # Test configurations: (num_frames, height, width, frame_rate)
    test_configs = [
        (9, 256, 256, 25),      # Small
        (25, 512, 768, 25),     # Medium
        (97, 512, 768, 25),     # Large (typical LTX)
        (161, 512, 768, 25),    # Very large
        (9, 512, 768, 30),      # Different frame rate
        (25, 768, 512, 24),     # Portrait
    ]
    
    vae_temporal_compression = 8
    vae_spatial_compression = 32
    batch_size = 1
    
    for idx, (num_frames, height, width, frame_rate) in enumerate(test_configs):
        print(f"\nConfig {idx}: frames={num_frames}, h={height}, w={width}, fps={frame_rate}")
        
        # Compute latent dimensions
        latent_num_frames = (num_frames - 1) // vae_temporal_compression + 1
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression
        seq_len = latent_num_frames * latent_height * latent_width
        
        print(f"  Latent dims: f={latent_num_frames}, h={latent_height}, w={latent_width}, seq={seq_len}")
        
        # Compute using diffusers style
        coords_diffusers = compute_video_coords_diffusers_style(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            frame_rate=frame_rate,
            vae_temporal_compression=vae_temporal_compression,
            vae_spatial_compression=vae_spatial_compression,
        )
        
        # Compute using Rust style
        coords_rust = compute_video_coords_rust_style(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            frame_rate=frame_rate,
            vae_temporal_compression=vae_temporal_compression,
            vae_spatial_compression=vae_spatial_compression,
        )
        
        # Verify they match
        mse = ((coords_diffusers - coords_rust) ** 2).mean().item()
        max_diff = (coords_diffusers - coords_rust).abs().max().item()
        
        print(f"  Diffusers vs Rust: MSE={mse:.2e}, max_diff={max_diff:.2e}")
        
        if mse > 1e-10:
            print(f"  WARNING: Mismatch detected!")
            print(f"    Diffusers first 5: {coords_diffusers[0, :5, :]}")
            print(f"    Rust first 5: {coords_rust[0, :5, :]}")
        
        # Store results
        key = f"coords_f{num_frames}_h{height}_w{width}_fps{frame_rate}"
        results[f"{key}_diffusers"] = coords_diffusers.contiguous().clone()
        results[f"{key}_rust_style"] = coords_rust.contiguous().clone()
        results[f"{key}_latent_dims"] = torch.tensor(
            [latent_num_frames, latent_height, latent_width], dtype=torch.int64
        )
        results[f"{key}_params"] = torch.tensor(
            [num_frames, height, width, frame_rate, vae_temporal_compression, vae_spatial_compression],
            dtype=torch.int64
        )
        
        # Also store individual components for debugging
        # Temporal component (first column)
        results[f"{key}_temporal"] = coords_diffusers[:, :, 0].contiguous().clone()
        # Spatial height component (second column)
        results[f"{key}_spatial_h"] = coords_diffusers[:, :, 1].contiguous().clone()
        # Spatial width component (third column)
        results[f"{key}_spatial_w"] = coords_diffusers[:, :, 2].contiguous().clone()
    
    return results


def main():
    results = capture_video_coords_parity()
    
    # Save to safetensors
    output_path = "gen_video_coords_parity.safetensors"
    save_file(results, output_path)
    print(f"\nSaved {len(results)} tensors to {output_path}")
    
    # Print summary
    print("\nSummary of captured data:")
    for key in sorted(results.keys()):
        tensor = results[key]
        print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
