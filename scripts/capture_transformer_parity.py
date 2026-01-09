#!/usr/bin/env python3
"""
Capture reference tensors for LTXVideoTransformer3DModel parity verification.

This script captures:
1. RoPE frequencies (cos/sin) for various video dimensions
2. Attention outputs for various inputs
3. Full forward pass outputs
4. AdaLayerNorm modulation values

Output: gen_transformer_parity.safetensors

Requirements: 11.1 - Provide Python scripts to capture reference tensors at each pipeline stage
"""

import os
import sys
import math
import torch
import torch.nn as nn
from safetensors.torch import save_file
from typing import Tuple, Optional

# Add local diffusers to path
tp_path = os.path.join(os.getcwd(), "tp", "diffusers", "src")
sys.path.insert(0, tp_path)

from diffusers.models.transformers.transformer_ltx import (
    LTXVideoTransformer3DModel,
    LTXVideoRotaryPosEmbed,
    apply_rotary_emb,
)


def capture_rope_frequencies():
    """Capture RoPE frequency computation for various video dimensions."""
    print("Capturing RoPE frequencies...")
    
    results = {}
    
    # Create RoPE module with default LTX-Video config
    rope = LTXVideoRotaryPosEmbed(
        dim=2048,  # inner_dim = 32 heads * 64 head_dim
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        patch_size=1,
        patch_size_t=1,
        theta=10000.0,
    )
    
    # Test various video dimensions (latent space dimensions)
    test_configs = [
        # (num_frames, height, width, rope_interpolation_scale)
        (2, 8, 8, (1.0, 1.0, 1.0)),      # Small test
        (4, 16, 16, (1.0, 1.0, 1.0)),    # Medium test
        (8, 16, 24, (1.0, 1.0, 1.0)),    # Typical LTX shape
        (21, 16, 24, (1.0, 1.0, 1.0)),   # Full video (161 frames -> 21 latent frames)
        # With interpolation scale (typical LTX-Video values)
        (8, 16, 24, (0.8, 32.0, 32.0)),  # With scaling
        (21, 16, 24, (0.8, 32.0, 32.0)), # Full video with scaling
    ]
    
    for idx, (num_frames, height, width, scale) in enumerate(test_configs):
        batch_size = 1
        seq_len = num_frames * height * width
        
        # Create dummy hidden states for shape
        hidden_states = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)
        
        # Get RoPE embeddings
        cos_freqs, sin_freqs = rope(
            hidden_states,
            num_frames=num_frames,
            height=height,
            width=width,
            rope_interpolation_scale=scale,
        )
        
        key = f"rope_f{num_frames}_h{height}_w{width}_s{scale[0]}_{scale[1]}_{scale[2]}"
        results[f"{key}_cos"] = cos_freqs.clone()
        results[f"{key}_sin"] = sin_freqs.clone()
        
        print(f"  Config {idx}: f={num_frames}, h={height}, w={width}, scale={scale}")
        print(f"    cos shape: {cos_freqs.shape}, sin shape: {sin_freqs.shape}")
        print(f"    cos range: [{cos_freqs.min().item():.4f}, {cos_freqs.max().item():.4f}]")
        print(f"    sin range: [{sin_freqs.min().item():.4f}, {sin_freqs.max().item():.4f}]")
    
    return results


def capture_rope_with_video_coords():
    """Capture RoPE with pre-computed video_coords (as used in pipeline)."""
    print("\nCapturing RoPE with video_coords...")
    
    results = {}
    
    rope = LTXVideoRotaryPosEmbed(
        dim=2048,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        patch_size=1,
        patch_size_t=1,
        theta=10000.0,
    )
    
    # Test with pre-computed video_coords
    test_configs = [
        (8, 16, 24),   # Typical shape
        (21, 16, 24),  # Full video
    ]
    
    for num_frames, height, width in test_configs:
        batch_size = 1
        seq_len = num_frames * height * width
        
        # Create video_coords as done in pipeline
        # Shape: [batch, 3, seq_len] where 3 = (frame, height, width)
        grid_f = torch.arange(num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)
        
        # Create meshgrid
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        video_coords = torch.stack(grid, dim=0)  # [3, F, H, W]
        video_coords = video_coords.unsqueeze(0)  # [1, 3, F, H, W]
        video_coords = video_coords.flatten(2, 4).transpose(1, 2)  # [1, seq_len, 3]
        
        # Apply temporal scaling as done in pipeline
        # (coord * 8 + 1 - 8).clamp(0) / frame_rate
        frame_rate = 25.0
        temporal_compression = 8
        video_coords_scaled = video_coords.clone()
        video_coords_scaled[:, :, 0] = (
            (video_coords[:, :, 0] * temporal_compression + 1 - temporal_compression)
            .clamp(min=0)
            / frame_rate
        )
        # Spatial scaling: coord * spatial_compression_ratio
        spatial_compression = 32
        video_coords_scaled[:, :, 1] = video_coords[:, :, 1] * spatial_compression
        video_coords_scaled[:, :, 2] = video_coords[:, :, 2] * spatial_compression
        
        # Transpose to [batch, 3, seq_len] for rope
        video_coords_for_rope = video_coords_scaled.transpose(1, 2)
        
        hidden_states = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)
        
        cos_freqs, sin_freqs = rope(
            hidden_states,
            video_coords=video_coords_for_rope,
        )
        
        key = f"rope_coords_f{num_frames}_h{height}_w{width}"
        results[f"{key}_video_coords"] = video_coords_for_rope.clone()
        results[f"{key}_cos"] = cos_freqs.clone()
        results[f"{key}_sin"] = sin_freqs.clone()
        
        print(f"  f={num_frames}, h={height}, w={width}")
        print(f"    video_coords shape: {video_coords_for_rope.shape}")
        print(f"    cos shape: {cos_freqs.shape}")
    
    return results


def capture_apply_rotary_emb():
    """Capture apply_rotary_emb function outputs."""
    print("\nCapturing apply_rotary_emb outputs...")
    
    results = {}
    
    # Test various input shapes
    test_shapes = [
        (1, 128, 2048),   # Small
        (1, 1024, 2048),  # Medium
        (1, 8192, 2048),  # Large (typical LTX)
    ]
    
    for idx, shape in enumerate(test_shapes):
        torch.manual_seed(42 + idx)
        
        x = torch.randn(shape, dtype=torch.float32)
        cos = torch.randn(shape, dtype=torch.float32)
        sin = torch.randn(shape, dtype=torch.float32)
        
        output = apply_rotary_emb(x, (cos, sin))
        
        results[f"rotary_input_{idx}"] = x.clone()
        results[f"rotary_cos_{idx}"] = cos.clone()
        results[f"rotary_sin_{idx}"] = sin.clone()
        results[f"rotary_output_{idx}"] = output.clone()
        
        print(f"  Shape {shape}: output mean={output.mean().item():.6f}")
    
    return results


def capture_transformer_forward_small():
    """Capture transformer forward pass with small model for fast verification."""
    print("\nCapturing transformer forward (small model)...")
    
    results = {}
    
    # Create small model for fast testing
    model = LTXVideoTransformer3DModel(
        in_channels=32,
        out_channels=32,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=4,
        attention_head_dim=16,
        cross_attention_dim=64,
        num_layers=2,
        caption_channels=64,
        qk_norm="rms_norm_across_heads",
    )
    model.eval()
    
    # Test configurations
    test_configs = [
        (2, 8, 8),    # Small
        (4, 8, 8),    # Medium
    ]
    
    for idx, (num_frames, height, width) in enumerate(test_configs):
        torch.manual_seed(42 + idx)
        
        batch_size = 1
        seq_len = num_frames * height * width
        
        hidden_states = torch.randn(batch_size, seq_len, 32, dtype=torch.float32)
        encoder_hidden_states = torch.randn(batch_size, 16, 64, dtype=torch.float32)
        timestep = torch.tensor([500.0], dtype=torch.float32)
        encoder_attention_mask = torch.ones(batch_size, 16, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                rope_interpolation_scale=(1.0, 1.0, 1.0),
                return_dict=False,
            )[0]
        
        key = f"small_f{num_frames}_h{height}_w{width}"
        results[f"{key}_hidden_states"] = hidden_states.clone()
        results[f"{key}_encoder_hidden_states"] = encoder_hidden_states.clone()
        results[f"{key}_timestep"] = timestep.clone()
        results[f"{key}_encoder_attention_mask"] = encoder_attention_mask.clone()
        results[f"{key}_output"] = output.clone()
        
        print(f"  Config {idx}: f={num_frames}, h={height}, w={width}")
        print(f"    output shape: {output.shape}, mean: {output.mean().item():.6f}")
    
    # Save model weights for Rust verification
    model_state = model.state_dict()
    for k, v in model_state.items():
        results[f"small_model.{k}"] = v.clone()
    
    return results


def capture_transformer_forward_full():
    """Capture transformer forward pass with full model (requires GPU and model weights)."""
    print("\nCapturing transformer forward (full model)...")
    
    results = {}
    
    # Check if model weights exist
    model_path = "models/models--Lightricks--LTX-Video-0.9.5/transformer"
    if not os.path.exists(model_path):
        print(f"  Skipping: model not found at {model_path}")
        return results
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use F32 for max precision comparison
    
    print(f"  Loading model from {model_path}...")
    model = LTXVideoTransformer3DModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()
    
    # Test with typical LTX-Video dimensions
    test_configs = [
        (2, 8, 8),     # Small test
        (4, 16, 16),   # Medium test
    ]
    
    for idx, (num_frames, height, width) in enumerate(test_configs):
        torch.manual_seed(42 + idx)
        
        batch_size = 1
        seq_len = num_frames * height * width
        
        hidden_states = torch.randn(batch_size, seq_len, 128, device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, 128, 4096, device=device, dtype=dtype)
        timestep = torch.tensor([500.0], device=device, dtype=dtype)
        encoder_attention_mask = torch.ones(batch_size, 128, device=device, dtype=dtype)
        
        rope_scale = (20.0 / 25.0, 32.0, 32.0)  # Typical LTX-Video values
        
        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0]
        
        key = f"full_f{num_frames}_h{height}_w{width}"
        results[f"{key}_hidden_states"] = hidden_states.cpu().clone()
        results[f"{key}_encoder_hidden_states"] = encoder_hidden_states.cpu().clone()
        results[f"{key}_timestep"] = timestep.cpu().clone()
        results[f"{key}_encoder_attention_mask"] = encoder_attention_mask.cpu().clone()
        results[f"{key}_output"] = output.cpu().clone()
        
        print(f"  Config {idx}: f={num_frames}, h={height}, w={width}")
        print(f"    output shape: {output.shape}, mean: {output.mean().item():.6f}")
    
    return results


def capture_adalayernorm_modulation():
    """Capture AdaLayerNorm modulation values (scale, shift, gate)."""
    print("\nCapturing AdaLayerNorm modulation...")
    
    results = {}
    
    # Create small model to extract modulation values
    model = LTXVideoTransformer3DModel(
        in_channels=32,
        out_channels=32,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=4,
        attention_head_dim=16,
        cross_attention_dim=64,
        num_layers=2,
        caption_channels=64,
        qk_norm="rms_norm_across_heads",
    )
    model.eval()
    
    inner_dim = 4 * 16  # num_heads * head_dim = 64
    
    # Test various timesteps
    timesteps = [0.0, 100.0, 500.0, 900.0, 1000.0]
    
    for t in timesteps:
        torch.manual_seed(42)
        
        batch_size = 1
        timestep = torch.tensor([t], dtype=torch.float32)
        
        # Get time embedding
        temb, embedded_timestep = model.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=torch.float32,
        )
        
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
        
        # Get scale_shift_table values
        scale_shift_values = model.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        
        results[f"adaln_timestep_{int(t)}"] = timestep.clone()
        results[f"adaln_temb_{int(t)}"] = temb.clone()
        results[f"adaln_embedded_timestep_{int(t)}"] = embedded_timestep.clone()
        results[f"adaln_shift_{int(t)}"] = shift.clone()
        results[f"adaln_scale_{int(t)}"] = scale.clone()
        
        print(f"  t={t}: shift mean={shift.mean().item():.6f}, scale mean={scale.mean().item():.6f}")
    
    # Also capture block-level modulation
    block = model.transformer_blocks[0]
    
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 64
    
    hidden_states = torch.randn(batch_size, seq_len, inner_dim, dtype=torch.float32)
    timestep = torch.tensor([500.0], dtype=torch.float32)
    
    temb, _ = model.time_embed(timestep.flatten(), batch_size=batch_size, hidden_dtype=torch.float32)
    temb = temb.view(batch_size, -1, temb.size(-1))
    
    # Extract block modulation values
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
    
    results["block_ada_shift_msa"] = shift_msa.clone()
    results["block_ada_scale_msa"] = scale_msa.clone()
    results["block_ada_gate_msa"] = gate_msa.clone()
    results["block_ada_shift_mlp"] = shift_mlp.clone()
    results["block_ada_scale_mlp"] = scale_mlp.clone()
    results["block_ada_gate_mlp"] = gate_mlp.clone()
    
    print(f"  Block modulation: shift_msa mean={shift_msa.mean().item():.6f}")
    
    return results


def capture_frequency_computation():
    """Capture the exact frequency computation for RoPE verification."""
    print("\nCapturing frequency computation details...")
    
    results = {}
    
    dim = 2048
    theta = 10000.0
    
    # Compute frequencies as done in LTXVideoRotaryPosEmbed
    start = 1.0
    end = theta
    
    freqs = theta ** torch.linspace(
        math.log(start, theta),
        math.log(end, theta),
        dim // 6,
        dtype=torch.float32,
    )
    freqs = freqs * math.pi / 2.0
    
    results["freq_base"] = freqs.clone()
    results["freq_dim"] = torch.tensor([dim], dtype=torch.int64)
    results["freq_theta"] = torch.tensor([theta], dtype=torch.float32)
    
    print(f"  dim={dim}, theta={theta}")
    print(f"  freqs shape: {freqs.shape}")
    print(f"  freqs range: [{freqs.min().item():.4f}, {freqs.max().item():.4f}]")
    
    # Test with various coordinate values
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [0.1, 0.2, 0.3],
    ], dtype=torch.float32)
    
    for i, coord in enumerate(coords):
        # Compute freqs * (coord * 2 - 1) for each dimension
        coord_scaled = coord * 2 - 1
        freq_f = freqs * coord_scaled[0]
        freq_h = freqs * coord_scaled[1]
        freq_w = freqs * coord_scaled[2]
        
        results[f"freq_coord{i}_f"] = freq_f.clone()
        results[f"freq_coord{i}_h"] = freq_h.clone()
        results[f"freq_coord{i}_w"] = freq_w.clone()
        results[f"freq_coord{i}_cos_f"] = freq_f.cos().clone()
        results[f"freq_coord{i}_sin_f"] = freq_f.sin().clone()
        
        print(f"  coord {i}: {coord.tolist()}")
    
    return results


def main():
    print("=" * 60)
    print("Capturing LTXVideoTransformer3DModel reference data")
    print("=" * 60)
    
    all_results = {}
    
    # Capture all reference data
    all_results.update(capture_rope_frequencies())
    all_results.update(capture_rope_with_video_coords())
    all_results.update(capture_apply_rotary_emb())
    all_results.update(capture_transformer_forward_small())
    all_results.update(capture_transformer_forward_full())
    all_results.update(capture_adalayernorm_modulation())
    all_results.update(capture_frequency_computation())
    
    # Save metadata
    metadata = {
        "description": "LTXVideoTransformer3DModel parity reference data",
        "diffusers_version": str(getattr(__import__('diffusers'), '__version__', 'unknown')),
        "torch_version": torch.__version__,
        "num_tensors": str(len(all_results)),
    }
    
    # Save to safetensors
    output_path = "gen_transformer_parity.safetensors"
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
