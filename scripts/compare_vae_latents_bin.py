#!/usr/bin/env python3
"""
Compare VAE decode outputs for the same `latents.bin`:
  - Rust VAE output (produced by `cargo run --bin vae_compare`) if available
  - diffusers VAE output (decoded locally from the same safetensors single-file)

No network is required; diffusers is imported from `tp/diffusers/src`.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_latents_bin(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        ndims = struct.unpack("<Q", f.read(8))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
        numel = 1
        for d in dims:
            numel *= d
        data = np.frombuffer(f.read(numel * 4), dtype=np.float32)
    return data.reshape(dims).copy()


def save_frame_png(video_bcthw: np.ndarray, path: Path, frame_idx: int = 0) -> None:
    # video in [-1, 1], shape (B, C, T, H, W)
    frame = video_bcthw[0, :, frame_idx, :, :]  # (C,H,W)
    frame = np.transpose(frame, (1, 2, 0))  # (H,W,C)
    frame = np.clip((frame + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, default=Path("output/latents.bin"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ltxv-2b-0.9.8-distilled/ltxv-2b-0.9.8-distilled.safetensors"),
        help="Single-file safetensors checkpoint (contains VAE).",
    )
    parser.add_argument("--timestep", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("output/vae_compare"))
    parser.add_argument(
        "--rust-video",
        type=Path,
        default=Path("output/vae_compare/rust_video.bin"),
        help="Optional Rust output produced by `vae_compare`.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.latents.exists():
        raise SystemExit(f"Missing latents file: {args.latents}")
    if not args.model.exists():
        raise SystemExit(f"Missing model file: {args.model}")

    latents = read_latents_bin(args.latents)
    print(f"latents: shape={latents.shape} range=[{latents.min():.4f}, {latents.max():.4f}]")

    # Import vendored diffusers (no network)
    sys.path.insert(0, os.path.abspath("tp/diffusers/src"))
    from diffusers import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_single_file(str(args.model), torch_dtype=torch.float32)
    vae.eval()

    latents_t = torch.from_numpy(latents).float()
    timestep_t = torch.tensor([args.timestep], dtype=torch.float32)

    with torch.no_grad():
        decoded = vae.decode(latents_t, temb=timestep_t, return_dict=False)[0]
    decoded_np = decoded.cpu().numpy().astype(np.float32)
    print(
        f"diffusers decoded: shape={decoded_np.shape} range=[{decoded_np.min():.4f}, {decoded_np.max():.4f}]"
    )

    diffusers_frame_path = args.out_dir / "diffusers_frame_0000.png"
    save_frame_png(decoded_np, diffusers_frame_path, frame_idx=0)
    print(f"saved: {diffusers_frame_path}")

    if args.rust_video.exists():
        rust_video = read_latents_bin(args.rust_video)
        print(
            f"rust decoded: shape={rust_video.shape} range=[{rust_video.min():.4f}, {rust_video.max():.4f}]"
        )

        if rust_video.shape != decoded_np.shape:
            raise SystemExit(f"shape mismatch: rust={rust_video.shape} diffusers={decoded_np.shape}")

        diff = np.abs(rust_video - decoded_np)
        print(f"abs diff: max={diff.max():.6f} mean={diff.mean():.6f}")

        rust_frame_path = args.out_dir / "rust_frame_0000.png"
        save_frame_png(rust_video, rust_frame_path, frame_idx=0)
        print(f"saved: {rust_frame_path}")

        diff_frame = diff[0, :, 0, :, :]  # (C,H,W)
        diff_frame = np.transpose(diff_frame, (1, 2, 0))
        diff_frame = np.clip(diff_frame / (diff_frame.max() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
        diff_path = args.out_dir / "abs_diff_frame_0000.png"
        Image.fromarray(diff_frame).save(diff_path)
        print(f"saved: {diff_path}")
    else:
        print(f"note: rust output not found at {args.rust_video} (run `cargo run --bin vae_compare` first)")


if __name__ == "__main__":
    main()

