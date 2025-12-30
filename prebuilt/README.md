# Prebuilt Artifacts

[![RU](README.RU.md)](README.RU.md) [![EN](README.md)](README.md)

This directory contains precompiled build artifacts for **Candle Video** dependencies (Flash Attention and Candle Kernels) to significantly reduce build times.

## 📦 Contents

The directory contains build artifacts for the following crates:

- **`candle-flash-attn-59809d6940b6f0bb/`**
  - Contains `libflashattention.a` (~230 MB) and object files.
- **`candle-kernels-00d4cbab2c6a62c1/`**
  - Contains compiled PTX files and `libmoe.a`.

## ℹ️ Build Information

These artifacts were compiled with the following configuration:

- **Candle Version**: 0.9.2-alpha.2
- **CUDA Version**: 12.6
- **Windows SDK**: 10.0.19041
- **OS**: Windows

## 🚀 Usage

These artifacts are intended to populate the `target/` directory to skip recompilation of CUDA kernels (which usually takes 15-20 minutes).

To use them:
1. Ensure your local environment matches the versions above.
2. Copy the contents of these directories into the corresponding `target/debug/build/` or `target/release/build/` directories created by Cargo.
   *Note: Cargo directory names contain hashes (e.g., `candle-flash-attn-<hash>`). You may need to copy the *contents* of the prebuilt folders into your locally generated folders.*

## ⚠️ Compatibility Warning

These binary artifacts are highly environment-specific. They will only work if:
- You are using the exact same library versions.
- Your CUDA toolkit version matches.
- Your GPU architecture is compatible.

If you encounter linking errors or runtime crashes, delete these artifacts from your `target` directory and allow Cargo to rebuild them from source.
