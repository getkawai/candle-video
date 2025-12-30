#!/usr/bin/env python3
"""
Compare reference tensors (from diffusers) with Rust implementation tensors.
Expects:
  - output/reference_tensors/*.npy (from dump_svd_reference.py)
  - output/rust_tensors/*.bin + *.shape (from Rust with DUMP_TENSORS=1)
"""
import numpy as np
from pathlib import Path
import sys


def load_rust_tensor(name: str, rust_dir: Path) -> np.ndarray:
    """Load tensor from Rust binary format (.bin + .shape files)."""
    bin_path = rust_dir / f"{name}.bin"
    shape_path = rust_dir / f"{name}.shape"
    
    if not bin_path.exists() or not shape_path.exists():
        raise FileNotFoundError(f"Rust tensor not found: {name}")
    
    # Read shape
    shape_str = shape_path.read_text().strip()
    shape = tuple(int(x) for x in shape_str.split(","))
    
    # Read binary data as float32
    data = np.fromfile(bin_path, dtype=np.float32)
    return data.reshape(shape)


def compare_tensors(name: str, ref: np.ndarray, rust: np.ndarray, atol: float = 1e-3, rtol: float = 1e-3):
    """Compare two tensors and return comparison results."""
    result = {
        "name": name,
        "ref_shape": ref.shape,
        "rust_shape": rust.shape,
        "shape_match": ref.shape == rust.shape,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "max_rel_diff": None,
        "close": False,
        "status": "❌"
    }
    
    if not result["shape_match"]:
        result["error"] = f"Shape mismatch: ref={ref.shape} vs rust={rust.shape}"
        return result
    
    # Compute differences
    abs_diff = np.abs(ref - rust)
    result["max_abs_diff"] = float(np.max(abs_diff))
    result["mean_abs_diff"] = float(np.mean(abs_diff))
    
    # Relative difference (avoid division by zero)
    ref_abs = np.abs(ref) + 1e-8
    rel_diff = abs_diff / ref_abs
    result["max_rel_diff"] = float(np.max(rel_diff))
    
    # Check if close
    result["close"] = np.allclose(ref, rust, atol=atol, rtol=rtol)
    result["status"] = "✅" if result["close"] else "⚠️"
    
    return result


def main():
    ref_dir = Path("output/reference_tensors")
    rust_dir = Path("output/rust_tensors")
    
    print("=" * 70)
    print("SVD Tensor Comparison: Diffusers vs Rust")
    print("=" * 70)
    
    # Check directories exist
    if not ref_dir.exists():
        print(f"\n❌ Reference tensors not found: {ref_dir}")
        print("   Run: python scripts/dump_svd_reference.py")
        sys.exit(1)
    
    if not rust_dir.exists():
        print(f"\n❌ Rust tensors not found: {rust_dir}")
        print("   Run: DUMP_TENSORS=1 cargo run --bin svd --release -- ...")
        sys.exit(1)
    
    # Get all reference files (.npy)
    ref_files = sorted(ref_dir.glob("*.npy"))
    # Get all rust files (.bin with matching .shape)
    rust_files = sorted(rust_dir.glob("*.bin"))
    
    ref_names = {f.stem for f in ref_files}
    rust_names = {f.stem for f in rust_files}
    
    print(f"\nReference tensors: {len(ref_names)}")
    print(f"Rust tensors: {len(rust_names)}")
    
    common = ref_names & rust_names
    only_ref = ref_names - rust_names
    only_rust = rust_names - ref_names
    
    if only_ref:
        print(f"\n⚠️  Only in reference: {sorted(only_ref)}")
    if only_rust:
        print(f"\n⚠️  Only in Rust: {sorted(only_rust)}")
    
    print("\n" + "-" * 70)
    print("Tensor Comparison Results")
    print("-" * 70)
    
    all_results = []
    for name in sorted(common):
        ref = np.load(ref_dir / f"{name}.npy")
        rust = load_rust_tensor(name, rust_dir)
        result = compare_tensors(name, ref, rust)
        all_results.append(result)
    
    # Print results in table format
    print(f"\n{'Tensor':<30} {'Status':<6} {'Shape Match':<12} {'Max Abs Diff':<14} {'Mean Abs Diff':<14}")
    print("-" * 80)
    
    passed = 0
    failed = 0
    for r in all_results:
        shape_str = "✅" if r["shape_match"] else f"❌ {r.get('error', '')}"
        max_diff = f"{r['max_abs_diff']:.6f}" if r['max_abs_diff'] is not None else "N/A"
        mean_diff = f"{r['mean_abs_diff']:.6f}" if r['mean_abs_diff'] is not None else "N/A"
        
        print(f"{r['name']:<30} {r['status']:<6} {shape_str:<12} {max_diff:<14} {mean_diff:<14}")
        
        if r["close"]:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n  Compared: {len(all_results)} tensors")
    print(f"  Passed:   {passed} ✅")
    print(f"  Failed:   {failed} ❌")
    
    if failed > 0:
        print("\n⚠️  Some tensors differ significantly!")
        print("   Check the detailed differences above.")
    else:
        print("\n✅ All tensors match within tolerance!")
    
    # Detailed diff for failed tensors
    failed_tensors = [r for r in all_results if not r["close"]]
    if failed_tensors:
        print("\n" + "-" * 70)
        print("Detailed Analysis of Failed Tensors")
        print("-" * 70)
        
        for r in failed_tensors[:5]:  # Limit to first 5
            name = r["name"]
            ref = np.load(ref_dir / f"{name}.npy")
            rust = load_rust_tensor(name, rust_dir)
            
            print(f"\n{name}:")
            print(f"  Reference: shape={ref.shape}, min={ref.min():.4f}, max={ref.max():.4f}, mean={ref.mean():.4f}")
            print(f"  Rust:      shape={rust.shape}, min={rust.min():.4f}, max={rust.max():.4f}, mean={rust.mean():.4f}")
            
            if r["shape_match"]:
                diff = np.abs(ref - rust)
                print(f"  Diff:      max={diff.max():.6f}, mean={diff.mean():.6f}")
                
                # Find location of max diff
                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  Max diff at index {max_idx}:")
                print(f"    Reference: {ref[max_idx]:.6f}")
                print(f"    Rust:      {rust[max_idx]:.6f}")


if __name__ == "__main__":
    main()
