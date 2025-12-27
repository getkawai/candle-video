"""
Compare DiT forward pass layer-by-layer between Python diffusers and Rust

This script:
1. Loads the same model weights
2. Runs one step of DiT with known inputs
3. Outputs intermediate values for comparison with Rust
"""
import torch
import numpy as np
from pathlib import Path

device = "cuda"
dtype = torch.bfloat16

print("Loading diffusers transformer...")
try:
    from diffusers import LTXVideoTransformer3DModel
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=dtype
    ).to(device)
    transformer.eval()
    print("Transformer loaded!")
except Exception as e:
    print(f"Error loading transformer: {e}")
    exit(1)

# Set up test inputs matching Rust
batch_size = 1
seq_len = 3 * 16 * 24  # T=3, H=16, W=24 = 1152
hidden_dim = 128
text_seq_len = 6
text_dim = 4096

# Random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create test inputs
hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim, device=device, dtype=dtype)
timestep = torch.tensor([1.0], device=device, dtype=dtype)  # First step
encoder_attention_mask = torch.ones(batch_size, text_seq_len, device=device)

print(f"Hidden states: {hidden_states.shape}, range: {hidden_states.float().min():.4f} to {hidden_states.float().max():.4f}")
print(f"Encoder hidden states: {encoder_hidden_states.shape}")
print(f"Timestep: {timestep}")

# Check transformer config
print("\nTransformer config:")
print(f"  in_channels: {transformer.config.in_channels}")
print(f"  num_layers: {transformer.config.num_layers}")
print(f"  num_attention_heads: {transformer.config.num_attention_heads}")
print(f"  attention_head_dim: {transformer.config.attention_head_dim}")

# Run forward pass with hooks to capture intermediate values
intermediates = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediates[name] = output[0].detach().clone()
        else:
            intermediates[name] = output.detach().clone()
    return hook

# Register hooks on key layers
transformer.proj_in.register_forward_hook(make_hook("proj_in"))
transformer.time_embed.register_forward_hook(make_hook("time_embed"))
transformer.caption_projection.register_forward_hook(make_hook("caption_projection"))
transformer.norm_out.register_forward_hook(make_hook("norm_out"))
transformer.proj_out.register_forward_hook(make_hook("proj_out"))

# Hook first transformer block
if len(transformer.transformer_blocks) > 0:
    first_block = transformer.transformer_blocks[0]
    first_block.register_forward_hook(make_hook("block_0_output"))
    first_block.norm1.register_forward_hook(make_hook("block_0_norm1"))
    first_block.attn1.register_forward_hook(make_hook("block_0_attn1"))
    first_block.norm2.register_forward_hook(make_hook("block_0_norm2"))
    first_block.attn2.register_forward_hook(make_hook("block_0_attn2"))
    first_block.ff.register_forward_hook(make_hook("block_0_ff"))

print("\nRunning forward pass...")
with torch.no_grad():
    output = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
        num_frames=3,
        height=16,
        width=24,
        rope_interpolation_scale=None,  # Match Rust for now
        return_dict=False,
    )[0]

print(f"\nOutput shape: {output.shape}")
print(f"Output range: {output.float().min():.4f} to {output.float().max():.4f}")

# Print intermediate values
print("\n=== Intermediate Values ===")
for name, tensor in intermediates.items():
    t = tensor.float()
    print(f"{name}: shape={list(tensor.shape)}, range=[{t.min():.4f}, {t.max():.4f}], mean={t.mean():.4f}")

# Save intermediates for Rust comparison
save_dir = Path("output/dit_debug")
save_dir.mkdir(exist_ok=True)

for name, tensor in intermediates.items():
    np.save(save_dir / f"{name}.npy", tensor.float().cpu().numpy())
    print(f"Saved {name} to {save_dir / f'{name}.npy'}")

# Save inputs for Rust to use
np.save(save_dir / "hidden_states.npy", hidden_states.float().cpu().numpy())
np.save(save_dir / "encoder_hidden_states.npy", encoder_hidden_states.float().cpu().numpy())
np.save(save_dir / "output.npy", output.float().cpu().numpy())

print(f"\nAll data saved to {save_dir}")
print("Use these files to compare with Rust DiT output layer by layer")
