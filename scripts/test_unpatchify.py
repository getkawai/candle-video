"""
Verify exact unpatchify behavior with small test tensors
"""
import torch
import numpy as np

def python_unpatchify(x, patch_size, patch_size_t=1):
    """
    Python decoder-style unpatchify
    """
    B, C, T, H, W = x.shape
    p_t = patch_size_t
    p = patch_size
    
    # Step 1: reshape to [B, c, p_t, p, p, T, H, W]
    x = x.reshape(B, -1, p_t, p, p, T, H, W)
    print(f"After reshape: {x.shape}")
    
    # Step 2: permute (0, 1, 5, 2, 6, 4, 7, 3) - decoder version
    x = x.permute(0, 1, 5, 2, 6, 4, 7, 3)
    print(f"After permute: {x.shape}")
    
    # Step 3: flatten
    x = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    print(f"After flatten: {x.shape}")
    
    return x

# Test with small tensor 
# Shape matching decoder output just before final unpatchify
# B=1, C=48, T=2, H=4, W=6 (but scaled down for testing)
B, C, T, H, W = 1, 48, 2, 4, 6
p = 4  # patch size
p_t = 1

# Create tensor with known values: channel * 1000 + T * 100 + H * 10 + W
x = torch.zeros(B, C, T, H, W)
for c in range(C):
    for t in range(T):
        for h in range(H):
            for w in range(W):
                x[0, c, t, h, w] = c * 1000 + t * 100 + h * 10 + w

print("Input shape:", x.shape)
print("Input sample values:")
print(f"  [0,0,0,0,0] = {x[0,0,0,0,0]:.0f}")  # channel 0
print(f"  [0,1,0,0,0] = {x[0,1,0,0,0]:.0f}")  # channel 1
print(f"  [0,16,0,0,0] = {x[0,16,0,0,0]:.0f}")  # channel 16
print(f"  [0,0,1,0,0] = {x[0,0,1,0,0]:.0f}")  # t=1

# Apply unpatchify
out = python_unpatchify(x, p, p_t)
print("\nOutput shape:", out.shape)

# Expected output: [B, c=3, T*p_t=2, H*?, W*?]
# With p=4, p_t=1:
# H*p_w + W*p_h... but since p_h=p_w=4:
# H_out = H * 4 = 16
# W_out = W * 4 = 24
# So [1, 3, 2, 16, 24]

print("\nOutput sample values at (0,0,0,y,x):")
for y in range(4):
    for x_idx in range(4):
        val = out[0, 0, 0, y, x_idx].item()
        print(f"  y={y}, x={x_idx}: {val:.0f}")
