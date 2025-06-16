#!/usr/bin/env python3
"""
Check why correlation is near zero when values match.
"""

import torch
import numpy as np

# Create two identical tensors
a = torch.randn(100)
b = a.clone()

# Check correlation
corr1 = np.corrcoef(a.numpy(), b.numpy())[0, 1]
print(f"Correlation of identical tensors: {corr1:.4f}")

# Add small noise
b_noisy = a + torch.randn_like(a) * 0.00001
corr2 = np.corrcoef(a.numpy(), b_noisy.numpy())[0, 1]
print(f"Correlation with tiny noise: {corr2:.4f}")

# Now check if the issue is with our comparison
print("\n\nChecking our comparison:")

# Simulate the data structure
vampnet_classifier = torch.randn(1, 4096, 10)
v11_output = torch.zeros(1, 4, 10, 1025)

# Copy values correctly
for cb in range(4):
    vamp_start = cb * 1024
    vamp_end = (cb + 1) * 1024
    v11_output[0, cb, :, :1024] = vampnet_classifier[0, vamp_start:vamp_end, :].t()

# Now compare
vamp_flat = vampnet_classifier[0, :4096, :].flatten()
v11_flat = v11_output[0, :, :, :1024].flatten()

print(f"VampNet flat shape: {vamp_flat.shape}")
print(f"V11 flat shape: {v11_flat.shape}")

# Check if they're the same
if vamp_flat.shape == v11_flat.shape:
    # They should NOT be the same because of different ordering
    corr = np.corrcoef(vamp_flat.numpy(), v11_flat.numpy())[0, 1]
    print(f"Correlation with wrong ordering: {corr:.4f}")

# Correct way to flatten
print("\n\nCorrect flattening:")
vamp_correct = []
v11_correct = []

for cb in range(4):
    for pos in range(10):
        vamp_start = cb * 1024
        vamp_end = (cb + 1) * 1024
        vamp_vec = vampnet_classifier[0, vamp_start:vamp_end, pos]
        v11_vec = v11_output[0, cb, pos, :1024]
        
        vamp_correct.append(vamp_vec)
        v11_correct.append(v11_vec)

vamp_correct = torch.cat(vamp_correct)
v11_correct = torch.cat(v11_correct)

corr_correct = np.corrcoef(vamp_correct.numpy(), v11_correct.numpy())[0, 1]
print(f"Correlation with correct ordering: {corr_correct:.4f}")

# Check max diff
diff = (vamp_correct - v11_correct).abs()
print(f"Max difference: {diff.max():.6f}")
print(f"Mean difference: {diff.mean():.6f}")