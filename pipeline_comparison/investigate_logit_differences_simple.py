#!/usr/bin/env python3
"""
Simple investigation of logit differences between VampNet and ONNX models.
Focus on the key issue: why outputs don't match.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("SIMPLE LOGIT DIFFERENCE INVESTIGATION")
print("=" * 80)

# Import interfaces
from vampnet_onnx import Interface as ONNXInterface
from vampnet.interface import Interface as VampNetInterface

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Initialize interfaces
print("\n1. Loading interfaces...")
device = torch.device('cpu')

# VampNet interface
vampnet_interface = VampNetInterface(
    codec_ckpt="../models/vampnet/codec.pth",
    coarse_ckpt="../models/vampnet/coarse.pth",
    coarse2fine_ckpt="../models/vampnet/c2f.pth",
    wavebeat_ckpt="../models/vampnet/wavebeat.pth",
)
vampnet_interface.to(device)
print("✓ VampNet interface initialized")

# ONNX interface
onnx_interface = ONNXInterface.from_default_models(device='cpu')
print("✓ ONNX interface initialized")

# Create test input
print("\n2. Creating test input...")
test_tokens = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
print(f"Test tokens shape: {test_tokens.shape}")
print(f"Masked positions: {mask.sum()}")

# Apply mask
masked_tokens = test_tokens.clone()
masked_tokens[mask] = 1024

# Get logits from both models
print("\n3. Getting model outputs...")

# VampNet
print("\nVampNet forward pass:")
with torch.no_grad():
    # VampNet's forward method does:
    # 1. x = self.embedding(x) - expects the concatenated latents
    # 2. The concatenated latents come from embedding.from_codes()
    
    vampnet_model = vampnet_interface.coarse
    
    # Get the concatenated latents
    z_latents = vampnet_model.embedding.from_codes(masked_tokens, vampnet_interface.codec)
    print(f"  Concatenated latents shape: {z_latents.shape}")
    print(f"  Latents stats: mean={z_latents.mean():.4f}, std={z_latents.std():.4f}")
    
    # Now use the model's forward, which expects these latents
    vampnet_out = vampnet_model(z_latents)
    print(f"  Output shape: {vampnet_out.shape}")
    print(f"  Output stats: mean={vampnet_out.mean():.4f}, std={vampnet_out.std():.4f}, range=[{vampnet_out.min():.4f}, {vampnet_out.max():.4f}]")

# ONNX
print("\nONNX forward pass:")
onnx_out = onnx_interface.coarse_session.run(None, {
    'codes': masked_tokens.numpy().astype(np.int64),
    'mask': mask.numpy()
})[0]
print(f"  Output shape: {onnx_out.shape}")
print(f"  Output stats: mean={onnx_out.mean():.4f}, std={onnx_out.std():.4f}, range=[{onnx_out.min():.4f}, {onnx_out.max():.4f}]")

# Compare
print("\n4. Comparison:")
print("-" * 60)

# Reshape VampNet output to match ONNX
# VampNet: [batch, vocab_size, n_codebooks * seq_len]
# ONNX: [batch, n_codebooks, seq_len, vocab_size + 1]
batch_size, vocab_size, total_positions = vampnet_out.shape
n_codebooks = 4
seq_len = total_positions // n_codebooks

vampnet_reshaped = vampnet_out.reshape(batch_size, vocab_size, n_codebooks, seq_len)
vampnet_reshaped = vampnet_reshaped.transpose(1, 2).transpose(2, 3)  # [batch, n_cb, seq, vocab]
print(f"VampNet reshaped: {vampnet_reshaped.shape}")

# Compare at masked positions
vampnet_masked = vampnet_reshaped[mask]
onnx_masked = onnx_out[mask][:, :1024]  # Remove mask token class

diff = (vampnet_masked - onnx_masked).abs()
print(f"\nAt masked positions:")
print(f"  Mean absolute difference: {diff.mean():.4f}")
print(f"  Max absolute difference: {diff.max():.4f}")
print(f"  Correlation: {np.corrcoef(vampnet_masked.flatten(), onnx_masked.flatten())[0,1]:.4f}")

# Check if it's just a scaling issue
vampnet_norm = vampnet_masked / (vampnet_masked.std() + 1e-8)
onnx_norm = onnx_masked / (onnx_masked.std() + 1e-8)
norm_diff = (vampnet_norm - onnx_norm).abs()
print(f"\nAfter normalization:")
print(f"  Mean absolute difference: {norm_diff.mean():.4f}")
print(f"  Correlation: {np.corrcoef(vampnet_norm.flatten(), onnx_norm.flatten())[0,1]:.4f}")

# Check a few specific values
print(f"\nSample logit values (first 5 masked positions):")
for i in range(min(5, vampnet_masked.shape[0])):
    print(f"  Position {i}:")
    print(f"    VampNet: {vampnet_masked[i, :5].tolist()}")
    print(f"    ONNX:    {onnx_masked[i, :5].tolist()}")

# Key finding
print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print("1. VampNet doesn't use explicit positional encoding")
print("2. ONNX model was exported with positional encoding")
print("3. This is likely the main source of the difference")
print("4. Need to check the ONNX export script for how PE is handled")