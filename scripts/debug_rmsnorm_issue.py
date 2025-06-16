#!/usr/bin/env python3
"""
Debug RMSNorm implementation differences.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from scripts.custom_ops.rmsnorm_onnx import SimpleRMSNorm

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

print("Comparing RMSNorm implementations...")

# Get VampNet's RMSNorm
vamp_norm = vampnet.transformer.layers[0].norm_1
print(f"VampNet RMSNorm type: {type(vamp_norm)}")

# Create our RMSNorm with same weights
our_norm = SimpleRMSNorm(1280)
our_norm.weight.data = vamp_norm.weight.data.clone()

# Test input
test_input = torch.randn(1, 10, 1280) * 10  # Large values to test stability

with torch.no_grad():
    vamp_out = vamp_norm(test_input)
    our_out = our_norm(test_input)

print(f"\nInput stats: mean={test_input.mean():.4f}, std={test_input.std():.4f}")
print(f"VampNet output: mean={vamp_out.mean():.4f}, std={vamp_out.std():.4f}")
print(f"Our output: mean={our_out.mean():.4f}, std={our_out.std():.4f}")

diff = (vamp_out - our_out).abs()
print(f"Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Check the actual implementation
print("\n\nChecking RMSNorm internals...")

# Manual RMSNorm calculation
eps = 1e-5
norm = test_input.norm(2, dim=-1, keepdim=True)
rms = norm * (1280 ** -0.5)
normalized = test_input / (rms + eps)
manual_out = normalized * vamp_norm.weight

manual_diff = (vamp_out - manual_out).abs()
print(f"Manual calculation difference: mean={manual_diff.mean():.6f}, max={manual_diff.max():.6f}")

# Check if VampNet has any special settings
if hasattr(vamp_norm, 'eps'):
    print(f"\nVampNet epsilon: {vamp_norm.eps}")
else:
    print("\nVampNet RMSNorm has no eps attribute")

# Test with extreme values
print("\n\nTesting with extreme values...")
extreme_input = torch.randn(1, 10, 1280) * 100

with torch.no_grad():
    vamp_extreme = vamp_norm(extreme_input)
    our_extreme = our_norm(extreme_input)

print(f"Extreme input range: [{extreme_input.min():.2f}, {extreme_input.max():.2f}]")
print(f"VampNet output range: [{vamp_extreme.min():.2f}, {vamp_extreme.max():.2f}]")
print(f"Our output range: [{our_extreme.min():.2f}, {our_extreme.max():.2f}]")