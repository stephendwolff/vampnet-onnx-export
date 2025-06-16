#!/usr/bin/env python3
"""
Debug weight transfer issue in V10.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

print("Debugging weight transfer...")

# Get classifier weights
conv_layer = vampnet.classifier.layers[0]
print(f"Conv1d weight shape: {conv_layer.weight.shape}")
print(f"Conv1d bias shape: {conv_layer.bias.shape}")

# Extract weights
classifier_weight = conv_layer.weight.data.squeeze(-1)  # Remove kernel dimension
print(f"\nSqueezed weight shape: {classifier_weight.shape}")

# Check first codebook weights
cb0_weight = classifier_weight[:1024]
print(f"\nCodebook 0 weight shape: {cb0_weight.shape}")
print(f"Weight stats: mean={cb0_weight.mean():.6f}, std={cb0_weight.std():.6f}")
print(f"Weight range: [{cb0_weight.min():.6f}, {cb0_weight.max():.6f}]")

# Sample some values
print(f"\nFirst 5 weights of first row: {cb0_weight[0, :5]}")
print(f"First 5 weights of last row: {cb0_weight[-1, :5]}")

# Check if there's any special initialization or modification
print(f"\n\nChecking for special patterns...")

# Check if weights are normalized
row_norms = cb0_weight.norm(dim=1)
print(f"Row norms: mean={row_norms.mean():.6f}, std={row_norms.std():.6f}")

col_norms = cb0_weight.norm(dim=0)
print(f"Col norms: mean={col_norms.mean():.6f}, std={col_norms.std():.6f}")

# Check if there's weight decay or other regularization
print(f"\n\nChecking VampNet configuration...")
if hasattr(vampnet, 'weight_decay'):
    print(f"Weight decay: {vampnet.weight_decay}")

# Let's also check the actual weight values we're copying
print("\n\nManual weight transfer test...")
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10

model = VampNetTransformerV10(n_layers=1)  # Just 1 layer for testing

# Manual transfer
for i in range(4):
    vamp_start = i * 1024
    vamp_end = (i + 1) * 1024
    
    # Get VampNet weights for this codebook
    vamp_w = classifier_weight[vamp_start:vamp_end]
    vamp_b = conv_layer.bias.data[vamp_start:vamp_end]
    
    # Set V10 weights
    model.output_projs[i].weight.data[:1024] = vamp_w
    model.output_projs[i].bias.data[:1024] = vamp_b
    
    # Compare immediately
    w_match = torch.allclose(model.output_projs[i].weight.data[:1024], vamp_w)
    b_match = torch.allclose(model.output_projs[i].bias.data[:1024], vamp_b)
    
    print(f"\nCodebook {i}:")
    print(f"  Weight match: {w_match}")
    print(f"  Bias match: {b_match}")
    
    if not w_match:
        diff = (model.output_projs[i].weight.data[:1024] - vamp_w).abs()
        print(f"  Weight diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
        
        # Check shapes
        print(f"  VampNet shape: {vamp_w.shape}")
        print(f"  V10 shape: {model.output_projs[i].weight.data[:1024].shape}")

# Check the 1025th row (mask token)
print(f"\n\nChecking mask token row:")
for i in range(4):
    mask_row = model.output_projs[i].weight.data[1024]
    print(f"Codebook {i} mask row: mean={mask_row.mean():.6f}, std={mask_row.std():.6f}, zeros={(mask_row == 0).sum()}")