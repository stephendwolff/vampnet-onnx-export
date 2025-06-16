#!/usr/bin/env python3
"""
Debug why weights don't match after transfer.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

print("Debugging weight mismatch...")

# Get VampNet classifier weights
conv_layer = vampnet.classifier.layers[0]
conv_weight = conv_layer.weight.data  # [4096, 1280, 1]
conv_bias = conv_layer.bias.data      # [4096]

print(f"Conv1d weight shape: {conv_weight.shape}")
print(f"Conv1d bias shape: {conv_bias.shape}")

# Create V10 model
model = VampNetTransformerV10(n_layers=1)

# Check initial state of V10 weights
print("\n1. Initial V10 weights (before transfer):")
for i in range(4):
    w = model.output_projs[i].weight.data
    print(f"   Proj {i}: mean={w.mean():.6f}, std={w.std():.6f}")

# Manual weight transfer
print("\n2. Manual weight transfer:")
classifier_weight_squeezed = conv_weight.squeeze(-1)  # [4096, 1280]

for i in range(4):
    start = i * 1024
    end = (i + 1) * 1024
    
    # Get the slice
    vamp_slice = classifier_weight_squeezed[start:end]  # [1024, 1280]
    
    print(f"\n   Codebook {i}:")
    print(f"   VampNet slice shape: {vamp_slice.shape}")
    print(f"   VampNet slice stats: mean={vamp_slice.mean():.6f}, std={vamp_slice.std():.6f}")
    
    # Before assignment
    v10_before = model.output_projs[i].weight.data.clone()
    
    # Assign
    model.output_projs[i].weight.data[:1024] = vamp_slice
    
    # After assignment
    v10_after = model.output_projs[i].weight.data[:1024]
    
    # Check if they match
    match = torch.allclose(vamp_slice, v10_after)
    print(f"   Assignment successful: {match}")
    
    if not match:
        diff = (vamp_slice - v10_after).abs()
        print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Now run the full transfer function
print("\n3. Running transfer_weights_v10:")
from scripts.export_vampnet_transformer_v10_all_relative import transfer_weights_v10

model2 = VampNetTransformerV10(n_layers=1)
transfer_weights_v10("models/vampnet/coarse.pth", model2, "models/vampnet/codec.pth")

# Compare
print("\n4. Comparing manual vs transfer function:")
for i in range(4):
    manual_w = model.output_projs[i].weight.data[:1024]
    transfer_w = model2.output_projs[i].weight.data[:1024]
    
    match = torch.allclose(manual_w, transfer_w)
    print(f"   Codebook {i} weights match: {match}")
    
    if not match:
        diff = (manual_w - transfer_w).abs()
        print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Final check - compare with original VampNet weights
print("\n5. Final comparison with VampNet:")
for i in range(4):
    start = i * 1024
    end = (i + 1) * 1024
    
    vamp_w = classifier_weight_squeezed[start:end]
    v10_w = model2.output_projs[i].weight.data[:1024]
    
    match = torch.allclose(vamp_w, v10_w)
    print(f"\n   Codebook {i}:")
    print(f"   Weights match: {match}")
    
    if not match:
        diff = (vamp_w - v10_w).abs()
        print(f"   Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
        
        # Check specific values
        print(f"   VampNet[0,0]: {vamp_w[0,0]:.6f}")
        print(f"   V10[0,0]:     {v10_w[0,0]:.6f}")
        print(f"   VampNet[0,1]: {vamp_w[0,1]:.6f}")
        print(f"   V10[0,1]:     {v10_w[0,1]:.6f}")
        
        # Check if it's a systematic issue
        ratio = v10_w[0,0] / vamp_w[0,0] if vamp_w[0,0] != 0 else 0
        print(f"   Ratio: {ratio:.6f}")