#!/usr/bin/env python3
"""
Check if VampNet classifier has bias.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

print("Checking VampNet classifier structure...")
print(f"Classifier type: {type(vampnet.classifier)}")
print(f"Classifier modules: {list(vampnet.classifier.named_modules())}")

# Get the Conv1d layer
conv_layer = vampnet.classifier.layers[0]
print(f"\nConv1d layer:")
print(f"  Weight shape: {conv_layer.weight.shape}")
print(f"  Has bias: {conv_layer.bias is not None}")

if conv_layer.bias is not None:
    print(f"  Bias shape: {conv_layer.bias.shape}")
    print(f"  Bias stats: mean={conv_layer.bias.mean():.4f}, std={conv_layer.bias.std():.4f}")
    
    # Check if bias is non-zero
    if conv_layer.bias.abs().max() > 1e-6:
        print("\n⚠️ Classifier has non-zero bias! This needs to be transferred.")
        
        # Show some bias values
        print(f"\nFirst 10 bias values: {conv_layer.bias[:10]}")

# Check our output projections
print("\n\nChecking our output projection structure...")
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9

model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=1
)

print("Our output projections:")
for i, proj in enumerate(model.output_projs):
    print(f"  Proj {i}: Linear layer with bias={proj.bias is not None}")

# Check Conv1d vs Linear difference
print("\n\nConv1d vs Linear comparison:")
print("VampNet uses Conv1d with kernel_size=1, which is equivalent to Linear")
print("But the weight shapes are different:")
print(f"  Conv1d: [out_channels, in_channels, kernel_size] = {conv_layer.weight.shape}")
print(f"  Linear: [out_features, in_features] = {model.output_projs[0].weight.shape}")
print("\nNeed to squeeze the kernel dimension when transferring weights!")