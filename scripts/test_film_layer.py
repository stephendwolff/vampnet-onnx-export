#!/usr/bin/env python3
"""
Test FiLM layer implementation.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Load VampNet
from vampnet.modules.transformer import VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()

# Test FiLM layer
print("Testing FiLM layer...")
layer = vampnet.transformer.layers[0]

# Create test input
torch.manual_seed(42)
x = torch.randn(1, 10, 1280)
x_norm = layer.norm_3(x)

print(f"Input shape: {x.shape}")
print(f"After norm: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")

# Apply FiLM
with torch.no_grad():
    # Check FiLM parameters
    film = layer.film_3
    print(f"\nFiLM input_dim: {film.input_dim}")
    print(f"FiLM output_dim: {film.output_dim}")
    
    # VampNet passes the same input for both x and r
    out = film(x_norm, x_norm)
    print(f"After FiLM: mean={out.mean():.4f}, std={out.std():.4f}")
    
    # Check what FiLM actually does
    if film.input_dim > 0:
        beta = film.beta(x_norm)
        gamma = film.gamma(x_norm) 
        print(f"\nBeta shape: {beta.shape}, mean={beta.mean():.4f}")
        print(f"Gamma shape: {gamma.shape}, mean={gamma.mean():.4f}")
        
        # Manual computation
        beta_view = beta.view(x_norm.size(0), film.output_dim, 1)
        gamma_view = gamma.view(x_norm.size(0), film.output_dim, 1)
        manual_out = x_norm * (gamma_view + 1) + beta_view
        
        # Transpose to match expected shape
        manual_out = manual_out.transpose(1, 2)
        
        print(f"\nManual FiLM output shape: {manual_out.shape}")
        print(f"Difference from FiLM output: {(out - manual_out).abs().max():.6f}")
    else:
        print("\nFiLM has input_dim=0, returning input unchanged")

# Now check our FiLM implementation
print("\n\nTesting our FiLM implementation...")
from scripts.custom_ops.film_onnx import SimpleFiLM

our_film = SimpleFiLM(1280, 1280)

# Transfer weights
if vampnet.transformer.layers[0].film_3.input_dim > 0:
    our_film.gamma_weight.data = layer.film_3.gamma.weight.data.t()
    our_film.gamma_bias.data = layer.film_3.gamma.bias.data
    our_film.beta_weight.data = layer.film_3.beta.weight.data.t()
    our_film.beta_bias.data = layer.film_3.beta.bias.data
    print("Transferred FiLM weights")

with torch.no_grad():
    our_out = our_film(x_norm, x_norm)
    print(f"Our FiLM output: mean={our_out.mean():.4f}, std={our_out.std():.4f}")
    
    diff = (our_out - out).abs()
    print(f"Difference - mean: {diff.mean():.6f}, max: {diff.max():.6f}")
    
    if diff.max() > 0.001:
        print("\n⚠️ FiLM implementations differ!")
    else:
        print("\n✓ FiLM implementations match!")