#!/usr/bin/env python3
"""
Test to understand VampNet's mask semantics and fix ONNX implementation.

Key findings:
1. In VampNet, mask value of 1 = MASKED (will be replaced)
2. In VampNet, mask value of 0 = UNMASKED (will be preserved)
3. The periodic_mask function SETS positions to 0 (unmasked) at periodic intervals
4. With rand_mask_intensity=0.0, linear_random returns all zeros (nothing masked)
5. The mask operations use AND logic to combine masks
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import VampNet's mask functions
from vampnet.mask import (
    full_mask, empty_mask, linear_random, periodic_mask, 
    codebook_mask, mask_and, mask_or
)

print("=" * 80)
print("VAMPNET MASK SEMANTICS TEST")
print("=" * 80)

# Create test tensor
test_shape = (1, 14, 20)  # Small for easy visualization
x = torch.zeros(test_shape)

print("\n1. Testing basic masks:")
print("-" * 40)

# Full mask (all ones)
full = full_mask(x)
print(f"full_mask: {full[0, 0, :].tolist()}")
print(f"  -> All {full[0, 0, 0].item()} = everything MASKED")

# Empty mask (all zeros)
empty = empty_mask(x)
print(f"empty_mask: {empty[0, 0, :].tolist()}")
print(f"  -> All {empty[0, 0, 0].item()} = everything UNMASKED")

print("\n2. Testing linear_random with different intensities:")
print("-" * 40)

# Set seed for reproducibility
torch.manual_seed(42)

# Test rand_mask_intensity = 0.0
mask_0 = linear_random(x, torch.tensor(0.0))
print(f"linear_random(0.0): {mask_0[0, 0, :].tolist()}")
print(f"  -> Intensity 0.0 produces all {mask_0[0, 0, 0].item()} (nothing masked)")

# Test rand_mask_intensity = 1.0
torch.manual_seed(42)
mask_1 = linear_random(x, torch.tensor(1.0))
print(f"linear_random(1.0): {mask_1[0, 0, :].tolist()}")
print(f"  -> Intensity 1.0 produces all {mask_1[0, 0, 0].item()} (everything masked)")

# Test rand_mask_intensity = 0.5
torch.manual_seed(42)
mask_5 = linear_random(x, torch.tensor(0.5))
print(f"linear_random(0.5): {mask_5[0, 0, :].tolist()}")
print(f"  -> Intensity 0.5 produces ~50% ones (masked)")

print("\n3. Testing periodic_mask:")
print("-" * 40)

# Periodic mask with period=5
pmask = periodic_mask(x, period=5, width=1)
print(f"periodic_mask(5): {pmask[0, 0, :].tolist()}")
print(f"  -> Positions 0, 5, 10, 15 are {pmask[0, 0, 0].item()} (UNMASKED)")
print(f"  -> Other positions are {pmask[0, 0, 1].item()} (MASKED)")

print("\n4. Testing mask combination (AND operation):")
print("-" * 40)

# Start with all masked (ones)
initial = full_mask(x)
print(f"Initial (all masked): {initial[0, 0, :].tolist()}")

# Apply periodic mask (has zeros at periodic positions)
combined = mask_and(initial, pmask)
print(f"After periodic AND:  {combined[0, 0, :].tolist()}")
print("  -> AND operation preserves zeros (unmasked positions)")

print("\n5. Testing VampNet's build_mask logic:")
print("-" * 40)

# Simulate VampNet's build_mask with rand_mask_intensity=0.0 and periodic_prompt=7
print("Parameters: rand_mask_intensity=0.0, periodic_prompt=7")

# Step 1: linear_random with intensity 0.0
step1 = linear_random(x, torch.tensor(0.0))
print(f"Step 1 - linear_random(0.0): {step1[0, 0, :].tolist()}")

# Step 2: periodic_mask
step2 = periodic_mask(x, period=7, width=1)
print(f"Step 2 - periodic_mask(7):   {step2[0, 0, :].tolist()}")

# Step 3: mask_and
final = mask_and(step1, step2)
print(f"Step 3 - mask_and:           {final[0, 0, :].tolist()}")
print("\nResult: All zeros! Nothing is masked because:")
print("  - linear_random(0.0) returns all zeros")
print("  - mask_and(zeros, anything) = zeros")

print("\n6. Testing codebook_mask:")
print("-" * 40)

# Test codebook mask
cb_mask = codebook_mask(empty_mask(x), val1=3)
print(f"Codebook mask shape: {cb_mask.shape}")
print(f"Codebook 0: {cb_mask[0, 0, :10].tolist()}")
print(f"Codebook 3: {cb_mask[0, 3, :10].tolist()}")
print("  -> Codebooks >= 3 are set to 1 (masked)")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("1. VampNet uses 1 = MASKED, 0 = UNMASKED")
print("2. periodic_mask creates ZEROS at periodic positions (preserves them)")
print("3. linear_random(0.0) returns all ZEROS (nothing masked)")
print("4. mask_and preserves zeros (unmasked positions)")
print("5. The ONNX implementation had this inverted!")