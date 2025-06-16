#!/usr/bin/env python3
"""
Verify that the fixed mask generator now matches VampNet's behavior.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vampnet.mask import linear_random, periodic_mask, mask_and, codebook_mask
from vampnet_onnx.mask_generator_proper import build_mask_proper

print("=" * 80)
print("MASK FIX VERIFICATION TEST")
print("=" * 80)

# Test configurations
test_configs = [
    {
        "name": "Test 1: rand=0.0, periodic=7, codebook=3",
        "params": {
            "rand_mask_intensity": 0.0,
            "periodic_prompt": 7,
            "upper_codebook_mask": 3,
        }
    },
    {
        "name": "Test 2: rand=1.0, periodic=0, codebook=4",
        "params": {
            "rand_mask_intensity": 1.0,
            "periodic_prompt": 0,
            "upper_codebook_mask": 4,
        }
    },
    {
        "name": "Test 3: rand=0.5, periodic=10, codebook=2",
        "params": {
            "rand_mask_intensity": 0.5,
            "periodic_prompt": 10,
            "upper_codebook_mask": 2,
        }
    },
]

# Test with larger tensor to match actual usage
test_shape = (1, 14, 100)

for config in test_configs:
    print(f"\n{config['name']}")
    print("-" * 60)
    
    # VampNet implementation
    x = torch.zeros(test_shape)
    torch.manual_seed(42)
    
    # Build VampNet mask step by step
    vampnet_mask = linear_random(x, torch.tensor(config['params']['rand_mask_intensity']))
    
    if config['params']['periodic_prompt'] > 0:
        pmask = periodic_mask(x, period=config['params']['periodic_prompt'], width=1, random_roll=True)
        vampnet_mask = mask_and(vampnet_mask, pmask)
    
    if config['params']['upper_codebook_mask'] > 0:
        vampnet_mask = codebook_mask(vampnet_mask, val1=config['params']['upper_codebook_mask'])
    
    vampnet_mask_np = vampnet_mask.numpy()
    
    # ONNX implementation
    np.random.seed(42)
    onnx_mask = build_mask_proper(
        np.zeros(test_shape),
        **config['params'],
        random_roll=True,
        seed=42
    )
    
    # Compare results
    match = np.array_equal(vampnet_mask_np, onnx_mask)
    overlap = np.mean(vampnet_mask_np == onnx_mask)
    
    print(f"Parameters: {config['params']}")
    print(f"Exact match: {match}")
    print(f"Overlap: {overlap:.1%}")
    print(f"VampNet mask ratio: {np.mean(vampnet_mask_np):.1%}")
    print(f"ONNX mask ratio: {np.mean(onnx_mask):.1%}")
    
    # Show samples from different codebooks
    print("\nSample mask values (first 20 positions):")
    for cb in [0, 2, 3, 13]:
        if cb < test_shape[1]:
            print(f"  Codebook {cb}:")
            print(f"    VampNet: {vampnet_mask_np[0, cb, :20].astype(int)}")
            print(f"    ONNX:    {onnx_mask[0, cb, :20].astype(int)}")
            if not np.array_equal(vampnet_mask_np[0, cb], onnx_mask[0, cb]):
                print(f"    MISMATCH!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Run a deterministic test without random roll
print("\nDeterministic test (no random roll):")
x = torch.zeros(test_shape)
torch.manual_seed(42)

# VampNet
vampnet_mask = linear_random(x, torch.tensor(0.0))
pmask = periodic_mask(x, period=7, width=1, random_roll=False)
vampnet_mask = mask_and(vampnet_mask, pmask)
vampnet_mask = codebook_mask(vampnet_mask, val1=3)

# ONNX
np.random.seed(42)
onnx_mask = build_mask_proper(
    np.zeros(test_shape),
    rand_mask_intensity=0.0,
    periodic_prompt=7,
    upper_codebook_mask=3,
    random_roll=False,
    seed=42
)

match = np.array_equal(vampnet_mask.numpy(), onnx_mask)
print(f"Exact match: {match}")
print(f"VampNet ratio: {np.mean(vampnet_mask.numpy()):.1%}")
print(f"ONNX ratio: {np.mean(onnx_mask):.1%}")

if match:
    print("\n✓ SUCCESS: ONNX mask generator now matches VampNet exactly!")
else:
    print("\n✗ FAILURE: Still have mismatches. Need further investigation.")