#!/usr/bin/env python3
"""
Test the proper mask generator implementation against VampNet.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("TESTING PROPER MASK GENERATOR")
print("=" * 80)

# Import the mask generators
from vampnet_onnx.mask_generator_proper import VampNetMaskGenerator
from vampnet.interface import Interface as VampNetInterface
from vampnet.mask import *

# Initialize VampNet interface
device = torch.device('cpu')
vampnet_interface = VampNetInterface(
    codec_ckpt="../models/vampnet/codec.pth",
    coarse_ckpt="../models/vampnet/coarse.pth",
    coarse2fine_ckpt="../models/vampnet/c2f.pth",
    wavebeat_ckpt="../models/vampnet/wavebeat.pth",
)
vampnet_interface.to(device)

# Create test tokens
np.random.seed(42)
torch.manual_seed(42)
test_tokens_np = np.random.randint(0, 1024, size=(1, 14, 100), dtype=np.int64)
test_tokens_torch = torch.tensor(test_tokens_np)

# Initialize our mask generator
mask_gen = VampNetMaskGenerator()

# Test configurations
configs = [
    {
        "name": "Random Only",
        "params": {
            "rand_mask_intensity": 0.8,
            "periodic_prompt": 0,
            "upper_codebook_mask": 0
        }
    },
    {
        "name": "Periodic Only",
        "params": {
            "rand_mask_intensity": 0.0,
            "periodic_prompt": 7,
            "periodic_prompt_width": 1,
            "upper_codebook_mask": 0
        }
    },
    {
        "name": "Combined",
        "params": {
            "rand_mask_intensity": 0.8,
            "periodic_prompt": 7,
            "periodic_prompt_width": 1,
            "upper_codebook_mask": 3
        }
    },
    {
        "name": "With Inpainting",
        "params": {
            "rand_mask_intensity": 1.0,
            "prefix_s": 0.5,
            "suffix_s": 0.5,
            "periodic_prompt": 10,
            "upper_codebook_mask": 4
        }
    }
]

# Compare masks
print("\n" + "-" * 60)
print("MASK COMPARISON")
print("-" * 60)

fig, axes = plt.subplots(len(configs), 3, figsize=(15, 4 * len(configs)))
if len(configs) == 1:
    axes = axes.reshape(1, -1)

for idx, config in enumerate(configs):
    print(f"\n--- {config['name']} ---")
    print(f"Parameters: {config['params']}")
    
    # Generate our mask
    our_mask = mask_gen.build_mask(
        test_tokens_np,
        **config['params'],
        seed=42  # For reproducibility
    )
    
    # Generate VampNet mask
    # VampNet's build_mask expects a signal, create a dummy one
    from audiotools import AudioSignal
    dummy_signal = AudioSignal(np.zeros((1, 44100)), 44100)
    
    vampnet_mask = vampnet_interface.build_mask(
        test_tokens_torch,
        sig=dummy_signal,
        **config['params']
    )
    
    # Convert to numpy
    if isinstance(vampnet_mask, torch.Tensor):
        vampnet_mask = vampnet_mask.cpu().numpy()
    
    # Ensure boolean
    our_mask = our_mask.astype(bool)
    vampnet_mask = vampnet_mask.astype(bool)
    
    # Calculate metrics
    mask_match = np.array_equal(our_mask, vampnet_mask)
    mask_overlap = np.mean(our_mask == vampnet_mask)
    our_ratio = np.mean(our_mask)
    vampnet_ratio = np.mean(vampnet_mask)
    
    print(f"  Our mask shape: {our_mask.shape}")
    print(f"  VampNet mask shape: {vampnet_mask.shape}")
    print(f"  Exact match: {mask_match}")
    print(f"  Overlap rate: {mask_overlap:.1%}")
    print(f"  Our mask ratio: {our_ratio:.1%}")
    print(f"  VampNet mask ratio: {vampnet_ratio:.1%}")
    
    # Visualize
    # Our mask
    ax = axes[idx, 0]
    im = ax.imshow(our_mask[0], aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"{config['name']} - Our Implementation\n(ratio: {our_ratio:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    
    # VampNet mask
    ax = axes[idx, 1]
    im = ax.imshow(vampnet_mask[0], aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"{config['name']} - VampNet\n(ratio: {vampnet_ratio:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    
    # Difference
    ax = axes[idx, 2]
    diff = our_mask[0].astype(float) - vampnet_mask[0].astype(float)
    im = ax.imshow(diff, aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=-1, vmax=1)
    ax.set_title(f"Difference\n(overlap: {mask_overlap:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('proper_mask_generator_comparison.png', dpi=150)
print("\n✓ Saved visualization to proper_mask_generator_comparison.png")

# Test individual mask components
print("\n" + "-" * 60)
print("TESTING INDIVIDUAL MASK COMPONENTS")
print("-" * 60)

# Test linear_random
print("\n1. Testing linear_random mask:")
for intensity in [0.0, 0.5, 1.0]:
    mask = mask_gen.linear_random((1, 14, 100), intensity, seed=42)
    ratio = np.mean(mask)
    print(f"   Intensity {intensity:.1f} -> Mask ratio: {ratio:.2%}")

# Test periodic_mask
print("\n2. Testing periodic mask:")
for period in [5, 10, 20]:
    mask = mask_gen.periodic_mask((1, 14, 100), period, width=1, random_roll=False, seed=42)
    ratio = np.mean(mask)
    n_preserved = np.sum(mask == 0) / 14  # per codebook
    print(f"   Period {period} -> Mask ratio: {ratio:.2%}, Preserved positions: {n_preserved}")

# Test inpaint mask
print("\n3. Testing inpaint mask:")
mask = mask_gen.inpaint((1, 14, 100), n_prefix=10, n_suffix=10)
ratio = np.mean(mask)
print(f"   Prefix=10, Suffix=10 -> Mask ratio: {ratio:.2%}")
print(f"   First 10 positions masked: {np.all(mask[:, :, :10] == 0)}")
print(f"   Last 10 positions masked: {np.all(mask[:, :, -10:] == 0)}")

# Test codebook mask
print("\n4. Testing codebook mask:")
for upper in [0, 3, 7]:
    mask = mask_gen.codebook_mask((1, 14, 100), upper)
    ratio = np.mean(mask)
    print(f"   Upper codebook {upper} -> Mask ratio: {ratio:.2%}")
    if upper > 0:
        print(f"   Lower codebooks masked: {np.all(mask[:, :upper, :] == 0)}")
        print(f"   Upper codebooks masked: {np.all(mask[:, upper:, :] == 1)}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)