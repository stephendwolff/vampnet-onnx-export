#!/usr/bin/env python3
"""
Pipeline Comparison: Step 4b
Compare Mask Generator with fixed seeds for deterministic comparison
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
print("STEP 4b: MASK GENERATOR COMPARISON (DETERMINISTIC)")
print("=" * 80)

# Set global seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Import interfaces
from vampnet_onnx import Interface as ONNXInterface
from vampnet_onnx.interface import AudioSignalCompat
from vampnet.interface import Interface as VampNetInterface

try:
    from audiotools import AudioSignal
    AUDIOTOOLS_AVAILABLE = True
except ImportError:
    AUDIOTOOLS_AVAILABLE = False

# Initialize interfaces
print("\nInitializing interfaces...")
device = torch.device('cpu')

# ONNX interface
onnx_interface = ONNXInterface.from_default_models(device='cpu')
print("✓ ONNX interface initialized")

# VampNet interface
vampnet_interface = VampNetInterface(
    codec_ckpt="../models/vampnet/codec.pth",
    coarse_ckpt="../models/vampnet/coarse.pth",
    coarse2fine_ckpt="../models/vampnet/c2f.pth",
    wavebeat_ckpt="../models/vampnet/wavebeat.pth",
)
vampnet_interface.to(device)
print("✓ VampNet interface initialized")

# Create test tokens with fixed seed
print("\nCreating test tokens...")
np.random.seed(SEED)
test_tokens = np.random.randint(0, 1024, size=(1, 14, 100), dtype=np.int64)
print(f"Test tokens shape: {test_tokens.shape}")

# Test different mask configurations with deterministic settings
mask_configs = [
    {
        "name": "Random 50%",
        "params": {"rand_mask_intensity": 0.5}
    },
    {
        "name": "Random 80%",
        "params": {"rand_mask_intensity": 0.8}
    },
    {
        "name": "Random 100%",
        "params": {"rand_mask_intensity": 1.0}
    },
    {
        "name": "Periodic Only",
        "params": {"rand_mask_intensity": 0.0, "periodic_prompt": 7}
    },
    {
        "name": "Upper Codebook Only",
        "params": {"rand_mask_intensity": 0.0, "upper_codebook_mask": 4}
    },
    {
        "name": "Combined Fixed",
        "params": {
            "rand_mask_intensity": 0.8,
            "periodic_prompt": 10,
            "upper_codebook_mask": 3,
            "_dropout": 0.0  # No dropout for deterministic comparison
        }
    }
]

# Create dummy audio signal for VampNet
if AUDIOTOOLS_AVAILABLE:
    dummy_audio = AudioSignal(np.zeros((1, 44100)), 44100)
else:
    dummy_audio = AudioSignalCompat(np.zeros(44100), 44100)

# Compare masks
print("\n" + "-" * 60)
print("DETERMINISTIC MASK COMPARISON")
print("-" * 60)

comparison_results = []

for config in mask_configs:
    print(f"\n--- {config['name']} ---")
    print(f"Parameters: {config['params']}")
    
    # Reset seeds before each mask generation
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Generate ONNX mask with seed
    onnx_params = config['params'].copy()
    onnx_params['seed'] = SEED  # Pass seed explicitly
    onnx_mask = onnx_interface.build_mask(
        test_tokens,
        signal=None,
        **onnx_params
    )
    
    # Reset seeds again for VampNet
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Generate VampNet mask
    vampnet_mask = vampnet_interface.build_mask(
        torch.tensor(test_tokens),
        dummy_audio,
        **config['params']
    )
    
    # Convert to numpy
    if isinstance(vampnet_mask, torch.Tensor):
        vampnet_mask = vampnet_mask.cpu().numpy()
    
    # Ensure boolean
    onnx_mask = onnx_mask.astype(bool)
    vampnet_mask = vampnet_mask.astype(bool)
    
    # Compare
    mask_match = np.array_equal(onnx_mask, vampnet_mask)
    mask_overlap = np.mean(onnx_mask == vampnet_mask)
    onnx_ratio = np.mean(onnx_mask)
    vampnet_ratio = np.mean(vampnet_mask)
    
    # Check specific patterns
    onnx_first_cb = onnx_mask[0, 0, :]
    vampnet_first_cb = vampnet_mask[0, 0, :]
    first_cb_match = np.array_equal(onnx_first_cb, vampnet_first_cb)
    
    print(f"  ONNX mask ratio: {onnx_ratio:.1%}")
    print(f"  VampNet mask ratio: {vampnet_ratio:.1%}")
    print(f"  Exact match: {mask_match}")
    print(f"  Overlap rate: {mask_overlap:.1%}")
    print(f"  First codebook match: {first_cb_match}")
    
    # Show first 20 values of first codebook for comparison
    print(f"  First 20 mask values (codebook 0):")
    print(f"    ONNX:    {onnx_first_cb[:20].astype(int)}")
    print(f"    VampNet: {vampnet_first_cb[:20].astype(int)}")
    
    comparison_results.append({
        "name": config['name'],
        "params": config['params'],
        "onnx_mask": onnx_mask,
        "vampnet_mask": vampnet_mask,
        "match": mask_match,
        "overlap": mask_overlap,
        "onnx_ratio": onnx_ratio,
        "vampnet_ratio": vampnet_ratio,
        "first_cb_match": first_cb_match
    })

# Create detailed visualization
print("\nCreating detailed visualizations...")

n_configs = len(comparison_results)
fig, axes = plt.subplots(n_configs, 3, figsize=(15, 4 * n_configs))
if n_configs == 1:
    axes = axes.reshape(1, -1)

for idx, result in enumerate(comparison_results):
    # Show only first 4 codebooks for clarity
    n_cb_show = 4
    
    # ONNX mask
    ax = axes[idx, 0]
    im = ax.imshow(result['onnx_mask'][0, :n_cb_show, :], aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"{result['name']} - ONNX\n(ratio: {result['onnx_ratio']:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    ax.set_yticks(range(n_cb_show))
    
    # VampNet mask
    ax = axes[idx, 1]
    im = ax.imshow(result['vampnet_mask'][0, :n_cb_show, :], aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"{result['name']} - VampNet\n(ratio: {result['vampnet_ratio']:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    ax.set_yticks(range(n_cb_show))
    
    # Difference (only first 4 codebooks)
    ax = axes[idx, 2]
    diff = result['onnx_mask'][0, :n_cb_show, :].astype(float) - result['vampnet_mask'][0, :n_cb_show, :].astype(float)
    im = ax.imshow(diff, aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=-1, vmax=1)
    ax.set_title(f"Difference\n(overlap: {result['overlap']:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    ax.set_yticks(range(n_cb_show))
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('mask_generator_deterministic_comparison.png', dpi=150)
print("✓ Saved visualization to mask_generator_deterministic_comparison.png")

# Analyze patterns
print("\n" + "-" * 60)
print("PATTERN ANALYSIS")
print("-" * 60)

# Check if periodic patterns match
periodic_result = next((r for r in comparison_results if r['name'] == 'Periodic Only'), None)
if periodic_result:
    print("\nPeriodic Pattern Analysis:")
    # Find unmasked positions in first codebook
    onnx_unmasked = np.where(~periodic_result['onnx_mask'][0, 0, :])[0]
    vampnet_unmasked = np.where(~periodic_result['vampnet_mask'][0, 0, :])[0]
    
    print(f"  ONNX unmasked positions: {onnx_unmasked[:10]}...")
    print(f"  VampNet unmasked positions: {vampnet_unmasked[:10]}...")
    
    if len(onnx_unmasked) > 1 and len(vampnet_unmasked) > 1:
        onnx_period = np.diff(onnx_unmasked).mean()
        vampnet_period = np.diff(vampnet_unmasked).mean()
        print(f"  ONNX average period: {onnx_period:.1f}")
        print(f"  VampNet average period: {vampnet_period:.1f}")

# Summary
print("\n" + "=" * 80)
print("DETERMINISTIC COMPARISON SUMMARY")
print("=" * 80)

exact_matches = sum(1 for r in comparison_results if r['match'])
high_overlap = sum(1 for r in comparison_results if r['overlap'] > 0.95)
ratio_match = sum(1 for r in comparison_results if abs(r['onnx_ratio'] - r['vampnet_ratio']) < 0.02)

print(f"With fixed seed ({SEED}):")
print(f"1. Exact matches: {exact_matches}/{len(comparison_results)}")
print(f"2. High overlap (>95%): {high_overlap}/{len(comparison_results)}")
print(f"3. Ratio match (±2%): {ratio_match}/{len(comparison_results)}")

print("\nKey findings:")
for result in comparison_results:
    status = "✓" if result['match'] else "✗" if result['overlap'] < 0.8 else "~"
    print(f"{status} {result['name']}: {result['overlap']:.1%} overlap, "
          f"ratios: ONNX={result['onnx_ratio']:.1%}, VampNet={result['vampnet_ratio']:.1%}")