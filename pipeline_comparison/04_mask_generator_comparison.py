#!/usr/bin/env python3
"""
Pipeline Comparison: Step 4
Compare Mask Generator - masking pattern creation
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
print("STEP 4: MASK GENERATOR COMPARISON")
print("=" * 80)

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

# Create test tokens (use same as encoder test)
print("\nCreating test tokens...")
# Create dummy tokens for testing masks
np.random.seed(42)  # For reproducibility
test_tokens = np.random.randint(0, 1024, size=(1, 14, 100), dtype=np.int64)
print(f"Test tokens shape: {test_tokens.shape}")

# Test different mask configurations
mask_configs = [
    {
        "name": "Default",
        "params": {}
    },
    {
        "name": "High Random Intensity",
        "params": {"rand_mask_intensity": 1.0}
    },
    {
        "name": "Periodic Prompt",
        "params": {"periodic_prompt": 7}
    },
    {
        "name": "Upper Codebook Only",
        "params": {"upper_codebook_mask": 4}
    },
    {
        "name": "Combined",
        "params": {
            "rand_mask_intensity": 0.8,
            "periodic_prompt": 10,
            "upper_codebook_mask": 3
        }
    }
]

# Create dummy audio signal for VampNet (it requires it for some mask types)
if AUDIOTOOLS_AVAILABLE:
    dummy_audio = AudioSignal(np.zeros((1, 44100)), 44100)
else:
    dummy_audio = AudioSignalCompat(np.zeros(44100), 44100)

# Compare masks
print("\n" + "-" * 60)
print("MASK COMPARISON")
print("-" * 60)

comparison_results = []

for config in mask_configs:
    print(f"\n--- {config['name']} ---")
    print(f"Parameters: {config['params']}")
    
    # Generate ONNX mask
    onnx_mask = onnx_interface.build_mask(
        test_tokens,
        signal=None,
        **config['params']
    )
    
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
    
    print(f"  ONNX mask shape: {onnx_mask.shape}")
    print(f"  VampNet mask shape: {vampnet_mask.shape}")
    print(f"  Exact match: {mask_match}")
    print(f"  Overlap rate: {mask_overlap:.1%}")
    print(f"  ONNX mask ratio: {onnx_ratio:.1%}")
    print(f"  VampNet mask ratio: {vampnet_ratio:.1%}")
    
    comparison_results.append({
        "name": config['name'],
        "params": config['params'],
        "onnx_mask": onnx_mask,
        "vampnet_mask": vampnet_mask,
        "match": mask_match,
        "overlap": mask_overlap,
        "onnx_ratio": onnx_ratio,
        "vampnet_ratio": vampnet_ratio
    })

# Visualizations
print("\nCreating visualizations...")

# Create figure with subplots for each configuration
n_configs = len(mask_configs)
fig, axes = plt.subplots(n_configs, 3, figsize=(15, 4 * n_configs))
if n_configs == 1:
    axes = axes.reshape(1, -1)

for idx, result in enumerate(comparison_results):
    # ONNX mask
    ax = axes[idx, 0]
    im = ax.imshow(result['onnx_mask'][0], aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"{result['name']} - ONNX\n(ratio: {result['onnx_ratio']:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    
    # VampNet mask
    ax = axes[idx, 1]
    im = ax.imshow(result['vampnet_mask'][0], aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"{result['name']} - VampNet\n(ratio: {result['vampnet_ratio']:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    
    # Difference
    ax = axes[idx, 2]
    diff = result['onnx_mask'][0].astype(float) - result['vampnet_mask'][0].astype(float)
    im = ax.imshow(diff, aspect='auto', cmap='RdBu', 
                   interpolation='nearest', vmin=-1, vmax=1)
    ax.set_title(f"Difference\n(overlap: {result['overlap']:.1%})")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Codebook')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('mask_generator_comparison.png', dpi=150)
print("✓ Saved visualization to mask_generator_comparison.png")

# Detailed analysis of differences
print("\n" + "-" * 60)
print("DIFFERENCE ANALYSIS")
print("-" * 60)

for result in comparison_results:
    if not result['match']:
        print(f"\n{result['name']} - Differences found:")
        diff_positions = np.where(result['onnx_mask'] != result['vampnet_mask'])
        n_diffs = len(diff_positions[0])
        print(f"  Total differences: {n_diffs}")
        
        # Analyze pattern of differences
        if n_diffs > 0 and n_diffs < 1000:  # Don't print too many
            # Check if differences are systematic
            diff_codebooks = np.unique(diff_positions[1])
            diff_timesteps = np.unique(diff_positions[2])
            
            print(f"  Affected codebooks: {diff_codebooks}")
            print(f"  Affected time range: [{diff_timesteps.min()}, {diff_timesteps.max()}]")
            
            # Sample a few differences
            n_samples = min(5, n_diffs)
            print(f"  Sample differences (first {n_samples}):")
            for i in range(n_samples):
                batch = diff_positions[0][i]
                codebook = diff_positions[1][i]
                timestep = diff_positions[2][i]
                onnx_val = result['onnx_mask'][batch, codebook, timestep]
                vampnet_val = result['vampnet_mask'][batch, codebook, timestep]
                print(f"    [{batch}, {codebook}, {timestep}]: ONNX={onnx_val}, VampNet={vampnet_val}")

# Pattern analysis
print("\n" + "-" * 60)
print("MASK PATTERN ANALYSIS")
print("-" * 60)

# Check temporal patterns
for result in comparison_results:
    print(f"\n{result['name']}:")
    
    # Temporal consistency
    onnx_temporal = np.mean(result['onnx_mask'][0], axis=0)  # Average across codebooks
    vampnet_temporal = np.mean(result['vampnet_mask'][0], axis=0)
    
    temporal_corr = np.corrcoef(onnx_temporal, vampnet_temporal)[0, 1] if np.std(onnx_temporal) > 0 and np.std(vampnet_temporal) > 0 else 0
    print(f"  Temporal pattern correlation: {temporal_corr:.4f}")
    
    # Codebook consistency
    onnx_codebook = np.mean(result['onnx_mask'][0], axis=1)  # Average across time
    vampnet_codebook = np.mean(result['vampnet_mask'][0], axis=1)
    
    codebook_corr = np.corrcoef(onnx_codebook, vampnet_codebook)[0, 1] if np.std(onnx_codebook) > 0 and np.std(vampnet_codebook) > 0 else 0
    print(f"  Codebook pattern correlation: {codebook_corr:.4f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

exact_matches = sum(1 for r in comparison_results if r['match'])
high_overlap = sum(1 for r in comparison_results if r['overlap'] > 0.95)
similar_ratio = sum(1 for r in comparison_results if abs(r['onnx_ratio'] - r['vampnet_ratio']) < 0.05)

print(f"1. Exact matches: {exact_matches}/{len(comparison_results)}")
print(f"2. High overlap (>95%): {high_overlap}/{len(comparison_results)}")
print(f"3. Similar mask ratios (±5%): {similar_ratio}/{len(comparison_results)}")

print("\n⚠️  Note: Mask generation often involves randomness or different strategies.")
print("Differences are expected and acceptable as long as:")
print("- Mask ratios are similar")
print("- Patterns follow the intended strategy")
print("- Both implementations produce valid masks for training/inference")