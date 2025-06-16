#!/usr/bin/env python3
"""
Pipeline Comparison: Step 3
Compare Codec Encoder - audio to tokens conversion
"""

import os
import sys
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("STEP 3: CODEC ENCODER COMPARISON")
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

# Create test audio
print("\nCreating test audio...")
sample_rate = 44100
duration = 1.74  # ~100 tokens at 44.1kHz with hop_length=768
t = np.linspace(0, duration, int(sample_rate * duration))

# Create a more complex test signal
audio = (0.3 * np.sin(2 * np.pi * 440 * t) +      # A4
         0.2 * np.sin(2 * np.pi * 554.37 * t) +   # C#5
         0.2 * np.sin(2 * np.pi * 659.25 * t) +   # E5
         0.1 * np.sin(2 * np.pi * 880 * t))       # A5

# Add some amplitude modulation
mod_freq = 4.0  # 4 Hz modulation
audio = audio * (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))

# Normalize
audio = audio / np.abs(audio).max() * 0.8
audio = audio.astype(np.float32)

print(f"Test audio: {duration:.2f}s at {sample_rate}Hz")

# Encode with both interfaces
print("\n" + "-" * 60)
print("ENCODING COMPARISON")
print("-" * 60)

# Create audio signals
if AUDIOTOOLS_AVAILABLE:
    audio_signal = AudioSignal(audio[np.newaxis, :], sample_rate)
else:
    audio_signal = AudioSignalCompat(audio, sample_rate)

# Encode with ONNX
print("\n--- ONNX Encoding ---")
onnx_tokens = onnx_interface.encode(audio_signal)
print(f"ONNX tokens shape: {onnx_tokens.shape}")
print(f"ONNX tokens dtype: {onnx_tokens.dtype}")
print(f"ONNX token range: [{onnx_tokens.min()}, {onnx_tokens.max()}]")
print(f"ONNX unique tokens: {len(np.unique(onnx_tokens))}")

# Encode with VampNet
print("\n--- VampNet Encoding ---")
vampnet_tokens = vampnet_interface.encode(audio_signal)

# Convert to numpy if needed
if isinstance(vampnet_tokens, torch.Tensor):
    vampnet_tokens = vampnet_tokens.cpu().numpy()

print(f"VampNet tokens shape: {vampnet_tokens.shape}")
print(f"VampNet tokens dtype: {vampnet_tokens.dtype}")
print(f"VampNet token range: [{vampnet_tokens.min()}, {vampnet_tokens.max()}]")
print(f"VampNet unique tokens: {len(np.unique(vampnet_tokens))}")

# Ensure same shape for comparison
if onnx_tokens.shape != vampnet_tokens.shape:
    print(f"\n⚠️  Shape mismatch! ONNX: {onnx_tokens.shape}, VampNet: {vampnet_tokens.shape}")
    # Truncate to minimum length
    min_seq_len = min(onnx_tokens.shape[2], vampnet_tokens.shape[2])
    onnx_tokens = onnx_tokens[:, :, :min_seq_len]
    vampnet_tokens = vampnet_tokens[:, :, :min_seq_len]
    print(f"Truncated to common length: {min_seq_len}")

# Token comparison
print("\n" + "-" * 60)
print("TOKEN COMPARISON")
print("-" * 60)

# Calculate exact matches
exact_matches = (onnx_tokens == vampnet_tokens)
match_rate = np.mean(exact_matches)
print(f"\nExact token match rate: {match_rate:.1%}")

# Per-codebook match rates
print("\nPer-codebook match rates:")
for i in range(onnx_tokens.shape[1]):
    codebook_match = np.mean(exact_matches[:, i, :])
    print(f"  Codebook {i:2d}: {codebook_match:.1%}")

# Token difference statistics
token_diff = np.abs(onnx_tokens - vampnet_tokens)
print(f"\nToken difference statistics:")
print(f"  Mean absolute difference: {np.mean(token_diff):.2f}")
print(f"  Max absolute difference: {np.max(token_diff)}")
print(f"  Tokens with difference > 0: {np.sum(token_diff > 0)} ({np.mean(token_diff > 0):.1%})")
print(f"  Tokens with difference > 10: {np.sum(token_diff > 10)} ({np.mean(token_diff > 10):.1%})")
print(f"  Tokens with difference > 100: {np.sum(token_diff > 100)} ({np.mean(token_diff > 100):.1%})")

# Visualizations
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Token distribution comparison
ax = axes[0, 0]
bins = np.linspace(0, max(onnx_tokens.max(), vampnet_tokens.max()), 50)
ax.hist(onnx_tokens.flatten(), bins=bins, alpha=0.5, label='ONNX', density=True)
ax.hist(vampnet_tokens.flatten(), bins=bins, alpha=0.5, label='VampNet', density=True)
ax.set_xlabel('Token Value')
ax.set_ylabel('Density')
ax.set_title('Token Distribution Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Per-codebook match heatmap
ax = axes[0, 1]
match_matrix = exact_matches[0].astype(float)
im = ax.imshow(match_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xlabel('Time Step')
ax.set_ylabel('Codebook')
ax.set_title('Token Match Heatmap (Green=Match, Red=Mismatch)')
plt.colorbar(im, ax=ax)

# 3. Token difference over time
ax = axes[1, 0]
for i in range(min(4, onnx_tokens.shape[1])):  # Show first 4 codebooks
    ax.plot(token_diff[0, i, :], label=f'Codebook {i}', alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Absolute Token Difference')
ax.set_title('Token Differences Over Time (First 4 Codebooks)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Codebook-wise statistics
ax = axes[1, 1]
codebook_stats = []
for i in range(onnx_tokens.shape[1]):
    codebook_match = np.mean(exact_matches[:, i, :])
    codebook_stats.append(codebook_match)

ax.bar(range(len(codebook_stats)), codebook_stats)
ax.set_xlabel('Codebook Index')
ax.set_ylabel('Match Rate')
ax.set_title('Per-Codebook Token Match Rate')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for i, v in enumerate(codebook_stats):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('codec_encoder_comparison.png', dpi=150)
print("✓ Saved visualization to codec_encoder_comparison.png")

# Additional analysis: Check if mismatches follow a pattern
print("\n" + "-" * 60)
print("MISMATCH PATTERN ANALYSIS")
print("-" * 60)

# Find positions where tokens don't match
mismatch_positions = np.where(~exact_matches)
if len(mismatch_positions[0]) > 0:
    print(f"\nFound {len(mismatch_positions[0])} mismatched tokens")
    
    # Sample some mismatches
    n_samples = min(10, len(mismatch_positions[0]))
    print(f"\nFirst {n_samples} mismatches:")
    for i in range(n_samples):
        batch_idx = mismatch_positions[0][i]
        codebook_idx = mismatch_positions[1][i]
        time_idx = mismatch_positions[2][i]
        onnx_val = onnx_tokens[batch_idx, codebook_idx, time_idx]
        vampnet_val = vampnet_tokens[batch_idx, codebook_idx, time_idx]
        print(f"  Position [{batch_idx}, {codebook_idx}, {time_idx}]: "
              f"ONNX={onnx_val}, VampNet={vampnet_val}, diff={abs(onnx_val - vampnet_val)}")

# Token correlation
print("\n" + "-" * 60)
print("TOKEN CORRELATION ANALYSIS")
print("-" * 60)

# Calculate correlation per codebook
print("\nPer-codebook correlations:")
for i in range(onnx_tokens.shape[1]):
    onnx_cb = onnx_tokens[0, i, :].flatten()
    vampnet_cb = vampnet_tokens[0, i, :].flatten()
    if len(onnx_cb) > 1:
        corr = np.corrcoef(onnx_cb, vampnet_cb)[0, 1]
        print(f"  Codebook {i:2d}: {corr:.4f}")

# Overall correlation
onnx_flat = onnx_tokens.flatten()
vampnet_flat = vampnet_tokens.flatten()
overall_corr = np.corrcoef(onnx_flat, vampnet_flat)[0, 1]
print(f"\nOverall token correlation: {overall_corr:.4f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Shape compatibility: {'✓' if onnx_tokens.shape == vampnet_tokens.shape else '✗'}")
print(f"2. Exact match rate: {match_rate:.1%} {'(excellent)' if match_rate > 0.95 else '(good)' if match_rate > 0.90 else '(needs improvement)'}")
print(f"3. Token correlation: {overall_corr:.4f} {'(excellent)' if overall_corr > 0.99 else '(good)' if overall_corr > 0.95 else '(poor)'}")
print(f"4. Max token difference: {np.max(token_diff)}")
print(f"5. Unique tokens - ONNX: {len(np.unique(onnx_tokens))}, VampNet: {len(np.unique(vampnet_tokens))}")

if match_rate < 0.99:
    print("\n⚠️  Token encoding shows differences between implementations.")
    print("This may be due to:")
    print("- Different model weights or initialization")
    print("- Numerical precision differences")
    print("- Different preprocessing steps")
else:
    print("\n✓ Excellent token encoding match between implementations!")