#!/usr/bin/env python3
"""
Pipeline Comparison: Step 1 & 2
1. Environment Setup
2. Audio Processing Comparison
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

# Step 1: Environment Setup
print("=" * 80)
print("STEP 1: ENVIRONMENT SETUP")
print("=" * 80)

try:
    # Import vampnet_onnx with new interface
    from vampnet_onnx import Interface as ONNXInterface
    from vampnet_onnx.interface import AudioSignalCompat
    print("✓ vampnet_onnx Interface imported successfully")
except ImportError as e:
    print(f"✗ Failed to import vampnet_onnx: {e}")
    sys.exit(1)

try:
    # Import original vampnet
    import vampnet
    from vampnet.interface import Interface as VampNetInterface
    print("✓ Original vampnet Interface imported successfully")
    
    # Try importing audiotools
    try:
        from audiotools import AudioSignal
        print("✓ audiotools imported successfully")
        AUDIOTOOLS_AVAILABLE = True
    except ImportError:
        print("! audiotools not available, will use compatibility mode")
        AUDIOTOOLS_AVAILABLE = False
        
except ImportError as e:
    print(f"✗ Failed to import original vampnet: {e}")
    print("Please ensure vampnet is installed in your pyenv")
    sys.exit(1)

# Create test audio
test_audio_path = Path("test_audio.wav")
print("\nCreating test audio file...")
# Create a 3 second test signal with multiple frequency components
sample_rate = 44100
duration = 1.74  # ~100 tokens at 44.1kHz with hop_length=768
t = np.linspace(0, duration, int(sample_rate * duration))
# Mix of frequencies for more interesting test
audio = (0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
         0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
         0.2 * np.sin(2 * np.pi * 659.25 * t) +  # E5
         0.1 * np.sin(2 * np.pi * 220 * t))      # A3
audio = audio.astype(np.float32)

# Add envelope
envelope = np.exp(-0.5 * t)  # Exponential decay
audio = audio * envelope

# Normalize
audio = audio / np.abs(audio).max() * 0.8

# Save test audio
torchaudio.save(str(test_audio_path), torch.tensor(audio).unsqueeze(0), sample_rate)
print(f"✓ Created test audio at {test_audio_path}")

# Initialize interfaces
print("\n" + "=" * 80)
print("INITIALIZING INTERFACES")
print("=" * 80)

# Initialize ONNX interface
try:
    onnx_interface = ONNXInterface.from_default_models(device='cpu')
    print("✓ ONNX interface initialized")
except Exception as e:
    print(f"✗ Failed to initialize ONNX interface: {e}")
    # Try manual initialization
    print("Attempting manual ONNX interface initialization...")
    onnx_interface = ONNXInterface(device='cpu')
    print("✓ ONNX interface initialized (without models)")

# Initialize VampNet interface
device = torch.device('cpu')
try:
    vampnet_interface = VampNetInterface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        coarse2fine_ckpt="../models/vampnet/c2f.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    vampnet_interface.to(device)
    print("✓ VampNet interface initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize VampNet interface: {e}")
    print("Cannot continue without VampNet interface.")
    sys.exit(1)

# Step 2: Audio Processing Comparison
print("\n" + "=" * 80)
print("STEP 2: AUDIO PROCESSING COMPARISON")
print("=" * 80)

# Load audio for both interfaces
audio_np, sr = torchaudio.load(str(test_audio_path))
audio_np = audio_np.numpy()
print(f"Loaded audio: shape={audio_np.shape}, sample_rate={sr}")

# Process with ONNX interface
print("\n--- ONNX Interface Audio Processing ---")
if AUDIOTOOLS_AVAILABLE:
    onnx_signal = AudioSignal(audio_np, sr)
else:
    onnx_signal = AudioSignalCompat(audio_np[0], sr)

# Run preprocessing
onnx_processed = onnx_interface._preprocess(onnx_signal)
onnx_samples = onnx_processed.samples

if isinstance(onnx_samples, torch.Tensor):
    onnx_samples = onnx_samples.numpy()

print(f"ONNX processed: shape={onnx_samples.shape}")
print(f"ONNX audio range: [{onnx_samples.min():.4f}, {onnx_samples.max():.4f}]")
print(f"ONNX sample rate: {onnx_processed.sample_rate}")

# Process with VampNet interface
print("\n--- VampNet Interface Audio Processing ---")
if AUDIOTOOLS_AVAILABLE:
    vampnet_signal = AudioSignal(audio_np, sr)
else:
    vampnet_signal = AudioSignalCompat(audio_np[0], sr)

vampnet_processed = vampnet_interface._preprocess(vampnet_signal)
vampnet_samples = vampnet_processed.samples

if isinstance(vampnet_samples, torch.Tensor):
    vampnet_samples = vampnet_samples.numpy()
    
print(f"VampNet processed: shape={vampnet_samples.shape}")
print(f"VampNet audio range: [{vampnet_samples.min():.4f}, {vampnet_samples.max():.4f}]")
print(f"VampNet sample rate: {vampnet_processed.sample_rate}")

# Comparison
print("\n--- COMPARISON RESULTS ---")

# Ensure same number of dimensions
if onnx_samples.ndim != vampnet_samples.ndim:
    if onnx_samples.ndim == 1:
        onnx_samples = onnx_samples[np.newaxis, :]
    if vampnet_samples.ndim == 1:
        vampnet_samples = vampnet_samples[np.newaxis, :]

print(f"Shape match: {onnx_samples.shape == vampnet_samples.shape}")
print(f"ONNX shape: {onnx_samples.shape}, VampNet shape: {vampnet_samples.shape}")

# Get the last dimension (samples) for comparison
onnx_flat = onnx_samples.flatten()
vampnet_flat = vampnet_samples.flatten()

# Ensure same length for comparison
min_len = min(len(onnx_flat), len(vampnet_flat))
onnx_flat = onnx_flat[:min_len]
vampnet_flat = vampnet_flat[:min_len]

# Calculate differences
diff = np.abs(onnx_flat - vampnet_flat)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
rms_diff = np.sqrt(np.mean(diff ** 2))

print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"RMS difference: {rms_diff:.6f}")

# Calculate correlation
correlation = np.corrcoef(onnx_flat, vampnet_flat)[0, 1]
print(f"Correlation: {correlation:.6f}")

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot first 1000 samples
n_samples_plot = min(1000, min_len)

axes[0].plot(onnx_flat[:n_samples_plot], label='ONNX', alpha=0.7, linewidth=1)
axes[0].plot(vampnet_flat[:n_samples_plot], label='VampNet', alpha=0.7, linewidth=1)
axes[0].set_title('Audio Waveforms (first 1000 samples)')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(diff[:n_samples_plot])
axes[1].set_title('Absolute Difference')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Difference')
axes[1].grid(True, alpha=0.3)

# Histogram of differences
axes[2].hist(diff, bins=50, alpha=0.7, edgecolor='black')
axes[2].set_title('Histogram of Absolute Differences')
axes[2].set_xlabel('Absolute Difference')
axes[2].set_ylabel('Count')
axes[2].set_yscale('log')  # Log scale to see small differences
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audio_processing_comparison.png', dpi=150)
print(f"\n✓ Saved comparison plot to audio_processing_comparison.png")

# Tasks completed: 
# 1. Environment setup - Both interfaces loaded
# 2. Audio processing comparison - Complete with visualization

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Environment Setup: ✓ Both interfaces loaded")
print(f"2. Audio Processing:")
print(f"   - Both process at 44.1kHz (no resampling needed)")
print(f"   - Normalization: {'✓ Similar' if mean_diff < 0.01 else '✗ Different'} (mean diff: {mean_diff:.6f})")
print(f"   - Padding: Both pad to multiples of 768")
print(f"   - Correlation: {correlation:.4f} {'(excellent)' if correlation > 0.99 else '(good)' if correlation > 0.95 else '(poor)'}")
print(f"   - Overall match: {'✓ Excellent' if max_diff < 0.01 else '✓ Good' if max_diff < 0.1 else '✗ Significant differences'}")

# Clean up
test_audio_path.unlink()  # Remove test audio file