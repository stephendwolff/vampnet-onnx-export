#!/usr/bin/env python3
"""Test the codec directly to understand its behavior."""

import torch
import numpy as np
import vampnet
from pathlib import Path

# Load VampNet
device = 'cpu'
interface = vampnet.interface.Interface(
    device=device,
    codec_ckpt="../models/vampnet/codec.pth",
    coarse_ckpt="../models/vampnet/coarse.pth",
    wavebeat_ckpt="../models/vampnet/wavebeat.pth"
)

codec = interface.codec

print("Testing codec.encode behavior:\n")

# Test different lengths
for seconds in [0.5, 1, 2, 3, 5, 10]:
    samples = int(seconds * 44100)
    audio = torch.randn(1, 1, samples)
    
    # Direct encode
    result = codec.encode(audio, 44100)
    codes = result["codes"] if isinstance(result, dict) else result
    
    expected = samples // 768
    print(f"{seconds:4.1f}s ({samples:6d} samples): {codes.shape[2]:3d} tokens (expected ~{expected:3d})")
    
    # Check actual vs expected
    if abs(codes.shape[2] - expected) > 1:
        print(f"      ^ MISMATCH! Difference: {codes.shape[2] - expected}")

# Let's also check the preprocess function
print("\nChecking codec.preprocess:")
test_audio = torch.randn(1, 1, 44100)
preprocessed, original_length = codec.preprocess(test_audio, 44100)
print(f"Input: {test_audio.shape}")
print(f"Output: {preprocessed.shape}, original_length={original_length}")
print(f"Padding added: {preprocessed.shape[2] - test_audio.shape[2]}")