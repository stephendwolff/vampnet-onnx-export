#!/usr/bin/env python3
"""Test what the quantizer returns."""

import torch
import vampnet

# Load VampNet
interface = vampnet.interface.Interface(
    device='cpu',
    codec_ckpt="../models/vampnet/codec.pth", 
    coarse_ckpt="../models/vampnet/coarse.pth",
    wavebeat_ckpt="../models/vampnet/wavebeat.pth"
)

# Test audio
audio = torch.randn(1, 1, 44544)  # Pre-padded

# Encode to latent
latent = interface.codec.encoder(audio)
print(f"Latent shape: {latent.shape}")

# Quantize
result = interface.codec.quantizer(latent)
print(f"\nQuantizer returns {len(result)} values:")
for i, r in enumerate(result):
    if torch.is_tensor(r):
        print(f"  {i}: Tensor with shape {r.shape}")
    else:
        print(f"  {i}: {type(r)} = {r}")

# Try the quantizer.from_codes method too
print("\n\nTesting from_codes method:")
codes = torch.randint(0, 1024, (1, 14, 58))
result2 = interface.codec.quantizer.from_codes(codes)
print(f"from_codes returns {len(result2)} values")
for i, r in enumerate(result2):
    if torch.is_tensor(r):
        print(f"  {i}: Tensor with shape {r.shape}")
    else:
        print(f"  {i}: {type(r)} = {r}")