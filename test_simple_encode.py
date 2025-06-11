#!/usr/bin/env python3
"""Simple test to understand the codec encoding flow."""

import torch
import numpy as np
import vampnet

# Load VampNet with correct paths
interface = vampnet.interface.Interface(
    device='cpu',
    codec_ckpt="models/vampnet/codec.pth",
    coarse_ckpt="models/vampnet/coarse.pth", 
    coarse2fine_ckpt="models/vampnet/c2f.pth",
    wavebeat_ckpt="models/vampnet/wavebeat.pth"
)

# Test 1: Use interface.encode (the working method)
print("Test 1: Using interface.encode()")
from audiotools import AudioSignal
test_audio_np = np.random.randn(1, 44100).astype(np.float32)
sig = AudioSignal(test_audio_np, sample_rate=44100)
z = interface.encode(sig)
print(f"Result shape: {z.shape}")
print(f"Result type: {type(z)}")

# Also test with tensor for codec.encode
test_audio = torch.from_numpy(test_audio_np).unsqueeze(0)

# Test 2: Use codec.encode directly  
print("\nTest 2: Using codec.encode()")
result = interface.codec.encode(test_audio, 44100)
print(f"Result type: {type(result)}")
if isinstance(result, dict):
    print(f"Result keys: {result.keys()}")
    codes = result["codes"]
    print(f"Codes shape: {codes.shape}")
else:
    print(f"Result shape: {result.shape}")

# Test 3: Manual encoding path
print("\nTest 3: Manual encoder + quantizer")
# Pad first
padded_len = ((44100 + 767) // 768) * 768
padded_audio = torch.nn.functional.pad(test_audio, (0, padded_len - 44100))
print(f"Padded shape: {padded_audio.shape}")

# Encode to latent
latent = interface.codec.encoder(padded_audio)
print(f"Latent shape: {latent.shape}")

# Quantize
quant_result = interface.codec.quantizer(latent)
print(f"Quantizer returns {len(quant_result)} values")

# Check if it's the ResidualVectorQuantize output format
if len(quant_result) == 5:
    quantized, codes, latents, commitment_loss, codebook_loss = quant_result
    print(f"Codes shape: {codes.shape}")
    print(f"Codes type: {type(codes)}")
    
    # The codes might be a list or need stacking
    if isinstance(codes, list):
        print(f"Codes is a list of {len(codes)} items")
        codes_tensor = torch.stack(codes, dim=1)
        print(f"Stacked codes shape: {codes_tensor.shape}")
elif len(quant_result) == 3:
    # Might be (quantized, codes, ...)
    print("Got 3 values from quantizer")
    for i, val in enumerate(quant_result):
        if torch.is_tensor(val):
            print(f"  {i}: Tensor {val.shape}")
        else:
            print(f"  {i}: {type(val)}")