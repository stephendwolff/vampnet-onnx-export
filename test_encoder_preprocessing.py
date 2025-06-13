#!/usr/bin/env python3
"""Test if preprocessing differences cause encoding mismatch."""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import vampnet

def test_preprocessing():
    """Test preprocessing and encoding step by step."""
    print("Loading models...")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        device='cpu',
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
    )
    
    # Load ONNX
    onnx_path = Path("scripts/models/vampnet_encoder_prepadded.onnx")
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Create test audio - exactly 100 tokens worth
    samples = 100 * 768  # 76800 samples
    test_audio = np.random.randn(samples).astype(np.float32)
    
    print(f"\nTest audio: {samples} samples ({samples/44100:.2f}s)")
    
    # Test 1: Direct encoding with VampNet
    print("\n1. VampNet encoding (direct):")
    audio_torch = torch.from_numpy(test_audio).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        result = interface.codec.encode(audio_torch, 44100)
        vampnet_codes = result["codes"]
    
    print(f"   Shape: {vampnet_codes.shape}")
    print(f"   First 10 tokens (cb0): {vampnet_codes[0, 0, :10].numpy()}")
    
    # Test 2: ONNX encoding
    print("\n2. ONNX encoding:")
    audio_onnx = test_audio.reshape(1, 1, -1)
    onnx_codes = onnx_session.run(None, {'audio_padded': audio_onnx})[0]
    
    print(f"   Shape: {onnx_codes.shape}")
    print(f"   First 10 tokens (cb0): {onnx_codes[0, 0, :10]}")
    
    # Compare
    print("\n3. Comparison:")
    if vampnet_codes.shape == onnx_codes.shape:
        matches = (vampnet_codes.numpy() == onnx_codes).astype(float)
        match_rate = matches.mean() * 100
        print(f"   Overall match rate: {match_rate:.1f}%")
        
        # Per codebook
        for i in range(min(4, vampnet_codes.shape[1])):
            cb_match = matches[0, i].mean() * 100
            print(f"   Codebook {i}: {cb_match:.1f}%")
    else:
        print(f"   Shape mismatch! VampNet: {vampnet_codes.shape}, ONNX: {onnx_codes.shape}")
    
    # Test with different audio characteristics
    print("\n4. Testing with different audio types:")
    
    # Silence
    silence = np.zeros(samples, dtype=np.float32)
    test_audio_type(interface, onnx_session, silence, "Silence")
    
    # Sine wave
    t = np.linspace(0, samples/44100, samples)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    test_audio_type(interface, onnx_session, sine, "440Hz Sine")
    
    # White noise
    noise = (0.1 * np.random.randn(samples)).astype(np.float32)
    test_audio_type(interface, onnx_session, noise, "White noise")


def test_audio_type(interface, onnx_session, audio, name):
    """Test a specific audio type."""
    print(f"\n   {name}:")
    
    # VampNet
    audio_torch = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        result = interface.codec.encode(audio_torch, 44100)
        vampnet_codes = result["codes"]
    
    # ONNX
    audio_onnx = audio.reshape(1, 1, -1)
    onnx_codes = onnx_session.run(None, {'audio_padded': audio_onnx})[0]
    
    # Compare
    matches = (vampnet_codes.numpy() == onnx_codes).astype(float)
    match_rate = matches.mean() * 100
    print(f"     Match rate: {match_rate:.1f}%")
    
    # Show first few tokens
    print(f"     VampNet: {vampnet_codes[0, 0, :5].numpy()}")
    print(f"     ONNX:    {onnx_codes[0, 0, :5]}")


if __name__ == "__main__":
    test_preprocessing()