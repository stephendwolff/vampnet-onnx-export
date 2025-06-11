#!/usr/bin/env python3
"""Fix the ONNX encoder to produce identical tokens to VampNet."""

import torch
import numpy as np
from pathlib import Path
from audiotools import AudioSignal
import vampnet

def test_codec_directly():
    """Test the codec directly to understand the encoding process."""
    
    # Load VampNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth",
        device=device
    )
    
    # Create test audio - 1 second
    test_audio = torch.randn(1, 1, 44100).to(device)
    print(f"Test audio shape: {test_audio.shape}")
    
    # Test 1: Direct codec.encode (what the ONNX wrapper does)
    print("\n1. Testing direct codec.encode:")
    try:
        # What the current ONNX wrapper does
        result = interface.codec.encode(test_audio, interface.codec.sample_rate)
        if isinstance(result, dict):
            codes = result["codes"]
        else:
            codes = result
        print(f"  Success! Shape: {codes.shape}")
        print(f"  Expected tokens: {44100 // interface.codec.hop_length} = {codes.shape[-1]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Type of error: {type(e)}")
    
    # Test 2: With codec.preprocess first
    print("\n2. Testing with codec.preprocess:")
    try:
        # Apply codec preprocessing
        preprocessed, length = interface.codec.preprocess(test_audio, interface.codec.sample_rate)
        print(f"  After preprocess: shape={preprocessed.shape}, length={length}")
        
        # Then encode
        result = interface.codec.encode(preprocessed, interface.codec.sample_rate)
        if isinstance(result, dict):
            codes = result["codes"]
        else:
            codes = result
        print(f"  Encoded shape: {codes.shape}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 3: Check what codec.encode actually expects
    print("\n3. Checking codec.encode signature:")
    import inspect
    if hasattr(interface.codec, 'encode'):
        sig = inspect.signature(interface.codec.encode)
        print(f"  Signature: {sig}")
        
        # Get the source if possible
        try:
            source_lines = inspect.getsourcelines(interface.codec.encode)[0]
            print(f"  First few lines of encode method:")
            for i, line in enumerate(source_lines[:5]):
                print(f"    {line.rstrip()}")
        except:
            pass
    
    # Test 4: Use AudioSignal like VampNet does
    print("\n4. Testing with AudioSignal (correct way):")
    sig = AudioSignal(test_audio.cpu().numpy()[0], sample_rate=44100)
    sig = sig.to(device)
    
    # Preprocess like Interface does
    sig = interface._preprocess(sig)
    print(f"  After preprocess: {sig.samples.shape}")
    
    # Encode
    z = interface.codec.encode(sig.samples, sig.sample_rate)["codes"]
    print(f"  Encoded shape: {z.shape}")
    print(f"  This is the correct output!")
    
    return interface

if __name__ == "__main__":
    interface = test_codec_directly()