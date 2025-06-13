#!/usr/bin/env python3
"""Test ONNX encoder with variable length inputs."""

import numpy as np
import onnxruntime as ort
from pathlib import Path

def test_variable_lengths():
    """Test ONNX encoder with different input lengths."""
    print("Testing ONNX encoder with variable length inputs...\n")
    
    # Load ONNX encoder
    onnx_path = Path("scripts/models/vampnet_encoder_prepadded.onnx")
    session = ort.InferenceSession(str(onnx_path))
    
    # Test different lengths (all multiples of 768)
    hop_length = 768
    test_cases = [
        (50, 50 * hop_length),   # 38400 samples = 50 tokens
        (100, 100 * hop_length), # 76800 samples = 100 tokens  
        (200, 200 * hop_length), # 153600 samples = 200 tokens
        (300, 300 * hop_length), # 230400 samples = 300 tokens
    ]
    
    for expected_tokens, samples in test_cases:
        # Create test audio
        audio = np.random.randn(1, 1, samples).astype(np.float32)
        
        # Run encoder
        outputs = session.run(None, {'audio_padded': audio})
        codes = outputs[0]
        
        # Check output
        actual_tokens = codes.shape[2]
        status = "✓" if actual_tokens == expected_tokens else "✗"
        
        print(f"Input: {samples:6d} samples → Output: {actual_tokens:3d} tokens (expected {expected_tokens:3d}) {status}")
        
        if actual_tokens != expected_tokens:
            print(f"  WARNING: Got {actual_tokens} tokens instead of {expected_tokens}!")
            print(f"  Shape: {codes.shape}")

if __name__ == "__main__":
    test_variable_lengths()