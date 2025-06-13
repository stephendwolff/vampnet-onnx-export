#!/usr/bin/env python3
"""Find which ONNX encoder works correctly."""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# Find all encoder ONNX files
encoder_files = list(Path(".").rglob("*encoder*.onnx"))
encoder_files = [f for f in encoder_files if "venv" not in str(f)]

print(f"Found {len(encoder_files)} encoder files:\n")

# Test each one
test_lengths = [44100, 88200, 132300, 220500, 441000]  # 1s, 2s, 3s, 5s, 10s
expected_tokens = [57, 114, 172, 287, 574]

for encoder_path in sorted(encoder_files):
    print(f"\n{'='*60}")
    print(f"Testing: {encoder_path}")
    print(f"{'='*60}")
    
    try:
        session = ort.InferenceSession(str(encoder_path))
        
        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"Inputs: {[inp.name for inp in inputs]}")
        print(f"Outputs: {[out.name for out in outputs]}")
        
        # Test with different lengths
        print("\nTesting different lengths:")
        all_correct = True
        
        for length, expected in zip(test_lengths, expected_tokens):
            audio = np.random.randn(1, 1, length).astype(np.float32)
            
            try:
                result = session.run(None, {inputs[0].name: audio})
                codes = result[0]
                
                is_correct = abs(codes.shape[2] - expected) <= 1
                status = "✓" if is_correct else "✗"
                all_correct &= is_correct
                
                print(f"  {length:6d} samples: {codes.shape[2]:3d} tokens (expected {expected:3d}) {status}")
                
            except Exception as e:
                print(f"  {length:6d} samples: ERROR - {str(e)[:50]}...")
                all_correct = False
        
        if all_correct:
            print(f"\n🎉 THIS ENCODER WORKS CORRECTLY!")
            
    except Exception as e:
        print(f"Failed to load: {e}")

print("\n\nSummary:")
print("Looking for an encoder that produces tokens proportional to input length.")
print("Expected ratio: ~768 samples per token (hop_length)")