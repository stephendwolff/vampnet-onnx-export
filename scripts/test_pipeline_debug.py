#!/usr/bin/env python3
"""
Debug the pipeline issue with transformer shapes.
"""

import numpy as np
from vampnet_onnx import VampNetONNXPipeline

# Create pipeline
pipeline = VampNetONNXPipeline(model_dir="onnx_models_test")

# Check transformer model info
if 'transformer' in pipeline.sessions:
    print("Transformer model info:")
    
    # Get inputs
    inputs = pipeline.sessions['transformer'].get_inputs()
    for inp in inputs:
        print(f"\nInput: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {type(inp.shape)}")
        print(f"  Shape[2]: {inp.shape[2]} (type: {type(inp.shape[2])})")
        
    # Test with different sequence lengths
    print("\n\nTesting sequence lengths:")
    for seq_len in [50, 100, 200]:
        codes = np.random.randint(0, 1024, (1, 4, seq_len), dtype=np.int64)
        mask = np.ones((1, 4, seq_len), dtype=np.int64)
        
        print(f"\nSequence length {seq_len}:")
        try:
            result = pipeline.sessions['transformer'].run(None, {'codes': codes, 'mask': mask})
            print(f"  ✓ Success! Output shape: {result[0].shape}")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:100]}...")

# Test a short audio to see the actual sequence length
print("\n\nTesting with actual audio:")
test_audio = np.random.randn(2, 44100 * 3).astype(np.float32)  # 3 seconds
result = pipeline.process_audio(test_audio, sample_rate=44100)

if 'codes' in result:
    print(f"Encoded sequence length: {result['codes'].shape[-1]}")