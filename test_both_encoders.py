#!/usr/bin/env python3
"""Quick test of both encoder versions."""

import numpy as np
import onnxruntime as ort

# Test audio - 1 second
test_audio = np.random.randn(1, 1, 44100).astype(np.float32)

print("Testing ONNX encoders with 1 second of audio (44100 samples)")
print("Expected tokens: 44100 / 768 = ~57\n")

# Test original encoder
encoder1 = ort.InferenceSession("scripts/models/vampnet_codec_encoder.onnx")
codes1 = encoder1.run(None, {"audio": test_audio})[0]
print(f"Original encoder output: {codes1.shape}")

# Test fixed encoder
encoder2 = ort.InferenceSession("scripts/models/vampnet_codec_encoder_fixed.onnx")
codes2 = encoder2.run(None, {"audio": test_audio})[0]
print(f"Fixed encoder output: {codes2.shape}")

# Test with longer audio
test_audio_long = np.random.randn(1, 1, 441600).astype(np.float32)  # 10 seconds
print(f"\nWith 10 seconds of audio (441600 samples)")
print("Expected tokens: 441600 / 768 = ~575")

codes1_long = encoder1.run(None, {"audio": test_audio_long})[0]
print(f"Original encoder: {codes1_long.shape}")

codes2_long = encoder2.run(None, {"audio": test_audio_long})[0]
print(f"Fixed encoder: {codes2_long.shape}")