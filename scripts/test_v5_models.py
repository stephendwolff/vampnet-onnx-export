#!/usr/bin/env python3
"""
Quick test of V5 models without positional encoding.
"""

import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

print("=" * 80)
print("TESTING V8 MODELS WITH ALL FIXES")
print("=" * 80)

# Load models directly
print("\n1. Loading models...")

# Original VampNet
import sys
sys.path.append(str(Path(__file__).parent.parent))
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet_model = VampNet.load("../models/vampnet/coarse.pth", map_location='cpu')
vampnet_model.eval()
codec = DAC.load(Path("../models/vampnet/codec.pth"))
codec.eval()
print("✓ VampNet loaded")

# ONNX V8 model (with all fixes)
onnx_session = ort.InferenceSession("../onnx_models_fixed/coarse_v8_film_fix.onnx")
print("✓ ONNX V8 model loaded")

# Create test input
print("\n2. Creating test input...")
torch.manual_seed(42)
np.random.seed(42)

test_tokens = torch.randint(0, 1024, (1, 4, 20))  # Small for quick test
mask = torch.zeros((1, 4, 20), dtype=torch.bool)
mask[:, :, 10:] = True
masked_tokens = test_tokens.clone()
masked_tokens[mask] = 1024

print(f"Test shape: {test_tokens.shape}")
print(f"Masked positions: {mask.sum()}")

# Run VampNet
print("\n3. Running VampNet...")
with torch.no_grad():
    # Get latents
    z_latents = vampnet_model.embedding.from_codes(masked_tokens, codec)
    # Forward
    vampnet_out = vampnet_model(z_latents)
    print(f"VampNet output: shape={vampnet_out.shape}, mean={vampnet_out.mean():.4f}, std={vampnet_out.std():.4f}")

# Run ONNX
print("\n4. Running ONNX V8...")
onnx_out = onnx_session.run(None, {
    'codes': masked_tokens.numpy().astype(np.int64),
    'mask': mask.numpy()
})[0]
print(f"ONNX output: shape={onnx_out.shape}, mean={onnx_out.mean():.4f}, std={onnx_out.std():.4f}")

# Compare
print("\n5. Comparison...")
print("-" * 60)

# Reshape VampNet output
batch_size, vocab_size, total_positions = vampnet_out.shape
n_codebooks = 4
seq_len = total_positions // n_codebooks

vampnet_reshaped = vampnet_out.reshape(batch_size, vocab_size, n_codebooks, seq_len)
vampnet_reshaped = vampnet_reshaped.transpose(1, 2).transpose(2, 3)

# Compare at masked positions
vampnet_masked = vampnet_reshaped[mask]
onnx_masked = onnx_out[mask][:, :1024]  # Remove mask token class

diff = (vampnet_masked - onnx_masked).abs()
corr = np.corrcoef(vampnet_masked.flatten(), onnx_masked.flatten())[0,1]

print(f"Mean absolute difference: {diff.mean():.4f}")
print(f"Max absolute difference: {diff.max():.4f}")
print(f"Correlation: {corr:.4f}")

# Check predictions
vampnet_preds = vampnet_masked.argmax(-1)
onnx_preds = onnx_masked.argmax(-1)
pred_match = (vampnet_preds == onnx_preds).float().mean()
print(f"Prediction match rate: {pred_match:.1%}")

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)

if corr > 0.9:
    print("✅ SUCCESS! The V8 models with all fixes match VampNet!")
    print(f"   Correlation: {corr:.4f}")
    print(f"   Prediction match: {pred_match:.1%}")
else:
    print("❌ FAILED! The models still don't match.")
    print(f"   Correlation: {corr:.4f}")
    print("   V8 has all fixes but still shows differences.")