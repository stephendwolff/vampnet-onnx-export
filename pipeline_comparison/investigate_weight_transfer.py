#!/usr/bin/env python3
"""
Investigate weight transfer issues between VampNet and ONNX models.
Focus on checking if weights were properly transferred.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("INVESTIGATING WEIGHT TRANSFER")
print("=" * 80)

# Import what we need
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')

# 1. Load VampNet model
print("\n1. Loading VampNet coarse model...")
coarse_model = VampNet.load("../models/vampnet/coarse.pth", map_location=device)
coarse_model.eval()

# 2. Load ONNX exported model (PyTorch version)
print("\n2. Loading ONNX export script model...")
sys.path.append("../scripts")
from export_vampnet_transformer_v4_correct import VampNetTransformerV4Correct, transfer_weights_v4

# Create model with same config as coarse
onnx_model = VampNetTransformerV4Correct(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=20,
    use_gated_ffn=True
)

# Transfer weights
print("\n3. Transferring weights...")
transfer_weights_v4("../models/vampnet/coarse.pth", onnx_model, "coarse")
onnx_model.eval()

# 3. Compare specific weights
print("\n4. Comparing specific weights...")
print("-" * 60)

# Check embedding weights
print("\nEmbedding weights:")
# Check first codebook embedding
vampnet_cb0 = None
onnx_cb0 = onnx_model.embedding.embeddings[0].weight.data[:1024]  # First 1024 are regular tokens

# Load checkpoint to get codec embeddings
ckpt = torch.load("../models/vampnet/coarse.pth", map_location='cpu')
if 'codec' in ckpt:
    codec_state = ckpt['codec']
    key = 'quantizer.quantizers.0.codebook.weight'
    if key in codec_state:
        vampnet_cb0 = codec_state[key]
        print(f"  VampNet codebook 0: shape {vampnet_cb0.shape}, mean={vampnet_cb0.mean():.4f}, std={vampnet_cb0.std():.4f}")
        print(f"  ONNX codebook 0: shape {onnx_cb0.shape}, mean={onnx_cb0.mean():.4f}, std={onnx_cb0.std():.4f}")
        print(f"  Difference: {(vampnet_cb0 - onnx_cb0).abs().max():.6f}")

# Check Conv1d projection
print("\nConv1d projection weights:")
vampnet_proj = coarse_model.embedding.out_proj.weight.data
onnx_proj = onnx_model.embedding.out_proj.weight.data
print(f"  VampNet proj: shape {vampnet_proj.shape}, mean={vampnet_proj.mean():.4f}, std={vampnet_proj.std():.4f}")
print(f"  ONNX proj: shape {onnx_proj.shape}, mean={onnx_proj.mean():.4f}, std={onnx_proj.std():.4f}")
print(f"  Difference: {(vampnet_proj - onnx_proj).abs().max():.6f}")

# Check positional encoding
print("\nPositional encoding:")
print("  Note: VampNet doesn't use explicit positional encoding!")
print("  It uses relative position bias in attention instead.")
print("  ONNX model has positional encoding that needs investigation.")

# Check first transformer layer
print("\nFirst transformer layer:")
vampnet_layer = coarse_model.transformer.layers[0]
onnx_layer = onnx_model.layers[0]

# RMSNorm weights
print("  RMSNorm1:")
vampnet_norm1 = vampnet_layer.norm_1.weight.data
onnx_norm1 = onnx_layer['norm1'].weight.data
print(f"    VampNet: shape {vampnet_norm1.shape}, mean={vampnet_norm1.mean():.4f}")
print(f"    ONNX: shape {onnx_norm1.shape}, mean={onnx_norm1.mean():.4f}")
print(f"    Difference: {(vampnet_norm1 - onnx_norm1).abs().max():.6f}")

# Attention weights
print("  Attention Q weights:")
vampnet_q = vampnet_layer.self_attn.w_qs.weight.data
onnx_q = onnx_layer['self_attn'].w_q.weight.data
print(f"    VampNet: shape {vampnet_q.shape}, mean={vampnet_q.mean():.4f}")
print(f"    ONNX: shape {onnx_q.shape}, mean={onnx_q.mean():.4f}")
print(f"    Difference: {(vampnet_q - onnx_q).abs().max():.6f}")

# Check classifier
print("\nClassifier weights:")
vampnet_cls = coarse_model.classifier.layers[0].weight.data.squeeze(-1)  # Remove kernel dimension
# For ONNX, we need to concatenate all output projections
onnx_cls_weights = []
for i in range(4):  # 4 codebooks
    onnx_cls_weights.append(onnx_model.output_projs[i].weight.data[:1024])  # Exclude mask token
onnx_cls = torch.cat(onnx_cls_weights, dim=0)

print(f"  VampNet classifier: shape {vampnet_cls.shape}, mean={vampnet_cls.mean():.4f}")
print(f"  ONNX classifier: shape {onnx_cls.shape}, mean={onnx_cls.mean():.4f}")
print(f"  Difference: {(vampnet_cls - onnx_cls).abs().max():.6f}")

# 5. Test forward pass with same input
print("\n5. Testing forward pass with identical input...")
print("-" * 60)

# Create simple input
test_tokens = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
masked_tokens = test_tokens.clone()
masked_tokens[mask] = 1024

# Load codec
codec = DAC.load(Path("../models/vampnet/codec.pth"))
codec.eval()
codec.to(device)

# VampNet forward
with torch.no_grad():
    # VampNet's forward method expects tokens, not embeddings
    # It will handle the embedding internally
    vampnet_out = coarse_model(masked_tokens)
    
    print(f"VampNet output: shape {vampnet_out.shape}, mean={vampnet_out.mean():.4f}, std={vampnet_out.std():.4f}")

# ONNX model forward
with torch.no_grad():
    onnx_out = onnx_model(masked_tokens, mask)
    print(f"ONNX output: shape {onnx_out.shape}, mean={onnx_out.mean():.4f}, std={onnx_out.std():.4f}")

# Reshape VampNet output to match ONNX format
vampnet_out_reshaped = vampnet_out.reshape(1, 1024, 4, 10).transpose(1, 2).transpose(2, 3)
print(f"\nVampNet reshaped: {vampnet_out_reshaped.shape}")

# Compare outputs at masked positions
vampnet_masked = vampnet_out_reshaped[mask][:, :1024]  # Exclude mask token class
onnx_masked = onnx_out[mask][:, :1024]

print(f"\nLogits at masked positions:")
print(f"  VampNet: mean={vampnet_masked.mean():.4f}, std={vampnet_masked.std():.4f}")
print(f"  ONNX: mean={onnx_masked.mean():.4f}, std={onnx_masked.std():.4f}")
print(f"  Mean absolute difference: {(vampnet_masked - onnx_masked).abs().mean():.4f}")
print(f"  Max absolute difference: {(vampnet_masked - onnx_masked).abs().max():.4f}")

# Summary
print("\n" + "=" * 80)
print("WEIGHT TRANSFER SUMMARY")
print("=" * 80)

if (vampnet_masked - onnx_masked).abs().mean() < 0.1:
    print("✓ Weights appear to be transferred correctly!")
    print("  The outputs match closely.")
else:
    print("✗ Weight transfer has issues!")
    print("  The outputs differ significantly.")
    print("\nPossible issues:")
    print("  1. Embedding weight transfer")
    print("  2. Positional encoding format")
    print("  3. Classifier weight arrangement")
    print("  4. Special token handling")