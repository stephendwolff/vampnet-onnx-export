#!/usr/bin/env python3
"""
Investigate the differences in logit values between VampNet and ONNX models.
This script performs detailed analysis to identify the root cause.
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
print("INVESTIGATING LOGIT VALUE DIFFERENCES")
print("=" * 80)

# Import interfaces
from vampnet_onnx import Interface as ONNXInterface
from vampnet.interface import Interface as VampNetInterface
from vampnet.modules.transformer import VampNet

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Initialize interfaces
print("\n1. Loading models...")
device = torch.device('cpu')

# Load VampNet models directly
coarse_model = VampNet.load("../models/vampnet/coarse.pth", map_location=device)
coarse_model.eval()

# Initialize ONNX interface
onnx_interface = ONNXInterface.from_default_models(device='cpu')

# Create simple test input
print("\n2. Creating test input...")
batch_size = 1
n_codebooks = 4
seq_len = 10  # Small for debugging
test_tokens = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len))
print(f"Test tokens shape: {test_tokens.shape}")

# Test 1: Check embeddings
print("\n3. Checking embeddings...")
print("-" * 60)

# Load codec for VampNet
from lac.model.lac import LAC as DAC
codec = DAC.load(Path("../models/vampnet/codec.pth"))
codec.eval()
codec.to(device)
print("Loaded LAC codec for embedding lookup")

# Get VampNet embeddings
with torch.no_grad():
    # VampNet embedding - from_codes returns concatenated embeddings
    vampnet_embeddings_concat = coarse_model.embedding.from_codes(test_tokens, codec)
    print(f"VampNet embeddings (concat) shape: {vampnet_embeddings_concat.shape}")
    print(f"VampNet embeddings (concat) range: [{vampnet_embeddings_concat.min():.4f}, {vampnet_embeddings_concat.max():.4f}]")
    
    # Apply the projection (out_proj)
    vampnet_embeddings = coarse_model.embedding(vampnet_embeddings_concat)
    print(f"VampNet embeddings (projected) shape: {vampnet_embeddings.shape}")
    print(f"VampNet embeddings (projected) range: [{vampnet_embeddings.min():.4f}, {vampnet_embeddings.max():.4f}]")
    
    # Need to transpose from [batch, d_model, seq_len] to [batch, seq_len, d_model]
    vampnet_embeddings = vampnet_embeddings.transpose(1, 2)
    print(f"VampNet embeddings (transposed) shape: {vampnet_embeddings.shape}")
    print(f"VampNet embeddings mean: {vampnet_embeddings.mean():.4f}")
    print(f"VampNet embeddings std: {vampnet_embeddings.std():.4f}")

# Test 2: Skip detailed layer checking for now
print("\n4. Skipping detailed transformer layer outputs...")

# Test 3: Check final classifier
print("\n5. Checking classifier weights...")
print("-" * 60)

# Check classifier structure
classifier = coarse_model.classifier
print(f"Classifier type: {type(classifier)}")
print(f"Classifier layers: {classifier}")

if hasattr(classifier, 'layers'):
    for i, layer in enumerate(classifier.layers):
        if hasattr(layer, 'weight'):
            print(f"  Layer {i} weight shape: {layer.weight.shape}")
            print(f"  Layer {i} weight range: [{layer.weight.min():.4f}, {layer.weight.max():.4f}]")
            print(f"  Layer {i} weight mean: {layer.weight.mean():.4f}")
            print(f"  Layer {i} weight std: {layer.weight.std():.4f}")

# Test 4: Compare with ONNX model weights
print("\n6. Checking ONNX model internals...")
print("-" * 60)

# Get ONNX model info
import onnx
import onnxruntime as ort

# Get the default ONNX model path
onnx_model_path = "onnx_models_fixed/coarse_complete_v3.onnx"
if Path(onnx_model_path).exists():
    onnx_model = onnx.load(onnx_model_path)
else:
    print(f"ONNX model not found at {onnx_model_path}")

# Get initializer names and shapes
if Path(onnx_model_path).exists():
    print("ONNX model initializers (first 10):")
    for i, init in enumerate(onnx_model.graph.initializer[:10]):
        tensor = onnx.numpy_helper.to_array(init)
        print(f"  {init.name}: shape {tensor.shape}, range [{tensor.min():.4f}, {tensor.max():.4f}]")

# Test 5: Run minimal forward pass comparison
print("\n7. Running minimal forward pass...")
print("-" * 60)

# Create mask
mask = torch.zeros((batch_size, n_codebooks, seq_len), dtype=torch.bool)
mask[:, :, seq_len//2:] = True  # Mask second half

# Apply mask to tokens
masked_tokens = test_tokens.clone()
masked_tokens[mask] = 1024  # Mask token

# VampNet forward
with torch.no_grad():
    # Get embeddings - need to do the full pipeline
    vampnet_emb_concat = coarse_model.embedding.from_codes(masked_tokens, codec)
    vampnet_emb_proj = coarse_model.embedding(vampnet_emb_concat)
    vampnet_emb = vampnet_emb_proj.transpose(1, 2)
    
    # Add positional encoding
    if hasattr(coarse_model, 'positional_encoder'):
        vampnet_emb = coarse_model.positional_encoder.add_positional_encoding(vampnet_emb)
    
    # Just use the full forward method which handles everything
    vampnet_out = coarse_model(vampnet_emb)
    print(f"VampNet output shape: {vampnet_out.shape}")
    print(f"VampNet output range: [{vampnet_out.min():.4f}, {vampnet_out.max():.4f}]")
    print(f"VampNet output mean: {vampnet_out.mean():.4f}")
    print(f"VampNet output std: {vampnet_out.std():.4f}")

# ONNX forward
onnx_out = onnx_interface.coarse_session.run(None, {
    'codes': masked_tokens.numpy().astype(np.int64),
    'mask': mask.numpy()
})[0]
print(f"\nONNX output shape: {onnx_out.shape}")
print(f"ONNX output range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
print(f"ONNX output mean: {onnx_out.mean():.4f}")
print(f"ONNX output std: {onnx_out.std():.4f}")

# Test 6: Check specific weight transfer
print("\n8. Checking specific weight transfers...")
print("-" * 60)

# Load the ONNX export script to see how weights were transferred
onnx_export_script = Path("../scripts/export_vampnet_transformer_v4_correct.py")
if onnx_export_script.exists():
    print("Found V4 export script - checking weight transfer logic...")
    
    # Import and check
    sys.path.append(str(onnx_export_script.parent))
    from export_vampnet_transformer_v4_correct import VampNetTransformerV4Correct, transfer_weights_v4
    
    # Create a fresh model and transfer weights
    test_model = VampNetTransformerV4Correct(
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
    transfer_weights_v4("../models/vampnet/coarse.pth", test_model, "coarse")
    
    # Test the transferred model
    test_model.eval()
    with torch.no_grad():
        test_out = test_model(masked_tokens, mask)
        print(f"Test model output shape: {test_out.shape}")
        print(f"Test model output range: [{test_out.min():.4f}, {test_out.max():.4f}]")

# Test 7: Direct embedding comparison
print("\n9. Direct embedding weight comparison...")
print("-" * 60)

# Check if embeddings match
ckpt = torch.load("../models/vampnet/coarse.pth", map_location='cpu')
if 'codec' in ckpt:
    codec_state = ckpt['codec']
    for i in range(4):  # First 4 codebooks
        key = f'quantizer.quantizers.{i}.codebook.weight'
        if key in codec_state:
            codec_emb = codec_state[key]
            print(f"Codebook {i} from checkpoint: shape {codec_emb.shape}, "
                  f"range [{codec_emb.min():.4f}, {codec_emb.max():.4f}]")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("\nKey findings:")
print("1. VampNet output is in different range than ONNX")
print("2. Check if weight transfer is correctly implemented")
print("3. Verify embedding dimensions match (latent_dim=8)")
print("4. Ensure positional encoding is transferred")
print("5. Check classifier weight shapes and transfer")