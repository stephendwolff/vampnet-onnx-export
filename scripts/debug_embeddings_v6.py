#!/usr/bin/env python3
"""
Debug embeddings in V6 model to understand the discrepancy.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Load models
print("Loading models...")
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet = VampNet.load("../models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("../models/vampnet/codec.pth"))
codec.eval()

onnx_session = ort.InferenceSession("../onnx_models_fixed/coarse_v6_proper.onnx")

# Create simple test input
print("\nCreating test input...")
torch.manual_seed(42)
test_tokens = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
masked_tokens = test_tokens.clone()
masked_tokens[mask] = 1024

print(f"Test tokens shape: {test_tokens.shape}")
print(f"Masked tokens: {masked_tokens}")

# Step 1: Test embeddings
print("\n1. Testing embeddings...")
with torch.no_grad():
    # VampNet embedding
    vampnet_latents = vampnet.embedding.from_codes(masked_tokens, codec)
    vampnet_embeddings = vampnet.embedding(vampnet_latents)
    print(f"VampNet latents shape: {vampnet_latents.shape}")
    print(f"VampNet embeddings shape: {vampnet_embeddings.shape}")
    print(f"VampNet embeddings mean: {vampnet_embeddings.mean():.4f}, std: {vampnet_embeddings.std():.4f}")
    
    # Let's check the raw latents
    print("\nRaw latent details:")
    for i in range(4):
        cb_latent = vampnet_latents[:, i*8:(i+1)*8, :]
        print(f"  Codebook {i} latent: mean={cb_latent.mean():.4f}, std={cb_latent.std():.4f}")

# Step 2: Check what ONNX embeddings produce
print("\n2. Checking ONNX embedding output...")
# We need to look at intermediate outputs - let's export a debug model
from scripts.export_vampnet_transformer_v6_proper import VampNetTransformerV6

debug_model = VampNetTransformerV6(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=1,  # Just 1 layer for debugging
    dropout=0.1,
)

# Load checkpoint and transfer embeddings only
ckpt = torch.load("../models/vampnet/coarse.pth", map_location='cpu')
if 'codec' in ckpt:
    codec_state = ckpt['codec']
    for i in range(4):
        key = f'quantizer.quantizers.{i}.codebook.weight'
        if key in codec_state:
            codec_emb = codec_state[key]
            debug_model.embedding.embeddings[i].weight.data[:1024] = codec_emb
            print(f"Transferred codebook {i} embeddings")

# Transfer special token
if hasattr(vampnet.embedding, 'special') and 'MASK' in vampnet.embedding.special:
    mask_emb = vampnet.embedding.special['MASK']
    for i in range(4):
        debug_model.embedding.embeddings[i].weight.data[1024] = mask_emb[i]
    print("Transferred MASK embeddings")

# Transfer projection
debug_model.embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data
debug_model.embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data

# Get embeddings from debug model
debug_model.eval()
with torch.no_grad():
    debug_embeddings = debug_model.embedding(masked_tokens)
    print(f"\nDebug model embeddings shape: {debug_embeddings.shape}")
    print(f"Debug model embeddings mean: {debug_embeddings.mean():.4f}, std: {debug_embeddings.std():.4f}")

# Compare embeddings
print("\n3. Comparing embeddings...")
# VampNet is [B, D, T], debug model is [B, T, D]
debug_embeddings_transposed = debug_embeddings.transpose(1, 2)
emb_diff = (vampnet_embeddings - debug_embeddings_transposed).abs()
print(f"Embedding difference - mean: {emb_diff.mean():.6f}, max: {emb_diff.max():.6f}")

# Check if issue is in the transformer layers
print("\n4. Testing single transformer layer...")
with torch.no_grad():
    # VampNet first layer
    x = vampnet_embeddings.transpose(1, 2)  # [B, T, D]
    x_norm = vampnet.transformer.layers[0].norm_1(x)
    attn_out, pos_bias = vampnet.transformer.layers[0].self_attn(x_norm, x_norm, x_norm)
    print(f"VampNet attention output: mean={attn_out.mean():.4f}, std={attn_out.std():.4f}")
    
    # Our model first layer
    x2 = debug_embeddings  # Already [B, T, D]
    x2_norm = debug_model.layers[0]['norm_1'](x2)
    attn_out2, pos_bias2 = debug_model.layers[0]['self_attn'](x2_norm, x2_norm, x2_norm)
    print(f"Debug model attention output: mean={attn_out2.mean():.4f}, std={attn_out2.std():.4f}")
    
    # Compare
    attn_diff = (attn_out - attn_out2).abs()
    print(f"Attention difference - mean: {attn_diff.mean():.6f}, max: {attn_diff.max():.6f}")

# Final check - run full forward pass
print("\n5. Full forward pass comparison...")
with torch.no_grad():
    vampnet_out = vampnet(vampnet_latents)
    print(f"VampNet full output: shape={vampnet_out.shape}, mean={vampnet_out.mean():.4f}")
    
    debug_out = debug_model(masked_tokens, mask)
    print(f"Debug model output: shape={debug_out.shape}, mean={debug_out.mean():.4f}")