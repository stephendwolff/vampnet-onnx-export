#!/usr/bin/env python3
"""
Debug full forward pass comparing VampNet and V7 ONNX model step by step.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

# Load models
print("Loading models...")
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Simple test case
torch.manual_seed(42)
test_tokens = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
masked_tokens = test_tokens.clone()
masked_tokens[mask] = 1024

print(f"\nTest tokens shape: {test_tokens.shape}")
print(f"Masked positions: {mask.sum()}")

# Step 1: Check embeddings through VampNet's full pipeline
print("\n1. VampNet full embedding process...")
with torch.no_grad():
    # Get latents via from_codes
    z_latents = vampnet.embedding.from_codes(masked_tokens, codec)
    print(f"Latents shape: {z_latents.shape}")
    print(f"Latents stats - mean: {z_latents.mean():.4f}, std: {z_latents.std():.4f}")
    
    # Project latents
    z_embeddings = vampnet.embedding(z_latents)
    print(f"Embeddings shape: {z_embeddings.shape}")
    print(f"Embeddings stats - mean: {z_embeddings.mean():.4f}, std: {z_embeddings.std():.4f}")
    
    # Rearrange for transformer
    x = rearrange(z_embeddings, "b d n -> b n d")
    print(f"Rearranged shape: {x.shape}")
    
    # Create mask (VampNet expects [batch, seq_len] mask)
    x_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)
    print(f"x_mask shape: {x_mask.shape}")
    
    # Pass through transformer
    out = vampnet.transformer(x=x, x_mask=x_mask)
    print(f"Transformer output shape: {out.shape}")
    print(f"Transformer output stats - mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    # Rearrange back
    out = rearrange(out, "b n d -> b d n")
    
    # Classifier
    final_out = vampnet.classifier(out, None)
    print(f"Classifier output shape: {final_out.shape}")
    print(f"Classifier output stats - mean: {final_out.mean():.4f}, std: {final_out.std():.4f}")
    
    # Final rearrange
    final_out = rearrange(final_out, "b (p c) t -> b p (t c)", c=vampnet.n_predict_codebooks)
    print(f"Final output shape: {final_out.shape}")

# Step 2: Debug first transformer layer
print("\n2. Debugging first transformer layer...")
with torch.no_grad():
    layer = vampnet.transformer.layers[0]
    
    # Pre-norm
    x_norm = layer.norm_1(x)
    print(f"After norm_1: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
    
    # Self-attention
    attn_out, pos_bias = layer.self_attn(x_norm, x_norm, x_norm)
    print(f"After self_attn: mean={attn_out.mean():.4f}, std={attn_out.std():.4f}")
    print(f"Position bias shape: {pos_bias.shape if pos_bias is not None else 'None'}")
    
    # Residual
    x = x + attn_out
    print(f"After residual: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # FFN block
    x_norm = layer.norm_3(x)
    x_norm = layer.film_3(x_norm, x_norm)
    ffn_out = layer.feed_forward(x_norm)
    print(f"After FFN: mean={ffn_out.mean():.4f}, std={ffn_out.std():.4f}")

# Step 3: Check weight transfer
print("\n3. Checking weight values...")
# Check a few weights from first layer
print("VampNet first layer weights:")
print(f"  norm_1 weight mean: {vampnet.transformer.layers[0].norm_1.weight.mean():.4f}")
print(f"  self_attn.w_qs weight mean: {vampnet.transformer.layers[0].self_attn.w_qs.weight.mean():.4f}")
print(f"  feed_forward.w_1 weight mean: {vampnet.transformer.layers[0].feed_forward.w_1.weight.mean():.4f}")

# Step 4: Check ONNX model
print("\n4. Testing ONNX V7 model...")
onnx_session = ort.InferenceSession("onnx_models_fixed/coarse_v7_codec_fix.onnx")

onnx_out = onnx_session.run(None, {
    'codes': masked_tokens.numpy().astype(np.int64),
    'mask': mask.numpy()
})[0]

print(f"ONNX output shape: {onnx_out.shape}")
print(f"ONNX output stats - mean: {onnx_out.mean():.4f}, std: {onnx_out.std():.4f}")

# Compare specific positions
print("\n5. Comparing specific logit values...")
# Get first unmasked position
unmasked_pos = None
for i in range(10):
    if not mask[0, 0, i]:
        unmasked_pos = i
        break

if unmasked_pos is not None:
    print(f"\nComparing logits at unmasked position {unmasked_pos}:")
    vampnet_logit = final_out[0, :, unmasked_pos*4][:5]  # First 5 values
    onnx_logit = onnx_out[0, 0, unmasked_pos, :5]
    print(f"VampNet: {vampnet_logit}")
    print(f"ONNX: {onnx_logit}")
    print(f"Difference: {(vampnet_logit - onnx_logit).abs()}")