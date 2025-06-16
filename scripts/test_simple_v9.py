#!/usr/bin/env python3
"""
Test V9 model with a single layer to isolate the issue.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, transfer_weights_v9

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V9 model with just 1 layer
model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=1  # Just 1 layer!
)

transfer_weights_v9("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Test input
torch.manual_seed(42)
codes = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
masked_codes = codes.clone()
masked_codes[mask] = 1024

print("Testing V9 with single layer...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Test embeddings first
    vampnet_emb = vampnet.embedding(latents)
    v9_emb = model.embedding(latents)
    
    print(f"\n1. Embeddings:")
    print(f"   VampNet shape: {vampnet_emb.shape}")
    print(f"   V9 shape: {v9_emb.shape}")
    print(f"   Embedding difference: {(vampnet_emb.transpose(1,2) - v9_emb).abs().max():.6f}")
    
    # Apply single layer manually
    from einops import rearrange
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    v9_x = v9_emb
    
    print(f"\n2. Before layer 0:")
    print(f"   VampNet: mean={vampnet_x.mean():.4f}, std={vampnet_x.std():.4f}")
    print(f"   V9:      mean={v9_x.mean():.4f}, std={v9_x.std():.4f}")
    
    # VampNet layer 0 - use transformer directly
    x_mask = torch.ones(1, 10, dtype=torch.bool)
    vampnet_out_1layer = vampnet.transformer(x=vampnet_x.clone(), x_mask=x_mask)
    
    # Extract after first layer manually for V9
    vamp_layer = vampnet.transformer.layers[0]
    
    # V9 layer 0
    v9_layer = model.layers[0]
    v9_norm1 = v9_layer['norm_1'](v9_x)
    v9_attn_out, position_bias = v9_layer['self_attn'](
        v9_norm1, v9_norm1, v9_norm1, None, None
    )
    v9_x = v9_x + v9_attn_out
    
    print(f"\n3. After attention:")
    print(f"   VampNet: mean={vampnet_x.mean():.4f}, std={vampnet_x.std():.4f}")
    print(f"   V9:      mean={v9_x.mean():.4f}, std={v9_x.std():.4f}")
    print(f"   Difference: {(vampnet_x - v9_x).abs().max():.6f}")
    
    # Continue with FFN
    vamp_norm3 = vamp_layer.norm_3(vampnet_x)
    vamp_norm3 = vamp_layer.film_3(vamp_norm3, vamp_norm3)
    vamp_ffn_out = vamp_layer.feed_forward(vamp_norm3)
    vampnet_x = vampnet_x + vamp_ffn_out
    
    v9_norm3 = v9_layer['norm_3'](v9_x)
    v9_norm3 = v9_layer['film_3'](v9_norm3, v9_norm3)
    v9_ffn_out = v9_layer['feed_forward'](v9_norm3)
    v9_x = v9_x + v9_ffn_out
    
    print(f"\n4. After FFN:")
    print(f"   VampNet: mean={vampnet_x.mean():.4f}, std={vampnet_x.std():.4f}")
    print(f"   V9:      mean={v9_x.mean():.4f}, std={v9_x.std():.4f}")
    print(f"   Difference: {(vampnet_x - v9_x).abs().max():.6f}")
    
    # Check specific values
    print(f"\n5. Sample values at position [0, 0]:")
    print(f"   VampNet: {vampnet_x[0, 0, :5]}")
    print(f"   V9:      {v9_x[0, 0, :5]}")
    print(f"   Diff:    {(vampnet_x[0, 0, :5] - v9_x[0, 0, :5]).abs()}")
    
    # Final outputs
    print(f"\n6. Full forward pass:")
    vampnet_out = vampnet(latents)
    v9_out = model(latents)
    print(f"   VampNet output: {vampnet_out.shape}")
    print(f"   V9 output: {v9_out.shape}")