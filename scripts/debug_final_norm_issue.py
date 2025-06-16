#!/usr/bin/env python3
"""
Debug why the final norm output differs so much.
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

# Create V9 model
model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=20
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

print("Debugging final norm issue...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Check if VampNet has a final norm
    print("\n1. Checking VampNet transformer structure:")
    print(f"Has final norm: {hasattr(vampnet.transformer, 'norm')}")
    if hasattr(vampnet.transformer, 'norm'):
        print(f"Final norm type: {type(vampnet.transformer.norm)}")
        print(f"Final norm weight shape: {vampnet.transformer.norm.weight.shape}")
        print(f"Final norm weight stats: mean={vampnet.transformer.norm.weight.mean():.4f}, std={vampnet.transformer.norm.weight.std():.4f}")
    
    # Check our final norm
    print("\n2. Checking V9 final norm:")
    print(f"Final norm type: {type(model.final_norm)}")
    print(f"Final norm weight shape: {model.final_norm.weight.shape}")
    print(f"Final norm weight stats: mean={model.final_norm.weight.mean():.4f}, std={model.final_norm.weight.std():.4f}")
    
    # Compare weights
    if hasattr(vampnet.transformer, 'norm'):
        weight_diff = (model.final_norm.weight - vampnet.transformer.norm.weight).abs()
        print(f"\nWeight difference: mean={weight_diff.mean():.6f}, max={weight_diff.max():.6f}")
    
    # Get embeddings
    vampnet_emb = vampnet.embedding(latents)
    v9_emb = model.embedding(latents)
    
    # Pass through transformers manually
    from einops import rearrange
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    
    # VampNet transformer
    vampnet_out = vampnet.transformer(x=vampnet_x, x_mask=torch.ones(1, 10, dtype=torch.bool))
    print(f"\n3. Before final norm:")
    print(f"VampNet output stats: mean={vampnet_out.mean():.4f}, std={vampnet_out.std():.4f}")
    
    # V9 transformer (manual)
    x = v9_emb
    position_bias = None
    for i, layer in enumerate(model.layers):
        x_norm = layer['norm_1'](x)
        if i == 0:
            attn_out, position_bias = layer['self_attn'](x_norm, x_norm, x_norm, None, position_bias)
        else:
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = layer['norm_3'](x)
        x_norm = layer['film_3'](x_norm, x_norm)
        ffn_out = layer['feed_forward'](x_norm)
        x = x + ffn_out
    
    print(f"V9 before final norm stats: mean={x.mean():.4f}, std={x.std():.4f}")
    diff_before = (x - vampnet_out).abs()
    print(f"Difference before final norm: mean={diff_before.mean():.6f}, max={diff_before.max():.6f}")
    
    # Apply final norms
    if hasattr(vampnet.transformer, 'norm'):
        vampnet_after_norm = vampnet.transformer.norm(vampnet_out)
        print(f"\n4. After final norm:")
        print(f"VampNet after norm stats: mean={vampnet_after_norm.mean():.4f}, std={vampnet_after_norm.std():.4f}")
    else:
        vampnet_after_norm = vampnet_out
        print(f"\n4. VampNet has no final norm")
    
    v9_after_norm = model.final_norm(x)
    print(f"V9 after norm stats: mean={v9_after_norm.mean():.4f}, std={v9_after_norm.std():.4f}")
    
    diff_after = (v9_after_norm - vampnet_after_norm).abs()
    print(f"Difference after final norm: mean={diff_after.mean():.6f}, max={diff_after.max():.6f}")
    
    # Check a few values
    print(f"\n5. Sample values at position [0, 0]:")
    print(f"VampNet: {vampnet_after_norm[0, 0, :5]}")
    print(f"V9: {v9_after_norm[0, 0, :5]}")
    
    # Check if the issue is in the accumulation
    print(f"\n6. Checking accumulation effect:")
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"After norm range: [{v9_after_norm.min():.2f}, {v9_after_norm.max():.2f}]")