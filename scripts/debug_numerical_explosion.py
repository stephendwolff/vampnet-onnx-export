#!/usr/bin/env python3
"""
Debug where the numerical explosion happens.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, transfer_weights_v9
from einops import rearrange

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

print("Tracking numerical explosion through layers...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Get embeddings
    vampnet_emb = vampnet.embedding(latents)
    v9_emb = model.embedding(latents)
    
    # Track through layers
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    v9_x = v9_emb
    
    print(f"\nInitial:")
    print(f"VampNet: mean={vampnet_x.mean():.4f}, std={vampnet_x.std():.4f}, range=[{vampnet_x.min():.2f}, {vampnet_x.max():.2f}]")
    print(f"V9:      mean={v9_x.mean():.4f}, std={v9_x.std():.4f}, range=[{v9_x.min():.2f}, {v9_x.max():.2f}]")
    
    # Apply layers one by one with a different approach
    # Just check every 5 layers to see the progression
    layer_indices = [0, 5, 10, 15, 19]
    
    # Store intermediate values
    vampnet_states = [vampnet_x.clone()]
    v9_states = [v9_x.clone()]
    
    # Apply all VampNet layers
    vampnet_full_out = vampnet.transformer(x=vampnet_x, x_mask=torch.ones(1, 10, dtype=torch.bool))
    
    # Apply V9 layers manually
    position_bias = None
    for i, layer in enumerate(model.layers):
        x_norm = layer['norm_1'](v9_x)
        if i == 0:
            attn_out, position_bias = layer['self_attn'](x_norm, x_norm, x_norm, None, position_bias)
        else:
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
        v9_x = v9_x + attn_out
        x_norm = layer['norm_3'](v9_x)
        x_norm = layer['film_3'](x_norm, x_norm)
        ffn_out = layer['feed_forward'](x_norm)
        v9_x = v9_x + ffn_out
        
        if i in layer_indices:
            print(f"\nAfter layer {i}:")
            print(f"  V9: mean={v9_x.mean():.4f}, std={v9_x.std():.4f}, range=[{v9_x.min():.2f}, {v9_x.max():.2f}]")
            
            # Track growth rate
            if len(v9_states) > 0:
                growth = v9_x.std() / v9_states[-1].std()
                print(f"  Growth rate: {growth:.3f}x")
            v9_states.append(v9_x.clone())
    
    print(f"\nFinal comparison:")
    print(f"  VampNet final: mean={vampnet_full_out.mean():.4f}, std={vampnet_full_out.std():.4f}")
    print(f"  V9 final:      mean={v9_x.mean():.4f}, std={v9_x.std():.4f}")
    print(f"  Std ratio (V9/VampNet): {v9_x.std() / vampnet_full_out.std():.2f}x")