#!/usr/bin/env python3
"""
Debug position bias handling in V9.
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

# Create V9 model with just 2 layers
model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=2  # Just 2 layers to see the transition
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

print("Debugging position bias...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Get embeddings
    v9_emb = model.embedding(latents)
    
    # Apply layers manually to track position bias
    x = v9_emb
    position_bias = None
    
    print(f"Initial x: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Layer 0 (relative attention)
    layer0 = model.layers[0]
    x_norm = layer0['norm_1'](x)
    print(f"\nLayer 0 after norm: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
    
    attn_out, position_bias = layer0['self_attn'](x_norm, x_norm, x_norm, None, position_bias)
    print(f"Layer 0 attention output: mean={attn_out.mean():.4f}, std={attn_out.std():.4f}")
    print(f"Position bias shape: {position_bias.shape if position_bias is not None else 'None'}")
    if position_bias is not None:
        print(f"Position bias stats: mean={position_bias.mean():.4f}, std={position_bias.std():.4f}")
    
    x = x + attn_out
    print(f"After residual: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # FFN
    x_norm = layer0['norm_3'](x)
    x_norm = layer0['film_3'](x_norm, x_norm)
    ffn_out = layer0['feed_forward'](x_norm)
    x = x + ffn_out
    print(f"After layer 0 complete: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Layer 1 (standard attention)
    layer1 = model.layers[1]
    x_norm = layer1['norm_1'](x)
    print(f"\nLayer 1 after norm: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
    
    attn_out, _ = layer1['self_attn'](x_norm, x_norm, x_norm)
    print(f"Layer 1 attention output: mean={attn_out.mean():.4f}, std={attn_out.std():.4f}")
    
    x = x + attn_out
    print(f"After residual: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Check VampNet for comparison
    print("\n\nVampNet comparison:")
    from einops import rearrange
    vampnet_emb = vampnet.embedding(latents)
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    
    # Get VampNet's first layer attention
    vamp_layer0 = vampnet.transformer.layers[0]
    print(f"VampNet layer 0 attention type: {type(vamp_layer0.self_attn)}")
    print(f"Has relative_attention_bias: {hasattr(vamp_layer0.self_attn, 'relative_attention_bias')}")
    
    # Check our relative attention
    print(f"\nOur layer 0 attention type: {type(layer0['self_attn'])}")
    print(f"Has relative_attention_bias: {hasattr(layer0['self_attn'], 'relative_attention_bias')}")
    
    # Check if position bias is being used correctly
    if hasattr(layer0['self_attn'], 'relative_attention_bias'):
        rab_weight = layer0['self_attn'].relative_attention_bias.weight
        print(f"\nRelative attention bias weight shape: {rab_weight.shape}")
        print(f"Weight stats: mean={rab_weight.mean():.4f}, std={rab_weight.std():.4f}")
    
    # Compare full forward passes
    vampnet_out = vampnet(latents)
    v9_out = model(latents)
    
    print(f"\n\nFinal outputs:")
    print(f"VampNet: {vampnet_out.shape}, mean={vampnet_out.mean():.4f}, std={vampnet_out.std():.4f}")
    print(f"V9: {v9_out.shape}, mean={v9_out.mean():.4f}, std={v9_out.std():.4f}")