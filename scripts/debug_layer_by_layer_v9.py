#!/usr/bin/env python3
"""
Debug V9 model layer by layer to find where it diverges from VampNet.
"""

import torch
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, transfer_weights_v9

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V9 model with full layers
model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=20  # Full model
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

print("Layer-by-layer comparison...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Initial embeddings
    vampnet_emb = vampnet.embedding(latents)
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    
    v9_x = model.embedding(latents)
    
    print(f"Initial embeddings match: {torch.allclose(vampnet_x, v9_x, atol=1e-5)}")
    
    # Check each transformer layer
    position_bias_vampnet = None
    position_bias_v9 = None
    
    for i in range(min(5, model.n_heads)):  # Check first 5 layers
        print(f"\nLayer {i}:")
        
        # VampNet layer
        vamp_layer = vampnet.transformer.layers[i]
        
        # Pre-norm
        vampnet_norm = vamp_layer.norm_1(vampnet_x)
        v9_norm = model.layers[i]['norm_1'](v9_x)
        
        norm_diff = (vampnet_norm - v9_norm).abs().max()
        print(f"  Norm1 diff: {norm_diff:.6f}")
        
        # Self-attention
        if i == 0:
            # First layer with relative attention
            vampnet_attn, position_bias_vampnet = vamp_layer.self_attn(
                vampnet_norm, vampnet_norm, vampnet_norm, None, position_bias_vampnet
            )
            v9_attn, position_bias_v9 = model.layers[i]['self_attn'](
                v9_norm, v9_norm, v9_norm, None, position_bias_v9
            )
        else:
            vampnet_attn, _ = vamp_layer.self_attn(
                vampnet_norm, vampnet_norm, vampnet_norm
            )
            v9_attn, _ = model.layers[i]['self_attn'](
                v9_norm, v9_norm, v9_norm
            )
        
        attn_diff = (vampnet_attn - v9_attn).abs().max()
        print(f"  Attention diff: {attn_diff:.6f}")
        
        # Residual
        vampnet_x = vampnet_x + vampnet_attn
        v9_x = v9_x + v9_attn
        
        # FFN
        vampnet_norm3 = vamp_layer.norm_3(vampnet_x)
        v9_norm3 = model.layers[i]['norm_3'](v9_x)
        
        # FiLM
        vampnet_film = vamp_layer.film_3(vampnet_norm3, vampnet_norm3)
        v9_film = model.layers[i]['film_3'](v9_norm3, v9_norm3)
        
        film_diff = (vampnet_film - v9_film).abs().max()
        print(f"  FiLM diff: {film_diff:.6f}")
        
        # FFN
        vampnet_ffn = vamp_layer.feed_forward(vampnet_film)
        v9_ffn = model.layers[i]['feed_forward'](v9_film)
        
        ffn_diff = (vampnet_ffn - v9_ffn).abs().max()
        print(f"  FFN diff: {ffn_diff:.6f}")
        
        # Final residual
        vampnet_x = vampnet_x + vampnet_ffn
        v9_x = v9_x + v9_ffn
        
        layer_diff = (vampnet_x - v9_x).abs().max()
        print(f"  Total layer diff: {layer_diff:.6f}")
        
        if layer_diff > 0.01:
            print(f"  ⚠️ Significant difference at layer {i}!")
            
            # More detailed check
            print(f"  VampNet x: mean={vampnet_x.mean():.4f}, std={vampnet_x.std():.4f}")
            print(f"  V9 x: mean={v9_x.mean():.4f}, std={v9_x.std():.4f}")
            break

# Final comparison
print("\nFinal outputs:")
with torch.no_grad():
    vampnet_out = vampnet(latents)
    v9_out = model(latents)
    
    # Reshape for comparison
    vampnet_reshaped = vampnet_out.reshape(1, 4, 10, 1024)
    v9_truncated = v9_out[:, :, :, :1024]
    
    diff = (vampnet_reshaped - v9_truncated).abs()
    print(f"Mean difference: {diff.mean():.6f}")
    print(f"Max difference: {diff.max():.6f}")
    
    # Check specific values
    print("\nSample values at position [0,0,0]:")
    print(f"VampNet: {vampnet_reshaped[0,0,0,:5]}")
    print(f"V9: {v9_truncated[0,0,0,:5]}")
    
    # Check shapes throughout
    print("\nShape check:")
    print(f"VampNet output: {vampnet_out.shape}")
    print(f"V9 output: {v9_out.shape}")
    print(f"VampNet reshaped: {vampnet_reshaped.shape}")
    print(f"V9 truncated: {v9_truncated.shape}")