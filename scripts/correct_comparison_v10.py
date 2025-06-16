#!/usr/bin/env python3
"""
Correct comparison between VampNet and V10.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10, transfer_weights_v10
from einops import rearrange

# Deterministic
torch.manual_seed(42)

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V10
model = VampNetTransformerV10()
transfer_weights_v10("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Test
codes = torch.randint(0, 1024, (1, 4, 10))
masked_codes = codes.clone()
masked_codes[:, :, 5:] = 1024

print("Correct comparison between VampNet and V10...")

with torch.no_grad():
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Run both
    vampnet_out = vampnet(latents)
    v10_out = model(latents)
    
    print(f"\nVampNet output shape: {vampnet_out.shape}")
    print(f"V10 output shape: {v10_out.shape}")
    
    # Understanding VampNet's output
    # vampnet_out is [1, 1024, 40]
    # This is [batch, vocab_size, seq_len * n_codebooks]
    # Rearranging: "b p (t c) -> b c t p"
    vampnet_reshaped = rearrange(vampnet_out, "b p (t c) -> b c t p", c=4, t=10)
    print(f"VampNet reshaped: {vampnet_reshaped.shape}")  # Should be [1, 4, 10, 1024]
    
    # V10 output is already [1, 4, 10, 1025]
    v10_truncated = v10_out[:, :, :, :1024]  # Remove mask token
    
    # Now compare
    diff = (vampnet_reshaped - v10_truncated).abs()
    corr = np.corrcoef(vampnet_reshaped.flatten(), v10_truncated.flatten())[0,1]
    
    print(f"\nMean absolute difference: {diff.mean():.6f}")
    print(f"Max absolute difference: {diff.max():.6f}")
    print(f"Correlation: {corr:.4f}")
    
    # Check specific values
    print(f"\nSample values at position [0, 0, 0, :5]:")
    print(f"VampNet: {vampnet_reshaped[0, 0, 0, :5]}")
    print(f"V10:     {v10_truncated[0, 0, 0, :5]}")
    
    if corr > 0.99:
        print("\n✅ SUCCESS! Models match!")
    else:
        print("\n❌ Still not matching...")
        
        # Let's check intermediate values
        print("\n\nChecking intermediate values...")
        
        # Trace through both models step by step
        x_vamp = vampnet.embedding(latents)
        x_v10 = model.embedding(latents)
        
        print(f"Embeddings shapes: VampNet {x_vamp.shape}, V10 {x_v10.shape}")
        
        # VampNet rearranges before transformer
        x_vamp = rearrange(x_vamp, "b d n -> b n d")
        
        print(f"After rearrange: VampNet {x_vamp.shape}, V10 {x_v10.shape}")
        print(f"Embeddings match: {torch.allclose(x_vamp, x_v10, atol=1e-5)}")
        
        # Check transformer output
        x_mask = torch.ones(1, 10, dtype=torch.bool)
        vamp_trans_out = vampnet.transformer(x=x_vamp, x_mask=x_mask)
        
        # V10 transformer (manual)
        x = x_v10
        position_bias = None
        for layer in model.layers:
            x_norm = layer['norm_1'](x)
            attn_out, position_bias = layer['self_attn'](x_norm, x_norm, x_norm, None, position_bias)
            x = x + attn_out
            x_norm = layer['norm_3'](x)
            x_norm = layer['film_3'](x_norm, x_norm)
            ffn_out = layer['feed_forward'](x_norm)
            x = x + ffn_out
        x = model.final_norm(x)
        
        print(f"\nTransformer outputs: VampNet {vamp_trans_out.shape}, V10 {x.shape}")
        print(f"Transformer match: {torch.allclose(vamp_trans_out, x, atol=1e-4)}")
        
        trans_diff = (vamp_trans_out - x).abs()
        print(f"Transformer diff: mean={trans_diff.mean():.6f}, max={trans_diff.max():.6f}")