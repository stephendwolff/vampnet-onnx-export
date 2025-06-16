#!/usr/bin/env python3
"""
Simple test to compare VampNet and our attention outputs.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10, transfer_weights_v10

# Create minimal test
torch.manual_seed(42)

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V10 with just 1 layer for debugging
model = VampNetTransformerV10(n_layers=1)
transfer_weights_v10("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Simple input
codes = torch.randint(0, 1024, (1, 4, 5))  # Smaller sequence
masked_codes = codes.clone()
masked_codes[:, :, 3:] = 1024

print("Testing attention layer differences...")

with torch.no_grad():
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Get embeddings
    from einops import rearrange
    vampnet_emb = vampnet.embedding(latents)
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    
    v10_emb = model.embedding(latents)
    
    print(f"Embeddings match: {torch.allclose(vampnet_x, v10_emb, atol=1e-5)}")
    
    # Test just first layer attention
    vamp_layer0 = vampnet.transformer.layers[0]
    v10_layer0 = model.layers[0]
    
    # Norm
    vamp_norm = vamp_layer0.norm_1(vampnet_x)
    v10_norm = v10_layer0['norm_1'](v10_emb)
    print(f"Norms match: {torch.allclose(vamp_norm, v10_norm, atol=1e-5)}")
    
    # Attention forward
    print("\nTesting attention forward...")
    
    # VampNet attention
    vamp_q = vamp_layer0.self_attn.w_qs(vamp_norm)
    vamp_k = vamp_layer0.self_attn.w_ks(vamp_norm)
    vamp_v = vamp_layer0.self_attn.w_vs(vamp_norm)
    
    # Our attention
    v10_q = v10_layer0['self_attn'].w_qs(v10_norm)
    v10_k = v10_layer0['self_attn'].w_ks(v10_norm)
    v10_v = v10_layer0['self_attn'].w_vs(v10_norm)
    
    print(f"Q match: {torch.allclose(vamp_q, v10_q, atol=1e-5)}")
    print(f"K match: {torch.allclose(vamp_k, v10_k, atol=1e-5)}")
    print(f"V match: {torch.allclose(vamp_v, v10_v, atol=1e-5)}")
    
    # Full attention
    vamp_attn_out, vamp_bias = vamp_layer0.self_attn(
        vamp_norm, vamp_norm, vamp_norm, 
        position_bias=None, use_cache=False
    )
    
    v10_attn_out, v10_bias = v10_layer0['self_attn'](
        v10_norm, v10_norm, v10_norm,
        mask=None, position_bias=None
    )
    
    print(f"\nAttention output shapes:")
    print(f"VampNet: {vamp_attn_out.shape}")
    print(f"V10: {v10_attn_out.shape}")
    
    print(f"\nAttention outputs match: {torch.allclose(vamp_attn_out, v10_attn_out, atol=1e-4)}")
    if not torch.allclose(vamp_attn_out, v10_attn_out, atol=1e-4):
        diff = (vamp_attn_out - v10_attn_out).abs()
        print(f"Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
        
        # Check position bias
        print(f"\nPosition bias shapes:")
        print(f"VampNet: {vamp_bias.shape if vamp_bias is not None else 'None'}")
        print(f"V10: {v10_bias.shape if v10_bias is not None else 'None'}")
        
        if vamp_bias is not None and v10_bias is not None:
            bias_match = torch.allclose(vamp_bias, v10_bias, atol=1e-5)
            print(f"Position biases match: {bias_match}")
            if not bias_match:
                bias_diff = (vamp_bias - v10_bias).abs()
                print(f"Bias difference: mean={bias_diff.mean():.6f}, max={bias_diff.max():.6f}")