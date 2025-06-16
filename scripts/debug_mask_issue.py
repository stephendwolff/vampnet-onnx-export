#!/usr/bin/env python3
"""
Debug if VampNet is using a mask that we're not applying.
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

print("Testing mask handling...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # VampNet forward with explicit mask
    vampnet_emb = vampnet.embedding(latents)
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    
    # Try different mask configurations
    print("\n1. With all-ones mask (no masking):")
    x_mask = torch.ones(1, 10, dtype=torch.bool)
    vampnet_out1 = vampnet.transformer(x=vampnet_x, x_mask=x_mask)
    print(f"   Output stats: mean={vampnet_out1.mean():.4f}, std={vampnet_out1.std():.4f}")
    
    print("\n2. With no mask (None):")
    vampnet_out2 = vampnet.transformer(x=vampnet_x, x_mask=None)
    print(f"   Output stats: mean={vampnet_out2.mean():.4f}, std={vampnet_out2.std():.4f}")
    
    print("\n3. With actual mask from codes:")
    # Create mask based on masked positions
    x_mask_actual = ~mask[0, 0]  # Invert because x_mask is for valid positions
    vampnet_out3 = vampnet.transformer(x=vampnet_x, x_mask=x_mask_actual)
    print(f"   Mask: {x_mask_actual}")
    print(f"   Output stats: mean={vampnet_out3.mean():.4f}, std={vampnet_out3.std():.4f}")
    
    # Compare outputs
    diff_1_2 = (vampnet_out1 - vampnet_out2).abs().max()
    diff_1_3 = (vampnet_out1 - vampnet_out3).abs().max()
    print(f"\n4. Differences:")
    print(f"   all-ones vs None: {diff_1_2:.6f}")
    print(f"   all-ones vs actual: {diff_1_3:.6f}")
    
    # Now test V9 without any mask
    print("\n5. V9 model (no mask):")
    v9_out = model(latents)
    print(f"   Output shape: {v9_out.shape}")
    
    # Compare specific positions
    print("\n6. Position-wise comparison:")
    vampnet_reshaped = vampnet_out1.reshape(1, 10, 1280)  # Use all-ones mask output
    v9_after_transformer = model.final_norm(
        model(latents)[0, 0, :, :1280].mean(dim=-1).unsqueeze(0).unsqueeze(0)
    )
    
    # Actually, let's trace through V9 properly
    x = model.embedding(latents)
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
    x = model.final_norm(x)
    
    print(f"   V9 after transformer: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Check if mask affects the explosion
    print("\n7. Checking if mask prevents explosion...")
    print(f"   VampNet uses x_mask in transformer: {vampnet.transformer.layers[0].self_attn.flash_attn}")
    
    # Check attention mask computation
    if hasattr(vampnet.transformer.layers[0].self_attn, 'flash_attn'):
        print(f"   Flash attention enabled: {vampnet.transformer.layers[0].self_attn.flash_attn}")